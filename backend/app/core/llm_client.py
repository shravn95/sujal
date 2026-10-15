"""LLM integration layer — NVIDIA Nemotron-3 Ultra 550B chart planning, streaming reasoning, narrative, and validation.

The LLM is constrained to choose chart types and fields from the real dataset schema.
It NEVER produces fabricated data values. Every number shown in the UI is computed locally
by DashboardBuilder from the user's cleaned dataframe.
"""
from __future__ import annotations

import json
import logging
import re
import time
from difflib import get_close_matches
from typing import Any, Dict, Generator, List, Optional

import openai
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

# Cap on how many representative rows are embedded in the dashboard-generation
# prompt. The profiler produces 50-100; the payload stays bounded regardless.
MAX_PROMPT_SAMPLE_ROWS = 100

# ---------------------------------------------------------------------------
# Supported Chart Types Catalog
# ---------------------------------------------------------------------------
ALLOWED_CHART_TYPES = {
    "Area Graph",
    "Line Graph",
    "Bar Chart",
    "Donut Chart",
    "Scatterplot",
    "Histogram",
}

_CHART_TYPE_ALIASES: Dict[str, str] = {
    "area": "Area Graph",
    "area chart": "Area Graph",
    "area graph": "Area Graph",
    "line": "Line Graph",
    "line chart": "Line Graph",
    "line graph": "Line Graph",
    "bar": "Bar Chart",
    "bar chart": "Bar Chart",
    "bar graph": "Bar Chart",
    "column": "Bar Chart",
    "column chart": "Bar Chart",
    "donut": "Donut Chart",
    "donut chart": "Donut Chart",
    "pie": "Donut Chart",
    "pie chart": "Donut Chart",
    "scatter": "Scatterplot",
    "scatterplot": "Scatterplot",
    "scatter plot": "Scatterplot",
    "histogram": "Histogram",
    "hist": "Histogram",
    "distribution": "Histogram",
}


class NemotronLLMClient:
    def __init__(self) -> None:
        self.api_key = settings.NVIDIA_API_KEY
        self.base_url = settings.NVIDIA_BASE_URL
        self.model = settings.NVIDIA_MODEL

        if self.api_key:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=90.0,
                max_retries=0,
            )
        else:
            self.client = None

    def _call_with_retry(self, fn, max_attempts: int = 3, initial_delay: float = 1.5):
        """Execute an API call with automatic backoff on rate limits."""
        for attempt in range(max_attempts):
            try:
                return fn()
            except openai.RateLimitError:
                if attempt == max_attempts - 1:
                    raise
                wait_time = initial_delay * (2 ** attempt)
                logger.warning("Rate limit encountered (429). Retrying in %.1fs (attempt %d/%d)...", wait_time, attempt + 1, max_attempts)
                time.sleep(wait_time)
            except Exception:
                raise

    # NOTE: KPI selection is now handled by local statistical engine in dataset_profiler.py
    # These methods are kept for backward compatibility but should not be called for new uploads
    def _statistical_kpi_fallback(self, kpi_candidates: List[Dict[str, Any]]) -> Optional[str]:
        """Deprecated: Use dataset_profiler._select_primary_kpi_local() instead."""
        if not kpi_candidates:
            return None

        scored = []
        for cand in kpi_candidates:
            stats = cand.get("statistics", {})
            var_norm = stats.get("variance_normalized", 0.0)
            cardinality = stats.get("cardinality_ratio", 0.0)
            null_ratio = stats.get("null_ratio", 1.0)
            entropy = stats.get("entropy", 0.0)

            score = var_norm * (1.0 - null_ratio) * cardinality * (1.0 + entropy)
            scored.append((cand["name"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0] if scored else None

    def _build_master_prompt(
        self,
        summary: Dict[str, Any],
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Build a rich, dataset-aware prompt for MiniMax-M3."""
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        primary_kpi = summary.get("primary_kpi")
        col_meta = summary.get("col_meta", {})
        row_count = summary.get("row_count", "unknown")
        quality = summary.get("quality_score", "unknown")
        samples = summary.get("sample_records", [])

        col_details: List[str] = []
        if isinstance(col_meta, dict) and col_meta:
            for col_name, col_info in col_meta.items():
                dtype = col_info.get("type", col_info.get("dtype", "?"))
                role = col_info.get("semantic_role", "unknown")
                missing_pct = col_info.get("missing_pct", 0)
                detail = f"  - {col_name} (type={dtype}, role={role}, missing={missing_pct}%"
                if "mean" in col_info:
                    detail += f", min={col_info.get('min', '?')}, max={col_info.get('max', '?')}, mean={col_info.get('mean', '?')}"
                if col_info.get("sample_values"):
                    detail += f", samples={col_info['sample_values'][:5]}"
                if col_info.get("unique_count") is not None:
                    detail += f", unique_values={col_info['unique_count']}"
                detail += ")"
                col_details.append(detail)

        if not col_details:
            for c in numeric:
                col_details.append(f"  - {c} (numeric)")
            for c in dates:
                col_details.append(f"  - {c} (date)")
            for c in categories:
                col_details.append(f"  - {c} (categorical)")

        schema_block = "\n".join(col_details) if col_details else "  (no column metadata available)"

        sample_rows_block = ""
        if samples:
            sample_rows_block = f"\nSample Data Records (Top 5 rows):\n{json.dumps(samples[:5], indent=2)}\n"

        conv_block = ""
        if conversation_context and len(conversation_context) > 0:
            recent = conversation_context[-5:]
            turns = "\n".join([f"  User: {t.get('query', '')}" for t in recent])
            conv_block = (
                "\n\nCONVERSATION HISTORY:\n"
                f"{turns}\n"
                "If the current query is a follow-up or refinement, build upon the previous context.\n"
            )

        prompt = f"""You are the Lead Data Scientist & Visualization Architect for the AutoInsights analytics platform.
Your task is to understand the user's dataset and question across ANY domain (Healthcare, IoT, HR, Finance, Operations, Science, Logistics, Education, etc.), then design a comprehensive, highly accurate interactive visual dashboard.

=== DATASET SCHEMA & PROFILE ===
Filename: {summary.get('filename', 'dataset')}
Total Rows: {row_count}
Data Quality Score: {quality}%
Primary Detected KPI: {primary_kpi or 'not detected'}

Numeric Columns: {json.dumps(numeric)}
Date Columns: {json.dumps(dates)}
Categorical Columns: {json.dumps(categories)}

Detailed Column Attributes:
{schema_block}
{sample_rows_block}
{conv_block}
=== CHART SELECTION RULES ===
- "Area Graph": Continuous time-series volume/trend (x=date, y=numeric metric)
- "Line Graph": Time-series trend or multi-period comparison (x=date, y=numeric metric)
- "Bar Chart": Categorical ranking, comparison, or top/bottom N (x=category, y=numeric metric)
- "Donut Chart": Proportional market/category share (x=category, y=numeric metric)
- "Scatterplot": Correlation and bivariate relationships (x=numeric1, y=numeric2)
- "Histogram": Statistical spread and frequency distribution (y=numeric metric)

=== QUERY TYPE DETECTION ===
- Trend queries: Use Area Graph or Line Graph over dates.
- Categorical / Breakdown queries: Use Bar Chart or Donut Chart.
- Correlation queries: Use Scatterplot with two numeric measures.
- Distribution queries: Use Histogram.
- Overview queries: Generate a diverse mix of 4-5 charts.

=== AGGREGATION & UNIT INTELLIGENCE ===
- For rates, ratings, satisfaction, temperatures, heart rates, conversion percentages, latencies, averages, scores, indexes: use aggregation "mean" or "median".
- For counts, inventory volumes, sales, revenue, totals, quantity, units, items: use aggregation "sum" or "count".
- For identifiers (patient IDs, order IDs, user IDs): use aggregation "count" or "count_distinct".
- Use appropriate format_type ("currency", "percentage", "integer", "decimal", "duration", "scientific") and unit ("$", "%", "°C", "mg/dL", "ms", "users", etc.).

=== STRICT OUTPUT CONSTRAINTS ===
1. ONLY use column names that EXACTLY match the schema above. Cross-verify every column name against the schema before including it. NEVER invent or hallucinate column names.
2. For the user's query, return between 3 to 5 diverse, high-value, complementary charts.
3. First chart must directly address the primary question.
4. Ensure axes are assigned accurately:
   - x_axis: Must be a valid date or categorical column from the schema.
   - y_axis: Must be a valid numeric measure from the schema.
5. Provide an insightful "insight_tooltip" for every chart highlighting key analytical takeaways.
6. Return ONLY raw JSON. Do NOT wrap in markdown code fences. Do NOT include any preamble, explanation, or commentary before or after the JSON.

=== EDGE CASES ===
- If no date columns exist, NEVER select Area Graph or Line Graph; use Bar Chart, Donut Chart, or Histogram.
- If only one numeric column exists, NEVER select Scatterplot.
- If the user asks for a chart type incompatible with available columns, substitute the closest valid alternative.

=== EXAMPLES ===
Example 1:
Query: "Show sales trend over time"
{{
  "dashboard_title": "Sales Trend Analysis",
  "charts": [
    {{
      "title": "Sales Over Time",
      "type": "Area Graph",
      "x_axis": "Date",
      "y_axis": "Sales",
      "aggregation": "sum",
      "format_type": "currency",
      "unit": "$",
      "insight_tooltip": "Displays revenue trajectory."
    }}
  ]
}}

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object matching this schema:
{{
  "dashboard_title": "Descriptive Dashboard Title",
  "charts": [
    {{
      "title": "Chart Title",
      "type": "Area Graph | Line Graph | Bar Chart | Donut Chart | Scatterplot | Histogram",
      "x_axis": "ExactColumnName",
      "y_axis": "ExactColumnName",
      "secondary_dimension": "OptionalExactColumnName",
      "aggregation": "sum | mean | median | count | min | max",
      "format_type": "currency | percentage | integer | decimal | duration | scientific",
      "unit": "unit string",
      "sort": "descending | ascending",
      "limit": 10,
      "insight_tooltip": "Analytical explanation of what this chart demonstrates."
    }}
  ],
  "resolved_intent": {{
    "time_filter": null,
    "dimensions": ["Column1"],
    "metrics": ["Metric1"],
    "is_followup": false
  }}
}}

User Query: "{query}"
"""
        return prompt

    def _normalize_chart_type(self, raw_type: str) -> Optional[str]:
        """Map raw chart type to canonical allowed type, returning None for unknown types."""
        if not raw_type or not isinstance(raw_type, str):
            return None
        clean = raw_type.strip()
        if clean in ALLOWED_CHART_TYPES:
            return clean
        lowered = clean.lower()
        if lowered in _CHART_TYPE_ALIASES:
            return _CHART_TYPE_ALIASES[lowered]
        return None

    def _fuzzy_match_column(self, requested: str, available: List[str]) -> Optional[str]:
        """Match requested column name against available columns."""
        if not requested or not available:
            return None
        lower_map = {c.lower(): c for c in available}
        if requested.lower() in lower_map:
            return lower_map[requested.lower()]
        matches = get_close_matches(requested.lower(), [c.lower() for c in available], n=1, cutoff=0.6)
        if matches:
            return lower_map[matches[0]]
        return None

    def _validate_and_repair_spec(self, spec: Dict[str, Any], summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and repair a single chart specification, dropping invalid specs."""
        if not isinstance(spec, dict):
            return None

        numeric_columns = summary.get("numeric_columns", [])
        date_columns = summary.get("date_columns", [])
        categorical_columns = summary.get("categorical_columns", [])
        all_columns = numeric_columns + date_columns + categorical_columns

        raw_type = spec.get("type", "")
        canonical_type = self._normalize_chart_type(str(raw_type))
        if not canonical_type:
            return None
        spec["type"] = canonical_type

        # Validate x_axis
        x_axis = spec.get("x_axis", "")
        if x_axis and isinstance(x_axis, str) and x_axis.strip():
            matched_x = self._fuzzy_match_column(x_axis, all_columns)
            spec["x_axis"] = matched_x if matched_x else ""
        else:
            spec["x_axis"] = ""

        # Validate y_axis
        y_axis = spec.get("y_axis", "")
        if y_axis and isinstance(y_axis, str) and y_axis.strip():
            matched_y = self._fuzzy_match_column(y_axis, numeric_columns)
            if matched_y:
                spec["y_axis"] = matched_y
            else:
                primary = summary.get("primary_kpi")
                spec["y_axis"] = primary if primary and primary in numeric_columns else ""
        else:
            primary = summary.get("primary_kpi")
            spec["y_axis"] = primary if primary and primary in numeric_columns else ""

        if not spec["y_axis"] and not numeric_columns:
            return None

        # Validate secondary_dimension if provided
        sec_dim = spec.get("secondary_dimension")
        if sec_dim and isinstance(sec_dim, str) and sec_dim.strip():
            matched_sec = self._fuzzy_match_column(sec_dim, categorical_columns + date_columns)
            spec["secondary_dimension"] = matched_sec

        # Validate aggregation method
        agg_raw = str(spec.get("aggregation", "sum")).lower().strip()
        allowed_aggs = {"sum", "mean", "median", "count", "count_distinct", "min", "max", "std"}
        spec["aggregation"] = agg_raw if agg_raw in allowed_aggs else "sum"

        # Format type & unit
        if "format_type" in spec and spec["format_type"]:
            spec["format_type"] = str(spec["format_type"]).lower().strip()
        if "unit" in spec and spec["unit"]:
            spec["unit"] = str(spec["unit"]).strip()

        # Sort & limit
        sort_raw = str(spec.get("sort", "descending")).lower().strip()
        spec["sort"] = "ascending" if sort_raw in ["asc", "ascending"] else "descending"
        try:
            spec["limit"] = max(1, min(50, int(spec.get("limit", 12))))
        except Exception:
            spec["limit"] = 12

        # Title & Tooltip
        x = spec.get("x_axis", "")
        y = spec.get("y_axis", "Value")
        if not spec.get("title") or not isinstance(spec.get("title"), str):
            spec["title"] = f"{y} by {x}" if x else f"{y} Analysis"

        if not spec.get("insight_tooltip") or not isinstance(spec.get("insight_tooltip"), str):
            spec["insight_tooltip"] = f"Analyzes {y} across {x or 'overall data'} dimensions."

        return spec

    def stream_chart_plan(
        self,
        dataset_summary: Dict[str, Any],
        user_query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream Nemotron-3 Ultra 550B reasoning chunks and yield final parsed plan.

        Yields:
            {"type": "thinking", "content": chunk_text}
            {"type": "result", "charts": validated, "intent": resolved_intent, "title": dashboard_title}
        """
        if not self.client or not self.api_key:
            yield {"type": "result", "charts": [], "intent": None, "title": None}
            return

        # Resolve primary KPI if pending
        if dataset_summary.get("kpi_selection_pending") and dataset_summary.get("kpi_candidates"):
            primary_kpi, selection_method = self.select_primary_kpi(dataset_summary)
            if primary_kpi:
                dataset_summary["primary_kpi"] = primary_kpi
                dataset_summary["kpi_selection_pending"] = False
                dataset_summary["kpi_selection_method"] = selection_method

        prompt = self._build_master_prompt(dataset_summary, user_query, conversation_context)

        full_content = ""
        full_reasoning = ""

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.35,
                top_p=0.95,
                max_tokens=16384,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    full_reasoning += reasoning
                    yield {"type": "thinking", "content": reasoning}
                if delta.content is not None:
                    full_content += delta.content

            # Parse JSON from content
            clean_content = full_content.strip()
            if "```" in clean_content:
                clean_content = clean_content.replace("```json", "").replace("```", "").strip()

            json_match = re.search(r"\{[\s\S]*\}", clean_content)
            if json_match:
                clean_content = json_match.group(0)

            parsed = json.loads(clean_content)
            dashboard_title = parsed.get("dashboard_title") if isinstance(parsed, dict) else None
            charts_raw = parsed.get("charts", []) if isinstance(parsed, dict) else []
            resolved_intent = parsed.get("resolved_intent") if isinstance(parsed, dict) else None

            validated: List[Dict[str, Any]] = []
            if isinstance(charts_raw, list):
                for spec in charts_raw:
                    repaired = self._validate_and_repair_spec(spec, dataset_summary)
                    if repaired:
                        validated.append(repaired)

            logger.info(
                "Nemotron-3 Ultra plan: %d charts validated, reasoning_len=%d",
                len(validated),
                len(full_reasoning),
            )
            yield {"type": "result", "charts": validated, "intent": resolved_intent, "title": dashboard_title}

        except openai.RateLimitError as e:
            logger.warning("NVIDIA API rate limit reached (429): %s", e)
            yield {"type": "thinking", "content": "⚠️ Model API quota/rate-limit hit on NVIDIA gateway. Activating high-precision local heuristic analytics engine...\n"}
            yield {"type": "result", "charts": [], "intent": None, "title": None}
        except Exception as e:
            logger.warning("LLM stream exception: %s", e)
            yield {"type": "thinking", "content": f"⚡ Note: LLM response unavailable ({type(e).__name__}). Using local verified analytics engine.\n"}
            yield {"type": "result", "charts": [], "intent": None, "title": None}

    def generate_chart_plan(
        self,
        dataset_summary: Dict[str, Any],
        user_query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous wrapper for chart plan generation."""
        charts = []
        for event in self.stream_chart_plan(dataset_summary, user_query, conversation_context):
            if event.get("type") == "result":
                charts = event.get("charts", [])
        return charts

    def generate_narrative_report(
        self,
        summary: Dict[str, Any],
        kpis: Dict[str, Any],
        charts: List[Dict[str, Any]],
        cleaning_report: Dict[str, Any],
        query: str,
    ) -> Tuple[Optional[str], str]:
        """Generate a rich, executive narrative analysis report narrating REAL pre-computed numbers from rendered charts."""
        if not self.client or not self.api_key:
            return None, "fallback"

        # Extract rendered chart data for grounding
        from app.core.dashboard_builder import extract_rendered_chart_data
        rendered_charts = extract_rendered_chart_data(charts)

        chart_blocks = []
        for chart in rendered_charts:
            title = chart.get("chart_title", "Chart")
            ctype = chart.get("chart_type", "")
            x = chart.get("x_axis", "")
            y = chart.get("y_axis", "")
            agg = chart.get("aggregation", "sum")
            data_points = chart.get("data_points", [])
            
            if not data_points:
                continue
                
            # Build detailed data description
            data_lines = []
            for dp in data_points:
                name = dp.get("name", "?")
                val = dp.get("value", 0)
                pct = dp.get("percentage")
                fmt = dp.get("formatted_value", f"{val:,.2f}")
                
                if pct is not None:
                    data_lines.append(f"  - {name}: {fmt} ({pct}%)")
                else:
                    data_lines.append(f"  - {name}: {fmt}")
            
            data_block = "\n".join(data_lines) if data_lines else "  (no data)"
            
            chart_blocks.append(
                f"### {title} ({ctype})\n"
                f"**X-Axis:** {x or 'N/A'} | **Y-Axis (Metric):** {y} | **Aggregation:** {agg}\n"
                f"**Data Points:**\n{data_block}\n"
                f"**Chart Insight:** {chart.get('insight_tooltip', 'No insight provided.')}"
            )

        chart_block = "\n\n".join(chart_blocks) if chart_blocks else "  (no computed chart data)"

        primary_kpi = kpis.get("primary_kpi", summary.get("primary_kpi", "Primary Metric"))
        # Use formatted_value for display (e.g., "51.9" or "$2,090")
        kpi_formatted = kpis.get("formatted_value", "N/A")
        kpi_value = kpis.get("value", 0)
        row_count = summary.get("row_count", 0)
        cleaned_rows = cleaning_report.get("cleaned_rows", row_count) if cleaning_report else row_count
        quality = summary.get("quality_score", 100)

        prompt = f"""You are a Principal Data Analyst writing an executive business intelligence report for decision-makers.
Narrate and interpret the REAL, pre-computed numbers from the ACTIVE CHARTS provided below.

CRITICAL RULES:
- Use ONLY the provided chart data, metrics, and facts. NEVER invent ungrounded data.
- Every finding, section, and recommendation MUST cite concrete numbers from the rendered charts.
- Reference specific data points: peak periods, category totals, percentages, category names.
- Ban generic pipeline statements (e.g., "Cleansed X rows", "Integrity score is Y%"). Every insight must explain WHAT THE ACTIVE CHART NUMBERS MEAN.
- Structure your response cleanly using GitHub Markdown:
  ### 1. Executive Summary & Overview
  ### 2. Key Performance Indicators & Core Trends
  ### 3. Dimensional & Category Highlights
  ### 4. Risk Factors, Outliers & Data Integrity
  ### 5. Strategic Recommendations & Next Steps

User Question: "{query}"

=== PRE-COMPUTED DATASET FACTS ===
- Source Records: {row_count:,} (Cleaned to {cleaned_rows:,} records)
- Data Quality Score: {quality}%
- Primary KPI: {primary_kpi} = {kpi_formatted}
- Numeric Columns: {summary.get('numeric_columns', [])}
- Date Columns: {summary.get('date_columns', [])}
- Categorical Dimensions: {summary.get('categorical_columns', [])}

=== ACTIVE RENDERED CHARTS (GROUND TRUTH) ===
{chart_block}

Generate the detailed analytical narrative now. Every claim must reference specific chart data.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.92,
                max_tokens=3072,
            )
            report_text = (response.choices[0].message.content or "").strip()
            if report_text:
                return report_text, "llm"
            return None, "fallback"
        except Exception as e:
            logger.warning("Narrative report generation failed: %s", e)
            return None, "fallback"

    def generate_initial_insights(
        self,
        summary: Dict[str, Any],
        cleaning_report: Dict[str, Any],
    ) -> Tuple[List[str], str]:
        """Generate 3 to 4 sharp, actionable initial insight bullets upon dataset ingestion."""
        if not self.client or not self.api_key:
            return [], "fallback"

        primary = summary.get("primary_kpi", "Primary Metric")
        row_count = summary.get("row_count", 0)
        col_count = summary.get("column_count", 0)
        quality = summary.get("quality_score", 100)
        cleaned_rows = cleaning_report.get("cleaned_rows", row_count) if cleaning_report else row_count
        dups = cleaning_report.get("duplicates_removed", 0) if cleaning_report else 0

        prompt = f"""You are an expert AI Data Scientist. A user just uploaded a dataset.
Write 3 concise, high-value bullet points summarizing the structure, primary metric, and analytical readiness.

RULES:
- Use only facts from below.
- Each bullet point must begin with '• ' and start with an active verb.
- Exactly 3 bullets. No preamble.

Facts:
- Rows: {row_count:,} raw, {cleaned_rows:,} cleaned ({dups:,} duplicates removed).
- Columns: {col_count} total (Numeric: {summary.get('numeric_columns', [])}, Dates: {summary.get('date_columns', [])}, Categories: {summary.get('categorical_columns', [])}).
- Primary KPI: {primary}
- Quality Score: {quality}%
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                top_p=0.92,
                max_tokens=512,
            )
            raw = (response.choices[0].message.content or "").strip()
            bullets = []
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                    bullets.append(line.lstrip("•-* ").strip())
                elif line and line[0].isdigit() and "." in line[:3]:
                    bullets.append(line.split(".", 1)[-1].strip())
                elif line:
                    bullets.append(line)
            if bullets:
                return bullets[:4], "llm"
            return [], "fallback"
        except Exception as e:
            logger.warning("Initial insights generation failed: %s", e)
            return [], "fallback"

    def generate_preprocessing_action(
        self,
        user_command: str,
        columns: List[str],
        column_types: Dict[str, str],
    ) -> Dict[str, Any]:
        """Convert a natural-language data transformation command into a structured preprocessing action."""
        if not self.client or not self.api_key:
            raise ValueError("LLM client not configured.")

        cols_desc = ", ".join([f"{c} ({column_types.get(c, 'unknown')})" for c in columns])
        action_schema = (
            "Allowed actions (choose exactly one):\n"
            "1. drop_column: params {\"column\": \"<existing column>\"}\n"
            "2. rename_column: params {\"column\": \"<existing column>\", \"new_name\": \"<new name>\"}\n"
            "3. fill_missing: params {\"column\": \"<existing column>\", \"strategy\": \"mean|median|mode|constant|zero|forward_fill\", \"value\": <optional>}\n"
            "4. change_type: params {\"column\": \"<existing column>\", \"target_type\": \"int|float|string|datetime\"}\n"
            "5. add_column: params {\"new_column\": \"<new column name>\", \"expression\": \"<arithmetic expression e.g. Sales - Profit>\"}\n"
            "6. filter_rows: params {\"column\": \"<existing column>\", \"operator\": \"==|!=|>|<|>=|<=\", \"value\": <value>}\n"
        )

        prompt = (
            "You are an automated data transformation agent. Convert the user command into ONE allowed action.\n"
            "Use ONLY column names from the provided schema. Return valid JSON only with keys: type, params, explanation.\n\n"
            f"Available Columns: {cols_desc}\n\n"
            f"{action_schema}\n"
            f"User command: {user_command}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.92,
                max_tokens=512,
            )
            content = (response.choices[0].message.content or "").strip()
            if "```" in content:
                content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            if not isinstance(parsed, dict) or "type" not in parsed:
                raise ValueError("Invalid JSON response from LLM.")
            return parsed
        except Exception as e:
            raise ValueError(f"Preprocessing action interpretation failed: {e}")

    @staticmethod
    def _format_profile_lines(profile: Dict[str, Any]) -> List[str]:
        """Render the enriched (pre-masked) profile columns as compact prompt lines."""
        lines: List[str] = []
        for col in profile.get("columns", []):
            if not isinstance(col, dict):
                continue
            name = col.get("name")
            dtype = col.get("dtype", "?")
            role = col.get("semantic_role", "unknown")
            missing = col.get("missing_pct", 0)
            detail = f"  - {name} (type={dtype}, role={role}, missing={missing}%"
            if "mean" in col:
                detail += (
                    f", min={col.get('min')}, max={col.get('max')}, mean={col.get('mean')}"
                    f", median={col.get('median')}, std={col.get('std')}, q1={col.get('q1')}, q3={col.get('q3')}"
                )
            if col.get("date_min") is not None or col.get("date_max") is not None:
                detail += f", range={col.get('date_min')} to {col.get('date_max')}"
            if col.get("samples"):
                detail += f", samples={col.get('samples')}"
            if col.get("unique_count") is not None and "mean" not in col:
                detail += f", unique_values={col.get('unique_count')}"
            detail += ")"
            lines.append(detail)

            top_values = col.get("top_values") or []
            distribution = ", ".join(
                f"{tv.get('value')} ({tv.get('count')}, {tv.get('pct')}%)"
                for tv in top_values if isinstance(tv, dict)
            )
            if distribution:
                lines.append(f"      distribution: {distribution}")
        return lines

    @staticmethod
    def _format_kpi_candidates(kpis: List[Dict[str, Any]]) -> str:
        """Render locally-scored KPI candidates as a compact prompt block."""
        rows = []
        for kpi in kpis or []:
            if not isinstance(kpi, dict) or not kpi.get("name"):
                continue
            rows.append(
                f"  - {kpi.get('name')} (aggregation={kpi.get('aggregation')}, "
                f"format={kpi.get('format_type')}, unit={kpi.get('unit')}, score={kpi.get('score')})"
            )
        if not rows:
            return ""
        return "KPI Candidates (locally scored):\n" + "\n".join(rows) + "\n"

    def generate_initial_dataset_architecture(
        self,
        summary: Dict[str, Any],
        sample_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Direct MiniMax-M3 to autonomously analyze the uploaded dataset,
        deduce its domain (Healthcare, IoT, HR, Finance, Operations, Science, etc.),
        select primary & secondary KPIs (with appropriate aggregation & format),
        and architect 4 to 6 domain-optimized visualization specifications.
        """
        if not self.client or not self.api_key:
            return None, "fallback"

        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        col_meta = summary.get("col_meta", {})
        llm_profile = summary.get("llm_profile") if isinstance(summary.get("llm_profile"), dict) else None

        # Prefer the enriched, PII-masked profile + representative sample produced
        # by the profiler. Fall back to the legacy col_meta path so direct callers
        # with hand-built summaries keep working unchanged.
        if llm_profile and llm_profile.get("columns"):
            schema_lines = self._format_profile_lines(llm_profile)
            kpi_block = self._format_kpi_candidates(llm_profile.get("kpi_candidates", []))
            samples = (
                summary.get("representative_sample")
                or sample_records
                or summary.get("sample_records", [])
            )
            sample_label = "representative"
        else:
            schema_lines = []
            if col_meta and isinstance(col_meta, dict):
                for name, meta in col_meta.items():
                    t = meta.get("type", "unknown")
                    r = meta.get("semantic_role", "unknown")
                    stat = ""
                    if "mean" in meta:
                        stat = f", min={meta.get('min')}, max={meta.get('max')}, mean={meta.get('mean')}"
                    if meta.get("sample_values"):
                        stat += f", samples={meta.get('sample_values')[:4]}"
                    schema_lines.append(f"  - {name} ({t}, role={r}{stat})")
            samples = sample_records or summary.get("sample_records", [])
            kpi_block = ""
            sample_label = "sample"

        schema_block = chr(10).join(schema_lines) if schema_lines else "  (standard profile)"

        sample_rows_str = ""
        if samples:
            shown = samples[:MAX_PROMPT_SAMPLE_ROWS]
            # Compact separators keep the payload small for wide datasets.
            sample_rows_str = (
                f"\nSample Records ({len(shown)} {sample_label} rows):\n"
                f"{json.dumps(shown, separators=(',', ':'), default=str)}\n"
            )

        prompt = f"""You are the Principal AI Data Architect.
A user has uploaded a new dataset for automated analytics and visualization.

=== DATASET PROFILE ===
Filename: {summary.get('filename', 'dataset')}
Rows: {summary.get('row_count', 'unknown')}
Columns: {summary.get('column_count', len(numeric) + len(dates) + len(categories))}
Numeric: {json.dumps(numeric)}
Dates: {json.dumps(dates)}
Categories: {json.dumps(categories)}

Schema Details:
{schema_block}
{kpi_block}{sample_rows_str}

=== YOUR MISSION ===
1. Deduce the exact operational domain (e.g. "Clinical Healthcare", "Industrial IoT Sensors", "HR & Workforce Intelligence", "E-Commerce", "Financial Risk", "Academic Education", "Logistics & Supply Chain", etc.).
2. Select the Primary and Secondary KPIs for this domain:
   - Choose appropriate aggregation method:
     * Use "mean" or "median" for ratings, satisfaction scores, temperatures, heart rates, conversion rates, percentages, latencies.
     * Use "sum" for sales volumes, revenue, expenses, units sold, total steps, item counts.
     * Use "count" or "count_distinct" for patient/user/ticket IDs.
   - Choose appropriate format_type ("currency", "percentage", "integer", "decimal", "duration", "scientific") and unit ("$", "%", "°C", "mg/dL", "ms", "users", etc.).
3. Design a cohesive initial visual architecture of 4 to 6 distinct, complementary charts:
   - Use canonical chart types: "Area Graph", "Line Graph", "Bar Chart", "Donut Chart", "Scatterplot", "Histogram", "Stacked Bar Graph", "Multi-set Bar Chart", "Heatmap", "Treemap", "Radar Chart", "Funnel Chart", "Box Plot".
   - Assign exact column names from the dataset.
   - Set aggregation method, format_type, unit, and deep insight tooltip for each chart.
4. Formulate 3-4 strategic domain takeaways.

=== OUTPUT FORMAT ===
Return ONLY valid JSON matching this exact structure:
{{
  "domain": "Domain Name",
  "dashboard_title": "Executive Domain Dashboard Title",
  "primary_kpi": {{
    "column": "ExactColumnName",
    "aggregation": "sum | mean | median | count",
    "format_type": "currency | percentage | integer | decimal | duration | scientific",
    "unit": "$ | % | °C | mg/dL | ms | etc",
    "label": "Display Title for Primary KPI"
  }},
  "secondary_kpi": {{
    "column": "ExactColumnName",
    "aggregation": "sum | mean | median | count",
    "format_type": "currency | percentage | integer | decimal | duration | scientific",
    "unit": "unit string",
    "label": "Display Title for Secondary KPI"
  }},
  "charts": [
    {{
      "title": "Chart Title",
      "type": "Bar Chart | Line Graph | Area Graph | Donut Chart | Scatterplot | Histogram | Stacked Bar Graph | Multi-set Bar Chart | Heatmap | Treemap | Radar Chart | Funnel Chart | Box Plot",
      "x_axis": "ExactColumnName",
      "y_axis": "ExactColumnName",
      "secondary_dimension": "OptionalExactColumnName",
      "aggregation": "sum | mean | median | count",
      "format_type": "currency | percentage | integer | decimal | duration | scientific",
      "unit": "unit string",
      "sort": "descending | ascending",
      "limit": 10,
      "insight_tooltip": "Deep analytical description of what this chart highlights."
    }}
  ],
  "insights": [
    "Insight bullet 1",
    "Insight bullet 2",
    "Insight bullet 3"
  ]
}}
"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.35,
                top_p=0.95,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                stream=True,
            )
            full_content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content

            content = full_content.strip()
            if "```" in content:
                if "```json" in content:
                    content = content.split("```json", 1)[1].split("```", 1)[0].strip()
                else:
                    content = content.split("```", 1)[1].split("```", 1)[0].strip()

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                content = json_match.group(0)

            data = json.loads(content)

            # Primary KPI is already resolved by local statistical engine in dataset_profiler
            # during file upload. No LLM call needed for KPI selection.
            primary_kpi = summary.get("primary_kpi")
            selection_method = summary.get("kpi_selection_method", "statistical_local")
            
            if primary_kpi:
                # Override the LLM architecture's primary_kpi with the locally-selected one
                data["primary_kpi"] = {
                    "column": primary_kpi,
                    "aggregation": data.get("primary_kpi", {}).get("aggregation", "sum"),
                    "format_type": data.get("primary_kpi", {}).get("format_type", "decimal"),
                    "unit": data.get("primary_kpi", {}).get("unit", ""),
                    "label": data.get("primary_kpi", {}).get("label", primary_kpi.replace("_", " ").title())
                }
                # Ensure summary is updated for downstream use
                summary["primary_kpi"] = primary_kpi
                summary["kpi_selection_pending"] = False
                summary["kpi_selection_method"] = selection_method

            # Validate and repair charts
            valid_charts = []
            for c in data.get("charts", []):
                rep = self._validate_and_repair_spec(c, summary)
                if rep:
                    valid_charts.append(rep)

            data["charts"] = valid_charts
            return data, "llm"
        except Exception as e:
            logger.warning("Initial dataset architecture generation failed via LLM: %s", e)
            return None, "fallback"


nemotron_client = NemotronLLMClient()
