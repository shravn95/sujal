"""Application service for the upload -> clean -> analyse -> dashboard workflow.

Coordinates query understanding, streaming reasoning with Nemotron-3 Ultra 550B,
data filtering, and verified local chart materialization.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from uuid import uuid4

import pandas as pd

from app.core.dataset_profiler import dataset_profiler
from app.core.data_cleaner import data_cleaner
from app.core.dashboard_builder import dashboard_builder, extract_rendered_chart_data
from app.core.llm_client import nemotron_client
from app.agents.forecast_agent import forecasting_agent

LARGE_FILE_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB
PROFILE_SAMPLE_MAX_ROWS = 200_000
CHUNK_SIZE = 100_000

_log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_telemetry_path = _log_dir / "query_telemetry.jsonl"
_telemetry_handler = logging.FileHandler(str(_telemetry_path), encoding="utf-8")
_telemetry_handler.setLevel(logging.INFO)
_telemetry_handler.setFormatter(logging.Formatter("%(message)s"))

_telemetry_logger = logging.getLogger("query_telemetry")
_telemetry_logger.setLevel(logging.INFO)
_telemetry_logger.addHandler(_telemetry_handler)
_telemetry_logger.propagate = False


def _log_query_telemetry(dataset_id: str, query: str, intent: Any = None, plan_source: str = "fallback", chart_count: int = 0) -> None:
    try:
        _telemetry_logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "dataset_id": dataset_id,
            "query": query,
            "intent": intent,
            "plan_source": plan_source,
            "chart_count": chart_count,
            "charts_materialized": chart_count,
        }))
    except Exception:
        pass


# ===================================================================
#  INTENT PARSER
# ===================================================================
class IntentParser:
    """Parse a natural-language query against a dataset schema to extract structured intent."""

    _RELATIVE_TIME_PATTERNS = [
        (re.compile(r"\blast\s+(\d+)\s+(month|year|quarter|week|day)s?\b", re.I), "last_n"),
        (re.compile(r"\bpast\s+(\d+)\s+(month|year|quarter|week|day)s?\b", re.I), "last_n"),
        (re.compile(r"\blast\s+quarter\b", re.I), "last_quarter"),
        (re.compile(r"\bthis\s+quarter\b", re.I), "this_quarter"),
        (re.compile(r"\blast\s+year\b", re.I), "last_year"),
        (re.compile(r"\b(?:this|current|latest)\s+year\b", re.I), "this_year"),
        (re.compile(r"\b(?:year\s+to\s+date|ytd)\b", re.I), "ytd"),
        (re.compile(r"\b(?:month\s+over\s+month|mom)\b", re.I), "mom"),
        (re.compile(r"\b(?:year\s+over\s+year|yoy)\b", re.I), "yoy"),
        (re.compile(r"\b(?:in|for|of)\s+(20\d{2})\b", re.I), "specific_year"),
    ]

    _COMPARE_PATTERN = re.compile(
        r"\b(?:compare|vs\.?|versus|against|difference\s+between)\b", re.I
    )

    _TOP_BOTTOM = re.compile(r"\b(top|bottom|best|worst|highest|lowest)\s*(\d+)?\b", re.I)

    @classmethod
    def parse(cls, query: str, summary: Dict[str, Any]) -> Dict[str, Any]:
        lower = query.lower().strip()
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        all_cols = numeric + dates + categories

        intent: Dict[str, Any] = {
            "raw_query": query,
            "time_filter": cls._parse_time_filter(lower),
            "comparison": None,
            "filters": [],
            "metrics": [],
            "dimensions": [],
            "top_bottom": None,
            "is_overview": False,
        }

        # Metrics extraction
        for col in numeric:
            if col.lower() in lower or col.lower().replace("_", " ") in lower:
                intent["metrics"].append(col)

        # Dimensions extraction
        for col in categories:
            if col.lower() in lower or col.lower().replace("_", " ") in lower:
                intent["dimensions"].append(col)

        # Comparison detection
        if cls._COMPARE_PATTERN.search(lower):
            vs_match = re.search(r"(\w+)\s+(?:vs\.?|versus|against)\s+(\w+)", lower, re.I)
            if vs_match:
                a, b = vs_match.group(1), vs_match.group(2)
                matched_a = cls._match_column(a, all_cols)
                matched_b = cls._match_column(b, all_cols)
                if matched_a and matched_b:
                    intent["comparison"] = {"columns": [matched_a, matched_b]}

            if not intent["comparison"] and len(intent["metrics"]) >= 2:
                intent["comparison"] = {"columns": intent["metrics"][:2]}

        # Filters extraction
        intent["filters"] = cls._parse_filters(lower, categories, summary)

        # Top / Bottom
        top_match = cls._TOP_BOTTOM.search(lower)
        if top_match:
            direction = top_match.group(1).lower()
            n = int(top_match.group(2)) if top_match.group(2) else 5
            intent["top_bottom"] = {
                "direction": "top" if direction in ("top", "best", "highest") else "bottom",
                "n": n,
            }

        # Overview keywords
        overview_keywords = ("everything", "overview", "analyze", "analyse", "dashboard", "all charts", "full analysis", "show me", "insights")
        if any(kw in lower for kw in overview_keywords) and not intent["metrics"]:
            intent["is_overview"] = True

        if not intent["metrics"] and summary.get("primary_kpi"):
            intent["metrics"] = [summary["primary_kpi"]]

        return intent

    @classmethod
    def _parse_time_filter(cls, lower: str) -> Optional[Dict[str, Any]]:
        for pattern, filter_type in cls._RELATIVE_TIME_PATTERNS:
            match = pattern.search(lower)
            if match:
                if filter_type == "last_n":
                    return {"type": "last_n", "n": int(match.group(1)), "unit": match.group(2).lower()}
                elif filter_type == "specific_year":
                    return {"type": "year", "value": int(match.group(1))}
                elif filter_type in ("last_quarter", "this_quarter", "last_year", "this_year", "ytd", "yoy", "mom"):
                    return {"type": filter_type}
        return None

    @classmethod
    def _parse_filters(cls, lower: str, categories: List[str], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        filters: List[Dict[str, Any]] = []
        col_meta = summary.get("col_meta", {})
        for cat_col in categories:
            meta = col_meta.get(cat_col, {}) if isinstance(col_meta, dict) else {}
            sample_values = meta.get("sample_values", [])
            for val in sample_values:
                val_str = str(val).strip()
                if len(val_str) >= 2 and val_str.lower() in lower:
                    filters.append({"column": cat_col, "operator": "==", "value": val_str})
                    break
        return filters

    @classmethod
    def _match_column(cls, text: str, columns: List[str]) -> Optional[str]:
        text_lower = text.lower().strip()
        for col in columns:
            if col.lower() == text_lower or col.lower().replace("_", " ") == text_lower:
                return col
        return None

    @classmethod
    def apply_time_filter(cls, df: pd.DataFrame, date_col: str, time_filter: Any) -> pd.DataFrame:
        """Apply a time filter safely whether passed as string or dict."""
        if not time_filter or date_col not in df.columns:
            return df

        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if not parsed.notna().any():
            return df

        latest_date = parsed.max()

        if isinstance(time_filter, str):
            ftype = time_filter.lower()
            fval = None
        elif isinstance(time_filter, dict):
            ftype = str(time_filter.get("type", "")).lower()
            fval = time_filter.get("value")
        else:
            return df

        filtered = df

        if ftype in ("this_year", "current_year", "latest_year", "ytd"):
            latest_year = int(latest_date.year)
            filtered = df.loc[parsed.dt.year == latest_year]

        elif ftype in ("last_year", "previous_year"):
            latest_year = int(latest_date.year)
            filtered = df.loc[parsed.dt.year == (latest_year - 1)]

        elif (ftype == "year" and fval) or (isinstance(fval, int)):
            filtered = df.loc[parsed.dt.year == int(fval)]

        elif ftype == "last_n" and isinstance(time_filter, dict):
            n = time_filter.get("n", 1)
            unit = time_filter.get("unit", "month")
            if unit == "month":
                cutoff = latest_date - pd.DateOffset(months=n)
            elif unit == "year":
                cutoff = latest_date - pd.DateOffset(years=n)
            elif unit == "quarter":
                cutoff = latest_date - pd.DateOffset(months=n * 3)
            elif unit == "week":
                cutoff = latest_date - timedelta(weeks=n)
            else:
                cutoff = latest_date - timedelta(days=n)
            filtered = df.loc[parsed >= cutoff]

        elif ftype == "last_quarter":
            q = (latest_date.month - 1) // 3
            if q == 0:
                start = pd.Timestamp(year=latest_date.year - 1, month=10, day=1)
                end = pd.Timestamp(year=latest_date.year - 1, month=12, day=31)
            else:
                start = pd.Timestamp(year=latest_date.year, month=(q - 1) * 3 + 1, day=1)
                end = pd.Timestamp(year=latest_date.year, month=q * 3, day=1) + pd.offsets.MonthEnd(0)
            filtered = df.loc[(parsed >= start) & (parsed <= end)]

        elif ftype == "this_quarter":
            q = (latest_date.month - 1) // 3
            start = pd.Timestamp(year=latest_date.year, month=q * 3 + 1, day=1)
            filtered = df.loc[parsed >= start]

        # Guard against zero-row result
        return filtered.copy() if not filtered.empty else df

    @classmethod
    def apply_value_filters(cls, df: pd.DataFrame, filters: List[Dict[str, Any]]) -> pd.DataFrame:
        result = df
        for f in filters:
            col = f.get("column")
            op = f.get("operator", "==")
            val = f.get("value")
            if col not in result.columns:
                continue
            try:
                if op == "==":
                    subset = result.loc[result[col].astype(str).str.lower() == str(val).lower()]
                elif op == "!=":
                    subset = result.loc[result[col].astype(str).str.lower() != str(val).lower()]
                elif op == ">":
                    subset = result.loc[result[col] > float(val)]
                elif op == "<":
                    subset = result.loc[result[col] < float(val)]
                elif op == ">=":
                    subset = result.loc[result[col] >= float(val)]
                elif op == "<=":
                    subset = result.loc[result[col] <= float(val)]
                else:
                    subset = result
                if not subset.empty:
                    result = subset.copy()
            except Exception:
                continue
        return result


# ===================================================================
#  CONVERSATION MEMORY
# ===================================================================
class ConversationMemory:
    def __init__(self) -> None:
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add(self, dataset_id: str, query: str, intent: Dict[str, Any]) -> None:
        self._history[dataset_id].append({
            "query": query,
            "intent": intent,
            "timestamp": datetime.utcnow().isoformat(),
        })
        if len(self._history[dataset_id]) > 20:
            self._history[dataset_id] = self._history[dataset_id][-20:]

    def get_context(self, dataset_id: str) -> List[Dict[str, str]]:
        return [{"query": h["query"]} for h in self._history.get(dataset_id, [])]

    def get_last_intent(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        history = self._history.get(dataset_id, [])
        return history[-1]["intent"] if history else None

    def resolve_followup(self, dataset_id: str, current_intent: Dict[str, Any], query: str) -> Dict[str, Any]:
        last = self.get_last_intent(dataset_id)
        if not last:
            return current_intent

        lower = query.lower().strip()
        is_followup = (
            len(lower.split()) <= 6
            or lower.startswith(("now ", "and ", "also ", "but ", "what about", "how about", "show only", "filter by"))
            or any(word in lower for word in ("that", "those", "instead", "same", "break down", "by"))
        )

        if not is_followup:
            return current_intent

        merged = dict(current_intent)
        if not merged["metrics"] and last.get("metrics"):
            merged["metrics"] = last["metrics"]
        if not merged["time_filter"] and last.get("time_filter"):
            merged["time_filter"] = last["time_filter"]
        if not merged["dimensions"] and last.get("dimensions"):
            merged["dimensions"] = last["dimensions"]
        if not merged["filters"] and last.get("filters"):
            merged["filters"] = last["filters"]

        return merged

    def clear(self, dataset_id: str) -> None:
        self._history.pop(dataset_id, None)


# ===================================================================
#  MASTER ORCHESTRATOR
# ===================================================================
class MasterOrchestrator:
    def __init__(self) -> None:
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.conversation_memory = ConversationMemory()

    def process_file_and_generate_initial_dashboard(
        self,
        file_bytes: bytes,
        filename: str,
        sheet_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        available_sheets: List[str] = []
        target_sheet = None
        is_large_file = len(file_bytes) > LARGE_FILE_THRESHOLD_BYTES

        if suffix == "csv":
            if is_large_file:
                chunks = pd.read_csv(BytesIO(file_bytes), chunksize=CHUNK_SIZE)
                raw_df = pd.concat(chunks, ignore_index=True)
            else:
                raw_df = pd.read_csv(BytesIO(file_bytes))
        elif suffix in {"xlsx", "xls"}:
            excel_file = pd.ExcelFile(BytesIO(file_bytes))
            available_sheets = excel_file.sheet_names
            target_sheet = sheet_name if sheet_name in available_sheets else available_sheets[0]
            raw_df = pd.read_excel(excel_file, sheet_name=target_sheet)
        else:
            raise ValueError("Please upload a CSV or Excel (.xlsx/.xls) file.")

        if raw_df.empty:
            raise ValueError("The uploaded dataset contains no records.")

        cleaned_df, cleaning_report = data_cleaner.clean_dataset(raw_df)

        profile_df = cleaned_df
        was_sampled = False
        sample_size = len(cleaned_df)
        if is_large_file and len(cleaned_df) > PROFILE_SAMPLE_MAX_ROWS:
            profile_df = cleaned_df.sample(n=PROFILE_SAMPLE_MAX_ROWS, random_state=42)
            was_sampled = True
            sample_size = len(profile_df)

        summary = dataset_profiler.profile_csv(
            file_bytes,
            filename,
            profile_df,
            cleaning_report=cleaning_report,
            was_sampled=was_sampled,
            sample_size=sample_size,
            total_rows=len(cleaned_df),
        )
        dataset_id = str(uuid4())
        self.datasets[dataset_id] = {
            "df": cleaned_df,
            "summary": summary,
            "cleaning_report": cleaning_report,
            "sheet_name": target_sheet,
            "filename": filename,
        }

        # Primary KPI already resolved by local statistical engine during profiling
        primary_kpi = summary.get("primary_kpi")
        kpi_selection_details = summary.get("kpi_selection_details")
        
        # Use local engine's formatting hints if available
        if kpi_selection_details:
            aggregation = kpi_selection_details.get("aggregation", "sum")
            format_type = kpi_selection_details.get("format_type", "decimal")
            unit = kpi_selection_details.get("unit", "")
            display_name = kpi_selection_details.get("display_name", primary_kpi.replace('_', ' ').title())
        else:
            aggregation = "sum"
            format_type = "currency" if primary_kpi and any(kw in primary_kpi.lower() for kw in ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'spend', 'income', 'budget', 'margin']) else "decimal"
            unit = "$" if primary_kpi and any(kw in primary_kpi.lower() for kw in ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'spend', 'income', 'budget', 'margin']) else ""
            display_name = primary_kpi.replace('_', ' ').title() if primary_kpi else "Key Metric"

        # Generate deterministic initial dashboard using local profiling only (no LLM calls)
        kpi_spec = {
            "column": primary_kpi,
            "aggregation": aggregation,
            "format_type": format_type,
            "unit": unit,
        }
        dashboard_title = f"{display_name} Overview & Analytics"

        charts = []
        for index, spec in enumerate(dashboard_builder.default_plan(summary)):
            chart = dashboard_builder.materialize_chart(cleaned_df, summary, spec, f"initial-{index + 1}")
            if chart:
                charts.append(chart)

        kpis = dashboard_builder.make_kpis(cleaned_df, summary, kpi_spec)
        rendered_charts = extract_rendered_chart_data(charts)
        forecast = self._forecast_for_dataset(cleaned_df, summary)
        insights = self._initial_insights(summary, cleaning_report, charts, use_llm=False)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "dashboard_title": dashboard_title,
            "summary": summary,
            "cleaning_report": cleaning_report,
            "charts": charts,
            "rendered_charts": rendered_charts,
            "kpi_summary": kpis,
            "forecast": forecast,
            "ai_insights": insights,
            "available_sheets": available_sheets,
            "plan_source": "deterministic",
        }

    async def trigger_deep_llm_analysis(self, dataset_id: str) -> Dict[str, Any]:
        """Trigger deep LLM analysis asynchronously - called by Copilot after dashboard mounts.
        
        Generates domain-optimized dashboard using LLM for chart architecture.
        Primary KPI is already resolved by local statistical engine during upload.
        """
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            return {"error": "Dataset not found"}

        summary = dataset["summary"]
        cleaned_df = dataset["df"]
        cleaning_report = dataset["cleaning_report"]

        # Primary KPI already resolved by local statistical engine during profiling
        primary_kpi = summary.get("primary_kpi")
        kpi_selection_details = summary.get("kpi_selection_details")
        
        # Use local engine's formatting hints
        if kpi_selection_details:
            aggregation = kpi_selection_details.get("aggregation", "sum")
            format_type = kpi_selection_details.get("format_type", "decimal")
            unit = kpi_selection_details.get("unit", "")
            display_name = kpi_selection_details.get("display_name", primary_kpi.replace('_', ' ').title())
        else:
            aggregation = "sum"
            format_type = "currency" if primary_kpi and any(kw in primary_kpi.lower() for kw in ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'spend', 'income', 'budget', 'margin']) else "decimal"
            unit = "$" if primary_kpi and any(kw in primary_kpi.lower() for kw in ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'spend', 'income', 'budget', 'margin']) else ""
            display_name = primary_kpi.replace('_', ' ').title() if primary_kpi else "Key Metric"

        # Generate LLM architecture (for charts only, not KPI selection)
        ai_arch, arch_source = nemotron_client.generate_initial_dataset_architecture(summary, summary.get("sample_records"))

        charts = []
        kpi_spec = None
        insights = []
        dashboard_title = None

        if arch_source == "llm" and ai_arch:
            dashboard_title = ai_arch.get("dashboard_title")
            kpi_spec = ai_arch.get("primary_kpi")
            if ai_arch.get("secondary_kpi") and isinstance(kpi_spec, dict):
                kpi_spec["secondary_column"] = ai_arch["secondary_kpi"].get("column")
                kpi_spec["secondary_aggregation"] = ai_arch["secondary_kpi"].get("aggregation", "mean")

            raw_charts = ai_arch.get("charts", [])
            for index, spec in enumerate(raw_charts):
                chart = dashboard_builder.materialize_chart(cleaned_df, summary, spec, f"llm-{index + 1}")
                if chart:
                    charts.append(chart)

            insights = ai_arch.get("insights", [])

        # Fallback to deterministic if LLM failed
        if not charts:
            for index, spec in enumerate(dashboard_builder.default_plan(summary)):
                chart = dashboard_builder.materialize_chart(cleaned_df, summary, spec, f"fallback-{index + 1}")
                if chart:
                    charts.append(chart)

        if not insights:
            insights = self._initial_insights(summary, cleaning_report, charts, use_llm=(arch_source == "llm"))

        # Use fallback kpi_spec if LLM didn't provide one (use local engine's hints)
        if not kpi_spec:
            kpi_spec = {
                "column": primary_kpi,
                "aggregation": aggregation,
                "format_type": format_type,
                "unit": unit,
            }

        kpis = dashboard_builder.make_kpis(cleaned_df, summary, kpi_spec)
        rendered_charts = extract_rendered_chart_data(charts)
        forecast = self._forecast_for_dataset(cleaned_df, summary)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "dashboard_title": dashboard_title or f"{display_name} Overview & Analytics",
            "charts": charts,
            "rendered_charts": rendered_charts,
            "kpi_summary": kpis,
            "forecast": forecast,
            "ai_insights": insights,
            "plan_source": arch_source,
            "kpi_selection_method": summary.get("kpi_selection_method"),
        }

    def _initial_insights(self, summary: Dict[str, Any], cleaning: Dict[str, Any], charts: List[Dict[str, Any]], use_llm: bool = True) -> List[str]:
        if use_llm:
            llm_bullets, source = nemotron_client.generate_initial_insights(summary, cleaning)
            if source == "llm" and llm_bullets:
                return llm_bullets
        primary = summary.get("primary_kpi") or "primary metric"
        return [
            f"Successfully cleansed {cleaning.get('cleaned_rows', summary.get('row_count', 0)):,} rows across {summary.get('column_count', 0)} attributes.",
            f"Identified {primary} as the primary business metric with {len(charts)} initial interactive visuals.",
            f"Computed overall data integrity score of {summary.get('quality_score', 100)}%.",
        ]

    def _forecast_for_dataset(self, df: pd.DataFrame, summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        metric, dates = summary.get("primary_kpi"), summary.get("date_columns", [])
        if not metric or not dates or metric not in df.columns:
            return None
        series = dashboard_builder._period_series(df, dates[0], metric)
        if len(series) < 2:
            return None
        values = [point["value"] for point in series]
        dates_list = [point["name"] for point in series]
        return forecasting_agent.forecast_metric(values, dates_list, periods=4, metric_name=metric)

    def _frame_for_query(self, df: pd.DataFrame, summary: Dict[str, Any], intent: Dict[str, Any]) -> pd.DataFrame:
        result = df
        time_filter = intent.get("time_filter")
        dates = summary.get("date_columns", [])
        if time_filter and dates:
            result = IntentParser.apply_time_filter(result, dates[0], time_filter)

        filters = intent.get("filters", [])
        if filters:
            result = IntentParser.apply_value_filters(result, filters)

        return result if not result.empty else df

    def process_query_stream(self, dataset_id: str, query: str) -> Generator[Dict[str, Any], None, None]:
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            yield {"type": "error", "message": "Dataset session expired. Please re-upload your file."}
            return

        summary = dataset["summary"]

        # Step 1: Initial Intent Parse
        intent = IntentParser.parse(query, summary)
        intent = self.conversation_memory.resolve_followup(dataset_id, intent, query)
        self.conversation_memory.add(dataset_id, query, intent)

        # Step 2: Stream reasoning tokens from Nemotron-3 Ultra 550B
        yield {"type": "thinking", "content": "Analyzing dataset schema & planning optimized visual architecture with NVIDIA Nemotron-3 Ultra 550B...\n"}

        conversation_context = self.conversation_memory.get_context(dataset_id)
        llm_plan: List[Dict[str, Any]] = []
        llm_intent: Optional[Dict[str, Any]] = None
        custom_dashboard_title: Optional[str] = None
        plan_source = "fallback"

        try:
            for event in nemotron_client.stream_chart_plan(summary, query, conversation_context):
                if event.get("type") == "thinking":
                    yield {"type": "thinking", "content": event["content"]}
                elif event.get("type") == "result":
                    llm_plan = event.get("charts", [])
                    llm_intent = event.get("intent")
                    custom_dashboard_title = event.get("title")

            if llm_plan:
                plan_source = "llm"
        except Exception as exc:
            logger.warning("LLM chart plan stream failed: %s", exc)
            plan_source = "fallback"

        # Step 3: Merge intent
        if plan_source == "llm" and llm_intent and isinstance(llm_intent, dict):
            merged = dict(intent)
            if llm_intent.get("time_filter") is not None:
                merged["time_filter"] = llm_intent["time_filter"]
            if llm_intent.get("dimensions"):
                merged["dimensions"] = llm_intent["dimensions"]
            if llm_intent.get("metrics"):
                merged["metrics"] = llm_intent["metrics"]
            if llm_intent.get("top_bottom") is not None:
                merged["top_bottom"] = llm_intent["top_bottom"]
            intent = merged

        # Step 4: Apply filters to dataframe
        filtered_df = self._frame_for_query(dataset["df"], summary, intent)

        # Step 5: Materialize real charts
        plan = llm_plan or dashboard_builder.heuristic_plan(summary, query, intent)
        charts = []
        for index, spec in enumerate(plan):
            chart = dashboard_builder.materialize_chart(filtered_df, summary, spec, f"query-{index + 1}")
            if chart:
                charts.append(chart)

        if not charts:
            charts = [
                chart
                for i, spec in enumerate(dashboard_builder.default_plan(summary))
                if (chart := dashboard_builder.materialize_chart(filtered_df, summary, spec, f"fallback-{i + 1}"))
            ]

        # Step 6: Compute KPIs & Narrative Report
        kpis = dashboard_builder.make_kpis(filtered_df, summary)
        rendered_charts = extract_rendered_chart_data(charts)
        report, report_source = self._generate_report(query, summary, dataset["cleaning_report"], charts, kpis, rendered_charts)
        forecast = self._forecast_for_dataset(filtered_df, summary)

        title = custom_dashboard_title or f"{kpis['primary_kpi']} Analysis & Insights"

        payload = {
            "dashboard_title": title,
            "suggested_charts": charts,
            "kpi_summary": kpis,
            "rendered_charts": rendered_charts,
            "detailed_report": report,
            "forecast": forecast,
            "plan_source": plan_source,
            "report_source": report_source,
        }

        yield {"type": "payload", "data": payload}

    def _generate_report(
        self,
        query: str,
        summary: Dict[str, Any],
        cleaning: Dict[str, Any],
        charts: List[Dict[str, Any]],
        kpis: Dict[str, Any],
        rendered_charts: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        llm_report, source = nemotron_client.generate_narrative_report(
            summary=summary,
            kpis=kpis,
            charts=charts,
            cleaning_report=cleaning,
            query=query,
        )
        if source == "llm" and llm_report:
            return llm_report, "llm"

        # Enhanced fallback report with full structure
        findings = []
        for chart in rendered_charts:
            data_points = chart.get("data_points", [])
            if data_points:
                top = data_points[0]
                name = top.get("name", "?")
                val = top.get("formatted_value", top.get("value", "?"))
                findings.append(f"- **{chart['chart_title']}**: {name} leads with {val}.")
            else:
                findings.append(f"- **{chart['chart_title']}**: No data available.")
        findings_str = "\n".join(findings) if findings else "No chart data available."

        # Build dimensional highlights from rendered charts
        dim_highlights = []
        for chart in rendered_charts:
            data_points = chart.get("data_points", [])
            if len(data_points) >= 2:
                top = data_points[0]
                bottom = data_points[-1]
                dim_highlights.append(
                    f"- **{chart['chart_title']}**: {top.get('name','?')} ({top.get('formatted_value','?')}) "
                    f"vs {bottom.get('name','?')} ({bottom.get('formatted_value','?')}) — "
                    f"spread of {len(data_points)} categories."
                )

        dim_highlights_str = "\n".join(dim_highlights) if dim_highlights else "No dimensional data available."

        # Build risk/outlier observations
        risk_obs = []
        for chart in rendered_charts:
            data_points = chart.get("data_points", [])
            if len(data_points) >= 3:
                values = [dp.get("value", 0) for dp in data_points if isinstance(dp.get("value"), (int, float))]
                if values:
                    avg_v = sum(values) / len(values)
                    outliers = [dp for dp in data_points if isinstance(dp.get("value"), (int, float)) and dp["value"] > avg_v * 2]
                    if outliers:
                        risk_obs.append(f"- **{chart['chart_title']}**: {len(outliers)} outlier(s) exceed 2× mean (e.g., {outliers[0].get('name','?')}: {outliers[0].get('formatted_value','?')}).")

        risk_obs_str = "\n".join(risk_obs) if risk_obs else "No significant outliers detected in active charts."

        # Build recommendations
        recs = []
        if findings:
            recs.append("- Prioritize investigation of top-performing segments identified in charts.")
        if risk_obs:
            recs.append("- Investigate outlier segments for data quality issues or exceptional performance drivers.")
        recs.append("- Monitor trend charts for inflection points indicating market shifts.")
        recs.append("- Validate data quality scores below 90% before critical decisions.")

        fallback_report = f"""### 1. Executive Summary & Overview

**Question:** {query}

The dataset contains **{summary['row_count']:,}** records across **{summary['column_count']}** dimensions and measures. 
After automated data cleaning and quality validation, **{cleaning.get('cleaned_rows', summary['row_count']):,}** records were analyzed.
The detected primary KPI is **{kpis['primary_kpi']}** with a total aggregated volume of **{kpis.get('formatted_value', 'N/A')}**.
Data integrity score: **{summary.get('quality_score', 100):.1f}%**.

### 2. Key Performance Indicators & Core Trends

Primary KPI: **{kpis['primary_kpi']} = {kpis.get('formatted_value', 'N/A')}** ({kpis.get('aggregation', 'sum').upper()} aggregation).
{findings_str}

### 3. Dimensional & Category Highlights

{dim_highlights_str}

### 4. Risk Factors, Outliers & Data Integrity

{risk_obs_str}

Data Quality Breakdown:
- Completeness: {summary.get('quality_score_breakdown', {}).get('completeness', 0):.1f}%
- Consistency: {summary.get('quality_score_breakdown', {}).get('consistency', 0):.1f}%
- Type Confidence: {summary.get('quality_score_breakdown', {}).get('type_confidence', 0):.1f}%

### 5. Strategic Recommendations & Next Steps

{chr(10).join(recs)}

*All metrics and aggregations are computed directly from the local cleaned dataset. AI-generated insights are grounded in verified data calculations.*
"""
        return fallback_report, "fallback"

    def get_dataset_preview(self, dataset_id: str, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise KeyError("Dataset session not found")

        df = dataset["df"]
        total_rows = len(df)
        page_size = min(page_size, 500)
        page = max(page, 1)
        total_pages = (total_rows + page_size - 1) // page_size
        page = min(page, total_pages) if total_pages > 0 else 1

        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_df = df.iloc[start_idx:end_idx]

        rows = (
            page_df.replace({pd.NA: None, pd.NaT: None, float("nan"): None})
            .where(pd.notnull(page_df), None)
            .to_dict(orient="records")
        )

        return {
            "columns": list(df.columns),
            "rows": rows,
            "total_rows": total_rows,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "cleaning_summary": dataset["summary"].get("cleaning_summary"),
        }

    def select_primary_kpi(self, dataset_id: str, column: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recalculate primary KPI summary card metrics for a user-selected column.
        Returns updated KPI spec with value, formatted_value, aggregation, format_type, unit.
        """
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise KeyError("Dataset session not found")

        df = dataset["df"]
        summary = dataset["summary"]
        cleaning_report = dataset["cleaning_report"]

        # Validate column exists and is a valid KPI option
        available_options = summary.get("available_kpi_options", [])
        option = next((opt for opt in available_options if opt["column"] == column), None)
        if not option:
            # Fallback: check if column is in numeric_columns
            if column not in summary.get("numeric_columns", []):
                raise ValueError(f"Column '{column}' is not a valid numeric metric for KPI selection")

        # Use option metadata or overrides
        aggregation = overrides.get("aggregation") if overrides else option.get("aggregation") if option else "sum"
        format_type = overrides.get("format_type") if overrides else option.get("format_type") if option else "decimal"
        unit = overrides.get("unit") if overrides else option.get("unit") if option else ""
        display_name = option.get("display_name", column.replace('_', ' ').title()) if option else column.replace('_', ' ').title()

        # Compute KPI value
        if column in df.columns:
            # Determine smart aggregation if not specified
            if not overrides or "aggregation" not in overrides:
                col_lower = column.lower()
                is_avg = any(kw in col_lower for kw in ['age', 'rating', 'score', 'temp', 'temperature', 'rate', 'pct', 'percentage', 'ratio', 'margin', 'latency', 'tenure', 'gpa', 'heartrate', 'heart_rate'])
                aggregation = "mean" if is_avg else "sum"
            
            val = dashboard_builder._apply_aggregation(df, column, aggregation)
            val = dashboard_builder._number(val)
        else:
            val = float(len(df))
            aggregation = "count"

        formatted = dashboard_builder._format_value(val, column, format_type=format_type, unit=unit)

        # Update the dataset summary with new primary KPI
        summary["primary_kpi"] = column

        # Rebuild KPI spec
        kpi_spec = {
            "column": column,
            "aggregation": aggregation,
            "format_type": format_type,
            "unit": unit,
        }

        # Recompute KPI summary
        kpis = dashboard_builder.make_kpis(df, summary, kpi_spec)

        return {
            "primary_kpi": kpis["primary_kpi"],
            "display_name": display_name,
            "value": val,
            "formatted_value": formatted,
            "aggregation": aggregation,
            "format_type": format_type,
            "unit": unit,
            "kpi_spec": kpi_spec,
            "kpi_summary": kpis,
        }

    def reaggregate_chart(self, dataset_id: str, chart_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recompute chart data with user-selected axes, aggregation, and chart type.
        Validates logical column pairings before materializing.
        """
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            raise KeyError("Dataset session not found")

        df = dataset["df"]
        summary = dataset["summary"]

        # Validate chart pairing using dashboard_builder's validation logic
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        all_cols = numeric + dates + categories

        chart_id = chart_spec.get("chart_id")
        chart_type = chart_spec.get("type", "Bar Chart")
        x_axis = chart_spec.get("x_axis")
        y_axis = chart_spec.get("y_axis")
        secondary_dimension = chart_spec.get("secondary_dimension")
        aggregation = chart_spec.get("aggregation", "sum")
        format_type = chart_spec.get("format_type")
        unit = chart_spec.get("unit")

        # Validate column existence
        if x_axis and x_axis not in all_cols:
            raise ValueError(f"X-axis column '{x_axis}' not found in dataset")
        if y_axis and y_axis not in numeric:
            raise ValueError(f"Y-axis column '{y_axis}' must be a numeric metric")
        if secondary_dimension and secondary_dimension not in all_cols:
            raise ValueError(f"Secondary dimension '{secondary_dimension}' not found in dataset")

        # Enforce logical column pairings
        normalized_type = chart_type.lower()
        is_id_like = lambda col: any(pattern in col.lower() for pattern in ['_id', ' id', 'uuid', 'code', 'index']) or col.lower() in ('id', 'uuid', 'code', 'index')

        if any(t in normalized_type for t in ["bar", "donut", "pie", "treemap", "funnel"]):
            # Categorical charts: X must be categorical (non-ID), Y must be numeric
            if x_axis and x_axis not in categories:
                raise ValueError(f"X-axis '{x_axis}' must be a valid categorical dimension for {chart_type}")
            if is_id_like(x_axis):
                raise ValueError(f"X-axis '{x_axis}' cannot be an ID column")
            if y_axis and y_axis not in numeric:
                raise ValueError(f"Y-axis '{y_axis}' must be a numeric metric")

        elif any(t in normalized_type for t in ["area", "line", "trend"]):
            # Time-series charts: X must be date/time, Y must be numeric
            if x_axis and x_axis not in dates:
                raise ValueError(f"X-axis '{x_axis}' must be a date/time column for {chart_type}")
            if y_axis and y_axis not in numeric:
                raise ValueError(f"Y-axis '{y_axis}' must be a numeric metric")

        elif "scatter" in normalized_type:
            # Scatterplot: both axes must be numeric (non-ID)
            if x_axis not in numeric or y_axis not in numeric:
                raise ValueError("Both X and Y axes must be numeric metrics for Scatterplot")
            if is_id_like(x_axis) or is_id_like(y_axis):
                raise ValueError("Scatterplot axes cannot be ID columns")

        elif "heatmap" in normalized_type:
            # Heatmap: both dimensions must be categorical (non-ID)
            if x_axis not in categories or (secondary_dimension and secondary_dimension not in categories):
                raise ValueError("Heatmap requires two categorical dimensions")
            if is_id_like(x_axis) or (secondary_dimension and is_id_like(secondary_dimension)):
                raise ValueError("Heatmap dimensions cannot be ID columns")

        # Materialize the chart with new spec
        spec = {
            "type": chart_type,
            "title": chart_spec.get("title"),
            "x_axis": x_axis,
            "y_axis": y_axis,
            "secondary_dimension": secondary_dimension,
            "aggregation": aggregation,
            "format_type": format_type,
            "unit": unit,
        }

        chart = dashboard_builder.materialize_chart(df, summary, spec, chart_id or f"custom-{uuid4().hex[:8]}")
        
        if not chart:
            raise ValueError("Failed to materialize chart with given specification")

        return chart


master_orchestrator = MasterOrchestrator()
