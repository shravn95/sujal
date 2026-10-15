import json
import re
from openai import OpenAI
from typing import Dict, Any
from app.config import settings
from app.models.schemas import ChartSpec, KPISpec, LLMAnalysisResponse
import time 
import httpx
import pandas as pd
from app.models.schemas import DatasetUnderstanding, DashboardPlan
import logging
from app.services.chart_validator import (
    validate_chart_specs,
    validate_kpi_specs,
    measure_set,
    ignored_columns,
)
from app.models.schemas import ChatSQLResponse
from typing import List, Tuple
from pydantic import ValidationError
from app.models.schemas import ChartSpec, KPISpec

logger = logging.getLogger(__name__)

client = OpenAI(
    base_url=settings.NVIDIA_BASE_URL,
    api_key=settings.NVIDIA_API_KEY,
    # read = max silence between chunks, so a stalled stream raises instead of hanging
    timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
    max_retries=0
)
def clean_json_string(raw_text: str) -> str:
    """
    Removes <think>...</think> blocks (if returned inline) and markdown ```json fences
    so json.loads() never fails.
    """
    # Remove <think>...</think> blocks if present in content
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    
    # Extract JSON from ```json ... ``` code blocks if present
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Fallback: grab from first '{' to last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1:
        return cleaned[first_brace:last_brace + 1]
    
    return cleaned


def generate_dataset_analysis(llm_profile: Dict[str, Any], filename: str) -> LLMAnalysisResponse:
    """
    Sends the compact dataset profile to nvidia/nemotron-3-ultra-550b-a55b
    and returns validated KPIs, observations, and chart configurations.
    """
    schema_json = json.dumps(LLMAnalysisResponse.model_json_schema(), indent=2)
    profile_json = json.dumps(llm_profile, indent=2)

    system_prompt = (
        "You are a Principal Data Analyst AI. Your job is to analyze a dataset profile "
        "and design an executive dashboard with high-signal KPIs, sharp analytical observations, "
        "and meaningful chart specifications.\n\n"
        "CRITICAL RULES:\n"
        "1. ONLY use exact column names listed in the provided dataset profile.\n"
        "2. For 'bar', 'line', 'area', and 'pie' charts, 'x_column' should be a categorical or datetime column, "
        "and 'y_column' MUST be a numeric column (unless aggregation is 'count').\n"
        "3. For 'scatter' charts, BOTH 'x_column' and 'y_column' MUST be numeric columns, and 'aggregation' MUST be 'none'.\n"
        "4. Choose 4 to 6 diverse charts (mix of bar, line/area if dates exist, pie for low-cardinality categories, scatter for correlated numerics).\n"
        "5. Respond ONLY with a valid JSON object matching the exact JSON Schema below. Do not include markdown explanations outside the JSON."
    )

    user_prompt = f"""
            Dataset Filename: {filename}

            Dataset Statistical Profile:
            {profile_json}

            Required Output JSON Schema:
            {schema_json}

            Return ONLY the raw JSON object matching the schema above.
            """

    completion = client.chat.completions.create(
        model=settings.NEMOTRON_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.6, # Slightly lower temperature for reliable structured JSON + strong reasoning
        top_p=0.95,
        max_tokens=8192,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        stream=True
    )

    full_content = ""
    full_reasoning = ""

    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            full_reasoning += reasoning
        if delta.content is not None:
            full_content += delta.content

    # Parse and validate with Pydantic
    cleaned_json_str = clean_json_string(full_content)
    parsed_dict = json.loads(cleaned_json_str)
    
    return LLMAnalysisResponse.model_validate(parsed_dict)


def generate_fallback_analysis(llm_profile: Dict[str, Any], filename: str) -> LLMAnalysisResponse:
    """Build a useful dashboard plan locally when the optional LLM is unavailable."""
    numeric = llm_profile.get("numeric_columns", [])
    categorical = llm_profile.get("categorical_columns", [])
    dates = llm_profile.get("datetime_columns", [])
    columns = numeric or categorical
    primary = numeric[0] if numeric else (categorical[0] if categorical else None)

    kpis = []
    if primary:
        kpis.append(KPISpec(label=f"Total {primary}", column=primary, operation="sum" if numeric else "count"))
    for column in numeric[1:4]:
        kpis.append(KPISpec(label=f"Average {column}", column=column, operation="avg"))
    if not kpis and categorical:
        kpis.append(KPISpec(label=f"Unique {categorical[0]}", column=categorical[0], operation="nunique"))

    charts = []
    if dates and numeric:
        charts.append(ChartSpec(
            id="trend_over_time",
            title=f"{numeric[0]} over time",
            description=f"Trend of {numeric[0]} across {dates[0]}.",
            chart_type="line",
            x_column=dates[0],
            y_column=numeric[0],
            aggregation="sum",
        ))
    for index, category in enumerate(categorical[:2]):
        if numeric:
            charts.append(ChartSpec(
                id=f"{numeric[0]}_by_{index}",
                title=f"{numeric[0]} by {category}",
                description=f"Comparison of {numeric[0]} across {category}.",
                chart_type="bar",
                x_column=category,
                y_column=numeric[0],
                aggregation="sum",
            ))
    if len(numeric) >= 2:
        charts.append(ChartSpec(
            id="numeric_relationship",
            title=f"{numeric[0]} vs {numeric[1]}",
            description=f"Relationship between {numeric[0]} and {numeric[1]}.",
            chart_type="scatter",
            x_column=numeric[0],
            y_column=numeric[1],
            aggregation="none",
        ))
    if not charts and columns:
        charts.append(ChartSpec(
            id="dataset_breakdown",
            title=f"Records by {columns[0]}",
            description=f"Record count grouped by {columns[0]}.",
            chart_type="bar",
            x_column=columns[0],
            y_column=columns[0],
            aggregation="count",
        ))

    return LLMAnalysisResponse(
        executive_summary=(
            f"{filename} contains {llm_profile.get('row_count', 0):,} rows and "
            f"{llm_profile.get('column_count', 0)} columns. The dashboard was generated locally "
            "because the AI analysis service was unavailable."
        ),
        key_observations=[
            f"The dataset contains {len(numeric)} numeric and {len(categorical)} categorical columns.",
            f"{len(dates)} date/time columns were detected.",
            f"{llm_profile.get('row_count', 0):,} cleaned records are available for analysis.",
            "Use the charts and KPI cards to inspect the strongest available dimensions and measures.",
        ],
        kpis=kpis[:4],
        charts=charts[:6],
    )




UNDERSTAND_SYSTEM = (
    "You are a principal data analyst doing first-contact analysis of a dataset from its statistical profile. "
    "Work out what the data is, what one row represents, and which columns are measures, dimensions, time, or noise.\n"
    "Pay special attention to LONG (tidy) layouts: if a categorical column lists metric names (e.g. Variable, Measure, "
    "Indicator) and a numeric column holds their values, that numeric column MIXES different metrics and often different "
    "units. Values of different metrics must never be summed or averaged together. The profile's 'mixed_scale_measures' "
    "shows where this happens.\n"
    "Ignore identifier, code, junk (e.g. 'Unnamed: N') and constant columns. Use exact column names from the profile. "
    "Be specific and grounded in the numbers, not generic.\n"
    "Respond ONLY with a JSON object matching the schema."
)

PLANNER_SYSTEM = (
    "You design the initial executive dashboard for a dataset. You are given the profile and an analyst's understanding of it. "
    "Each chart must answer one of the analyst's key_questions and tell the viewer something real.\n"
    "RULES:\n"
    "1. NEVER use aggregation 'count' (row counts) when numeric measures exist. Aggregate a real measure.\n"
    "2. y_column must be one of 'allowed_y_columns' and x_column one of 'allowed_x_columns' in the column contract "
    "(scatter: both from allowed_y_columns). Never use 'never_use' columns.\n"
    "3. If 'mixed_scale_measures' lists the measure, you MUST add filters that pin ONE metric/unit "
    "(copy exact values from the column's all_values). Put the filter in the chart title, e.g. 'Total income by industry (2025)'.\n"
    "4. line/area charts: x_column is the time column, sort 'x_axis' as 'x_asc'. For snapshots (rankings, shares) "
    "filter the time column to its latest value.\n"
    "5. scatter: two DIFFERENT numeric columns with a meaningful relationship, aggregation 'none'. "
    "NEVER scatter two metrics that live inside the same value column (long format); use bar/line with a filter instead.\n"
    "scatter: two numeric measures with a meaningful relationship, aggregation 'none'.\n"
    "6. Choose 4-6 charts that complement each other: one trend over time (if a time column exists), one or two rankings, "
    "one composition, and one relationship/distribution only if it is meaningful. Do not repeat the same idea.\n"
    "7. Choose 3-4 KPIs on different measures, each pinned with filters when needed; the FIRST is the headline KPI. "
    "At most one count/nunique KPI.\n"
    "8. 'description' is one sentence stating the insight or question the chart answers.\n"
    "Use exact column names and exact filter values. Respond ONLY with a JSON object matching the schema."
    "9. Use only the allowed values from the schema: chart_type (bar, line, scatter, pie, area), "
    "aggregation (sum, avg, count, min, max, none), operation (sum, avg, count, nunique, max, min), "
    "sort (y_desc, y_asc, x_asc), limit between 1 and 30.\n"
)


def _stream_json(system_prompt: str, user_prompt: str, thinking: bool, max_tokens: int,  timeout_s: float) -> dict:
    completion = client.chat.completions.create(
        model=settings.NEMOTRON_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.4,
        top_p=0.95,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        stream=True,
    )
    deadline = time.monotonic() +  + timeout_s
    content = ""
    for chunk in completion:
        if time.monotonic() > deadline:
            completion.close()
            raise TimeoutError(f"LLM exceeded {timeout_s}s")
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
    return json.loads(clean_json_string(content))


def understand_dataset(profile: Dict[str, Any], filename: str) -> DatasetUnderstanding:
    user = (
        f"Dataset filename: {filename}\n\nProfile:\n{json.dumps(profile)}\n\n"
        f"Output JSON Schema:\n{json.dumps(DatasetUnderstanding.model_json_schema())}\n\nReturn ONLY the JSON object."
    )
    raw = _stream_json(UNDERSTAND_SYSTEM, user, thinking=True, max_tokens=8192,
                   timeout_s=settings.LLM_UNDERSTAND_TIMEOUT_SECONDS)
    return DatasetUnderstanding.model_validate(raw)


def _parse_plan(raw) -> Tuple[DashboardPlan, List[str]]:
    """Parse KPIs and charts one by one so a single bad item can't discard the whole plan."""
    if not isinstance(raw, dict):
        raise ValueError(f"Planner returned {type(raw).__name__}, expected a JSON object")

    def short(e: ValidationError) -> str:
        return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())

    kpis, charts, errors = [], [], []
    for i, item in enumerate(raw.get("kpis") or []):
        try:
            kpis.append(KPISpec.model_validate(item))
        except ValidationError as e:
            errors.append(f"KPI #{i + 1} invalid: {short(e)}")
    for i, item in enumerate(raw.get("charts") or []):
        try:
            charts.append(ChartSpec.model_validate(item))
        except ValidationError as e:
            name = item.get("title", f"#{i + 1}") if isinstance(item, dict) else f"#{i + 1}"
            errors.append(f"Chart '{name}' invalid: {short(e)}")
    return DashboardPlan(kpis=kpis, charts=charts), errors

def plan_dashboard(profile: Dict[str, Any], understanding: DatasetUnderstanding, df: pd.DataFrame) -> DashboardPlan:
    measures = measure_set(df, profile)
    dims = set(profile.get("categorical_columns", []))
    time_cols = set(profile.get("time_columns", []))

    # Trust the data, not the model: drop columns the analyst invented or mislabelled
    understanding = understanding.model_copy(update={
        "measure_columns": [c for c in understanding.measure_columns if c in measures],
        "dimension_columns": [c for c in understanding.dimension_columns if c in dims],
        "time_column": understanding.time_column if understanding.time_column in time_cols
                       else (sorted(time_cols)[0] if time_cols else None),
    })
    contract = {
        "allowed_y_columns": sorted(measures),
        "allowed_x_columns": sorted(dims | time_cols),
        "never_use": sorted(ignored_columns(profile)),
    }

    schema = json.dumps(DashboardPlan.model_json_schema())
    seen: set = set()
    good_charts, good_kpis, errors, feedback = [], [], [], ""

    for attempt in range(2):
        user = (
            f"Analyst understanding:\n{understanding.model_dump_json()}\n\n"
            f"Column contract (hard limits):\n{json.dumps(contract)}\n\n"
            f"Profile:\n{json.dumps(profile)}\n\n"
            f"Output JSON Schema:\n{schema}\n{feedback}\nReturn ONLY the JSON object."
        )
        raw = _stream_json(PLANNER_SYSTEM, user, thinking=False, max_tokens=4096,
                           timeout_s=settings.LLM_PLAN_TIMEOUT_SECONDS)
        try:
            plan, parse_errs = _parse_plan(raw)
        except Exception as e:
            raise ValueError(f"Planner output did not match schema: {e}; raw={json.dumps(raw)[:800]}")
        logger.warning("planner proposed: %s", json.dumps(raw)[:2500])
        charts, chart_errs = validate_chart_specs(df, profile, plan.charts, seen)
        kpis, kpi_errs = validate_kpi_specs(df, profile, plan.kpis)
        good_charts += charts
        if len(kpis) > len(good_kpis):
            good_kpis = kpis
        errors = chart_errs + kpi_errs + parse_errs
        if errors:
            logger.warning("plan attempt %d rejected %d item(s): %s", attempt + 1, len(errors), errors)
        if len(good_charts) >= 4 and len(good_kpis) >= 3:
            break
        feedback = ("\nYour previous plan had these problems:\n- " + "\n- ".join(errors[:10]) +
                    "\nFix them and return the FULL plan again.\n")

    if not good_charts:
                raise ValueError(f"No valid charts. allowed_y={sorted(measures)}; " + "; ".join(errors))

    ids = set()
    for i, c in enumerate(good_charts):
        if c.id in ids:
            c.id = f"{c.id}-{i}"
        ids.add(c.id)
    return DashboardPlan(kpis=good_kpis[:4], charts=good_charts[:6])



def generate_chat_sql(user_message: str, dataset_summary: Dict[str, Any]) -> ChatSQLResponse:
    """
    Asks Nemotron-3-Ultra to translate natural language into a DuckDB SQL query.
    """
    schema_json = json.dumps(dataset_summary, indent=2)
    schema_schema = json.dumps(ChatSQLResponse.model_json_schema(), indent=2)

    system_prompt = (
        "You are an expert Text-to-SQL Data Analyst AI. You are querying a DuckDB database "
        "where the table name is 'dataset' (backed by a Parquet file).\n\n"
        "CRITICAL RULES:\n"
        "1. ONLY use exact column names from the dataset schema below.\n"
        "2. Write valid DuckDB SQL queries (SELECT statements only).\n"
        "3. If the user asks for aggregations, use GROUP BY correctly.\n"
        "4. Return ONLY a valid JSON object matching the provided JSON schema."
    )

    user_prompt = f"""
Dataset Schema & Profile:
{schema_json}

User Question: {user_message}

Required Output JSON Schema:
{schema_schema}

Return ONLY the raw JSON object.
"""

    completion = client.chat.completions.create(
        model=settings.NEMOTRON_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2, # Low temperature for precise SQL syntax
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        stream=False
    )

    content = completion.choices[0].message.content
    cleaned = clean_json_string(content)
    return ChatSQLResponse.model_validate(json.loads(cleaned))


def synthesize_chat_answer(user_message: str, sql_query: str, query_results: List[Dict[str, Any]]) -> str:
    """
    Synthesizes a natural language answer from the query results.
    """
    results_json = json.dumps(query_results[:20], indent=2)
    
    prompt = f"""
User Question: {user_message}
SQL Executed: {sql_query}
Query Results (top rows):
{results_json}

Provide a clear, concise, professional analytical answer summarizing these findings. Highlight key numbers or anomalies.
"""

    completion = client.chat.completions.create(
        model=settings.NEMOTRON_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=2048,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        stream=False
    )
    return completion.choices[0].message.content.strip()