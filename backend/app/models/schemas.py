from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any, Dict, Union
from pydantic import BaseModel, Field, field_validator, AliasChoices

class DatasetChatRequest(BaseModel):
    dataset_id: str = Field(
        validation_alias=AliasChoices("dataset_id", "datasetId", "id")
    )
    message: str = Field(
        validation_alias=AliasChoices("message", "query", "prompt", "question", "text")
    )
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        default=[],
        validation_alias=AliasChoices("chat_history", "history", "messages")
    )

    class Config:
        populate_by_name = True

class FilterSpec(BaseModel):
    column: str = Field(description="Exact column name to filter on")
    op: Literal["eq", "ne", "in", "gte", "lte"] = "eq"
    value: Union[str, float, List[str]] = Field(
        description="Exact value copied from the profile (a list when op is 'in')"
    )
# --- What we ask Nemotron to generate ---
class ChartSpec(BaseModel):
    id: str = Field(description="Unique slug ID for the chart, e.g., 'sales_by_region'")
    title: str = Field(description="Clear business title for the chart")
    description: str = Field(description="1-sentence insight explaining why this chart matters")
    chart_type: Literal["bar", "line", "scatter", "pie", "area"]
    x_column: str = Field(description="Exact column name from the dataset for X-axis / categories")
    y_column: str = Field(description="Exact column name from the dataset for Y-axis / values")
    aggregation: Literal["sum", "avg", "count", "min", "max", "none"] = Field(
        default="sum",
        description="Use 'none' only for scatter plots. For bar/line/pie/area, specify how to aggregate y_column grouped by x_column."
    )
    filters: List[FilterSpec] = Field(
        default_factory=list,
        description="Row filters applied BEFORE aggregating. Required to pin one metric/unit when a column mixes metrics."
    )
    sort: Literal["y_desc", "y_asc", "x_asc"] = Field(
        default="y_desc", description="x_asc for time axes, y_desc for rankings"
    )
    limit: int = Field(default=12, ge=1, le=30, description="Max categories shown on non-time charts (1-30)")

    @field_validator("limit", mode="before")
    @classmethod
    def _clamp_limit(cls, v):
        try:
            return max(1, min(30, int(v)))   # 0, 2, 50, "12" all become something valid
        except (TypeError, ValueError):
            return 12

class KPISpec(BaseModel):
    label: str = Field(description="Short title for KPI card, e.g., 'Total Revenue'")
    column: str = Field(description="Exact column name to calculate from")
    operation: Literal["sum", "avg", "count", "nunique", "max", "min"]
    filters: List[FilterSpec] = Field(default_factory=list, description="Row filters applied before computing the KPI")

class LLMAnalysisResponse(BaseModel):
    executive_summary: str = Field(description="2-3 sentence overview of what this dataset is about and its primary shape.")
    key_observations: List[str] = Field(description="4 to 6 sharp, actionable analytical observations or anomalies found in the data profile.")
    kpis: List[KPISpec] = Field(description="3 to 4 top-level summary metrics for KPI cards.")
    charts: List[ChartSpec] = Field(description="4 to 6 diverse charts to visualize the dataset.")

class DatasetUnderstanding(BaseModel):
    dataset_purpose: str = Field(description="What this dataset is about, in 1-2 sentences")
    grain: str = Field(description="What one row represents")
    structure: Literal["wide", "long", "mixed"] = Field(description="'long' if a column lists metric names and another holds their values")
    measure_columns: List[str] = Field(description="Numeric columns worth aggregating")
    dimension_columns: List[str] = Field(description="Categorical columns worth grouping by")
    time_column: Optional[str] = Field(default=None, description="Best time axis, if any")
    ignore_columns: List[str] = Field(description="Identifier, code, junk, or constant columns not to chart")
    caveats: List[str] = Field(description="Mixed units/metrics, sparsity, or anything that makes naive charts misleading")
    key_questions: List[str] = Field(description="4 to 6 business questions this data can answer")
    executive_summary: str = Field(description="2-3 sentence overview")
    key_observations: List[str] = Field(description="4 to 6 specific, data-grounded observations")


class DashboardPlan(BaseModel):
    kpis: List[KPISpec] = Field(description="3 to 4 headline KPIs; the first is the most important")
    charts: List[ChartSpec] = Field(description="4 to 6 charts, each answering one key question")


# --- Hydrated Models Sent to React Frontend ---
class HydratedKPI(BaseModel):
    label: str
    value: Any
    column: str
    operation: str

class HydratedChart(ChartSpec):
    # Recharts-friendly array: [{"x": "Category A", "y": 120}, ...]
    data: List[Dict[str, Any]]

class CleaningReport(BaseModel):
    original_rows: int
    original_columns: int
    cleaned_rows: int
    cleaned_columns: int
    duplicates_removed: int
    empty_columns_dropped: List[str]
    type_conversions: List[str]
    missing_value_summary: Dict[str, float] # column -> percentage missing

class DashboardResponse(BaseModel):
    dataset_id: str
    filename: str
    cleaning_report: CleaningReport
    executive_summary: str
    key_observations: List[str]
    kpis: List[HydratedKPI]
    charts: List[HydratedChart]
    sample_rows: List[Dict[str, Any]]


# --- Phase 2: Chatbot Schemas ---
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "ai"]
    content: str

class ChatSQLResponse(BaseModel):
    sql_query: str = Field(description="The read-only DuckDB SQL query to execute against table 'dataset'.")
    explanation: str = Field(description="1-sentence explanation of what this query calculates.")
    needs_chart: bool = Field(default=False, description="True ONLY if the user explicitly asks for a chart/plot/graph/visualization or comparison breakdown.")
    chart_type: Optional[Literal["bar", "line", "pie", "scatter", "area"]] = None
    chart_title: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None

class DatasetChatResponse(BaseModel):
    answer: str = Field(description="Clear natural language answer synthesized from the query results.")
    sql_query: str
    query_results: List[Dict[str, Any]] = Field(description="Top rows resulting from the SQL execution.")
    chart: Optional[HydratedChart] = None