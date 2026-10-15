import io
import json
import logging
import os
import traceback
import uuid
import inngest
from app.inngest_app.client import inngest_client, ANALYZE_EVENT
from app.services.pipeline_status import read_status, write_status
from datetime import datetime, timezone
from app.services import workspace as ws

import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import settings
from app.models.schemas import CleaningReport, LLMAnalysisResponse
from app.services.data_cleaner import clean_and_profile_dataset
from app.services.nemotron_service import generate_fallback_analysis
from app.services.chart_hydrator import hydrate_kpis, hydrate_charts

router = APIRouter()
logger = logging.getLogger(__name__)


def _build_frontend_response(
    dataset_id: str,
    filename: str,
    cleaned_df: pd.DataFrame,
    cleaning_report,
    llm_profile: dict,
    ai_analysis,
    hydrated_kpis,
    hydrated_charts,
    plan_source: str,
):
    """
    Bridge between the Phase 1 plan's DashboardResponse and the shape the
    React frontend actually consumes (summary, charts, kpi_summary, etc.).
    """
    # Identify column categories from the LLM profile
    numeric_cols = llm_profile.get("numeric_columns", [])
    categorical_cols = llm_profile.get("categorical_columns", [])
    datetime_cols = llm_profile.get("datetime_columns", [])

    # Build summary object expected by the frontend
    summary = {
        "filename": filename,
        "row_count": llm_profile.get("row_count", len(cleaned_df)),
        "column_count": llm_profile.get("column_count", len(cleaned_df.columns)),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "date_columns": llm_profile.get("time_columns", datetime_cols),
        "identifier_columns": llm_profile.get("identifier_columns", []), # columns with very high cardinality relative to row count
        "quality_score": round(100 - sum(cleaning_report.missing_value_summary.values()) / max(len(cleaned_df.columns), 1), 1),
        "primary_kpi": hydrated_kpis[0].column if hydrated_kpis else (numeric_cols[0] if numeric_cols else ""),
        "available_kpi_options": [
            {
                "column": kpi.column,
                "display_name": kpi.label,
                "aggregation": kpi.operation,
                "format_type": "number",
                "unit": "",
            }
            for kpi in hydrated_kpis
        ],
    }

    # Build KPI summary for the top cards
    primary_kpi = hydrated_kpis[0] if hydrated_kpis else None
    secondary_kpi = hydrated_kpis[1] if len(hydrated_kpis) > 1 else None
    kpi_summary = {
        "primary_kpi": primary_kpi.label if primary_kpi else "—",
        "value": primary_kpi.value if primary_kpi else "—",
        "formatted_value": _format_number(primary_kpi.value) if primary_kpi and isinstance(primary_kpi.value, (int, float)) else "—",
        "data_quality": summary["quality_score"],
        "total_rows": llm_profile.get("row_count", len(cleaned_df)),
    }
    if secondary_kpi:
        kpi_summary["secondary_kpi"] = secondary_kpi.label
        kpi_summary["secondary_formatted_value"] = (
            _format_number(secondary_kpi.value) if isinstance(secondary_kpi.value, (int, float)) else str(secondary_kpi.value)
        )

    # Adapt chart objects to the frontend shape
    charts_out = []
    for i, chart in enumerate(hydrated_charts):
        chart_obj = {
            "id": chart.id or f"auto-{i}",
            "title": chart.title,
            "type": _map_chart_type_to_frontend(chart.chart_type),
            "x_axis": chart.x_column,
            "y_axis": chart.y_column,
            "secondary_dimension": None,
            "aggregation": chart.aggregation,
            "data": chart.data,
            "insight_tooltip": chart.description,
            "filters": [f.model_dump() for f in chart.filters],
        }
        charts_out.append(chart_obj)

    # Build the dashboard title from the executive summary
    dashboard_title = f"{filename.rsplit('.', 1)[0]} — Analytics Dashboard"

    # Build sample rows
    sample_rows = cleaned_df.head(10).fillna("").to_dict(orient="records")
    # Stringify all values for safe JSON serialization
    for row in sample_rows:
        for k, v in row.items():
            if pd.isna(v):
                row[k] = ""
            elif not isinstance(v, (str, int, float, bool)):
                row[k] = str(v)

    return {
        "dataset_id": dataset_id,
        "summary": summary,
        "dashboard_title": dashboard_title,
        "kpi_summary": kpi_summary,
        "charts": charts_out,
        "executive_summary": ai_analysis.executive_summary,
        "key_observations": ai_analysis.key_observations,
        "cleaning_report": cleaning_report.model_dump(),
        "sample_rows": sample_rows,
        "plan_source": plan_source,
        "forecast": None,
    }


def _map_chart_type_to_frontend(chart_type: str) -> str:
    """Map plan chart types to frontend ChartComponents types."""
    mapping = {
        "bar": "Bar Chart",
        "line": "Line Graph",
        "area": "Area Graph",
        "pie": "Pie Chart",
        "scatter": "Scatterplot",
    }
    return mapping.get(chart_type, "Bar Chart")


def _format_number(val) -> str:
    """Format a number with K/M/B abbreviations."""
    if not isinstance(val, (int, float)):
        return str(val)
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{val / 1_000:.1f}K"
    if val == int(val):
        return f"{int(val):,}"
    return f"{val:,.2f}"

def _path(dataset_id: str, suffix: str) -> str:
    """Build a storage path; rejects non-UUID ids so nobody can traverse directories."""
    try:
        uuid.UUID(dataset_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset id.")
    return os.path.join(settings.STORAGE_DIR, f"{dataset_id}{suffix}")


# ── Step 1: upload → clean → Parquet → profile (no LLM, fast) ─────────────
@router.post("/upload")
def upload_step(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    try:
        dataset_id, df, report, profile = clean_and_profile_dataset(
            file_bytes=file.file.read(), filename=file.filename
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {e}")

    # Persist what the later steps need so they only require the dataset_id
    with open(_path(dataset_id, ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "filename": file.filename,
                "title": file.filename,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cleaning_report": report.model_dump(),
                "llm_profile": profile,
            },
            f, default=str,
        )

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "cleaning_report": report.model_dump(),
    }


# ── Step 2: LLM analysis (summary, observations, KPI + chart specs) ───────
@router.post("/dataset/{dataset_id}/analyze", status_code=202)
def analyze_step(dataset_id: str):
    """Enqueue the LLM analysis in Inngest and return immediately."""
    if not os.path.exists(_path(dataset_id, ".meta.json")):
        raise HTTPException(status_code=404, detail="Dataset not found. Upload it again.")

    # Reset first so the poller never reads a stale 'done' or 'failed' from a previous run
    write_status(dataset_id, state="queued", attempt=0, message=None, fallback_reason=None)
    try:
        inngest_client.send_sync(
            inngest.Event(name=ANALYZE_EVENT, data={"dataset_id": dataset_id})
        )
    except Exception as e:
        write_status(dataset_id, state="failed", message=str(e)[:200])
        raise HTTPException(
            status_code=503,
            detail="Could not reach Inngest. Is the dev server running (port 8288)?",
        )
    return {"state": "queued"}


@router.get("/dataset/{dataset_id}/status")
def analyze_status(dataset_id: str):
    status = read_status(dataset_id)
    if status is None:
        raise HTTPException(status_code=404, detail="No analysis in progress.")
    if status.get("state") == "done":
        with open(_path(dataset_id, ".analysis.json"), encoding="utf-8") as f:
            saved = json.load(f)
        a = saved["analysis"]
        status = {
            **status,
            "executive_summary": a["executive_summary"],
            "key_observations": a["key_observations"],
            "plan_source": saved["plan_source"],
        }
    return status

# ── Step 3: hydrate KPIs/charts → the same payload the old /upload returned ─
@router.post("/dataset/{dataset_id}/dashboard")
def dashboard_step(dataset_id: str):
    try:
        with open(_path(dataset_id, ".meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(_path(dataset_id, ".analysis.json"), encoding="utf-8") as f:
            saved = json.load(f)
        df = pd.read_parquet(_path(dataset_id, ".parquet"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run the analysis step first.")

    try:
        analysis = LLMAnalysisResponse.model_validate(saved["analysis"])
        report = CleaningReport.model_validate(meta["cleaning_report"])

        response = _build_frontend_response(
            dataset_id=dataset_id,
            filename=meta["filename"],
            cleaned_df=df,
            cleaning_report=report,
            llm_profile=meta["llm_profile"],
            ai_analysis=analysis,
            hydrated_kpis=hydrate_kpis(df, analysis.kpis),
            hydrated_charts=hydrate_charts(df, analysis.charts),
            plan_source=saved["plan_source"],
        )
        ws.write_json(_path(dataset_id, ".dashboard.json"), response)
        return JSONResponse(content=response)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dashboard build failed: {e}")