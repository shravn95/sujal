from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Any, List, Optional
from app.agents.orchestrator import master_orchestrator
from app.core.pdf_generator import pdf_generator

router = APIRouter()


class ExportPDFRequest(BaseModel):
    dataset_id: str
    query: Optional[str] = None
    report_text: Optional[str] = None
    charts: Optional[List[Any]] = None


@router.post("/report/export-pdf")
async def export_pdf_report(payload: ExportPDFRequest = Body(...)):
    """
    Generates and returns a PDF executive report for the given dataset.
    """
    dataset = master_orchestrator.datasets.get(payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset session not found. Please re-upload your file.")

    summary = dataset["summary"]
    cleaning_report = dataset.get("cleaning_report", {})
    charts = payload.charts or []  # Use charts from request if provided
    kpis = {
        "primary_kpi": summary.get("primary_kpi", "Primary Metric"),
        "formatted_value": "N/A",
        "value": "N/A",
        "data_quality": summary.get("quality_score", 100),
        "total_rows": summary.get("row_count", 0),
        "secondary_kpi": None,
        "secondary_value": None,
        "secondary_formatted_value": None,
    }

    # Try to get forecast if available
    forecast = None
    if "forecast" in dataset and dataset["forecast"]:
        forecast = dataset["forecast"]

    filename = summary.get("filename", "dataset")
    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    export_filename = f"{base_name}_executive_report.pdf"

    pdf_bytes = pdf_generator.generate_report(
        dataset_summary=summary,
        cleaning_report=cleaning_report,
        charts=charts,
        kpis=kpis,
        forecast=forecast,
        query=payload.query,
        report_text=payload.report_text,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{export_filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition",
        },
    )