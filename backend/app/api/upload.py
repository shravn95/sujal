from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import Response
from app.agents.orchestrator import master_orchestrator
from app.config import settings
import io
import pandas as pd

router = APIRouter()

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...), sheet_name: str | None = Query(default=None)):
    """
    Receives CSV dataset upload, parses schema & stats locally using Pandas/DuckDB,
    and returns dataset profile + auto-generated initial dashboard deterministically (no LLM calls).
    Returns in under 2 seconds for instant UI loading.
    """
    if not file.filename or not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Upload a CSV or Excel (.xlsx/.xls) dataset.")

    try:
        contents = await file.read()

        # Enforce upload size limit
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(contents) / (1024*1024):.1f}MB). Maximum allowed: {settings.MAX_UPLOAD_SIZE_MB}MB."
            )

        result = master_orchestrator.process_file_and_generate_initial_dashboard(contents, file.filename, sheet_name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing dataset: {str(e)}")


@router.post("/dataset/{dataset_id}/analyze")
async def trigger_deep_analysis(dataset_id: str):
    """
    Trigger deep LLM analysis asynchronously - called by Copilot after dashboard mounts.
    This endpoint performs the heavy LLM reasoning (planner, verifier, synthesizer)
    that was removed from the initial upload for instant response.
    """
    try:
        result = await master_orchestrator.trigger_deep_llm_analysis(dataset_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering deep analysis: {str(e)}")


@router.get("/dataset/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, page: int = Query(default=1, ge=1), page_size: int = Query(default=50, ge=1, le=500)):
    """
    Returns a paginated preview of the cleaned dataset.
    """
    try:
        result = master_orchestrator.get_dataset_preview(dataset_id, page, page_size)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving dataset preview: {str(e)}")

@router.get("/dataset/{dataset_id}/download")
async def download_dataset(dataset_id: str):
    """
    Exports and downloads the latest cleaned/preprocessed dataset as a CSV file.
    """
    try:
        ds = master_orchestrator.datasets.get(dataset_id)
        if not ds:
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")
        
        df: pd.DataFrame = ds["df"]
        filename = ds.get("filename", "dataset")
        base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
        export_filename = f"{base_name}_cleaned.csv"
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")
        
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{export_filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting dataset: {str(e)}")


@router.post("/dataset/{dataset_id}/select-kpi")
async def select_primary_kpi(dataset_id: str, payload: dict = Body(...)):
    """
    Update the primary KPI to a user-selected column and recalculate KPI summary card metrics.
    Request body: {"column": "fare", "aggregation": "mean", "format_type": "currency", "unit": "$"}
    """
    try:
        column = payload.get("column")
        if not column:
            raise HTTPException(status_code=400, detail="Column name is required")
        
        overrides = {
            "aggregation": payload.get("aggregation"),
            "format_type": payload.get("format_type"),
            "unit": payload.get("unit"),
        }
        # Remove None values
        overrides = {k: v for k, v in overrides.items() if v is not None}
        
        result = master_orchestrator.select_primary_kpi(dataset_id, column, overrides)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error selecting primary KPI: {str(e)}")


@router.post("/dataset/{dataset_id}/chart/aggregate")
async def aggregate_chart(dataset_id: str, payload: dict = Body(...)):
    """
    Recompute chart data with user-selected axes, aggregation, and chart type.
    Validates logical column pairings before materializing.
    Request body: {"chart_id": "auto-0", "type": "Bar Chart", "x_axis": "ward", "y_axis": "fare", "aggregation": "mean", "format_type": "currency", "unit": "$", "secondary_dimension": null}
    """
    try:
        chart_spec = {
            "chart_id": payload.get("chart_id"),
            "type": payload.get("type"),
            "x_axis": payload.get("x_axis"),
            "y_axis": payload.get("y_axis"),
            "secondary_dimension": payload.get("secondary_dimension"),
            "aggregation": payload.get("aggregation"),
            "format_type": payload.get("format_type"),
            "unit": payload.get("unit"),
            "title": payload.get("title"),
        }
        # Remove None values
        chart_spec = {k: v for k, v in chart_spec.items() if v is not None}
        
        result = master_orchestrator.reaggregate_chart(dataset_id, chart_spec)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aggregating chart: {str(e)}")

