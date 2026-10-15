import json
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import StreamingResponse
from app.agents.orchestrator import master_orchestrator

router = APIRouter()

@router.post("/query")
async def process_natural_language_query(payload: dict = Body(...)):
    """
    Receives user natural language query (e.g. 'Show yearly sales performance by region')
    along with dataset summary, streaming reasoning thinking thoughts and updated chart layout.
    """
    query = payload.get("query")
    dataset_id = payload.get("dataset_id")

    if not query:
        raise HTTPException(status_code=400, detail="Query string is required.")
    if not dataset_id:
        raise HTTPException(status_code=400, detail="Upload a dataset before asking for analysis.")

    def event_stream():
        try:
            for chunk in master_orchestrator.process_query_stream(dataset_id, query):
                yield f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n"
        except KeyError as e:
            err_msg = {"type": "error", "error_type": "DATA_ERROR", "content": f"Dataset or column not found: {str(e)}"}
            yield f"data: {json.dumps(err_msg, ensure_ascii=True)}\n\n"
        except TimeoutError as e:
            err_msg = {"type": "error", "error_type": "TIMEOUT", "content": f"AI model timed out: {str(e)}"}
            yield f"data: {json.dumps(err_msg, ensure_ascii=True)}\n\n"
        except Exception as e:
            err_type = "RATE_LIMIT" if "rate" in str(e).lower() else "INTERNAL_ERROR"
            err_msg = {"type": "error", "error_type": err_type, "content": f"Analysis error: {str(e)}"}
            yield f"data: {json.dumps(err_msg, ensure_ascii=True)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
