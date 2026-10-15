from fastapi import APIRouter, HTTPException, Body
from app.agents.orchestrator import master_orchestrator
from app.core.llm_client import nemotron_client
from app.core.preprocessing_actions import PreprocessingExecutor
from app.core.dataset_profiler import dataset_profiler
from typing import Dict, Any
import pandas as pd

router = APIRouter()

ALLOWED_ACTIONS = {"drop_column", "rename_column", "fill_missing", "change_type", "add_column", "filter_rows"}

@router.post("/dataset/{dataset_id}/preprocess")
async def preprocess_dataset(dataset_id: str, payload: Dict[str, Any] = Body(...)):
    """
    Accepts a natural-language command, asks the LLM for a single preprocessing action,
    validates it, executes it on a copy of the cleaned DataFrame, re-profiles, and stores the result.
    """
    command = payload.get("command", "").strip()
    if not command:
        raise HTTPException(status_code=400, detail="Missing 'command' in request body.")

    # Retrieve dataset
    try:
        ds = master_orchestrator.datasets[dataset_id]
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    df_clean: pd.DataFrame = ds["df"]
    summary: Dict[str, Any] = ds["summary"]

    # Build column list and types from profile
    columns = list(df_clean.columns)
    # Determine simple type labels
    col_types = {}
    for c in columns:
        dtype = str(df_clean[c].dtype)
        if "int" in dtype or "float" in dtype:
            col_types[c] = "numeric"
        elif "datetime" in dtype:
            col_types[c] = "datetime"
        else:
            col_types[c] = "categorical"

    # Ask LLM for action
    try:
        action = nemotron_client.generate_preprocessing_action(command, columns, col_types)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    action_type = action["type"]
    if action_type not in ALLOWED_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Disallowed action type: {action_type}")

    # Execute action on a copy
    try:
        new_df, confirmation_message = PreprocessingExecutor.apply_action(df_clean.copy(), {"type": action_type, "params": action["params"]})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Action execution failed: {e}")

    # Re-profile the new dataframe
    try:
        # For re-profiling, we need raw file bytes; but we have cleaned df. Use profile_csv with dataframe param.
        # The profile_csv method accepts dataframe argument.
        new_summary = dataset_profiler.profile_csv(
            b"", ds.get("filename", "dataset"), new_df,
            cleaning_report=None,
            was_sampled=False,
            sample_size=len(new_df),
            total_rows=len(new_df)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-profiling failed: {e}")

    # Update stored dataset
    master_orchestrator.datasets[dataset_id]["df"] = new_df
    master_orchestrator.datasets[dataset_id]["summary"] = new_summary

    return {
        "explanation": action["explanation"],
        "confirmation_message": confirmation_message,
        "updated_summary": new_summary,
    }