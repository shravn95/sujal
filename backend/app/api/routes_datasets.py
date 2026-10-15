import os
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import workspace as ws

router = APIRouter()


def _valid(dataset_id: str) -> str:
    try:
        meta_path = ws.path(dataset_id, ".meta.json")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid dataset id.")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return dataset_id


class MessageIn(BaseModel):
    role: Literal["user", "ai"]
    text: str = Field(min_length=1, max_length=20000)
    meta: Dict[str, Any] = Field(default_factory=dict)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class DashboardStateIn(BaseModel):
    charts: List[Dict[str, Any]]
    dashboard_title: Optional[str] = None


@router.get("/datasets")
def list_all():
    return {"datasets": ws.list_datasets()}


@router.get("/datasets/{dataset_id}")
def open_dataset(dataset_id: str):
    return ws.get_workspace(_valid(dataset_id))


@router.patch("/datasets/{dataset_id}")
def rename(dataset_id: str, body: RenameIn):
    ws.rename_dataset(_valid(dataset_id), body.title)
    return {"ok": True}


@router.delete("/datasets/{dataset_id}")
def delete(dataset_id: str):
    ws.delete_dataset(_valid(dataset_id))
    return {"ok": True}


@router.post("/datasets/{dataset_id}/messages", status_code=201)
def add_message(dataset_id: str, body: MessageIn):
    return ws.add_message(_valid(dataset_id), body.role, body.text, body.meta)


@router.delete("/datasets/{dataset_id}/messages")
def clear_chat(dataset_id: str):
    ws.clear_messages(_valid(dataset_id))
    return {"ok": True}


@router.put("/datasets/{dataset_id}/dashboard")
def save_dashboard(dataset_id: str, body: DashboardStateIn):
    _valid(dataset_id)
    if not os.path.exists(ws.path(dataset_id, ".dashboard.json")):
        raise HTTPException(status_code=404, detail="Dashboard not built yet.")
    changes: Dict[str, Any] = {"charts": body.charts}
    if body.dashboard_title:
        changes["dashboard_title"] = body.dashboard_title
    ws.save_dashboard_state(dataset_id, changes)
    return {"ok": True}