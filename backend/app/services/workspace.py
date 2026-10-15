import glob
import json
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.config import settings
from app.services.pipeline_status import read_status

DB_PATH = os.path.join(settings.STORAGE_DIR, "autoinsight.db")


# ── paths and JSON helpers ───────────────────────────────────────────────
def path(dataset_id: str, suffix: str) -> str:
    uuid.UUID(dataset_id)  # raises ValueError for non-UUIDs (blocks path traversal)
    return os.path.join(settings.STORAGE_DIR, f"{dataset_id}{suffix}")


def read_json(p: str) -> Any:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(p: str, data: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=settings.STORAGE_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, default=str)
    os.replace(tmp, p)  # atomic: readers never see a half-written file


# ── chat messages (SQLite) ───────────────────────────────────────────────
@contextmanager
def _db():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    with _db() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   dataset_id TEXT NOT NULL,
                   role TEXT NOT NULL,
                   text TEXT NOT NULL,
                   meta TEXT,
                   created_at TEXT NOT NULL)"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_messages_dataset ON messages(dataset_id, id)")


def add_message(dataset_id: str, role: str, text: str, meta: Optional[dict] = None) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _db() as c:
        cur = c.execute(
            "INSERT INTO messages (dataset_id, role, text, meta, created_at) VALUES (?,?,?,?,?)",
            (dataset_id, role, text, json.dumps(meta) if meta else None, now),
        )
        return {"id": cur.lastrowid, "role": role, "text": text, "meta": meta or {}, "created_at": now}


def list_messages(dataset_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    with _db() as c:
        rows = c.execute(
            "SELECT id, role, text, meta, created_at FROM messages WHERE dataset_id=? ORDER BY id DESC LIMIT ?",
            (dataset_id, limit),
        ).fetchall()
    rows.reverse()
    return [
        {"id": r["id"], "role": r["role"], "text": r["text"],
         "meta": json.loads(r["meta"]) if r["meta"] else {}, "created_at": r["created_at"]}
        for r in rows
    ]


def clear_messages(dataset_id: str) -> None:
    with _db() as c:
        c.execute("DELETE FROM messages WHERE dataset_id=?", (dataset_id,))


# ── datasets ─────────────────────────────────────────────────────────────
def dataset_state(dataset_id: str) -> str:
    if os.path.exists(path(dataset_id, ".dashboard.json")):
        return "ready"
    st = read_status(dataset_id) or {}
    if st.get("state") in ("queued", "running", "retrying"):
        return "analyzing"
    if os.path.exists(path(dataset_id, ".analysis.json")):
        return "analyzed"  # AI part finished; dashboard still has to be built (fast, no LLM)
    if st.get("state") == "failed":
        return "failed"
    return "uploaded"


def _summary(dataset_id: str, meta: dict, meta_path: str) -> Dict[str, Any]:
    rep = meta.get("cleaning_report", {})
    created = meta.get("created_at") or datetime.fromtimestamp(
        os.path.getmtime(meta_path), timezone.utc
    ).isoformat()
    return {
        "dataset_id": dataset_id,
        "title": meta.get("title") or meta.get("filename") or dataset_id,
        "filename": meta.get("filename"),
        "row_count": rep.get("cleaned_rows"),
        "column_count": rep.get("cleaned_columns"),
        "state": dataset_state(dataset_id),
        "created_at": created,
    }


def list_datasets() -> List[Dict[str, Any]]:
    out = []
    for p in glob.glob(os.path.join(settings.STORAGE_DIR, "*.meta.json")):
        ds = os.path.basename(p)[: -len(".meta.json")]
        try:
            uuid.UUID(ds)
            out.append(_summary(ds, read_json(p), p))
        except Exception:
            continue
    out.sort(key=lambda d: d["created_at"], reverse=True)
    return out


def get_workspace(dataset_id: str) -> Dict[str, Any]:
    mp = path(dataset_id, ".meta.json")
    meta = read_json(mp)
    dp = path(dataset_id, ".dashboard.json")
    return {
        **_summary(dataset_id, meta, mp),
        "dashboard": read_json(dp) if os.path.exists(dp) else None,
        "messages": list_messages(dataset_id),
    }


def rename_dataset(dataset_id: str, title: str) -> None:
    mp = path(dataset_id, ".meta.json")
    meta = read_json(mp)
    meta["title"] = title.strip()[:120]
    write_json(mp, meta)


def save_dashboard_state(dataset_id: str, changes: dict) -> None:
    dp = path(dataset_id, ".dashboard.json")
    data = read_json(dp)
    data.update(changes)
    write_json(dp, data)


def delete_dataset(dataset_id: str) -> None:
    uuid.UUID(dataset_id)
    for p in glob.glob(os.path.join(settings.STORAGE_DIR, f"{dataset_id}.*")):
        try:
            os.remove(p)
        except OSError:
            pass
    clear_messages(dataset_id)