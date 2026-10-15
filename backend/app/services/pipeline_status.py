import json
import os
import tempfile

from app.config import settings


def _file(dataset_id: str) -> str:
    return os.path.join(settings.STORAGE_DIR, f"{dataset_id}.status.json")


def read_status(dataset_id: str):
    try:
        with open(_file(dataset_id), encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_status(dataset_id: str, **fields) -> None:
    data = read_status(dataset_id) or {}
    data.update(fields)
    fd, tmp = tempfile.mkstemp(dir=settings.STORAGE_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, _file(dataset_id))  # atomic, so the poller never reads half a file