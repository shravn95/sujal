import os
import re
import duckdb
import pandas as pd
from typing import Dict, Any, Tuple, List
from app.config import settings


def get_parquet_path(dataset_id: str) -> str:
    """Finds the parquet file for dataset_id in STORAGE_DIR."""
    direct = os.path.join(settings.STORAGE_DIR, f"{dataset_id}.parquet")
    if os.path.exists(direct):
        return direct
    # Fallback if stored inside a workspace subfolder
    nested = os.path.join(settings.STORAGE_DIR, dataset_id, "cleaned.parquet")
    if os.path.exists(nested):
        return nested
    raise FileNotFoundError(f"Dataset '{dataset_id}' parquet file not found.")


def execute_dataset_sql(dataset_id: str, sql_query: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Executes a read-only DuckDB SQL query against the dataset's parquet file.
    The table can be referenced as 'dataset', 'data', or 'df'.
    """
    parquet_path = get_parquet_path(dataset_id)

    # Strip trailing semicolons or markdown fences if any
    clean_sql = re.sub(r"```(?:sql)?|```", "", sql_query, flags=re.IGNORECASE).strip().rstrip(";")

    # Safety check: ensure read-only SELECT / WITH query
    first_word = clean_sql.split()[0].lower() if clean_sql.split() else ""
    if first_word not in ("select", "with", "describe", "summarize", "show"):
        return [], f"Only read-only SELECT queries are allowed (got '{first_word}')."

    forbidden = {"drop", "delete", "insert", "update", "alter", "truncate", "create", "grant", "copy"}
    tokens = set(re.findall(r"\b[a-z_]+\b", clean_sql.lower()))
    blocked = tokens.intersection(forbidden)
    if blocked:
        return [], f"Forbidden SQL keyword detected: {', '.join(blocked).upper()}"

    conn = duckdb.connect(database=":memory:")
    try:
        escaped_path = parquet_path.replace("'", "''")
        # Create views for common table aliases so 'FROM dataset', 'FROM data', or 'FROM df' all work seamlessly
        conn.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{escaped_path}')")
        conn.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{escaped_path}')")
        conn.execute(f"CREATE VIEW df AS SELECT * FROM read_parquet('{escaped_path}')")

        result_df = conn.execute(clean_sql).df()
        records = result_df.head(100).fillna("").astype(str).to_dict(orient="records")
        return records, ""
    except Exception as e:
        return [], str(e)
    finally:
        conn.close()