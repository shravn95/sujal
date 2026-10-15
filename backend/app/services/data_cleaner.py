import os
import re
import uuid
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from app.config import settings
from app.models.schemas import CleaningReport

def _to_numeric_text(s: pd.Series) -> pd.Series:
    """Text -> numbers: strips currency/commas/%/spaces and turns '(1,234)' into -1234."""
    t = s.astype(str).str.strip()
    neg = t.str.match(r"^\(.*\)$")
    t = t.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    t = t.str.replace(r"[$€£₹,%\s]", "", regex=True)
    out = pd.to_numeric(t, errors="coerce")
    out[neg] = -out[neg]
    return out

def clean_and_profile_dataset(file_bytes: bytes, filename: str) -> Tuple[str, pd.DataFrame, CleaningReport, Dict[str, Any]]:
    """
    1. Loads CSV or Excel into Pandas.
    2. Cleans column names, drops empty rows/cols, removes duplicates, coerces currencies/dates.
    3. Saves as Parquet for fast DuckDB querying.
    4. Returns dataset_id, cleaned DataFrame, CleaningReport, and a compact LLM profile dict.
    """
    # 1. Load Data
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(pd.io.common.BytesIO(file_bytes), low_memory=False)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(pd.io.common.BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")

    orig_rows, orig_cols = df.shape
    type_conversions = []

    # 2. Clean Column Names (strip whitespace, replace spaces with underscores if needed, keep readable)
    df.columns = [str(col).strip() for col in df.columns]

    # 3. Drop 100% empty rows and columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    df.dropna(how="all", axis=1, inplace=True)
    df.dropna(how="all", axis=0, inplace=True)
        # Drop export-artifact columns like "Unnamed: 9" that are almost entirely empty
    artifact_cols = [c for c in df.columns if re.match(r"^Unnamed: \d+$", str(c)) and df[c].isna().mean() > 0.9]
    df.drop(columns=artifact_cols, inplace=True)
    empty_cols += artifact_cols

    # 4. Drop exact duplicate rows
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = before_dedup - len(df)

    # 5. Smart Type Coercion (Strings -> Numbers & Dates)
    for col in df.columns:
        if df[col].dtype == "object":
            # Strip whitespace from strings
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan, "null": np.nan, "N/A": np.nan})

            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # Try currency/numeric coercion (e.g., "$1,234.50" or "45%")
             # Numeric coercion over ALL values (not a 100-row sample). Columns that are mostly numbers
            # plus a few suppression markers (S, .., np, ...) are treated as numeric.
            num = _to_numeric_text(non_null)
            share = num.notna().mean()
            leftovers = non_null[num.isna()]
            sentinel_like = (
                share > 0.5 and leftovers.nunique() <= 8 and leftovers.str.len().max() <= 12
            )
            if share > 0.85 or sentinel_like:
                df[col] = _to_numeric_text(df[col])
                note = f"Converted '{col}' from text to numeric"
                if len(leftovers):
                    note += f" ({len(leftovers)} non-numeric entries such as {sorted(leftovers.unique())[:3]} set to missing)"
                type_conversions.append(note)
                continue

            # Try datetime coercion if column name hints at date/time or values look like dates
            if any(k in col.lower() for k in ["date", "time", "year", "month", "day", "created", "timestamp"]):
                try:
                    parsed_dates = pd.to_datetime(df[col], errors="coerce", format="mixed")
                    if parsed_dates.notna().mean() > 0.8:
                        df[col] = parsed_dates
                        type_conversions.append(f"Converted '{col}' to datetime")
                except Exception:
                    pass

    # 6. Calculate missing value percentages (only for columns that have missing values)
    missing_summary = {}
    for col in df.columns:
        pct = round(float(df[col].isna().mean() * 100), 2)
        if pct > 0:
            missing_summary[col] = pct

    # 7. Save cleaned dataset as Parquet for fast DuckDB access in Phase 1 & Phase 2
    dataset_id = str(uuid.uuid4())
    parquet_path = os.path.join(settings.STORAGE_DIR, f"{dataset_id}.parquet")
    df.to_parquet(parquet_path, index=False)

    # 8. Build Cleaning Report
    cleaning_report = CleaningReport(
        original_rows=orig_rows,
        original_columns=orig_cols,
        cleaned_rows=len(df),
        cleaned_columns=len(df.columns),
        duplicates_removed=duplicates_removed,
        empty_columns_dropped=empty_cols,
        type_conversions=type_conversions,
        missing_value_summary=missing_summary
    )

    # 9. Build Compact Statistical Profile for Nemotron
    llm_profile = build_llm_profile(df)

    return dataset_id, df, cleaning_report, llm_profile


_ID_NAME = re.compile(r"(^|_)(id|uuid|guid|code)($|_)", re.I)
_YEAR_NAME = re.compile(r"(^|_)(year|yr|fy|fiscal)($|_)", re.I)


def _is_year_like(s: pd.Series, name: str) -> bool:
    if not _YEAR_NAME.search(name):
        return False
    s = s.dropna()
    return len(s) > 0 and bool((s % 1 == 0).all()) and bool(s.between(1900, 2100).all())


def _mixed_scale(df, measure_cols, dimension_cols):
    """Find numeric columns whose typical size differs hugely between groups of a categorical column
    (e.g. a 'value' column holding both counts and dollars). Summing those together is meaningless."""
    found = []
    for cat in dimension_cols:
        n = df[cat].nunique(dropna=True)
        if not 2 <= n <= 30:
            continue
        for m in measure_cols:
            med = df.groupby(cat)[m].median().abs().dropna()
            med = med[med > 0]
            if len(med) >= 2 and med.max() / med.min() > 50:
                found.append({
                    "measure_column": m,
                    "split_by": cat,
                    "group_medians": {str(k): round(float(v), 2) for k, v in med.sort_values(ascending=False).head(15).items()},
                })
    return found[:6]

def build_llm_profile(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows = len(df)
    columns_info, numeric_cols, categorical_cols, datetime_cols = [], [], [], []

    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        meta: Dict[str, Any] = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_pct": round(float(df[col].isna().mean() * 100), 1),
            "unique_values": nunique,
        }
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            numeric_cols.append(col)
            meta["role"] = "numeric"
            if nunique > 0:
                meta["stats"] = {
                    "min": round(float(df[col].min()), 2),
                    "max": round(float(df[col].max()), 2),
                    "mean": round(float(df[col].mean()), 2),
                    "median": round(float(df[col].median()), 2),
                    "std": round(float(df[col].std()), 2) if n_rows > 1 else 0.0,
                }
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            meta["role"] = "datetime"
            if nunique > 0:
                meta["range"] = {"min_date": str(df[col].min()), "max_date": str(df[col].max())}
        else:
            categorical_cols.append(col)
            meta["role"] = "categorical"
            vc = df[col].value_counts(dropna=True)
            meta["top_categories"] = {str(k): int(v) for k, v in vc.head(5).items()}
            if nunique <= 40:  # the model needs EXACT values to write valid filters
                meta["all_values"] = [str(v)[:60] for v in vc.index.tolist()]
        columns_info.append(meta)

    # Classify noise so the model (and the validator) can ignore it
    junk_cols = [c for c in df.columns if re.match(r"^Unnamed: \d+$", str(c))]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    identifier_cols = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        by_name = bool(_ID_NAME.search(str(c)))
        by_cardinality = (not pd.api.types.is_numeric_dtype(df[c])) and n_rows > 50 and nun / n_rows > 0.95
        if by_name or by_cardinality:
            identifier_cols.append(c)
    time_like = [c for c in numeric_cols if _is_year_like(df[c], str(c))]

    excluded = set(junk_cols) | set(constant_cols) | set(identifier_cols) | set(time_like)
    for m in columns_info:
        n = m["name"]
        if n in junk_cols: m["role"] = "junk"
        elif n in constant_cols: m["role"] = "constant"
        elif n in identifier_cols: m["role"] = "identifier"
        elif n in time_like: m["role"] = "time"

    measure_cols = [c for c in numeric_cols if c not in excluded]
    dimension_cols = [c for c in categorical_cols if c not in excluded]

    correlations = []
    if len(measure_cols) >= 2:
        corr = df[measure_cols].corr().abs()
        for i in range(len(measure_cols)):
            for j in range(i + 1, len(measure_cols)):
                v = corr.iloc[i, j]
                if not np.isnan(v) and v > 0.3:
                    correlations.append({"col_1": measure_cols[i], "col_2": measure_cols[j], "correlation": round(float(v), 2)})

    return {
        "row_count": n_rows,
        "column_count": len(df.columns),
        "numeric_columns": measure_cols,          # now: real measures only
        "categorical_columns": dimension_cols,    # now: real dimensions only
        "datetime_columns": datetime_cols,
        "time_columns": datetime_cols + time_like,
        "identifier_columns": identifier_cols,
        "junk_columns": junk_cols,
        "constant_columns": constant_cols,
        "mixed_scale_measures": _mixed_scale(df, measure_cols[:6], dimension_cols[:8]),
        "columns": columns_info,
        "strong_correlations": correlations[:10],
        "sample_rows": df.head(3).fillna("").astype(str).to_dict(orient="records"),
    }