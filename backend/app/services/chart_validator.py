import json
from typing import List, Set, Tuple, Optional

import pandas as pd

from app.models.schemas import ChartSpec, KPISpec
from app.services.chart_hydrator import apply_filters, hydrate_charts, hydrate_kpis


def ignored_columns(profile) -> Set[str]:
    return (
        set(profile.get("identifier_columns", []))
        | set(profile.get("junk_columns", []))
        | set(profile.get("constant_columns", []))
    )


def measure_set(df: pd.DataFrame, profile) -> Set[str]:
    """Single source of truth for 'what can be aggregated': profile measures plus any column
    that is numeric or coerces to numeric the same way the hydrator does."""
    blocked = ignored_columns(profile) | set(profile.get("time_columns", []))
    measures = set(profile.get("numeric_columns", []))
    for c in df.columns:
        if c in measures or c in blocked:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_numeric_dtype(s):
            measures.add(c)
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            cleaned = s.dropna().astype(str).str.replace(",", "", regex=False).str.strip()
            if len(cleaned) > 0 and pd.to_numeric(cleaned, errors="coerce").notna().mean() > 0.8:
                measures.add(c)
    return measures


def _filter_problem(df, filters) -> Optional[str]:
    try:
        sub = apply_filters(df, filters)
    except Exception as e:
        return f"invalid filter ({e})"
    return "filters match no rows; use exact values listed in the profile" if sub.empty else None


def _scale_problem(profile, measure, filters) -> Optional[str]:
    splits = {m["split_by"] for m in profile.get("mixed_scale_measures", []) if m["measure_column"] == measure}
    if splits and not any(f.column in splits for f in filters):
        return (
            f"'{measure}' mixes different metrics/units across {sorted(splits)}; "
            f"add a filter pinning one value of one of those columns"
        )
    return None


def _normalize(s: ChartSpec, time_cols: Set[str]) -> ChartSpec:
    """Fix harmless mistakes instead of rejecting the chart."""
    upd = {}
    if s.chart_type != "scatter" and s.aggregation == "none":
        upd["aggregation"] = "sum"
    if s.chart_type in ("line", "area") and s.x_column not in time_cols:
        upd["chart_type"] = "bar"
    elif s.chart_type in ("line", "area"):
        upd["sort"] = "x_asc"
    return s.model_copy(update=upd) if upd else s


def _chart_problem(df, profile, s: ChartSpec, measures: Set[str]) -> Optional[str]:
    time_cols = set(profile.get("time_columns", []))
    ignored = ignored_columns(profile)
    dims = set(profile.get("categorical_columns", []))

    for c in [s.x_column] + ([s.y_column] if s.aggregation != "count" else []):
        if c not in df.columns:
            return f"column '{c}' does not exist"
        if c in ignored:
            return f"column '{c}' is an identifier/junk/constant column and must not be charted"

    if s.aggregation == "count" and measures:
        return "row-count charts are not insightful; aggregate a numeric measure instead"

    if s.chart_type == "scatter":
        if s.x_column not in measures or s.y_column not in measures or s.x_column == s.y_column:
            return (
                f"scatter needs two DIFFERENT numeric columns from {sorted(measures)}. Two metrics stored "
                f"inside one value column cannot be scattered; use a bar or line chart with a filter instead"
            )
    else:
        if s.aggregation != "count" and s.y_column not in measures:
            return f"y_column must be one of {sorted(measures)}, got '{s.y_column}'"
        if s.x_column in measures:
            return "x_column must be a categorical or time column, not a measure"
        if s.x_column not in dims and s.x_column not in time_cols:
            return f"x_column must be one of {sorted(dims | time_cols)}, got '{s.x_column}'"
        if s.chart_type == "pie" and s.aggregation not in ("sum", "count"):
            return "pie charts only make sense with aggregation 'sum' (shares of a whole)"

    p = _filter_problem(df, s.filters)
    if p:
        return p
    for m in {s.x_column, s.y_column} & measures:
        p = _scale_problem(profile, m, s.filters)
        if p:
            return p

    n = apply_filters(df, s.filters)[s.x_column].nunique(dropna=True)
    if s.chart_type != "scatter" and n < 2:
        return "x_column has fewer than 2 distinct values after filtering"
    if s.chart_type == "pie" and n > 12:
        return "too many slices for a pie; use a bar chart"
    if not hydrate_charts(df, [s]):
        return "produces no varying data after aggregation"
    return None


def _key(s: ChartSpec) -> str:
    return json.dumps(
        [s.chart_type, s.x_column, s.y_column, s.aggregation, [f.model_dump() for f in s.filters]],
        sort_keys=True,
        default=str,
    )


def validate_chart_specs(df, profile, specs: List[ChartSpec], seen: Set[str]) -> Tuple[List[ChartSpec], List[str]]:
    measures = measure_set(df, profile)
    time_cols = set(profile.get("time_columns", []))
    ok, errors = [], []
    for raw in specs:
        s = _normalize(raw, time_cols)
        k = _key(s)
        if k in seen:
            continue
        problem = _chart_problem(df, profile, s, measures)
        if problem:
            errors.append(f"Chart '{s.title}': {problem}")
        else:
            seen.add(k)
            ok.append(s)
    return ok, errors


def validate_kpi_specs(df, profile, specs: List[KPISpec]) -> Tuple[List[KPISpec], List[str]]:
    measures = measure_set(df, profile)
    ok, errors, counts = [], [], 0
    for k in specs:
        problem = None
        if k.column not in df.columns:
            problem = f"column '{k.column}' does not exist"
        elif k.operation in ("sum", "avg", "max", "min"):
            problem = (None if k.column in measures else f"needs one of {sorted(measures)}") or _scale_problem(
                profile, k.column, k.filters
            )
        else:
            counts += 1
            if counts > 1:
                problem = "at most one count/unique KPI"
        problem = problem or _filter_problem(df, k.filters)
        if not problem and not hydrate_kpis(df, [k]):
            problem = "could not be computed"
        if problem:
            errors.append(f"KPI '{k.label}': {problem}")
        else:
            ok.append(k)
    return ok, errors