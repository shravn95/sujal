import pandas as pd
import numpy as np
from typing import List, Dict, Any
from app.models.schemas import (
    KPISpec, ChartSpec, HydratedKPI, HydratedChart
)


def _to_numeric(s: pd.Series) -> pd.Series:
    """Safely coerce a Series to numeric, stripping commas and whitespace from string/object dtypes."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    cleaned = (
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    for f in filters or []:
        if f.column not in df.columns:
            raise ValueError(f"filter column '{f.column}' not found")
        s = df[f.column]
        vals = f.value if isinstance(f.value, list) else [f.value]
        if f.op in ("gte", "lte"):
            num = _to_numeric(s)
            v = float(str(vals[0]).replace(",", "").strip())
            df = df[num >= v] if f.op == "gte" else df[num <= v]
            continue
        if pd.api.types.is_numeric_dtype(s):
            try:
                mask = s.isin([float(str(v).replace(",", "").strip()) for v in vals])
            except (TypeError, ValueError):
                mask = pd.Series(False, index=df.index)
        else:
            mask = s.astype(str).isin([str(v) for v in vals])
        df = df[~mask] if f.op == "ne" else df[mask]
    return df


def hydrate_kpis(df: pd.DataFrame, kpi_specs: List[KPISpec]) -> List[HydratedKPI]:
    """
    Compute actual KPI values from the cleaned DataFrame.
    Returns a list of HydratedKPI objects ready for the frontend.
    """
    hydrated = []
    for kpi in kpi_specs:
        col = kpi.column
        if col not in df.columns:
            continue

        try:
            series = apply_filters(df, kpi.filters)[col].dropna()
            if len(series) == 0:
                continue

            if kpi.operation in ("sum", "avg", "max", "min"):
                num_series = _to_numeric(series).dropna()
                if len(num_series) == 0:
                    continue
                if kpi.operation == "sum":
                    val = round(float(num_series.sum()), 2)
                elif kpi.operation == "avg":
                    val = round(float(num_series.mean()), 2)
                elif kpi.operation == "max":
                    val = round(float(num_series.max()), 2)
                elif kpi.operation == "min":
                    val = round(float(num_series.min()), 2)
            elif kpi.operation == "nunique":
                val = int(series.nunique())
            else:  # count
                val = int(len(series))

            hydrated.append(HydratedKPI(
                label=kpi.label,
                value=val,
                column=col,
                operation=kpi.operation
            ))
        except Exception:
            continue

    return hydrated


def hydrate_charts(df: pd.DataFrame, chart_specs: List[ChartSpec]) -> List[HydratedChart]:
    """
    Execute the exact aggregations requested by Nemotron against the cleaned DataFrame
    and format the results as Recharts-compatible flat arrays.

    Each data point gets both generic keys (name/value or x/y) AND the original column
    names so React components can bind to dataKey="value" effortlessly.
    """
    hydrated_charts = []

    for spec in chart_specs:
        x_col = spec.x_column
        y_col = spec.y_column

        if x_col not in df.columns:
            continue
        if y_col and y_col not in df.columns and spec.aggregation != "count":
            continue

        try:
            chart_data = []
            base_df = apply_filters(df, spec.filters)

            # 1. Scatter Plots (unaggregated numeric pairs, capped at 200 points)
            if spec.chart_type == "scatter" or spec.aggregation == "none":
                if not y_col or y_col not in df.columns:
                    continue
                sub = base_df[[x_col, y_col]].dropna().copy()
                sub[x_col] = _to_numeric(sub[x_col])
                sub[y_col] = _to_numeric(sub[y_col])
                sub = sub.dropna().head(200)

                for _, row in sub.iterrows():
                    x_val = round(float(row[x_col]), 2)
                    y_val = round(float(row[y_col]), 2)
                    chart_data.append({
                        "name": str(x_val),
                        "value": y_val,
                        "x": x_val,
                        "y": y_val,
                        x_col: x_val,
                        y_col: y_val
                    })

            # 2. Aggregated Charts (Bar, Line, Area, Pie)
            else:
                work_df = base_df.dropna(subset=[x_col]).copy()
                col = work_df[x_col]
                if pd.api.types.is_datetime64_any_dtype(col):
                    work_df["_x_group"] = col.dt.strftime("%Y-%m-%d")
                    is_date = True
                elif pd.api.types.is_numeric_dtype(col) and (col % 1 == 0).all():
                    work_df["_x_group"] = col.astype("int64").astype(str)   # 2011 not 2011.0
                    is_date = False
                else:
                    work_df["_x_group"] = col.astype(str)
                    is_date = False

                if spec.aggregation == "count" or not y_col:
                    grouped = work_df.groupby("_x_group").size().reset_index(name="_y_val")
                    y_label = "Count"
                else:
                    work_df["_y_num"] = _to_numeric(work_df[y_col])
                    work_df = work_df.dropna(subset=["_y_num"])

                    agg_map = {
                        "sum": "sum",
                        "avg": "mean",
                        "min": "min",
                        "max": "max"
                    }
                    agg_func = agg_map.get(spec.aggregation, "sum")
                    grouped = work_df.groupby("_x_group")["_y_num"].agg(agg_func).reset_index(name="_y_val")
                    y_label = y_col

                # Sort and limit results
                if is_date or spec.chart_type in ("line", "area") or spec.sort == "x_asc":
                    grouped = grouped.sort_values("_x_group", ascending=True).head(60)
                else:
                    limit = min(spec.limit, 8) if spec.chart_type == "pie" else spec.limit
                    grouped = grouped.sort_values("_y_val", ascending=(spec.sort == "y_asc")).head(limit)

                # Calculate total for percentage computation
                total = grouped["_y_val"].sum() if len(grouped) > 0 else 1

                for _, row in grouped.iterrows():
                    name_val = str(row["_x_group"])
                    num_val = round(float(row["_y_val"]), 2)
                    pct = round((num_val / total) * 100, 1) if total > 0 else 0
                    chart_data.append({
                        "name": name_val,
                        "value": num_val,
                        "percentage": pct,
                        "formatted_value": _format_number(num_val),
                        x_col: name_val,
                        y_label: num_val
                    })

            if len(chart_data) >= 2 and len({d["value"] for d in chart_data}) > 1:
                hydrated_charts.append(HydratedChart(
                    **spec.model_dump(),
                    data=chart_data
                ))

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Skipping chart {spec.id} due to hydration error: {e}")
            continue

    return hydrated_charts


def _format_number(val: float) -> str:
    """Format a number with K/M/B abbreviations for display."""
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{val / 1_000:.1f}K"
    if val == int(val):
        return f"{int(val):,}"
    return f"{val:,.2f}"