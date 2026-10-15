"""Turn a validated dataframe into rich, interactive chart payloads for the UI.

The LLM is intentionally not allowed to produce fabricated numbers.
It specifies the visualization intent and fields, and DashboardBuilder performs
every aggregation, cross-tabulation, percentage, and statistical calculation locally.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DashboardBuilder:
    chart_type_map = {
        "area": "Area Graph",
        "area graph": "Area Graph",
        "line": "Line Graph",
        "line graph": "Line Graph",
        "bar": "Bar Chart",
        "bar chart": "Bar Chart",
        "stacked bar": "Stacked Bar Graph",
        "stacked bar graph": "Stacked Bar Graph",
        "multiset bar": "Multi-set Bar Chart",
        "multi-set bar": "Multi-set Bar Chart",
        "donut": "Donut Chart",
        "donut chart": "Donut Chart",
        "pie": "Donut Chart",
        "pie chart": "Donut Chart",
        "scatter": "Scatterplot",
        "scatterplot": "Scatterplot",
        "histogram": "Histogram",
        "treemap": "Treemap",
        "radar": "Radar Chart",
        "heatmap": "Heatmap",
        "funnel": "Funnel Chart",
        "box plot": "Box Plot",
    }

    @staticmethod
    def _number(value: Any) -> float:
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return round(val, 4)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _apply_aggregation(df_or_series: Any, metric: str, agg_func: str = "sum", grouped_by: Optional[str] = None) -> Any:
        """Dynamically compute aggregations across any mathematical method (sum, mean, median, count, min, max, std)."""
        agg = (agg_func or "sum").lower().strip()
        if grouped_by:
            gb = df_or_series.groupby(grouped_by, dropna=False)
            if agg in ["mean", "avg", "average"]:
                return gb[metric].mean()
            elif agg in ["median", "med"]:
                return gb[metric].median()
            elif agg in ["count", "cnt", "volume"]:
                return gb[metric].count()
            elif agg in ["count_distinct", "unique", "distinct"]:
                return gb[metric].nunique()
            elif agg in ["min", "minimum"]:
                return gb[metric].min()
            elif agg in ["max", "maximum"]:
                return gb[metric].max()
            elif agg in ["std", "stddev"]:
                return gb[metric].std().fillna(0)
            else:
                return gb[metric].sum()
        else:
            s = df_or_series[metric]
            if agg in ["mean", "avg", "average"]:
                return s.mean()
            elif agg in ["median", "med"]:
                return s.median()
            elif agg in ["count", "cnt", "volume"]:
                return s.count()
            elif agg in ["count_distinct", "unique", "distinct"]:
                return s.nunique()
            elif agg in ["min", "minimum"]:
                return s.min()
            elif agg in ["max", "maximum"]:
                return s.max()
            elif agg in ["std", "stddev"]:
                return s.std()
            else:
                return s.sum()

    @staticmethod
    def _format_value(value: float, metric_name: str = "", format_type: Optional[str] = None, unit: Optional[str] = None) -> str:
        """Format numbers dynamically across all domains (Healthcare, IoT, HR, Finance, etc.)."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "\u2014"
        num = float(value)
        fmt = (format_type or "").lower().strip()
        u = unit.strip() if unit else ""

        # Only mark as currency if explicitly told via format_type or unit — avoid false positives
        is_currency = fmt == "currency" or u in ("$", "\u20ac", "\u00a3", "\u00a5")
        is_pct = fmt == "percentage" or u == "%"

        if is_currency:
            prefix, suffix = u if u in ("$", "\u20ac", "\u00a3", "\u00a5") else "$", ""
        elif is_pct:
            prefix, suffix = "", "%"
        elif u:
            prefix, suffix = "", f" {u}" if len(u) > 1 else u
        else:
            prefix, suffix = "", ""

        abs_val = abs(num)
        if abs_val >= 1_000_000_000:
            val_str = f"{num / 1_000_000_000:.2f}B"
        elif abs_val >= 1_000_000:
            val_str = f"{num / 1_000_000:.2f}M"
        elif abs_val >= 10_000 and not is_pct:
            val_str = f"{num / 1_000:.1f}K"
        elif abs_val >= 100 or (num == int(num) and abs_val >= 1):
            val_str = f"{int(round(num)):,}"
        elif abs_val < 0.01 and abs_val > 0:
            val_str = f"{num:.4f}"
        else:
            val_str = f"{num:.2f}"

        return f"{prefix}{val_str}{suffix}"

    @classmethod
    def _format_currency_or_number(cls, value: float, metric_name: str = "") -> str:
        """Backward compatibility wrapper for legacy callers."""
        return cls._format_value(value, metric_name)

    def _field(self, requested: Optional[str], available: List[str], fallback: Optional[str] = None) -> Optional[str]:
        if requested:
            for column in available:
                if column.lower() == str(requested).lower() or column.lower().replace("_", " ") == str(requested).lower():
                    return column
        return fallback or (available[0] if available else None)

    # -----------------------------------------------------------------------
    # Aggregation Helpers
    # -----------------------------------------------------------------------
    def _period_series(
        self,
        df: pd.DataFrame,
        date_col: str,
        metric: str,
        agg_func: str = "sum",
        format_type: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if date_col not in df.columns or metric not in df.columns:
            return []
        frame = df[[date_col, metric]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col])
        if frame.empty:
            return []

        # Determine best grouping frequency
        date_range_days = (frame[date_col].max() - frame[date_col].min()).days
        if date_range_days > 730:
            frame["_period"] = frame[date_col].dt.to_period("Q")
        elif date_range_days > 60:
            frame["_period"] = frame[date_col].dt.to_period("M")
        else:
            frame["_period"] = frame[date_col].dt.to_period("W")

        grouped = self._apply_aggregation(frame, metric, agg_func, grouped_by="_period").tail(36)
        # Compute percentage based on the displayed values (not the underlying raw total)
        displayed_values = [self._number(v) for v in grouped.values]
        total_displayed = sum(abs(v) for v in displayed_values) or 1.0

        series = []
        for period, value in grouped.items():
            num = self._number(value)
            pct = round((abs(num) / total_displayed) * 100, 1) if total_displayed else 0.0
            series.append({
                "name": str(period),
                "value": num,
                metric: num,
                "percentage": pct,
                "formatted_value": self._format_value(num, metric, format_type, unit),
            })
        return series

    def _category_series(
        self,
        df: pd.DataFrame,
        dimension: str,
        metric: str,
        agg_func: str = "sum",
        max_items: int = 12,
        sort_order: str = "descending",
        format_type: Optional[str] = None,
        unit: Optional[str] = None,
        cap_donut: bool = False,  # Cap at top 5 + Other for donut/pie
    ) -> List[Dict[str, Any]]:
        if dimension not in df.columns or metric not in df.columns:
            return []
        frame = df[[dimension, metric]].dropna()
        if frame.empty:
            return []

        grouped = self._apply_aggregation(frame, metric, agg_func, grouped_by=dimension)
        if sort_order == "ascending":
            grouped = grouped.sort_values(ascending=True)
        else:
            grouped = grouped.sort_values(ascending=False)

        # For donut/pie charts, cap at top 5 + Other
        donut_cap = 5 if cap_donut else max_items
        top_items = grouped.head(donut_cap)
        
        # If we have more items than the cap, create "Other" bucket
        if cap_donut and len(grouped) > donut_cap:
            other_items = grouped.iloc[donut_cap:]
            other_value = other_items.sum()
            # Compute percentage based on total (including Other)
            total_all = grouped.sum()
            # Replace last item with "Other" or append
            if len(top_items) == donut_cap:
                # Replace the last item with the combined "Other" if we exactly hit the cap
                # Actually, we should add "Other" as a new entry
                pass
            # Better approach: create new series with top 5 + Other
            top_names = list(top_items.index)
            top_values = [self._number(v) for v in top_items.values]
            
            # Add "Other" if there are more items
            if len(grouped) > donut_cap:
                other_value = other_items.sum()
                top_names.append("Other")
                top_values.append(self._number(other_value))
            
            total_displayed = sum(abs(v) for v in top_values) or 1.0
            series = []
            for name, value in zip(top_names, top_values):
                pct = round((abs(value) / total_displayed) * 100, 1) if total_displayed else 0.0
                series.append({
                    "name": str(name),
                    "value": value,
                    metric: value,
                    "percentage": pct,
                    "formatted_value": self._format_value(value, metric, format_type, unit),
                })
            return series
        
        # Original logic for non-donut capped
        top_items = grouped.head(max_items)
        # Compute percentage based on the displayed subset values
        displayed_values = [self._number(v) for v in top_items.values]
        total_displayed = sum(abs(v) for v in displayed_values) or 1.0
        series = []
        for name, value in top_items.items():
            num = self._number(value)
            pct = round((abs(num) / total_displayed) * 100, 1) if total_displayed else 0.0
            series.append({
                "name": str(name),
                "value": num,
                metric: num,
                "percentage": pct,
                "formatted_value": self._format_value(num, metric, format_type, unit),
            })
        return series

    # -----------------------------------------------------------------------
    # Deep Analytical Insight Generation
    # -----------------------------------------------------------------------
    def _generate_deep_insight(
        self,
        chart_type: str,
        data: List[Dict[str, Any]],
        x_axis: str,
        y_axis: str,
        secondary_dimension: Optional[str] = None,
        metric: Optional[str] = None,
        agg_func: str = "sum",
    ) -> str:
        """
        Generate a rich 3-4 sentence analytical breakdown for a chart.
        Covers: Trend/Pattern, Extremes, Distribution/Anomalies, Domain Meaning.
        """
        if not data or not isinstance(data, list):
            return "Insufficient data for analysis."

        # Extract values for analysis
        values = [self._number(d.get("value", d.get(y_axis, 0))) for d in data if isinstance(d, dict)]
        names = [str(d.get("name", d.get(x_axis, "?"))) for d in data if isinstance(d, dict)]
        formatted = [d.get("formatted_value", "") for d in data if isinstance(d, dict)]
        percentages = [d.get("percentage", 0) for d in data if isinstance(d, dict)]

        if not values:
            return "No computable values in chart data."

        max_val = max(values)
        min_val = min(values)
        max_idx = values.index(max_val)
        min_idx = values.index(min_val)
        max_name = names[max_idx] if max_idx < len(names) else "?"
        min_name = names[min_idx] if min_idx < len(names) else "?"
        max_fmt = formatted[max_idx] if max_idx < len(formatted) else f"{max_val:,.2f}"
        min_fmt = formatted[min_idx] if min_idx < len(formatted) else f"{min_val:,.2f}"

        avg_val = sum(values) / len(values)
        total = sum(values)

        # --- Chart-type specific analysis ---
        chart_type_lower = chart_type.lower()

        # 1. TIME-SERIES (Area/Line/Trend)
        if any(t in chart_type_lower for t in ["area", "line", "trend"]):
            # Trend direction
            if len(values) >= 3:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                trend = "upward" if sum(second_half) > sum(first_half) else ("downward" if sum(second_half) < sum(first_half) else "flat")
                velocity = (second_half[-1] - first_half[0]) / max(abs(first_half[0]), 1) * 100
            else:
                trend = "flat"
                velocity = 0

            # Volatility
            if len(values) > 1:
                changes = [abs(values[i] - values[i-1]) / max(abs(values[i-1]), 1) * 100 for i in range(1, len(values))]
                avg_change = sum(changes) / len(changes)
                volatility = "high" if avg_change > 25 else ("moderate" if avg_change > 10 else "low")
            else:
                volatility = "low"

            # Peak/Trough timing
            peak_period = max_name
            trough_period = min_name

            return (
                f"Trend & Pattern: The {y_axis} series shows a {trend} trajectory ({velocity:+.1f}% net change) with {volatility} period-over-period volatility. "
                f"Extremes: Peak of {max_fmt} occurred at {peak_period}; trough of {min_fmt} at {trough_period}. "
                f"Distribution: Values oscillate around {self._format_value(avg_val, y_axis)} mean. "
                f"Domain Implication: This {'growth' if trend == 'upward' else 'decline' if trend == 'downward' else 'stability'} in {metric or y_axis} "
                f"suggests {'investment in scaling' if trend == 'upward' else 'root-cause investigation of drivers' if trend == 'downward' else 'steady-state monitoring'}."
            )

        # 2. CATEGORICAL (Bar/Donut/Pie/Treemap/Funnel)
        elif any(t in chart_type_lower for t in ["bar", "donut", "pie", "treemap", "funnel"]):
            # Concentration analysis
            top3_pct = sum(percentages[:3]) if len(percentages) >= 3 else sum(percentages)
            concentration = "highly concentrated" if top3_pct > 70 else ("moderately concentrated" if top3_pct > 45 else "well distributed")

            # Gap analysis
            gaps = []
            for i in range(len(values) - 1):
                if values[i] > 0:
                    drop = (values[i] - values[i+1]) / values[i] * 100
                    if drop > 30:
                        gaps.append(f"{names[i]}→{names[i+1]} ({drop:.0f}% drop)")

            gap_str = f" Notable gaps: {', '.join(gaps[:2])}." if gaps else ""

            # Dominant category
            dominant = f"{max_name} ({max_fmt}, {percentages[max_idx]:.1f}%)" if max_idx < len(percentages) else max_name

            return (
                f"Key Visual Takeaway: {concentration.capitalize()} distribution — top 3 categories capture {top3_pct:.1f}% of total {metric or y_axis}. "
                f"Extremes & Distribution: {dominant} dominates; {min_name} ({min_fmt}) is the smallest contributor.{gap_str} "
                f"Anomalies: {len([v for v in values if v > avg_val * 2])} outlier(s) exceed 2× mean. "
                f"Business Impact: {'Focus resources on ' + max_name + ' segment' if top3_pct > 60 else 'Balanced portfolio allows diversified strategy'}. "
                f"{'Reallocate from ' + min_name + ' to high-growth areas.' if min_val < avg_val * 0.1 else 'Low performers warrant diagnostic review.'}"
            )

        # 3. SCATTERPLOT / CORRELATION
        elif "scatter" in chart_type_lower:
            # Compute correlation if we have x,y pairs
            x_vals = [self._number(d.get("x", d.get("x_label", 0))) for d in data if isinstance(d, dict)]
            y_vals = [self._number(d.get("y", d.get("y_label", 0))) for d in data if isinstance(d, dict)]
            corr_str = ""
            if len(x_vals) == len(y_vals) and len(x_vals) > 2:
                import numpy as np
                corr = np.corrcoef(x_vals, y_vals)[0,1] if np.std(x_vals) > 0 and np.std(y_vals) > 0 else 0
                corr_dir = "positive" if corr > 0 else "negative"
                corr_strength = "strong" if abs(corr) > 0.7 else ("moderate" if abs(corr) > 0.3 else "weak")
                corr_str = f" Correlation: {corr_dir} ({corr:.2f}, {corr_strength})."

            # Outliers in scatter
            outliers = []
            for i, d in enumerate(data):
                if isinstance(d, dict):
                    xv = self._number(d.get("x", d.get("x_label", 0)))
                    yv = self._number(d.get("y", d.get("y_label", 0)))
                    if xv > 2 * (max(x_vals) if x_vals else 1) or yv > 2 * (max(y_vals) if y_vals else 1):
                        outliers.append(d.get("name", f"point {i}"))

            outlier_str = f" Outliers: {', '.join(outliers[:2])}." if outliers else ""

            return (
                f"Key Visual Takeaway: Bivariate relationship between {x_axis or 'X'} and {y_axis or 'Y'} shows {corr_strength if 'corr_strength' in locals() else 'no clear'} association.{corr_str}"
                f" Extremes: Max Y at {max_name} ({max_fmt}); Min Y at {min_name} ({min_fmt}).{outlier_str} "
                f"Distribution: Y values span {self._format_value(min_val, y_axis)}–{self._format_value(max_val, y_axis)}. "
                f"Business Impact: {'Leverage ' + corr_dir + ' correlation for predictive modeling' if 'corr_dir' in locals() and abs(corr) > 0.5 else 'No actionable linear relationship detected — explore non-linear drivers'}."
            )

        # 4. HISTOGRAM / DISTRIBUTION
        elif "histogram" in chart_type_lower or "distribution" in chart_type_lower:
            # Skewness
            if len(values) > 2:
                import numpy as np
                skew = np.mean(((np.array(values) - avg_val) / max(np.std(values), 1)) ** 3)
                skew_dir = "right" if skew > 0.5 else ("left" if skew < -0.5 else "symmetric")
            else:
                skew_dir = "symmetric"

            # Modal bin
            modal_idx = values.index(max_val)
            modal_name = names[modal_idx] if modal_idx < len(names) else "?"

            return (
                f"Key Visual Takeaway: {metric or y_axis} distribution is {skew_dir}-skewed with mode at {modal_name} ({max_fmt}). "
                f"Extremes: Range spans {self._format_value(min_val, y_axis)} to {self._format_value(max_val, y_axis)}. "
                f"Distribution: {len([v for v in values if v > avg_val * 1.5])} bins in upper tail; {len([v for v in values if v < avg_val * 0.5])} in lower tail. "
                f"Domain Implication: {'Long tail indicates high-value outliers — investigate top segments' if skew_dir == 'right' else 'Left skew suggests floor effects or censoring' if skew_dir == 'left' else 'Symmetric spread supports parametric assumptions'}."
            )

        # 5. HEATMAP
        elif "heatmap" in chart_type_lower:
            # Find hotspots
            hotspots = [d for d in data if isinstance(d, dict) and d.get("intensity", 0) > 0.7]
            hotspot_names = [f"{d.get('x','?')}×{d.get('y','?')}" for d in hotspots[:3]]

            return (
                f"Key Visual Takeaway: {len(hotspots)} high-intensity cells ({len(hotspots)/max(len(data),1)*100:.0f}%) indicate strong interaction zones. "
                f"Extremes: Hottest intersection at {hotspot_names[0] if hotspot_names else 'N/A'}; coldest regions show minimal overlap. "
                f"Distribution: Intensity clusters around {hotspot_names[:2] if len(hotspot_names) >= 2 else hotspot_names}. "
                f"Business Impact: {'Prioritize cross-functional initiatives at ' + ', '.join(hotspot_names[:2]) if hotspot_names else 'No strong cross-dimensional signals — dimensions operate independently'}."
            )

        # 6. BOX PLOT
        elif "box" in chart_type_lower or "whisker" in chart_type_lower:
            # data contains min, q1, median, q3, max per category
            if data and isinstance(data[0], dict) and "q1" in data[0]:
                iqrs = [d.get("q3", 0) - d.get("q1", 0) for d in data if isinstance(d, dict)]
                medians = [d.get("median", 0) for d in data if isinstance(d, dict)]
                max_iqr_idx = iqrs.index(max(iqrs)) if iqrs else 0
                max_iqr_name = data[max_iqr_idx].get("name", "?") if max_iqr_idx < len(data) else "?"
                overall_median = sum(medians) / len(medians) if medians else 0

                return (
                    f"Key Visual Takeaway: {max_iqr_name} shows highest variance (IQR={self._format_value(iqrs[max_iqr_idx], y_axis) if iqrs else 'N/A'}). "
                    f"Extremes: Overall median {self._format_value(overall_median, y_axis)}; widest spread in {max_iqr_name}. "
                    f"Distribution: {len([i for i in iqrs if i > sum(iqrs)/len(iqrs)*1.5])} categories exceed 1.5× average IQR. "
                    f"Domain Implication: High-variance segments ({max_iqr_name}) need segmented strategies; low-variance segments allow standardized treatment."
                )

        # Default fallback
        return (
            f"Key Visual Takeaway: {metric or y_axis} spans {self._format_value(min_val, y_axis)}–{self._format_value(max_val, y_axis)}. "
            f"Extremes: Peak {max_fmt} at {max_name}; trough {min_fmt} at {min_name}. "
            f"Distribution: Mean {self._format_value(avg_val, y_axis)} across {len(values)} data points. "
            f"Business Impact: Monitor {max_name} for sustainability; investigate {min_name} for improvement potential."
        )

    # -----------------------------------------------------------------------
    # Aggregation Helpers
    # -----------------------------------------------------------------------
    def _stacked_category_series(
        self,
        df: pd.DataFrame,
        dim1: str,
        dim2: str,
        metric: str,
        agg_func: str = "sum",
        format_type: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Cross-tabulate two dimensions for Stacked Bar and Grouped Bar charts."""
        if dim1 not in df.columns or dim2 not in df.columns or metric not in df.columns:
            return [], []
        frame = df[[dim1, dim2, metric]].dropna()
        if frame.empty:
            return [], []

        top_dim1 = frame.groupby(dim1)[metric].count().nlargest(8).index
        top_dim2 = frame.groupby(dim2)[metric].count().nlargest(5).index
        filtered = frame[frame[dim1].isin(top_dim1) & frame[dim2].isin(top_dim2)]

        agg = "mean" if agg_func in ["mean", "avg", "average"] else "sum"
        pivot = filtered.pivot_table(index=dim1, columns=dim2, values=metric, aggfunc=agg, fill_value=0)
        secondary_keys = [str(c) for c in pivot.columns]

        data = []
        for name, row in pivot.iterrows():
            row_dict = {
                "name": str(name),
                "value": self._number(row.sum()),
                "secondary_keys": secondary_keys,
            }
            for sec_col in secondary_keys:
                row_dict[sec_col] = self._number(row.get(sec_col, 0))
            data.append(row_dict)
        return data, secondary_keys

    def _scatter_series(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        format_type_x: Optional[str] = None,
        format_type_y: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if x not in df.columns or y not in df.columns:
            return []
        frame = df[[x, y]].dropna().head(300)
        return [
            {
                "x": self._number(row[x]),
                "y": self._number(row[y]),
                "z": 100,
                "x_label": x,
                "y_label": y,
                "name": f"Record {idx + 1}",
                "formatted_x": self._format_value(self._number(row[x]), x, format_type_x),
                "formatted_y": self._format_value(self._number(row[y]), y, format_type_y),
            }
            for idx, (_, row) in enumerate(frame.iterrows())
        ]

    def _histogram_series(self, df: pd.DataFrame, metric: str, format_type: Optional[str] = None, unit: Optional[str] = None) -> List[Dict[str, Any]]:
        if metric not in df.columns:
            return []
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            return []

        # Use 5 to 7 clean bins to avoid squished/overlapping labels
        bins_count = 6 if len(values) >= 50 else min(5, max(3, len(values.unique())))
        counts, bin_edges = np.histogram(values, bins=bins_count)
        data = []

        is_all_ints = (values % 1 == 0).all()

        for i in range(len(counts)):
            b_min = float(bin_edges[i])
            b_max = float(bin_edges[i + 1])

            if is_all_ints:
                min_str = f"{int(round(b_min)):,}"
                max_str = f"{int(round(b_max)):,}"
            else:
                min_str = self._format_value(self._number(b_min), metric, format_type, unit)
                max_str = self._format_value(self._number(b_max), metric, format_type, unit)

            label = f"{min_str} - {max_str}"
            data.append({
                "name": label,
                "value": int(counts[i]),
                "count": int(counts[i]),
                "range_min": b_min,
                "range_max": b_max,
                "formatted_value": f"{int(counts[i]):,} records",
            })
        return data

    def _heatmap_series(self, df: pd.DataFrame, dim1: str, dim2: str, metric: str, agg_func: str = "sum") -> Dict[str, Any]:
        if dim1 not in df.columns or dim2 not in df.columns or metric not in df.columns:
            return {"x_labels": [], "y_labels": [], "data": [], "matrix": []}
        frame = df[[dim1, dim2, metric]].dropna()
        if frame.empty:
            return {"x_labels": [], "y_labels": [], "data": [], "matrix": []}

        top_dim1 = frame.groupby(dim1)[metric].count().nlargest(6).index.tolist()
        top_dim2 = frame.groupby(dim2)[metric].count().nlargest(5).index.tolist()
        filtered = frame[frame[dim1].isin(top_dim1) & frame[dim2].isin(top_dim2)]

        agg = "mean" if agg_func in ["mean", "avg"] else "sum"
        pivot = filtered.pivot_table(index=dim2, columns=dim1, values=metric, aggfunc=agg, fill_value=0)
        max_val = float(pivot.max().max()) or 1.0

        items = []
        matrix = []
        for y_idx, (y_name, row) in enumerate(pivot.iterrows()):
            row_vals = []
            for x_idx, (x_name, val) in enumerate(row.items()):
                num = self._number(val)
                intensity = round((num / max_val), 2) if max_val else 0
                row_vals.append(num)
                items.append({
                    "x": str(x_name),
                    "y": str(y_name),
                    "value": num,
                    "intensity": intensity,
                    "formatted_value": self._format_value(num, metric),
                })
            matrix.append(row_vals)

        return {
            "x_labels": [str(x) for x in pivot.columns],
            "y_labels": [str(y) for y in pivot.index],
            "matrix": matrix,
            "data": items,
        }

    def _box_plot_series(self, df: pd.DataFrame, dimension: Optional[str], metric: str) -> List[Dict[str, Any]]:
        if metric not in df.columns:
            return []
        if dimension and dimension in df.columns:
            top_cats = df.groupby(dimension)[metric].count().nlargest(5).index
            data = []
            for cat in top_cats:
                vals = df[df[dimension] == cat][metric].dropna()
                if len(vals) < 2:
                    continue
                q1, median, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.50)), float(vals.quantile(0.75))
                min_val, max_val = float(vals.min()), float(vals.max())
                data.append({
                    "name": str(cat),
                    "min": self._number(min_val),
                    "q1": self._number(q1),
                    "median": self._number(median),
                    "q3": self._number(q3),
                    "max": self._number(max_val),
                    "value": self._number(median),
                    "formatted_value": f"Median: {self._format_value(median, metric)}",
                })
            if data:
                return data

        vals = df[metric].dropna()
        if len(vals) < 2:
            return []
        q1, median, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.50)), float(vals.quantile(0.75))
        return [{
            "name": metric,
            "min": self._number(vals.min()),
            "q1": self._number(q1),
            "median": self._number(median),
            "q3": self._number(q3),
            "max": self._number(vals.max()),
            "value": self._number(median),
            "formatted_value": f"Median: {self._format_value(median, metric)}",
        }]

    # -----------------------------------------------------------------------
    # Master Materializer
    # -----------------------------------------------------------------------
    def materialize_chart(
        self,
        df: pd.DataFrame,
        summary: Dict[str, Any],
        spec: Dict[str, Any],
        chart_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Translate a chart specification into real computed dataframe aggregations using AI parameters."""
        numeric = summary.get("numeric_columns", [])
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])

        metric = self._field(spec.get("y_axis") or spec.get("metric"), numeric, summary.get("primary_kpi"))
        chart_type = str(spec.get("type", "Bar Chart"))
        normalized = chart_type.lower()
        title = spec.get("title")

        agg_func = spec.get("aggregation", "sum")
        format_type = spec.get("format_type")
        unit = spec.get("unit")
        limit = spec.get("limit", 12)
        sort_order = spec.get("sort", "descending")

        # Helper to generate deep insight for a chart
        def gen_insight(chart_type: str, data: List[Dict], x_axis: str, y_axis: str, secondary_dim: Optional[str] = None, metric_name: Optional[str] = None, agg: str = "sum") -> str:
            if spec.get("insight_tooltip"):
                return spec["insight_tooltip"]
            return self._generate_deep_insight(chart_type, data, x_axis, y_axis, secondary_dim, metric_name, agg)

        # 1. SCATTERPLOT
        if "scatter" in normalized and len(numeric) >= 2:
            x = self._field(spec.get("x_axis"), numeric, numeric[0])
            y = self._field(spec.get("y_axis"), numeric, numeric[1] if len(numeric) > 1 else numeric[0])
            if x and y and x in df.columns and y in df.columns:
                data = self._scatter_series(df, x, y, format_type_x=format_type, format_type_y=format_type)
                return {
                    "id": chart_id,
                    "title": title or f"{y} vs {x} Correlation",
                    "type": "Scatterplot",
                    "x_axis": x,
                    "y_axis": y,
                    "data": data,
                    "insight_tooltip": gen_insight("Scatterplot", data, x, y, None, metric, agg_func),
                }

        if not metric or metric not in df.columns:
            return None

        # 2. TIME-SERIES: AREA GRAPH & LINE GRAPH
        if any(t in normalized for t in ["area", "line", "trend"]) and dates:
            dimension = self._field(spec.get("x_axis"), dates, dates[0])
            if dimension and dimension in df.columns:
                data = self._period_series(df, dimension, metric, agg_func=agg_func, format_type=format_type, unit=unit)
                final_type = "Area Graph" if "area" in normalized else "Line Graph"
                if data:
                    return {
                        "id": chart_id,
                        "title": title or f"{metric} Trend over Time",
                        "type": final_type,
                        "x_axis": dimension,
                        "y_axis": metric,
                        "data": data,
                        "insight_tooltip": gen_insight(final_type, data, dimension, metric, None, metric, agg_func),
                    }

        # 3. STACKED BAR GRAPH & MULTI-SET BAR
        if any(t in normalized for t in ["stacked", "multi-set", "multiset", "grouped"]) and len(categories) >= 2:
            dim1 = self._field(spec.get("x_axis"), categories, categories[0])
            dim2_candidates = [c for c in categories if c != dim1]
            dim2 = self._field(spec.get("secondary_dimension"), dim2_candidates, dim2_candidates[0] if dim2_candidates else None)
            if dim1 and dim2:
                data, sec_keys = self._stacked_category_series(df, dim1, dim2, metric, agg_func=agg_func, format_type=format_type, unit=unit)
                final_type = "Stacked Bar Graph" if "stacked" in normalized else "Multi-set Bar Chart"
                if data:
                    return {
                        "id": chart_id,
                        "title": title or f"{metric} by {dim1} and {dim2}",
                        "type": final_type,
                        "x_axis": dim1,
                        "y_axis": metric,
                        "secondary_dimension": dim2,
                        "secondary_keys": sec_keys,
                        "data": data,
                        "insight_tooltip": gen_insight(final_type, data, dim1, metric, dim2, metric, agg_func),
                    }

        # 4. HEATMAP
        if "heatmap" in normalized and len(categories) >= 2:
            dim1 = self._field(spec.get("x_axis"), categories, categories[0])
            dim2_candidates = [c for c in categories if c != dim1]
            dim2 = self._field(spec.get("secondary_dimension"), dim2_candidates, dim2_candidates[0] if dim2_candidates else None)
            if dim1 and dim2:
                heat_dict = self._heatmap_series(df, dim1, dim2, metric, agg_func=agg_func)
                if heat_dict["data"]:
                    return {
                        "id": chart_id,
                        "title": title or f"{metric} Density: {dim1} × {dim2}",
                        "type": "Heatmap",
                        "x_axis": dim1,
                        "y_axis": metric,
                        "secondary_dimension": dim2,
                        "data": heat_dict["data"],
                        "matrix_data": heat_dict,
                        "insight_tooltip": gen_insight("Heatmap", heat_dict["data"], dim1, metric, dim2, metric, agg_func),
                    }

        # 5. BOX PLOT
        if any(t in normalized for t in ["box", "whisker"]):
            dim = self._field(spec.get("x_axis"), categories, categories[0] if categories else None)
            data = self._box_plot_series(df, dim, metric)
            if data:
                return {
                    "id": chart_id,
                    "title": title or f"{metric} Distribution Variance",
                    "type": "Box Plot",
                    "x_axis": dim or "Overall",
                    "y_axis": metric,
                    "data": data,
                    "insight_tooltip": gen_insight("Box Plot", data, dim or "Overall", metric, None, metric, agg_func),
                }

        # 6. HISTOGRAM / DISTRIBUTION
        if any(t in normalized for t in ["histogram", "distribution", "spread", "freq"]):
            data = self._histogram_series(df, metric, format_type=format_type, unit=unit)
            if data:
                return {
                    "id": chart_id,
                    "title": title or f"Distribution of {metric}",
                    "type": "Histogram",
                    "x_axis": "Bins",
                    "y_axis": metric,
                    "data": data,
                    "insight_tooltip": gen_insight("Histogram", data, "Bins", metric, None, metric, agg_func),
                }

        # 7. TREEMAP
        if "treemap" in normalized and categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            cat_data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=min(limit, 10), sort_order=sort_order, format_type=format_type, unit=unit)
            if cat_data:
                treemap_items = [{"name": d["name"], "size": d["value"], "value": d["value"], "percentage": d["percentage"]} for d in cat_data]
                return {
                    "id": chart_id,
                    "title": title or f"{dimension} Volume Breakdown ({metric})",
                    "type": "Treemap",
                    "x_axis": dimension,
                    "y_axis": metric,
                    "data": treemap_items,
                    "insight_tooltip": gen_insight("Treemap", treemap_items, dimension, metric, None, metric, agg_func),
                }

        # 8. RADAR CHART
        if "radar" in normalized and categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            cat_data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=min(limit, 6), sort_order=sort_order, format_type=format_type, unit=unit)
            if cat_data:
                max_val = max(d["value"] for d in cat_data) or 1.0
                radar_items = [{"subject": d["name"], "A": round((d["value"] / max_val) * 100, 1), "value": d["value"], "fullMark": 100} for d in cat_data]
                return {
                    "id": chart_id,
                    "title": title or f"{metric} Multi-Axis Radar",
                    "type": "Radar Chart",
                    "x_axis": dimension,
                    "y_axis": metric,
                    "data": radar_items,
                    "insight_tooltip": gen_insight("Radar Chart", radar_items, dimension, metric, None, metric, agg_func),
                }

        # 9. FUNNEL CHART
        if "funnel" in normalized and categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            cat_data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=min(limit, 6), sort_order=sort_order, format_type=format_type, unit=unit)
            if cat_data:
                return {
                    "id": chart_id,
                    "title": title or f"{metric} Funnel by {dimension}",
                    "type": "Funnel Chart",
                    "x_axis": dimension,
                    "y_axis": metric,
                    "data": cat_data,
                    "insight_tooltip": gen_insight("Funnel Chart", cat_data, dimension, metric, None, metric, agg_func),
                }

        # 10. CATEGORICAL BAR, DONUT, PIE CHART
        if categories:
            dimension = self._field(spec.get("x_axis"), categories, categories[0])
            # Cap donut/pie at top 5 + Other
            is_donut = any(t in normalized for t in ["donut", "pie"])
            data = self._category_series(df, dimension, metric, agg_func=agg_func, max_items=limit, sort_order=sort_order, format_type=format_type, unit=unit, cap_donut=is_donut)
            if not data:
                return None
            final_type = "Donut Chart" if any(t in normalized for t in ["donut", "pie"]) else "Bar Chart"
            return {
                "id": chart_id,
                "title": title or f"{metric} by {dimension}",
                "type": final_type,
                "x_axis": dimension,
                "y_axis": metric,
                "data": data,
                "insight_tooltip": gen_insight(final_type, data, dimension, metric, None, metric, agg_func),
            }

        # Fallback to Histogram
        data = self._histogram_series(df, metric, format_type=format_type, unit=unit)
        if data:
            return {
                "id": chart_id,
                "title": title or f"Distribution of {metric}",
                "type": "Histogram",
                "x_axis": "Bins",
                "y_axis": metric,
                "data": data,
                "insight_tooltip": gen_insight("Histogram", data, "Bins", metric, None, metric, agg_func),
            }

        return None

    def default_plan(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a rich, balanced 4-5 chart default layout on dataset ingestion."""
        # Initialize primary_kpi with safe fallback
        primary_kpi = None
        if summary.get("primary_kpi"):
            primary_kpi = summary["primary_kpi"]
        elif summary.get("numeric_columns"):
            # Fallback to highest variance numeric column
            numeric_cols = summary["numeric_columns"]
            if len(numeric_cols) == 1:
                primary_kpi = numeric_cols[0]
            else:
                # Could compute variance here, but use first for simplicity
                primary_kpi = numeric_cols[0]
        
        metric = primary_kpi
        dates = summary.get("date_columns", [])
        categories = summary.get("categorical_columns", [])
        numeric = summary.get("numeric_columns", [])

        # Determine metric's semantic aggregation
        metric_lower = (metric or "").lower()
        is_avg_metric = any(kw in metric_lower for kw in ['age', 'rating', 'score', 'temperature', 'heartrate', 'rate', 'pct', 'percentage', 'percent', 'ratio', 'margin', 'latency', 'tenure', 'gpa'])
        default_agg = "mean" if is_avg_metric else "sum"

        plan: List[Dict[str, Any]] = []

        # 1. Primary Time Series
        if dates and metric:
            plan.append({
                "type": "Area Graph",
                "title": f"{metric} Trend over Time",
                "x_axis": dates[0],
                "y_axis": metric,
                "aggregation": default_agg,
            })

        # 2. Primary Category Breakdown
        if categories and metric:
            plan.append({
                "type": "Bar Chart",
                "title": f"Average {metric} by {categories[0]}" if default_agg == "mean" else f"{metric} by {categories[0]}",
                "x_axis": categories[0],
                "y_axis": metric,
                "aggregation": default_agg,
            })

        # 3. Market Share / Donut
        if categories:
            cat_dim = categories[1] if len(categories) >= 2 else categories[0]
            if default_agg == "sum" and metric:
                plan.append({
                    "type": "Donut Chart",
                    "title": f"{cat_dim} Share of {metric}",
                    "x_axis": cat_dim,
                    "y_axis": metric,
                    "aggregation": "sum",
                })
            else:
                # For demographic/average metrics, Donut displays category record distribution count
                plan.append({
                    "type": "Donut Chart",
                    "title": f"{cat_dim} Distribution",
                    "x_axis": cat_dim,
                    "y_axis": metric or "Records",
                    "aggregation": "count",
                })

        # 4. Distribution / Histogram
        if metric:
            plan.append({
                "type": "Histogram",
                "title": f"Distribution of {metric}",
                "y_axis": metric,
            })

        # 5. Correlation Scatter (pick 2 numeric metrics, never ID)
        if len(numeric) >= 2:
            x_m = numeric[0]
            y_m = numeric[1]
            plan.append({
                "type": "Scatterplot",
                "title": f"{y_m} vs {x_m} Correlation",
                "x_axis": x_m,
                "y_axis": y_m,
            })
        elif len(categories) >= 2 and metric and len(plan) < 5:
            plan.append({
                "type": "Bar Chart",
                "title": f"{metric} by {categories[1]}",
                "x_axis": categories[1],
                "y_axis": metric,
                "aggregation": default_agg,
            })

        return plan[:5]

    def heuristic_plan(
        self,
        summary: Dict[str, Any],
        query: str,
        intent: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Robust offline rule-based planner covering all query patterns."""
        # Initialize primary_kpi with safe fallback
        primary_kpi = None
        if summary.get("primary_kpi"):
            primary_kpi = summary["primary_kpi"]
        elif summary.get("numeric_columns"):
            primary_kpi = summary["numeric_columns"][0]
        
        lower = query.lower()
        metric = primary_kpi
        dates = summary.get("date_columns", [])
        cats = summary.get("categorical_columns", [])
        numeric = summary.get("numeric_columns", [])

        metric_lower = (metric or "").lower()
        is_avg_metric = any(kw in metric_lower for kw in ['age', 'rating', 'score', 'temp', 'rate', 'pct', 'ratio', 'margin', 'latency', 'tenure'])
        default_agg = "mean" if is_avg_metric else "sum"

        intent_metrics = (intent or {}).get("metrics", [])
        intent_dimensions = (intent or {}).get("dimensions", [])
        time_filter = (intent or {}).get("time_filter")
        comparison = (intent or {}).get("comparison")
        top_bottom = (intent or {}).get("top_bottom")
        is_overview = (intent or {}).get("is_overview", False)

        selected_metric = intent_metrics[0] if intent_metrics else next((c for c in numeric if c.lower() in lower), metric)
        selected_cat = (
            intent_dimensions[0]
            if intent_dimensions
            else next((c for c in cats if c.lower() in lower), cats[0] if cats else None)
        )
        second_cat = intent_dimensions[1] if len(intent_dimensions) > 1 else (cats[1] if len(cats) > 1 else None)

        plans: List[Dict[str, Any]] = []

        # Overview query
        if is_overview or any(w in lower for w in ["everything", "overview", "all charts", "dashboard", "insights"]):
            return self.default_plan(summary)

        # Distribution / Spread / Histogram
        if any(w in lower for w in ["distribution", "spread", "histogram", "freq", "frequency", "variance"]):
            target_metric = selected_metric or (numeric[0] if numeric else None)
            if target_metric:
                plans.append({"type": "Histogram", "title": f"Distribution of {target_metric}", "y_axis": target_metric})

        # Comparison query
        if comparison:
            comp_cols = comparison.get("columns", [])
            for c_metric in comp_cols:
                if c_metric in numeric and selected_cat:
                    plans.append({"type": "Bar Chart", "title": f"{c_metric} by {selected_cat}", "x_axis": selected_cat, "y_axis": c_metric, "aggregation": default_agg})
            if len(comp_cols) >= 2 and all(c in numeric for c in comp_cols[:2]):
                plans.append({"type": "Scatterplot", "title": f"{comp_cols[1]} vs {comp_cols[0]}", "x_axis": comp_cols[0], "y_axis": comp_cols[1]})

        # Time-based query
        has_time = time_filter or any(w in lower for w in ["trend", "forecast", "over time", "month", "year", "growth", "quarter", "timeline", "sales forecast"])
        if has_time and dates and selected_metric:
            ctype = "Area Graph" if "area" in lower else "Line Graph"
            plans.append({"type": ctype, "title": f"{selected_metric} Trend", "x_axis": dates[0], "y_axis": selected_metric, "aggregation": default_agg})

        # Top / Bottom
        if top_bottom and selected_cat and selected_metric:
            direction = top_bottom.get("direction", "top").title()
            n = top_bottom.get("n", 5)
            plans.append({"type": "Bar Chart", "title": f"{direction} {n} {selected_cat} by {selected_metric}", "x_axis": selected_cat, "y_axis": selected_metric, "aggregation": default_agg})

        # Stacked / Heatmap breakdown
        if selected_cat and second_cat and selected_metric and any(w in lower for w in ["breakdown", "stacked", "cross", "matrix", "segment"]):
            plans.append({"type": "Stacked Bar Graph", "title": f"{selected_metric} by {selected_cat} & {second_cat}", "x_axis": selected_cat, "secondary_dimension": second_cat, "y_axis": selected_metric, "aggregation": default_agg})

        # Category breakdown & Donut
        if selected_cat and selected_metric:
            if not any(p.get("type") == "Bar Chart" and p.get("x_axis") == selected_cat for p in plans):
                plans.append({"type": "Bar Chart", "title": f"{selected_metric} by {selected_cat}", "x_axis": selected_cat, "y_axis": selected_metric, "aggregation": default_agg})
            if len(plans) < 5:
                plans.append({"type": "Donut Chart", "title": f"{selected_cat} Share of {selected_metric}", "x_axis": selected_cat, "y_axis": selected_metric, "aggregation": default_agg})

        # Scatter
        if len(numeric) >= 2 and not any(p.get("type") == "Scatterplot" for p in plans) and len(plans) < 5:
            plans.append({"type": "Scatterplot", "title": f"{numeric[1]} vs {numeric[0]}", "x_axis": numeric[0], "y_axis": numeric[1]})

        return (plans or self.default_plan(summary))[:5]

    def make_kpis(self, df: pd.DataFrame, summary: Dict[str, Any], ai_kpi_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Initialize primary_kpi with safe fallback
        primary_kpi = None
        if summary.get("primary_kpi"):
            primary_kpi = summary["primary_kpi"]
        elif summary.get("numeric_columns"):
            primary_kpi = summary["numeric_columns"][0]
        
        metric = (ai_kpi_spec.get("column") if ai_kpi_spec else None) or primary_kpi or "Records"
        
        # Smart aggregation inference:
        metric_lower = (metric or "").lower()
        is_avg = any(kw in metric_lower for kw in ['age', 'rating', 'score', 'temp', 'temperature', 'rate', 'pct', 'percentage', 'ratio', 'margin', 'latency', 'tenure', 'gpa'])
        
        if ai_kpi_spec and ai_kpi_spec.get("aggregation"):
            agg_func = ai_kpi_spec["aggregation"]
        elif is_avg:
            agg_func = "mean"
        else:
            agg_func = "sum"

        format_type = (ai_kpi_spec.get("format_type") if ai_kpi_spec else None) or summary.get("format_type")
        unit = (ai_kpi_spec.get("unit") if ai_kpi_spec else None) or summary.get("unit")

        if metric in df.columns:
            val = self._number(self._apply_aggregation(df, metric, agg_func))
        else:
            val = float(len(df))

        formatted = self._format_value(val, metric, format_type=format_type, unit=unit)

        secondary_metric = (ai_kpi_spec.get("secondary_column") if ai_kpi_spec else None) or summary.get("secondary_kpi")
        secondary_val_str = "—"
        if secondary_metric and secondary_metric in df.columns:
            sec_lower = secondary_metric.lower()
            sec_default_agg = "mean" if any(kw in sec_lower for kw in ['age', 'rating', 'score', 'rate', 'pct', 'salary', 'income']) else "sum"
            sec_agg = (ai_kpi_spec.get("secondary_aggregation") if ai_kpi_spec else None) or sec_default_agg
            sec_val = self._number(self._apply_aggregation(df, secondary_metric, sec_agg))
            secondary_val_str = self._format_value(sec_val, secondary_metric)

        kpi_title = f"Avg {metric}" if (agg_func == "mean" and is_avg) else metric

        return {
            "primary_kpi": kpi_title,
            "metric_column": metric,
            "value": val,
            "formatted_value": formatted,
            "aggregation": agg_func,
            "secondary_kpi": secondary_metric,
            "secondary_formatted_value": secondary_val_str,
            "total_rows": len(df),
            "data_quality": summary.get("quality_score", 100.0),
        }


dashboard_builder = DashboardBuilder()


def extract_rendered_chart_data(charts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract the current rendered state of charts for executive report grounding.
    Returns simplified chart data with title, type, axes, aggregation, and top 10 data points.
    """
    rendered = []
    for chart in charts:
        data = chart.get("data", [])
        if not data or not isinstance(data, list):
            continue
        
        # Extract top 10 data points with key fields
        top_data = []
        for row in data[:10]:
            if not isinstance(row, dict):
                continue
            point = {
                "name": row.get("name", row.get("x", row.get("y", "?"))),
                "value": row.get("value", row.get("y", row.get("value", 0))),
                "percentage": row.get("percentage"),
                "formatted_value": row.get("formatted_value"),
            }
            # Include secondary dimension if present
            if "secondary_keys" in chart:
                for sec_key in chart["secondary_keys"]:
                    if sec_key in row:
                        point[sec_key] = row[sec_key]
            top_data.append(point)
        
        rendered.append({
            "chart_title": chart.get("title", "Chart"),
            "chart_type": chart.get("type", "Unknown"),
            "x_axis": chart.get("x_axis", ""),
            "y_axis": chart.get("y_axis", ""),
            "secondary_dimension": chart.get("secondary_dimension"),
            "aggregation": chart.get("aggregation", "sum"),
            "insight_tooltip": chart.get("insight_tooltip", ""),
            "data_points": top_data,
        })
    return rendered
