"""Predictive Forecasting Agent for Time-Series Business Metrics."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class ForecastingAgent:
    def forecast_metric(
        self,
        historical_values: List[float],
        historical_dates: Optional[List[str]] = None,
        periods: int = 4,
        metric_name: str = "Metric",
    ) -> Dict[str, Any]:
        """Generate predictive time-series projections with 95% confidence intervals and date awareness."""
        if not historical_values or len(historical_values) < 2:
            return None

        vals = np.array([float(v) for v in historical_values if not math.isnan(v)])
        n = len(vals)
        if n < 2:
            return None

        x = np.arange(n)
        slope, intercept = np.polyfit(x, vals, 1)

        residuals = vals - (slope * x + intercept)
        std_err = float(np.std(residuals)) if len(residuals) > 1 else float(vals.mean() * 0.05)
        if std_err == 0:
            std_err = float(vals.mean() * 0.05)

        # Generate future date labels if historical dates are provided
        future_labels = []
        if historical_dates and len(historical_dates) == n:
            last_date_str = historical_dates[-1]
            try:
                # Try parsing last date
                last_dt = pd.to_datetime(last_date_str)
                # Check frequency based on last 2 dates if possible
                if n >= 2:
                    prev_dt = pd.to_datetime(historical_dates[-2])
                    diff_days = max(1, (last_dt - prev_dt).days)
                else:
                    diff_days = 30

                if diff_days > 75:  # Quarterly
                    for i in range(1, periods + 1):
                        next_dt = last_dt + pd.DateOffset(months=3 * i)
                        future_labels.append(f"Q{((next_dt.month - 1) // 3) + 1} {next_dt.year}")
                elif diff_days > 20:  # Monthly
                    for i in range(1, periods + 1):
                        next_dt = last_dt + pd.DateOffset(months=i)
                        future_labels.append(next_dt.strftime("%b %Y"))
                elif diff_days > 5:  # Weekly
                    for i in range(1, periods + 1):
                        next_dt = last_dt + timedelta(weeks=i)
                        future_labels.append(f"Wk of {next_dt.strftime('%b %d')}")
                else:  # Daily
                    for i in range(1, periods + 1):
                        next_dt = last_dt + timedelta(days=i)
                        future_labels.append(next_dt.strftime("%b %d"))
            except Exception:
                pass

        if len(future_labels) < periods:
            future_labels = [f"Period +{i}" for i in range(1, periods + 1)]

        forecast_points = []
        for i in range(1, periods + 1):
            future_x = n - 1 + i
            expected = max(0.0, float(slope * future_x + intercept))
            ci_spread = float(1.96 * std_err * np.sqrt(i))
            best_case = expected + ci_spread
            worst_case = max(0.0, expected - ci_spread)

            forecast_points.append({
                "period": future_labels[i - 1],
                "expected": round(expected, 2),
                "best_case": round(best_case, 2),
                "worst_case": round(worst_case, 2),
            })

        mean_val = float(vals.mean()) or 1.0
        growth_rate_pct = round(float((slope / mean_val) * 100), 2)
        trend_direction = "Bullish Growth" if growth_rate_pct > 2.0 else ("Declining" if growth_rate_pct < -2.0 else "Stable")

        return {
            "forecast_type": "Linear Trend with 95% Confidence Intervals",
            "historical_mean": round(mean_val, 2),
            "projected_growth_rate": f"{growth_rate_pct:+.1f}%",
            "trend_direction": trend_direction,
            "forecast_points": forecast_points,
            "ai_summary": f"{metric_name} demonstrates a {trend_direction.lower()} trajectory projected at {growth_rate_pct:+.1f}% change across the next {periods} periods.",
        }


forecasting_agent = ForecastingAgent()
