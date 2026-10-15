import pandas as pd
import numpy as np
import io
import re
from typing import Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# LLM payload sizing / privacy controls
#
# These bound the size of the dataset summary that is sent to the LLM so we
# never ship the full dataset, and guarantee that detected PII is masked
# before it leaves the process.
# ---------------------------------------------------------------------------
REPRESENTATIVE_SAMPLE_MIN = 50
REPRESENTATIVE_SAMPLE_MAX = 100
REPRESENTATIVE_SAMPLE_DEFAULT = 60

TOP_VALUES_MIN = 5
TOP_VALUES_MAX = 10
TOP_VALUES_DEFAULT = 8

# Any column whose PII confidence meets this bar is masked in LLM payloads.
PII_MASK_CONFIDENCE_THRESHOLD = 0.6


class LocalDatasetProfiler:
    def __init__(self):
        self._pii_name_patterns = [
            (r'(email|e_mail|mail)', 'email', 0.9),
            (r'(phone|tel|mobile|cell)', 'phone', 0.9),
            (r'(ssn|social_security|socialsecurity)', 'ssn', 0.95),
            (r'(credit_card|creditcard|cc_num|card_number|cc\b|card\b)', 'credit_card', 0.9),
            (r'(passport|passport_num)', 'passport', 0.85),
            (r'(driver_license|driverlicense|dl_num)', 'driver_license', 0.85),
            (r'(address|street|zip|postal)', 'address', 0.7),
            (r'(name|first_name|last_name|full_name|fname|lname)', 'name', 0.6),
            (r'(dob|date_of_birth|birth)', 'dob', 0.8),
            (r'(ip_address|ip\b)', 'ip_address', 0.7),
        ]

        self._pii_value_patterns = [
            (re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'), 'email', 0.95),
            (re.compile(r'^(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$'), 'phone', 0.9),
            (re.compile(r'^\d{3}-\d{2}-\d{4}$'), 'ssn', 0.95),
            (re.compile(r'^(\d{4}[- ]?){3}\d{4}$|^\d{13,16}$'), 'credit_card', 0.85),
            (re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'), 'ip_address', 0.8),
        ]

    def _calculate_skewness(self, series: pd.Series) -> float:
        """Calculate skewness of a numeric series using pandas built-in."""
        clean = series.dropna()
        if len(clean) < 3:
            return 0.0
        try:
            return float(clean.skew())
        except Exception:
            return 0.0

    def _calculate_column_statistics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate statistical metrics for KPI ranking: variance, cardinality, null ratio, entropy."""
        clean = series.dropna()
        n = len(clean)
        if n == 0:
            return {
                "variance_normalized": 0.0,
                "cardinality_ratio": 0.0,
                "null_ratio": 1.0,
                "entropy": 0.0
            }

        # Variance normalized by mean^2 for scale invariance (coefficient of variation squared)
        mean = clean.mean()
        var = clean.var()
        variance_normalized = float(var / (mean**2 + 1e-10)) if mean != 0 else 0.0

        # Cardinality ratio (unique values / total non-null)
        cardinality_ratio = float(clean.nunique() / n)

        # Null ratio
        null_ratio = float(series.isnull().mean())

        # Entropy: bin continuous values into 10 quantiles, compute Shannon entropy
        entropy = 0.0
        try:
            if n >= 10:
                binned = pd.qcut(clean, q=10, duplicates='drop')
                value_counts = binned.value_counts(normalize=True)
                # Filter out zero probabilities to avoid log2(0)
                value_counts = value_counts[value_counts > 0]
                entropy = float(-(value_counts * np.log2(value_counts)).sum())
            else:
                # For small series, use value counts directly
                value_counts = clean.value_counts(normalize=True)
                value_counts = value_counts[value_counts > 0]
                entropy = float(-(value_counts * np.log2(value_counts)).sum())
        except Exception:
            entropy = 0.0

        return {
            "variance_normalized": round(variance_normalized, 6),
            "cardinality_ratio": round(cardinality_ratio, 6),
            "null_ratio": round(null_ratio, 6),
            "entropy": round(entropy, 6)
        }

    def _is_id_column(self, col_name: str, series: pd.Series, row_count: int) -> bool:
        """Check if column is an ID column (excluded from KPI selection)."""
        col_lower = col_name.lower()
        # Name-based patterns
        if col_lower in ('id', 'uuid', 'code', 'index'):
            return True
        if col_lower.endswith(('_id', '-id', ' id', '_uuid', '-uuid', '_code', '-code', '_index', '-index')):
            return True
        if col_lower.startswith(('id_', 'id-', 'uuid_', 'uuid-', 'code_', 'code-', 'index_', 'index-')):
            return True
        # Unique count equals row count (perfect unique identifier) - ONLY for discrete/integer columns
        # Continuous float metrics naturally have high uniqueness
        is_discrete = pd.api.types.is_integer_dtype(series.dtype) or (series.dropna() % 1 == 0).all()
        if is_discrete and series.nunique() == row_count:
            return True
        return False

    def _is_binary_target(self, series: pd.Series) -> bool:
        """Check if a numeric column is a binary target (values strictly in {0, 1})."""
        clean = series.dropna()
        if len(clean) == 0:
            return False
        unique_vals = set(clean.unique())
        return unique_vals.issubset({0, 1, 0.0, 1.0}) and len(unique_vals) <= 2

    def _select_primary_kpi_local(self, df: pd.DataFrame, numeric_cols: List[str], col_meta_dict: Dict, row_count: int) -> Optional[Dict[str, Any]]:
        """
        Instant local statistical KPI selection engine.
        Returns dict with primary_kpi info including score, formatting hints, etc.
        """
        candidates = []
        
        for col in numeric_cols:
            meta = col_meta_dict.get(col, {})
            series = df[col]
            
            # ID Exclusion: skip ID columns
            if self._is_id_column(col, series, row_count):
                continue
                
            stats = meta.get("statistics", {})
            mean = meta.get("mean", 0.0)
            std = meta.get("std", 0.0)
            null_ratio = stats.get("null_ratio", 1.0)
            cardinality_ratio = stats.get("cardinality_ratio", 0.0)
            entropy = stats.get("entropy", 0.0)
            
            # Binary Outcome Detection
            if self._is_binary_target(series):
                # Score based on entropy around 0.5 mean (maximally informative)
                score = 2.0 * (1.0 - abs(0.5 - mean))
                candidates.append({
                    "name": col,
                    "score": round(score, 6),
                    "is_binary_target": True,
                    "display_name": f"{col.replace('_', ' ').title()} Rate",
                    "format_type": "percentage",
                    "unit": "%",
                    "aggregation": "mean",
                    "semantic_role": "target_rate",
                    "kpi_metadata": {
                        "mean": mean,
                        "entropy_score": round(1.0 - abs(0.5 - mean), 4)
                    }
                })
            else:
                # Continuous Metric Scoring
                # score = (std / (mean + 1e-6)) * (1 - null_ratio) * normalized_entropy
                cv = std / (abs(mean) + 1e-6) if mean != 0 else 0.0
                
                # Normalize entropy to 0-1 range (max entropy for 10 bins = log2(10) ≈ 3.32)
                max_entropy = np.log2(10)
                normalized_entropy = min(entropy / max_entropy, 1.0) if max_entropy > 0 else 0.0
                
                score = cv * (1.0 - null_ratio) * normalized_entropy
                
                # Cardinality penalty ONLY for discrete integer-like columns with cardinality_ratio > 0.90
                # (potential unrecognized ID/timestamp). Continuous float metrics naturally have high cardinality.
                is_integer_like = pd.api.types.is_integer_dtype(series.dtype) or (series.dropna() % 1 == 0).all()
                cardinality_penalty_applied = False
                if cardinality_ratio > 0.90 and is_integer_like:
                    score *= 0.1  # Heavy penalty for near-unique discrete columns
                    cardinality_penalty_applied = True
                
                if score > 0:
                    candidates.append({
                        "name": col,
                        "score": round(score, 6),
                        "is_binary_target": False,
                        "display_name": col.replace('_', ' ').title(),
                        "format_type": "currency" if any(kw in col.lower() for kw in ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'spend', 'income', 'budget', 'margin', 'salary', 'balance']) else "decimal",
                        "unit": "$" if any(kw in col.lower() for kw in ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'spend', 'income', 'budget', 'margin', 'salary', 'balance']) else "",
                        "aggregation": "sum" if any(kw in col.lower() for kw in ['revenue', 'sales', 'profit', 'amount', 'price', 'cost', 'spend', 'income', 'budget', 'margin', 'quantity', 'units', 'volume', 'count']) else "mean",
                        "semantic_role": "measure",
                        "kpi_metadata": {
                            "cv": round(cv, 4),
                            "null_ratio": null_ratio,
                            "normalized_entropy": round(normalized_entropy, 4),
                            "cardinality_penalty": cardinality_penalty_applied
                        }
                    })
        
        if not candidates:
            return None
            
        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        
        return {
            "primary_kpi": best["name"],
            "score": best["score"],
            "is_binary_target": best["is_binary_target"],
            "display_name": best["display_name"],
            "format_type": best["format_type"],
            "unit": best["unit"],
            "aggregation": best["aggregation"],
            "semantic_role": best["semantic_role"],
            "kpi_metadata": best.get("kpi_metadata", {}),
            "all_candidates": candidates  # For debugging/transparency
        }

    def _build_available_kpi_options(self, kpi_selection_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build simplified KPI options for frontend dropdown from kpi_selection_details.
        Returns a list of available KPI candidates with their formatting metadata.
        """
        if not kpi_selection_details or "all_candidates" not in kpi_selection_details:
            return []
        
        options = []
        for candidate in kpi_selection_details["all_candidates"]:
            # Only include candidates with positive scores
            if candidate.get("score", 0) > 0:
                options.append({
                    "column": candidate["name"],
                    "display_name": candidate["display_name"],
                    "format_type": candidate["format_type"],
                    "unit": candidate["unit"],
                    "aggregation": candidate["aggregation"],
                    "semantic_role": candidate["semantic_role"],
                    "is_binary": candidate.get("is_binary_target", False),
                    "score": candidate["score"]
                })
        
        # Sort by score descending (best first)
        options.sort(key=lambda x: x.get("score", 0), reverse=True)
        return options

    def _detect_anomaly_spikes(self, series: pd.Series, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomaly spikes using z-score method (manual implementation)."""
        clean = series.dropna()
        if len(clean) < 10:
            return []
        try:
            mean = clean.mean()
            std = clean.std()
            if std == 0:
                return []
            z_scores = np.abs((clean - mean) / std)
            spike_indices = np.where(z_scores > threshold)[0]
            spikes = []
            for idx in spike_indices[:5]:
                spikes.append({
                    "index": int(idx),
                    "value": float(clean.iloc[idx]),
                    "z_score": round(float(z_scores.iloc[idx]), 2)
                })
            return spikes
        except Exception:
            return []
            return []

    def _calculate_segment_concentration(self, df: pd.DataFrame, cat_col: str, metric_col: str) -> Dict[str, Any]:
        """Calculate segment concentration (e.g., top N categories share of total)."""
        try:
            grouped = df.groupby(cat_col)[metric_col].sum().sort_values(ascending=False)
            total = grouped.sum()
            if total == 0:
                return {}
            
            cumsum = grouped.cumsum()
            top2_share = (cumsum.iloc[1] / total * 100) if len(cumsum) > 1 else 0
            top3_share = (cumsum.iloc[2] / total * 100) if len(cumsum) > 2 else 0
            top5_share = (cumsum.iloc[4] / total * 100) if len(cumsum) > 4 else 0
            
            return {
                "top_1_share_pct": round(float(grouped.iloc[0] / total * 100), 1),
                "top_2_share_pct": round(float(top2_share), 1),
                "top_3_share_pct": round(float(top3_share), 1),
                "top_5_share_pct": round(float(top5_share), 1),
                "concentration_risk": "high" if top2_share > 60 else ("medium" if top2_share > 40 else "low"),
                "herfindahl_index": round(float(((grouped / total) ** 2).sum()), 3)
            }
        except Exception:
            return {}

    def _calculate_period_over_period_growth(self, df: pd.DataFrame, date_col: str, metric_col: str, 
                                               periods: int = 4) -> List[Dict[str, Any]]:
        """Calculate period-over-period growth trends."""
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[date_col])
            if df_temp.empty:
                return []
            
            df_temp['period'] = df_temp[date_col].dt.to_period('M')
            monthly = df_temp.groupby('period', sort=True)[metric_col].sum()
            
            if len(monthly) < 2:
                return []
            
            growth_data = []
            for i in range(1, min(len(monthly), periods + 1)):
                curr_period = monthly.index[-i]
                prev_period = monthly.index[-i-1] if i < len(monthly) else None
                curr_val = monthly.iloc[-i]
                prev_val = monthly.iloc[-i-1] if i < len(monthly) else 0
                
                if prev_val != 0:
                    growth_pct = round(float((curr_val - prev_val) / abs(prev_val) * 100), 1)
                else:
                    growth_pct = 0.0
                
                growth_data.append({
                    "period": str(curr_period),
                    "value": round(float(curr_val), 2),
                    "previous_value": round(float(prev_val), 2),
                    "growth_pct": growth_pct,
                    "direction": "up" if growth_pct > 0 else ("down" if growth_pct < 0 else "flat")
                })
            
            return list(reversed(growth_data))
        except Exception:
            return []

    def _calculate_rolling_trend(self, df: pd.DataFrame, date_col: str, metric_col: str, window: int = 3) -> Dict[str, Any]:
        """Calculate rolling trend statistics."""
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[date_col])
            if df_temp.empty:
                return {}
            
            df_temp['period'] = df_temp[date_col].dt.to_period('M')
            monthly = df_temp.groupby('period', sort=True)[metric_col].sum().reset_index()
            
            if len(monthly) < window:
                return {}
            
            monthly['rolling_mean'] = monthly[metric_col].rolling(window=window).mean()
            monthly['rolling_std'] = monthly[metric_col].rolling(window=window).std()
            
            latest_trend = monthly.iloc[-1]
            prev_trend = monthly.iloc[-window] if len(monthly) >= window else monthly.iloc[0]
            
            trend_direction = "improving" if latest_trend['rolling_mean'] > prev_trend['rolling_mean'] else "declining"
            volatility = round(float(latest_trend['rolling_std'] / latest_trend['rolling_mean'] * 100), 1) if latest_trend['rolling_mean'] != 0 else 0
            
            return {
                "current_rolling_avg": round(float(latest_trend['rolling_mean']), 2),
                "trend_direction": trend_direction,
                "volatility_pct": volatility,
                "window_months": window
            }
        except Exception:
            return {}

    def _detect_pii(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        pii_flags = []
        for col in df.columns:
            col_lower = str(col).lower()
            max_confidence = 0.0
            reasons = []

            for pattern, ptype, conf in self._pii_name_patterns:
                if re.search(pattern, col_lower):
                    if conf > max_confidence:
                        max_confidence = conf
                    reasons.append(f"column name matches {ptype} pattern")

            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                sample_vals = df[col].dropna().astype(str).head(50)
                for regex, ptype, conf in self._pii_value_patterns:
                    matches = sample_vals.apply(lambda x: bool(regex.match(x))).sum()
                    if matches > 0:
                        match_ratio = matches / len(sample_vals)
                        val_conf = conf * min(1.0, match_ratio * 2)
                        if val_conf > max_confidence:
                            max_confidence = val_conf
                        reasons.append(f"{matches}/{len(sample_vals)} values match {ptype} format")

            if max_confidence > 0.0:
                pii_flags.append({
                    "column": str(col),
                    "reason": "; ".join(reasons),
                    "confidence": round(max_confidence, 2)
                })
        return pii_flags

    def _mask_value(self, value: str, ptype: str) -> str:
        if ptype == 'email':
            parts = value.split('@')
            if len(parts) == 2:
                return f"{parts[0][0]}***@***.{parts[1].split('.')[-1]}"
            return "***@***.***"
        elif ptype == 'phone':
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
            return "***-***-****"
        elif ptype == 'ssn':
            return "***-**-" + value[-4:] if len(value) >= 4 else "***-**-****"
        elif ptype == 'credit_card':
            return "**** **** **** " + value[-4:] if len(value) >= 4 else "**** **** **** ****"
        elif ptype == 'ip_address':
            parts = value.split('.')
            return f"***.***.***.{parts[-1]}" if len(parts) == 4 else "***.***.***.***"
        else:
            return value[0] + "***" if len(value) > 0 else "***"

    def _get_pii_type(self, reasons: str) -> str:
        for ptype in ['email', 'phone', 'ssn', 'credit_card', 'ip_address', 'passport', 'driver_license', 'address', 'name', 'dob']:
            if ptype in reasons.lower():
                return ptype
        return 'unknown'

    # ------------------------------------------------------------------
    # Enriched profile helpers + representative sampling (LLM payloads)
    # ------------------------------------------------------------------
    def _quartiles(self, series: pd.Series) -> Tuple[float, float]:
        """Return (q1, q3) for a numeric series, robust to empty/all-null."""
        clean = series.dropna()
        if clean.empty:
            return 0.0, 0.0
        try:
            return round(float(clean.quantile(0.25)), 4), round(float(clean.quantile(0.75)), 4)
        except Exception:
            return 0.0, 0.0

    def _categorical_top_values(self, series: pd.Series,
                                limit: int = TOP_VALUES_DEFAULT) -> List[Dict[str, Any]]:
        """Top-N value counts with counts + percentages (bounded output)."""
        limit = max(TOP_VALUES_MIN, min(TOP_VALUES_MAX, int(limit)))
        clean = series.dropna()
        if clean.empty:
            return []
        try:
            counts = clean.astype(str).value_counts()
        except Exception:
            return []
        total = int(counts.sum()) or 1
        return [
            {"value": str(value), "count": int(count), "pct": round(float(count) / total * 100, 2)}
            for value, count in counts.head(limit).items()
        ]

    def _date_range(self, series: pd.Series) -> Dict[str, Any]:
        """Min/max date (ISO) and span in days for a date-like column."""
        clean = series.dropna()
        if clean.empty:
            return {}
        try:
            parsed = pd.to_datetime(clean, errors="coerce", format="mixed").dropna()
        except Exception:
            parsed = pd.to_datetime(clean, errors="coerce").dropna()
        if parsed.empty:
            return {}
        lo, hi = parsed.min(), parsed.max()
        try:
            span = int((hi - lo).days)
        except Exception:
            span = None
        return {"date_min": str(lo)[:10], "date_max": str(hi)[:10], "span_days": span}

    def _pii_column_map(self, pii_flags: List[Dict[str, Any]]) -> Dict[str, str]:
        """Map column name -> PII type for columns that clear the mask threshold."""
        mapping: Dict[str, str] = {}
        for flag in pii_flags or []:
            try:
                confidence = float(flag.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence >= PII_MASK_CONFIDENCE_THRESHOLD:
                mapping[str(flag.get("column"))] = self._get_pii_type(flag.get("reason", ""))
        return mapping

    def _mask_scalar(self, value: Any, pii_type: Optional[str]) -> Any:
        """Mask a single value when its column is flagged as PII."""
        if not pii_type or value is None:
            return value
        text = str(value)
        if not text.strip():
            return value
        return self._mask_value(text, pii_type)

    def _mask_records(self, records: List[Dict[str, Any]],
                      pii_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Return a copy of records with PII columns masked."""
        if not pii_map or not records:
            return records
        masked: List[Dict[str, Any]] = []
        for record in records:
            masked.append({
                key: self._mask_scalar(value, pii_map.get(str(key)))
                for key, value in record.items()
            })
        return masked

    def _records_from_df(self, frame: pd.DataFrame) -> List[Dict[str, Any]]:
        """Serialize a frame to JSON-safe records (datetimes as strings, NaN->None)."""
        if frame is None or frame.empty:
            return []
        out = frame.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].astype(str)
        try:
            return out.replace({np.nan: None}).to_dict(orient="records")
        except Exception:
            return []

    @staticmethod
    def _first_parseable_date_col(frame: pd.DataFrame, date_cols: List[str]) -> Optional[str]:
        for col in date_cols or []:
            if col in frame.columns and frame[col].dropna().nunique() > 1:
                return col
        return None

    @staticmethod
    def _first_stratify_col(frame: pd.DataFrame, categorical_cols: List[str],
                            target: int) -> Optional[str]:
        for col in categorical_cols or []:
            if col not in frame.columns:
                continue
            try:
                nunique = frame[col].nunique(dropna=True)
            except Exception:
                continue
            if 2 <= nunique <= max(2, target):
                return col
        return None

    def build_representative_sample(self, df: pd.DataFrame,
                                    numeric_cols: List[str],
                                    date_cols: List[str],
                                    categorical_cols: List[str],
                                    n: int = REPRESENTATIVE_SAMPLE_DEFAULT,
                                    random_state: int = 42) -> Tuple[List[Dict[str, Any]], str]:
        """
        Select a small, representative slice of the dataset for the LLM.

        Returns ``(records, method)`` where method is one of
        ``temporal`` / ``stratified`` / ``random`` / ``empty``. Sampling is
        reproducible (fixed ``random_state``) and de-duplicated so the same row
        is never sent twice.
        """
        n = int(max(REPRESENTATIVE_SAMPLE_MIN, min(REPRESENTATIVE_SAMPLE_MAX, n)))
        if df is None or len(df) == 0:
            return [], "empty"

        try:
            pool = df.drop_duplicates()
        except Exception:
            pool = df
        if pool is None or pool.empty:
            pool = df
        target = min(n, len(pool))

        # 1) Temporal sampling — even coverage across the time axis.
        date_col = self._first_parseable_date_col(pool, date_cols)
        if date_col is not None and len(pool) > target:
            try:
                parsed = pd.to_datetime(pool[date_col], errors="coerce", format="mixed")
                valid = pool.loc[parsed.notna()].copy()
                valid["__ts__"] = parsed.loc[parsed.notna()].astype("int64")
                if len(valid) >= target:
                    valid = valid.sort_values("__ts__")
                    buckets = pd.qcut(valid["__ts__"], q=target, duplicates="drop")
                    picked = valid.groupby(buckets, observed=True).head(1)
                    if len(picked) < target:
                        remaining = valid.drop(index=picked.index)
                        if not remaining.empty:
                            step = max(1, len(remaining) // max(1, target - len(picked)))
                            picked = pd.concat([picked, remaining.iloc[::step]]).drop_duplicates().head(target)
                    if len(picked) >= max(1, target // 2):
                        return self._records_from_df(picked.drop(columns="__ts__")), "temporal"
            except Exception:
                pass

        # 2) Stratified sampling across a low-cardinality categorical column.
        cat_col = self._first_stratify_col(pool, categorical_cols, target)
        if cat_col is not None:
            try:
                total = len(pool)
                frames = []
                for _, group in pool.groupby(cat_col, dropna=False, observed=True):
                    k = min(max(1, int(round(target * len(group) / total))), len(group))
                    frames.append(group.sample(n=k, random_state=random_state))
                sample = pd.concat(frames).drop_duplicates().head(target)
                if len(sample) < target:
                    remaining = pool.drop(index=sample.index)
                    if not remaining.empty:
                        sample = pd.concat([
                            sample,
                            remaining.sample(n=min(target - len(sample), len(remaining)),
                                             random_state=random_state),
                        ])
                return self._records_from_df(sample.head(target)), "stratified"
            except Exception:
                pass

        # 3) Uniform random sample (deterministic).
        try:
            sample = pool.sample(n=target, random_state=random_state)
        except Exception:
            sample = pool.head(target)
        return self._records_from_df(sample), "random"

    def build_llm_profile(self, summary: Dict[str, Any], pii_map: Dict[str, str],
                          top_n: int = TOP_VALUES_DEFAULT) -> Dict[str, Any]:
        """Compact, PII-masked column profile for the LLM prompt."""
        numeric = set(summary.get("numeric_columns", []))
        dates = set(summary.get("date_columns", []))
        cats = set(summary.get("categorical_columns", []))

        columns = []
        for col in summary.get("columns", []):
            if not isinstance(col, dict):
                continue
            name = str(col.get("name"))
            pii_type = pii_map.get(name)
            entry: Dict[str, Any] = {
                "name": name,
                "dtype": col.get("type"),
                "semantic_role": col.get("semantic_role"),
                "missing_pct": col.get("missing_pct"),
                "samples": [self._mask_scalar(v, pii_type) for v in (col.get("sample_values") or [])[:3]],
            }
            if name in numeric:
                for key in ("min", "max", "mean", "median", "std", "q1", "q3"):
                    if key in col:
                        entry[key] = col[key]
            if name in cats or name in dates:
                entry["unique_count"] = col.get("unique_count")
            if name in cats:
                entry["top_values"] = [
                    {
                        "value": self._mask_scalar(tv.get("value"), pii_type),
                        "count": tv.get("count"),
                        "pct": tv.get("pct"),
                    }
                    for tv in (col.get("top_values") or [])[:top_n]
                    if isinstance(tv, dict)
                ]
            if name in dates:
                for key in ("date_min", "date_max"):
                    if key in col:
                        entry[key] = col[key]
            columns.append(entry)

        kpi_options = sorted(
            summary.get("available_kpi_options", []) or [],
            key=lambda x: x.get("score", 0),
            reverse=True,
        )[:8]
        kpi_candidates = [
            {
                "name": k.get("column"),
                "score": round(float(k["score"]), 4) if isinstance(k.get("score"), (int, float, np.number)) else k.get("score"),
                "aggregation": k.get("aggregation"),
                "format_type": k.get("format_type"),
                "unit": k.get("unit"),
                "semantic_role": k.get("semantic_role"),
            }
            for k in kpi_options
        ]

        return {
            "filename": summary.get("filename"),
            "row_count": summary.get("row_count"),
            "column_count": summary.get("column_count"),
            "quality_score": summary.get("quality_score"),
            "primary_kpi": summary.get("primary_kpi"),
            "numeric_columns": summary.get("numeric_columns", []),
            "date_columns": summary.get("date_columns", []),
            "categorical_columns": summary.get("categorical_columns", []),
            "columns": columns,
            "kpi_candidates": kpi_candidates,
        }

    def profile_csv(self, file_contents: bytes, filename: str, dataframe: pd.DataFrame | None = None,
                    kpi_keywords: Optional[List[str]] = None,
                    geo_keywords: Optional[List[str]] = None,
                    cleaning_report: Optional[Dict[str, Any]] = None,
                    mask_pii: bool = False,
                    was_sampled: bool = False,
                    sample_size: int = 0,
                    total_rows: int = 0) -> Dict[str, Any]:
        """
        Parses a CSV dataset locally using Pandas/NumPy, extracting a rich,
        compact JSON summary to send to NVIDIA Nemotron without sending raw files.
        """
        df = dataframe.copy() if dataframe is not None else pd.read_csv(io.BytesIO(file_contents))

        row_count, col_count = df.shape
        columns_info: List[Dict[str, Any]] = []
        numeric_cols = []
        date_cols = []
        categorical_cols = []
        detected_kpis = []

        # Default geo keywords only (KPI keywords no longer used - LLM selects primary KPI)
        default_geo_keywords = ['city', 'state', 'country', 'region', 'zip', 'lat', 'lon', 'location']
        geo_kw = [kw.lower() for kw in (geo_keywords or default_geo_keywords)]

        # Pre-detect date columns (fast sampled check)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(str(col))
            elif not pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                col_lower = str(col).lower()
                is_date_name = any(kw in col_lower for kw in ["date", "time", "year", "month", "day", "timestamp", "period"])
                sample = df[col].dropna().head(20)
                try:
                    parsed_dates = pd.to_datetime(sample, errors='coerce', format='mixed')
                    if parsed_dates.notna().mean() >= 0.8 or (is_date_name and parsed_dates.notna().mean() >= 0.5):
                        date_cols.append(str(col))
                except Exception:
                    pass

        for col in df.columns:
            col_lower = str(col).lower()
            dtype_str = str(df[col].dtype)
            missing_pct = round(float(df[col].isnull().mean() * 100), 2)
            unique_count = int(df[col].nunique())
            sample_vals = df[col].dropna().unique()[:3].tolist()
            sample_vals_clean = [str(v) for v in sample_vals]

            col_meta = {
                "name": str(col),
                "type": dtype_str,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "sample_values": sample_vals_clean
            }

            # Date detection (already collected)
            if str(col) in date_cols:
                col_meta["semantic_role"] = "date"
                col_meta.update(self._date_range(df[col]))
            # Numeric analysis
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_meta["mean"] = float(df[col].mean()) if not df[col].isnull().all() else 0.0
                col_meta["std"] = float(df[col].std()) if not df[col].isnull().all() else 0.0
                col_meta["min"] = float(df[col].min()) if not df[col].isnull().all() else 0.0
                col_meta["max"] = float(df[col].max()) if not df[col].isnull().all() else 0.0
                col_meta["median"] = float(df[col].median()) if not df[col].isnull().all() else 0.0
                q1, q3 = self._quartiles(df[col])
                col_meta["q1"] = q1
                col_meta["q3"] = q3

                # Calculate statistical metrics for KPI ranking
                col_meta["statistics"] = self._calculate_column_statistics(df[col])

                # Robust ID detection (handles 'User ID', 'user_id', 'ID', 'CustomerID', etc.)
                is_id_like = bool(
                    re.search(r'(^|[_\s\-])(id|key|code|zip|postal|row|index|num|number|identifier)($|[_\s\-])', col_lower)
                    or col_lower.endswith((' id', '_id', '-id', ' key', '_key'))
                    or col_lower.startswith(('id ', 'id_', 'id-'))
                    or col_lower in ('id', 'userid', 'user_id', 'user id', 'customerid', 'customer_id', 'customer id', 'orderid', 'order_id', 'order id', 'patientid', 'patient_id', 'patient id')
                )
                is_discrete_date = bool(re.search(r'(^|[_\s\-])(year|yr|month|day|quarter|qtr|week|hour|minute)($|[_\s\-])', col_lower))

                unique_ratio = unique_count / row_count if row_count else 1.0

                if is_id_like:
                    kpi_conf = 0.0
                    col_meta["semantic_role"] = "identifier"
                    col_meta["default_aggregation"] = "count"
                elif is_discrete_date:
                    kpi_conf = 0.0
                    col_meta["semantic_role"] = "temporal_discrete"
                    col_meta["default_aggregation"] = "count"
                    numeric_cols.append(str(col))
                else:
                    numeric_cols.append(str(col))
                    # Neutral confidence - LLM will select primary KPI based on statistics and domain
                    kpi_conf = 0.5
                    col_meta["semantic_role"] = "measure"
                    col_meta["default_aggregation"] = "sum"

                col_meta["kpi_confidence"] = round(kpi_conf, 2)
                if kpi_conf > 0.0:
                    detected_kpis.append((str(col), kpi_conf))
            # Categorical / geo analysis
            elif any(kw in col_lower for kw in geo_kw):
                categorical_cols.append(str(col))
                col_meta["semantic_role"] = "geographical"
                col_meta["geo_confidence"] = 1.0
            else:
                # fallback pattern detection for geo
                pattern_match = False
                if (pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object) and df[col].notna().any():
                    for val in sample_vals_clean:
                        if re.fullmatch(r'[A-Z]{2}|[A-Z]{3}', val):
                            pattern_match = True
                            break
                        if val.lower() in {"united states", "usa", "canada", "mexico", "uk", "germany", "france", "india", "china", "japan", "australia", "brazil"}:
                            pattern_match = True
                            break
                if pattern_match:
                    categorical_cols.append(str(col))
                    col_meta["semantic_role"] = "geographical"
                    col_meta["geo_confidence"] = 0.5
                else:
                    categorical_cols.append(str(col))
                    col_meta["semantic_role"] = "dimension"
                    col_meta["geo_confidence"] = 0.0

            # Categorical distribution (bounded to TOP_VALUES_* entries) so the
            # LLM can reason about dominant values without seeing the whole column.
            if col_meta.get("semantic_role") in ("dimension", "geographical"):
                col_meta["top_values"] = self._categorical_top_values(df[col])

            columns_info.append(col_meta)

        # PII Detection
        pii_flags = self._detect_pii(df)
        # Columns whose PII confidence clears the bar are masked in every
        # LLM-bound structure (samples, distributions, sample rows).
        pii_map = self._pii_column_map(pii_flags)

        # Build masked preview if requested
        masked_preview = {}
        if mask_pii:
            for flag in pii_flags:
                if flag["confidence"] >= 0.7:
                    col_name = flag["column"]
                    ptype = self._get_pii_type(flag["reason"])
                    sample_vals = df[col_name].dropna().astype(str).head(5).tolist()
                    masked_preview[col_name] = [self._mask_value(v, ptype) for v in sample_vals]

        # Build kpi_candidates for local statistical primary KPI selection
        col_meta_dict = {c["name"]: c for c in columns_info}
        kpi_candidates = []
        for col in numeric_cols:
            meta = col_meta_dict.get(col, {})
            kpi_candidates.append({
                "name": col,
                "dtype": meta.get("type", "unknown"),
                "semantic_role": meta.get("semantic_role", "measure"),
                "default_aggregation": meta.get("default_aggregation", "sum"),
                "current_confidence": meta.get("kpi_confidence", 0.5),
                "statistics": meta.get("statistics", {}),
                "sample_values": meta.get("sample_values", []),
                "unique_count": meta.get("unique_count", 0),
                "missing_pct": meta.get("missing_pct", 0.0)
            })

        # Build detected_kpis list ordered by confidence descending (for backward compatibility)
        if detected_kpis:
            detected_kpis.sort(key=lambda x: x[1], reverse=True)
            detected_kpis = [col for col, _ in detected_kpis]
        else:
            # fallback to non-id numeric columns
            detected_kpis = [c for c in numeric_cols if not re.search(r'(^|_)(id|key|code|zip|postal|row)($|_)', c.lower())] or numeric_cols[:4]

        # LOCAL STATISTICAL KPI SELECTION (instant, zero-network)
        kpi_selection_result = self._select_primary_kpi_local(df, numeric_cols, col_meta_dict, row_count)
        if kpi_selection_result:
            primary_kpi = kpi_selection_result["primary_kpi"]
            kpi_selection_pending = False
            kpi_selection_method = "statistical_local"
            kpi_selection_details = kpi_selection_result
        else:
            primary_kpi = numeric_cols[0] if numeric_cols else None
            kpi_selection_pending = False
            kpi_selection_method = "fallback_first_numeric"
            kpi_selection_details = None

        # Correlation Matrix for Numeric Columns
        correlation_matrix = {}
        if len(numeric_cols) > 1:
            try:
                corr_df = df[numeric_cols].corr().fillna(0).round(2)
                correlation_matrix = corr_df.to_dict()
            except Exception:
                pass

        # Advanced Statistical Metrics
        advanced_stats = {}

        # Skewness for numeric columns
        skewness = {}
        for col in numeric_cols:
            skewness[col] = round(self._calculate_skewness(df[col]), 2)
        advanced_stats["skewness"] = skewness

        # Anomaly spikes for primary KPI
        if primary_kpi and primary_kpi in numeric_cols:
            advanced_stats["anomaly_spikes"] = self._detect_anomaly_spikes(df[primary_kpi])

        # Segment concentration for top categorical columns with primary KPI
        segment_concentration = {}
        for cat_col in categorical_cols[:3]:
            if primary_kpi and primary_kpi in df.columns:
                concentration = self._calculate_segment_concentration(df, cat_col, primary_kpi)
                if concentration:
                    segment_concentration[cat_col] = concentration
        advanced_stats["segment_concentration"] = segment_concentration

        # Period-over-period growth for date columns with primary KPI
        pop_growth = {}
        for date_col in date_cols:
            if primary_kpi and primary_kpi in df.columns:
                growth = self._calculate_period_over_period_growth(df, date_col, primary_kpi)
                if growth:
                    pop_growth[date_col] = growth
                    
                    # Add rolling trend
                    trend = self._calculate_rolling_trend(df, date_col, primary_kpi)
                    if trend:
                        pop_growth[f"{date_col}_trend"] = trend
        advanced_stats["period_over_period_growth"] = pop_growth

        # Extract Aggregated Series for Actual Chart Rendering
        extracted_chart_data = {}

        if date_cols and primary_kpi:
            try:
                date_col = date_cols[0]
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                df_temp = df_temp.dropna(subset=[date_col])
                df_temp['period'] = df_temp[date_col].dt.strftime('%b %Y')
                ts_grouped = df_temp.groupby('period', sort=False)[primary_kpi].sum().reset_index()
                extracted_chart_data["time_series"] = [
                    {"name": str(r['period']), "value": round(float(r[primary_kpi]), 2), primary_kpi: round(float(r[primary_kpi]), 2)}
                    for _, r in ts_grouped.head(12).iterrows()
                ]
            except Exception:
                pass

        if categorical_cols and primary_kpi:
            try:
                cat_col = categorical_cols[0]
                cat_grouped = df.groupby(cat_col)[primary_kpi].sum().reset_index().sort_values(by=primary_kpi, ascending=False).head(8)
                extracted_chart_data["categorical"] = [
                    {"name": str(r[cat_col]), "value": round(float(r[primary_kpi]), 2), "count": round(float(r[primary_kpi]), 2)}
                    for _, r in cat_grouped.iterrows()
                ]
            except Exception:
                pass

        if len(numeric_cols) >= 2:
            try:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                sample_scatter = df[[col1, col2]].dropna().head(30)
                extracted_chart_data["scatter"] = [
                    {"x": round(float(r[col1]), 2), "y": round(float(r[col2]), 2), "z": 100}
                    for _, r in sample_scatter.iterrows()
                ]
            except Exception:
                pass

        # High-level Summary object
        col_meta_dict = {c["name"]: c for c in columns_info}
        effective_row_count = total_rows if total_rows > 0 else row_count

        # Build available KPI options for frontend dropdown
        available_kpi_options = self._build_available_kpi_options(kpi_selection_details) if kpi_selection_details else []

        summary = {
            "filename": filename,
            "row_count": effective_row_count,
            "column_count": col_count,
            "columns": columns_info,
            "col_meta": col_meta_dict,
            "numeric_columns": numeric_cols,
            "date_columns": date_cols,
            "categorical_columns": categorical_cols,
            "detected_kpis": detected_kpis,
            "primary_kpi": primary_kpi,
            "kpi_candidates": kpi_candidates,
            "kpi_selection_pending": kpi_selection_pending,
            "kpi_selection_method": kpi_selection_method if 'kpi_selection_method' in locals() else "unknown",
            "kpi_selection_details": kpi_selection_details if 'kpi_selection_details' in locals() else None,
            "available_kpi_options": available_kpi_options,
            "correlation_matrix": correlation_matrix,
            "extracted_chart_data": extracted_chart_data,
            "quality_score": round(float((1 - df.isnull().mean().mean()) * 100), 1),
            "pii_flags": pii_flags,
            "advanced_stats": advanced_stats,
        }

        # Sample records for LLM semantic domain understanding (PII-masked).
        try:
            summary["sample_records"] = self._mask_records(self._records_from_df(df.head(5)), pii_map)
        except Exception:
            summary["sample_records"] = []

        # Representative sample (50-100 rows) + compact masked profile for the
        # dashboard-generation LLM payload. Raw data stays local; only these
        # bounded, masked structures are ever serialized into the prompt.
        try:
            representative, sampling_method = self.build_representative_sample(
                df, numeric_cols, date_cols, categorical_cols
            )
        except Exception:
            representative, sampling_method = [], "error"
        summary["representative_sample"] = self._mask_records(representative, pii_map)
        summary["sampling"] = {"method": sampling_method, "n": len(summary["representative_sample"])}
        summary["llm_profile"] = self.build_llm_profile(summary, pii_map)
        if pii_map:
            summary["pii_columns"] = pii_map

        if was_sampled:
            summary["was_sampled"] = True
            summary["sample_size"] = sample_size
        if masked_preview:
            summary["masked_preview"] = masked_preview

        # If a cleaning report is supplied, attach a human‑readable cleaning summary
        # and a detailed quality‑score breakdown.
        if cleaning_report is not None:
            # ---- cleaning_summary ----
            missing_handled = cleaning_report.get("missing_values_handled", {})
            imputation_used = cleaning_report.get("imputation_strategy_used", {})
            high_missing = cleaning_report.get("high_missing_flagged", [])
            outlier_treat = cleaning_report.get("outlier_treatment", {})

            imputation_details = []
            for col, miss_cnt in missing_handled.items():
                if col in high_missing:
                    pct = round(miss_cnt / cleaning_report["initial_rows"] * 100, 1)
                    imputation_details.append(f"{col}: {pct}% missing — left as-is, flagged for review")
                else:
                    strategy = imputation_used.get(col, "median")
                    friendly = {
                        "forward_fill": "forward-filled",
                        "median": "median-imputed",
                        "mode": "mode-imputed",
                        "skipped_high_missing": "left as-is, flagged for review"
                    }.get(strategy, strategy)
                    imputation_details.append(f"{col}: {friendly} ({miss_cnt} missing values)")

            high_missing_details = [
                f"{col}: {round(cleaning_report['missing_values_handled'][col] / cleaning_report['initial_rows'] * 100, 1)}% missing — left as-is, flagged for review"
                for col in high_missing
            ]

            outlier_details = []
            for col, info in outlier_treat.items():
                cnt = info.get("count", 0)
                action = info.get("action", "none")
                if cnt and action != "none":
                    if action == "winsorized_1_99":
                        outlier_details.append(f"{col}: {cnt} outliers capped at 1st/99th percentile")
                    else:
                        outlier_details.append(f"{col}: {cnt} outliers treated ({action})")

            cleaning_summary = {
                "rows_before": cleaning_report.get("initial_rows"),
                "rows_after": cleaning_report.get("cleaned_rows"),
                "duplicates_removed": cleaning_report.get("duplicates_removed"),
                "imputation_details": imputation_details,
                "high_missing_flagged": high_missing_details,
                "outlier_treatment_details": outlier_details
            }
            summary["cleaning_summary"] = cleaning_summary

            # ---- quality_score_breakdown ----
            # completeness: 100 - average missing %
            avg_missing = sum(col["missing_pct"] for col in columns_info) / len(columns_info) if columns_info else 0
            completeness = max(0.0, 100.0 - avg_missing)

            # consistency: proportion of numeric columns without outlier treatment
            treated_cols = sum(1 for info in outlier_treat.values() if info.get("action") != "none" and info.get("count", 0) > 0)
            total_numeric = len(numeric_cols) if numeric_cols else 1
            consistency = max(0.0, 100.0 * (1.0 - treated_cols / total_numeric))

            # type_confidence: average of kpi_confidence, geo_confidence, date confidence (1.0)
            conf_vals = []
            for col_meta in columns_info:
                if "kpi_confidence" in col_meta:
                    conf_vals.append(col_meta["kpi_confidence"])
                elif "geo_confidence" in col_meta:
                    conf_vals.append(col_meta["geo_confidence"])
                elif col_meta.get("semantic_role") == "date":
                    conf_vals.append(1.0)
                else:
                    conf_vals.append(0.0)
            type_confidence = (sum(conf_vals) / len(conf_vals) * 100) if conf_vals else 0.0

            summary["quality_score_breakdown"] = {
                "completeness": round(completeness, 1),
                "consistency": round(consistency, 1),
                "type_confidence": round(type_confidence, 1)
            }

        return summary

dataset_profiler = LocalDatasetProfiler()
