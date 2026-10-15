import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class AutomatedDataCleaner:
    def __init__(self):
        self._default_synonym_map = {
            "usa": "United States",
            "u.s.a.": "United States",
            "us": "United States",
            "uk": "United Kingdom",
            "u.k.": "United Kingdom",
        }

    def _normalize_categories(self, df: pd.DataFrame, synonym_map: Optional[Dict[str, str]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Normalize categorical columns: trim whitespace, unify casing, apply synonym map."""
        report = {}
        if synonym_map is None:
            synonym_map = self._default_synonym_map

        for col in df.columns:
            if not (pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object):
                continue
            non_null = df[col].dropna()
            if non_null.empty:
                continue

            original_vals = non_null.astype(str)
            # 1. Trim whitespace
            trimmed = original_vals.str.strip()
            # 2. Canonical casing: use most frequent trimmed form for each lowercased value
            lowered = trimmed.str.lower()
            canonical_map = lowered.groupby(lowered).apply(
                lambda s: trimmed.loc[s.index].mode().iloc[0] if not trimmed.loc[s.index].mode().empty else s.iloc[0]
            )
            # Build mapping from trimmed to canonical
            trim_to_canonical = {orig: canonical_map.get(orig.lower(), orig) for orig in trimmed.unique()}
            # Apply canonical casing
            normalized = trimmed.map(trim_to_canonical)

            # 3. Synonym map for country-ish columns
            col_lower = col.lower()
            is_countryish = any(kw in col_lower for kw in ["country", "nation", "region"])
            if not is_countryish:
                # Heuristic: check if >30% of unique non-null values match synonym keys (case-insensitive)
                unique_vals = normalized.unique()
                matches = sum(1 for v in unique_vals if v.lower() in synonym_map)
                if len(unique_vals) > 0 and matches / len(unique_vals) >= 0.3:
                    is_countryish = True

            if is_countryish:
                # Apply synonym map case-insensitively
                def apply_synonym(v):
                    return synonym_map.get(v.lower(), v)
                normalized = normalized.apply(apply_synonym)

            # Count changes
            changed_mask = (original_vals != normalized)
            if changed_mask.any():
                changed_indices = changed_mask[changed_mask].index
                examples = []
                for idx in changed_indices[:5]:
                    examples.append({"before": original_vals.loc[idx], "after": normalized.loc[idx]})
                df[col] = df[col].astype(object)
                df.loc[changed_indices, col] = normalized.loc[changed_indices].values
                report[col] = {
                    "values_normalized": int(changed_mask.sum()),
                    "examples": examples,
                }

        return df, report

    def clean_dataset(self, df: pd.DataFrame, treat_outliers: bool = True, category_synonym_map: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Performs automated cleaning: duplicate removal, missing value imputation,
        outlier treatment, and column standardization.
        """
        initial_rows = len(df)

        # 1. Remove duplicates
        df_cleaned = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_cleaned)

        # Detect datetime columns (fast sampled check)
        datetime_cols: List[str] = []
        for col in df_cleaned.columns:
            if pd.api.types.is_datetime64_any_dtype(df_cleaned[col]):
                datetime_cols.append(col)
            elif (pd.api.types.is_string_dtype(df_cleaned[col]) or df_cleaned[col].dtype == object) and df_cleaned[col].notna().any():
                col_lower = str(col).lower()
                is_date_name = any(kw in col_lower for kw in ["date", "time", "year", "month", "day", "timestamp", "period"])
                sample = df_cleaned[col].dropna().head(20)
                try:
                    parsed_sample = pd.to_datetime(sample, errors='coerce', format='mixed')
                    if parsed_sample.notna().mean() >= 0.8 or (is_date_name and parsed_sample.notna().mean() >= 0.5):
                        datetime_cols.append(col)
                except Exception:
                    pass

        row_index = np.arange(len(df_cleaned))

        # 2. Handle missing values with column-aware strategies
        missing_report = {}
        imputation_strategy_used = {}
        high_missing_flagged = []

        for col in df_cleaned.columns:
            null_count = df_cleaned[col].isnull().sum()
            if null_count == 0:
                continue
            missing_report[col] = int(null_count)
            missing_rate = null_count / len(df_cleaned)

            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Decide imputation strategy
                use_forward_fill = False
                if datetime_cols:
                    # Correlation with row order as proxy for time-series
                    numeric_series = df_cleaned[col].astype(float)
                    corr = numeric_series.corr(pd.Series(row_index, index=df_cleaned.index), method='spearman')
                    if pd.notna(corr) and abs(corr) > 0.5:
                        use_forward_fill = True
                if use_forward_fill:
                    df_cleaned[col] = df_cleaned[col].ffill()
                    imputation_strategy_used[col] = "forward_fill"
                else:
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    imputation_strategy_used[col] = "median"
            else:
                # Categorical column
                if missing_rate > 0.4:
                    high_missing_flagged.append(col)
                    imputation_strategy_used[col] = "skipped_high_missing"
                    # leave NaNs as‑is
                else:
                    mode_val = df_cleaned[col].mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                    df_cleaned[col] = df_cleaned[col].fillna(fill_val)
                    imputation_strategy_used[col] = "mode"

        # 2b. Category normalization for categorical columns
        df_cleaned, category_normalization = self._normalize_categories(df_cleaned, category_synonym_map)

        # 3. Outlier detection and treatment (winsorization)
        outlier_treatment = {}
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = df_cleaned[col].std()
            outlier_count = 0
            if std > 0:
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / std)
                outliers = z_scores > 3
                outlier_count = int(outliers.sum())
            if treat_outliers and outlier_count > 0:
                lower = df_cleaned[col].quantile(0.01)
                upper = df_cleaned[col].quantile(0.99)
                df_cleaned[col] = df_cleaned[col].clip(lower, upper)
                action = "winsorized_1_99"
            else:
                action = "none"
            outlier_treatment[col] = {"count": outlier_count, "action": action}

        report = {
            "initial_rows": initial_rows,
            "cleaned_rows": len(df_cleaned),
            "duplicates_removed": duplicates_removed,
            "missing_values_handled": missing_report,
            "imputation_strategy_used": imputation_strategy_used,
            "high_missing_flagged": high_missing_flagged,
            "outlier_treatment": outlier_treatment,
            "category_normalization": category_normalization,
            "status": "Success"
        }

        return df_cleaned, report

data_cleaner = AutomatedDataCleaner()
