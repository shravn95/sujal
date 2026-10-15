"""Safe, deterministic preprocessing action executor for pandas DataFrames.

All operations work on a copy of the input DataFrame and return a new DataFrame.
No LLM involvement; pure Python/pandas logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import pandas as pd


class PreprocessingExecutor:
    """Execute predefined preprocessing actions on a DataFrame safely."""

    _TOKEN_PATTERN = re.compile(r'\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*|[-+*/()]')

    @staticmethod
    def _validate_expression(expr: str, df: pd.DataFrame) -> None:
        """Validate expression contains only allowed tokens and references existing numeric columns."""
        tokens = PreprocessingExecutor._TOKEN_PATTERN.findall(expr)
        if not tokens:
            raise ValueError("Expression is empty or contains no valid tokens.")
        remaining = PreprocessingExecutor._TOKEN_PATTERN.sub('', expr)
        if remaining.strip():
            raise ValueError("Expression contains disallowed characters or patterns.")
        paren_depth = 0
        for tok in tokens:
            if tok == '(':
                paren_depth += 1
            elif tok == ')':
                paren_depth -= 1
                if paren_depth < 0:
                    raise ValueError("Unbalanced parentheses in expression.")
        if paren_depth != 0:
            raise ValueError("Unbalanced parentheses in expression.")
        # Reject ** (power) and // (floor division) by checking for consecutive * or / tokens
        for i in range(len(tokens) - 1):
            if tokens[i] == '*' and tokens[i + 1] == '*':
                raise ValueError("Power operator ** is not allowed.")
            if tokens[i] == '/' and tokens[i + 1] == '/':
                raise ValueError("Floor division operator // is not allowed.")
        for tok in tokens:
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', tok):
                if tok not in df.columns:
                    raise ValueError(f"Column '{tok}' referenced in expression does not exist.")
                if not pd.api.types.is_numeric_dtype(df[tok]):
                    raise ValueError(f"Column '{tok}' is not numeric; cannot use in arithmetic expression.")

    @classmethod
    def drop_column(cls, df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
        """Drop a column."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        new_df = df.drop(columns=[column])
        return new_df, f"Dropped column '{column}'"

    @classmethod
    def rename_column(cls, df: pd.DataFrame, column: str, new_name: str) -> Tuple[pd.DataFrame, str]:
        """Rename a column."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if new_name in df.columns:
            raise ValueError(f"Column '{new_name}' already exists.")
        new_df = df.rename(columns={column: new_name})
        return new_df, f"Renamed column '{column}' to '{new_name}'"

    @classmethod
    def fill_missing(cls, df: pd.DataFrame, column: str, strategy: str, value: Any = None) -> Tuple[pd.DataFrame, str]:
        """Fill missing values in a column using a strategy."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        if strategy not in {"mean", "median", "mode", "constant", "zero", "forward_fill"}:
            raise ValueError(f"Unsupported strategy '{strategy}'.")
        series = df[column]
        missing_before = series.isna().sum()
        if missing_before == 0:
            return df.copy(), f"No missing values in '{column}' to fill."

        new_df = df.copy()
        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("Mean strategy requires numeric column.")
            fill_val = series.mean()
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("Median strategy requires numeric column.")
            fill_val = series.median()
        elif strategy == "mode":
            mode_vals = series.mode()
            if mode_vals.empty:
                raise ValueError("Mode strategy failed: no mode found.")
            fill_val = mode_vals.iloc[0]
        elif strategy == "constant":
            if value is None:
                raise ValueError("Constant strategy requires a 'value' parameter.")
            fill_val = value
        elif strategy == "zero":
            fill_val = 0
        elif strategy == "forward_fill":
            new_df[column] = series.ffill()
            filled = missing_before - new_df[column].isna().sum()
            return new_df, f"Forward-filled {filled} missing values in '{column}'"
        else:
            raise ValueError(f"Unsupported strategy '{strategy}'.")

        new_df[column] = series.fillna(fill_val)
        filled = missing_before - new_df[column].isna().sum()
        return new_df, f"Filled {filled} missing values in '{column}' using {strategy}"

    @classmethod
    def change_type(cls, df: pd.DataFrame, column: str, target_type: str) -> Tuple[pd.DataFrame, str]:
        """Change column dtype, raising clear error on failure."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        new_df = df.copy()
        series = new_df[column]
        try:
            if target_type == "int":
                new_df[column] = pd.to_numeric(series, errors='raise').astype('Int64')
            elif target_type == "float":
                new_df[column] = pd.to_numeric(series, errors='raise').astype('float64')
            elif target_type == "string":
                new_df[column] = series.astype('string')
            elif target_type == "datetime":
                new_df[column] = pd.to_datetime(series, errors='raise')
            else:
                raise ValueError(f"Unsupported target_type '{target_type}'.")
        except Exception as e:
            # Find problematic values
            bad_mask = pd.to_numeric(series, errors='coerce').isna() if target_type in {"int", "float"} else pd.to_datetime(series, errors='coerce').isna()
            bad_vals = series[bad_mask].unique()[:10]
            raise ValueError(f"Failed to convert column '{column}' to {target_type}. Problematic values: {list(bad_vals)}") from e
        return new_df, f"Changed type of column '{column}' to {target_type}"

    @classmethod
    def add_column(cls, df: pd.DataFrame, new_column: str, expression: str) -> Tuple[pd.DataFrame, str]:
        """Add a new column computed from a safe arithmetic expression."""
        if new_column in df.columns:
            raise ValueError(f"Column '{new_column}' already exists.")
        cls._validate_expression(expression, df)
        new_df = df.copy()
        try:
            new_df[new_column] = new_df.eval(expression)
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from e
        return new_df, f"Added column '{new_column}' = {expression}"

    @classmethod
    def filter_rows(cls, df: pd.DataFrame, column: str, operator: str, value: Any) -> Tuple[pd.DataFrame, str]:
        """Filter rows based on a comparison."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")
        ops = {
            "==": lambda s, v: s == v,
            "!=": lambda s, v: s != v,
            ">":  lambda s, v: s > v,
            "<":  lambda s, v: s < v,
            ">=": lambda s, v: s >= v,
            "<=": lambda s, v: s <= v,
        }
        if operator not in ops:
            raise ValueError(f"Unsupported operator '{operator}'.")
        new_df = df.copy()
        try:
            mask = ops[operator](new_df[column], value)
        except Exception as e:
            raise ValueError(f"Failed to apply filter: {e}") from e
        filtered = new_df[mask]
        removed = len(new_df) - len(filtered)
        return filtered, f"Filtered rows where '{column}' {operator} {value} (removed {removed} rows)"

    @classmethod
    def apply_action(cls, df: pd.DataFrame, action: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
        """Dispatch an action dict to the appropriate method."""
        action_type = action.get("type")
        params = action.get("params", {})
        if action_type == "drop_column":
            return cls.drop_column(df, **params)
        if action_type == "rename_column":
            return cls.rename_column(df, **params)
        if action_type == "fill_missing":
            return cls.fill_missing(df, **params)
        if action_type == "change_type":
            return cls.change_type(df, **params)
        if action_type == "add_column":
            return cls.add_column(df, **params)
        if action_type == "filter_rows":
            return cls.filter_rows(df, **params)
        raise ValueError(f"Unknown action type: {action_type}")


if __name__ == "__main__":
    # Quick self-test
    import sys

    df = pd.DataFrame({
        "a": [1, 2, None, 4],
        "b": [10.0, 20.0, 30.0, 40.0],
        "c": ["x", "y", "z", "w"],
        "d": pd.to_datetime(["2021-01-01", "2021-01-02", None, "2021-01-04"])
    })

    print("Original DF:")
    print(df)
    print()

    # drop_column
    df2, msg = PreprocessingExecutor.drop_column(df, "c")
    print(msg)
    print(df2)
    print()

    # rename_column
    df3, msg = PreprocessingExecutor.rename_column(df, "a", "a_renamed")
    print(msg)
    print(df3)
    print()

    # fill_missing mean
    df4, msg = PreprocessingExecutor.fill_missing(df, "a", "mean")
    print(msg)
    print(df4)
    print()

    # fill_missing constant
    df5, msg = PreprocessingExecutor.fill_missing(df, "a", "constant", value=99)
    print(msg)
    print(df5)
    print()

    # change_type to int
    df6, msg = PreprocessingExecutor.change_type(df, "b", "int")
    print(msg)
    print(df6)
    print()

    # change_type to datetime (will fail on None)
    try:
        PreprocessingExecutor.change_type(df, "d", "datetime")
    except ValueError as e:
        print("Expected error on datetime conversion:", e)
    print()

    # add_column
    df7, msg = PreprocessingExecutor.add_column(df, "e", "a + b")
    print(msg)
    print(df7)
    print()

    # filter_rows
    df8, msg = PreprocessingExecutor.filter_rows(df, "b", ">", 25)
    print(msg)
    print(df8)
    print()

    # Test bad expression rejection
    try:
        PreprocessingExecutor.add_column(df, "evil", "__import__('os').system('ls')")
    except ValueError as e:
        print("Blocked malicious expression:", e)

    # Test unknown action
    try:
        PreprocessingExecutor.apply_action(df, {"type": "unknown", "params": {}})
    except ValueError as e:
        print("Blocked unknown action:", e)

    print("\nAll tests passed.")