"""
data_cleaning.py
------------------
Deterministic (non-AI) data hygiene functions:
  - Standardize column names to snake_case
  - Normalize null-like strings ("", "nan", "None") to real NaN
  - Strip whitespace from text columns
  - Drop duplicate rows (ignoring surrogate/primary key columns)
  - Impute missing values (median for numeric, mode for categorical)

The main entry point is clean_data(), which returns the cleaned
DataFrame plus a metadata dictionary describing what changed. The
metadata is what powers the "before vs after" metric cards in the
Streamlit dashboard.
"""

import re
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

NULL_LITERALS = {"", "nan", "none", "null", "na", "n/a"}


def _to_snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s]", "_", name)     # non-alphanumeric -> underscore
    name = re.sub(r"\s+", "_", name)          # spaces -> underscore
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)  # camelCase -> snake_case
    name = re.sub(r"_+", "_", name)           # collapse multiple underscores
    return name.strip("_").lower()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake_case(str(c)) for c in df.columns]
    return df


def _is_surrogate_key(col_name: str) -> bool:
    """Columns that look like primary/foreign keys should be ignored when
    deduplicating, since two legitimately different rows might otherwise
    look identical once the ID column is dropped from comparison, or a
    non-unique-looking ID could falsely cause rows to be treated as dupes."""
    col_name = col_name.lower()
    return col_name == "id" or col_name.endswith("_id")


def _normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in NULL_LITERALS
            else (v.strip() if isinstance(v, str) else v)
        )
    return df


def _drop_duplicates_ignoring_keys(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    compare_cols = [c for c in df.columns if not _is_surrogate_key(c)]
    before = len(df)
    df = df.drop_duplicates(subset=compare_cols if compare_cols else None)
    dropped = before - len(df)
    return df, dropped


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if _is_surrogate_key(col):
            continue
        if df[col].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            mode_series = df[col].mode(dropna=True)
            fill_val = mode_series.iloc[0] if not mode_series.empty else "unknown"
            df[col] = df[col].fillna(fill_val)
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the full deterministic cleaning pipeline and return the cleaned
    DataFrame along with a metadata dict summarizing the changes made."""

    initial_rows = len(df)
    nulls_before = int(df.isna().sum().sum())

    df = _standardize_columns(df)
    df = _normalize_nulls(df)
    df, duplicates_dropped = _drop_duplicates_ignoring_keys(df)
    df = _impute_missing(df)

    nulls_after = int(df.isna().sum().sum())
    final_rows = len(df)

    metadata = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "duplicates_dropped": duplicates_dropped,
        "nulls_before": nulls_before,
        "nulls_after": nulls_after,
        "columns": list(df.columns),
    }

    return df, metadata
