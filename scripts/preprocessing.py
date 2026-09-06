"""
preprocessing.py
------------------
Feature-engineering steps that turn cleaned data into a model-ready
matrix:
  - handle_outliers_iqr: caps (winsorizes) numeric outliers instead of
    deleting rows, so no data is lost.
  - encode_categoricals: one-hot encodes text/category columns, dropping
    the first level of each to avoid the "dummy variable trap".
  - scale_features: standardizes numeric columns (mean 0, std 1) while
    leaving ID columns and binary 0/1 target columns untouched.
"""

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler


def _is_surrogate_key(col_name: str) -> bool:
    col_name = col_name.lower()
    return col_name == "id" or col_name.endswith("_id")


def handle_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Cap numeric values outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    df = df.copy()
    capped_counts: Dict[str, int] = {}

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if not _is_surrogate_key(c)
    ]

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - factor * iqr
        upper_fence = q3 + factor * iqr

        below = df[col] < lower_fence
        above = df[col] > upper_fence
        count_capped = int(below.sum() + above.sum())

        if count_capped > 0:
            df[col] = df[col].clip(lower=lower_fence, upper=upper_fence)
            capped_counts[col] = count_capped

    return df, capped_counts


def encode_categoricals(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """One-hot encode categorical columns with drop_first=True to avoid
    multicollinearity between dummy variables."""
    df = df.copy()

    if columns is None:
        columns = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if not _is_surrogate_key(c)
        ]

    if not columns:
        return df

    return pd.get_dummies(df, columns=columns, drop_first=True)


def scale_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """Apply StandardScaler (z-score) to numeric columns, excluding ID
    columns and any explicitly named target/binary columns."""
    df = df.copy()
    exclude_cols = set(exclude_cols or [])

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in exclude_cols and not _is_surrogate_key(c)
    ]

    # Also skip columns that are already binary (0/1) e.g. a target label
    # or a one-hot encoded dummy column, since scaling them adds no value.
    numeric_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) > 2]

    if not numeric_cols:
        return df

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
