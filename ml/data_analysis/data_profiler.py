
import pandas as pd
import numpy as np

from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype
)


def detect_feature_type(series):

    if is_bool_dtype(series):
        return "boolean"

    if is_datetime64_any_dtype(series):
        return "datetime"

    if is_numeric_dtype(series):
        return "numerical"

    return "categorical"


def analyze_column(df, column):

    series = df[column]

    feature_type = detect_feature_type(series)

    missing = int(series.isnull().sum())

    result = {
        "name": str(column),
        "dtype": str(series.dtype),
        "feature_type": feature_type,
        "missing_values": missing,
        "missing_percentage": round(
            (missing / len(series)) * 100, 2
        ),
        "unique_values": int(series.nunique()),
    }

    # Numerical column analysis
    if feature_type == "numerical":

        result["statistics"] = {
            "mean": float(series.mean()) if not series.isnull().all() else None,
            "median": float(series.median()) if not series.isnull().all() else None,
            "std": float(series.std()) if not series.isnull().all() else None,
            "min": float(series.min()) if not series.isnull().all() else None,
            "max": float(series.max()) if not series.isnull().all() else None
        }

    # Categorical / Boolean analysis
    elif feature_type in ["categorical", "boolean"]:

        top_values = series.value_counts(
            dropna=True
        ).head(5)

        result["top_values"] = {
            str(key): int(value)
            for key, value in top_values.items()
        }

    # Datetime analysis
    elif feature_type == "datetime":

        valid = series.dropna()

        result["statistics"] = {
            "min": str(valid.min()) if not valid.empty else None,
            "max": str(valid.max()) if not valid.empty else None
        }

    return result


def analyze_dataset(df):

    columns = []

    for column in df.columns:
        columns.append(
            analyze_column(df, column)
        )

    profile = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns
    }

    return profile
