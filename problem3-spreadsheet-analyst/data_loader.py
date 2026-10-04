from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

NULL_TOKENS = {"", "null", "none", "n/a", "na", "nan", "-", "--", "unknown"}

@dataclass
class DataProfile:
    rows: int
    columns: int
    column_types: dict
    null_pct: dict
    sample_values: dict

def _clean_column_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name)).strip()
    return name or "unnamed"

def _maybe_numeric(series: pd.Series) -> pd.Series:
    if series.dtype != "object":
        return series
    raw = series.astype("string")
    cleaned = raw.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("%", "", regex=False).str.strip()
    parsed = pd.to_numeric(cleaned, errors="coerce")
    non_null = raw.notna().sum()
    if non_null and parsed.notna().sum() / non_null >= 0.92:
        return parsed
    return series

def _maybe_datetime(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.dtype != "object":
        return series
    hint = any(x in name.lower() for x in ("date", "time", "created", "closed", "due"))
    if not hint:
        return series
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null = series.notna().sum()
    if non_null and parsed.notna().sum() / non_null >= 0.70:
        return parsed
    return series

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_column_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for col in df.select_dtypes(include="object").columns:
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.str.lower().isin(NULL_TOKENS), pd.NA)
    for col in df.columns:
        df[col] = _maybe_datetime(df[col], col)
        df[col] = _maybe_numeric(df[col])
    return df

def load_csv(source, max_rows: int | None = 300_000) -> pd.DataFrame:
    kwargs = dict(low_memory=False, on_bad_lines="skip")
    if max_rows:
        kwargs["nrows"] = max_rows
    try:
        df = pd.read_csv(source, **kwargs)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, encoding="latin-1", **kwargs)
    if df.empty:
        raise ValueError("The CSV contains no usable rows.")
    return clean_dataframe(df)

def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    samples = {c: df[c].dropna().astype(str).head(4).tolist() for c in df.columns}
    return DataProfile(
        rows=len(df),
        columns=len(df.columns),
        column_types={c: str(df[c].dtype) for c in df.columns},
        null_pct={c: round(float(df[c].isna().mean() * 100), 1) for c in df.columns},
        sample_values=samples,
    )
