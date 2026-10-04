from __future__ import annotations

import pandas as pd

def _apply_filter(df, f):
    col, op, value = f["column"], f["op"], f["value"]
    s = df[col]
    if op == "contains":
        return df[s.astype("string").str.contains(str(value), case=False, na=False, regex=False)]
    if op == "eq":
        if pd.api.types.is_string_dtype(s):
            return df[s.astype("string").str.lower() == str(value).lower()]
        return df[s == value]
    if op == "neq":
        return df[s != value]

    if pd.api.types.is_datetime64_any_dtype(s):
        value = pd.to_datetime(value, errors="coerce")
        value2 = pd.to_datetime(f.get("value2"), errors="coerce")
    else:
        value = pd.to_numeric(value, errors="coerce")
        value2 = pd.to_numeric(f.get("value2"), errors="coerce")

    if op == "gt": return df[s > value]
    if op == "gte": return df[s >= value]
    if op == "lt": return df[s < value]
    if op == "lte": return df[s <= value]
    if op == "between": return df[s.between(value, value2)]
    raise ValueError("Unsupported filter.")

def execute_plan(df: pd.DataFrame, plan: dict) -> pd.DataFrame:
    work = df.copy()
    for f in plan["filters"]:
        work = _apply_filter(work, f)

    tc, grain = plan.get("time_column"), plan.get("time_grain")
    if tc and grain:
        dates = pd.to_datetime(work[tc], errors="coerce")
        work = work.loc[dates.notna()].copy()
        dates = dates.loc[dates.notna()]
        if grain == "day": work["__time__"] = dates.dt.floor("D")
        elif grain == "week": work["__time__"] = dates.dt.to_period("W").dt.start_time
        elif grain == "month": work["__time__"] = dates.dt.to_period("M").dt.to_timestamp()
        elif grain == "quarter": work["__time__"] = dates.dt.to_period("Q").dt.start_time
        else: work["__time__"] = dates.dt.to_period("Y").dt.start_time

    groups = list(plan["group_by"])
    if tc and grain:
        groups = ["__time__"] + groups

    agg, col = plan["metric"]["agg"], plan["metric"]["column"]
    if groups:
        gb = work.groupby(groups, dropna=False)
        if agg == "count" and col == "*":
            out = gb.size().rename("value").reset_index()
        else:
            if col == "*":
                raise ValueError(f"Aggregation '{agg}' requires a metric column.")
            out = getattr(gb[col], agg)().rename("value").reset_index()
    else:
        if agg == "count":
            out = pd.DataFrame({"metric": ["count"], "value": [len(work)]})
        else:
            if col == "*":
                raise ValueError(f"Aggregation '{agg}' requires a metric column.")
            out = pd.DataFrame({"metric": [f"{agg}({col})"], "value": [getattr(work[col], agg)()]})

    if len(out) > 1 and "value" in out.columns:
        out = out.sort_values("value", ascending=plan["sort"] == "asc")
    if tc and grain and "__time__" in out.columns:
        out = out.sort_values("__time__")
    return out.head(plan["limit"]).reset_index(drop=True)

def make_insight(result: pd.DataFrame) -> str:
    if result.empty:
        return "No rows matched the question after cleaning and filtering."
    if len(result) == 1:
        return f"The result is {result.iloc[0]['value']:,}."
    labels = [c for c in result.columns if c != "value"]
    if labels:
        idx = result["value"].astype(float).idxmax()
        row = result.loc[idx]
        label = " / ".join(str(row[c]) for c in labels)
        return f"The highest result is {label} with {row['value']:,}."
    return f"The query returned {len(result)} summarized rows."
