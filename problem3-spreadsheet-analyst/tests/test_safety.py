import pandas as pd
import pytest

from data_loader import clean_dataframe
from executor import execute_plan
from planner import validate_plan

def test_rejects_unknown_column():
    with pytest.raises(ValueError):
        validate_plan({
            "filters": [],
            "group_by": ["__import__('os').system('bad')"],
            "metric": {"column": "*", "agg": "count"}
        }, ["Borough", "Complaint Type"])

def test_rejects_unsupported_operation():
    with pytest.raises(ValueError):
        validate_plan({
            "filters": [{"column": "Borough", "op": "eval", "value": "x"}],
            "group_by": [],
            "metric": {"column": "*", "agg": "count"}
        }, ["Borough"])

def test_cleaner_handles_null_tokens_and_dates():
    df = pd.DataFrame({
        "Created Date": ["01/01/2026 10:00:00 AM", "null"],
        "Value": ["1,200", "N/A"],
    })
    out = clean_dataframe(df)
    assert pd.api.types.is_datetime64_any_dtype(out["Created Date"])
    assert out["Value"].iloc[0] == 1200

def test_executor_counts_group_safely():
    df = pd.DataFrame({"Borough": ["BROOKLYN", "QUEENS", "BROOKLYN"]})
    plan = validate_plan({
        "filters": [],
        "group_by": ["Borough"],
        "metric": {"column": "*", "agg": "count"},
        "sort": "desc",
        "limit": 10,
        "chart": "bar"
    }, list(df.columns))
    out = execute_plan(df, plan)
    assert out.iloc[0]["value"] == 2
