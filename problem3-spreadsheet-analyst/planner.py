from __future__ import annotations

import json
import os
import re
from difflib import get_close_matches

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ALLOWED_AGGS = {"count", "sum", "mean", "median", "min", "max", "nunique"}
ALLOWED_OPS = {"eq", "neq", "contains", "gt", "gte", "lt", "lte", "between"}
ALLOWED_CHARTS = {"auto", "bar", "line", "pie", "none"}
ALLOWED_GRAINS = {None, "day", "week", "month", "quarter", "year"}

SYSTEM = """You are a data query planner. Convert the user's question into JSON only.
Never produce Python, SQL, shell commands, imports, file paths, URLs, or executable code.
Use only columns present in the supplied schema.
Return exactly these fields: filters, group_by, time_column, time_grain, metric, sort, limit, chart.
filters is a list of objects with column, op, value, value2.
metric is an object with column and agg.
Allowed ops: eq, neq, contains, gt, gte, lt, lte, between.
Allowed aggs: count, sum, mean, median, min, max, nunique.
For how many use count. For most/common/top sort descending.
For trends use a date column plus a day/week/month/quarter/year grain.
Do not invent columns.
"""

def _client():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"), os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=key), os.getenv("OPENAI_MODEL", "gpt-4o-mini")

def _find_col(columns, terms):
    for c in columns:
        if any(term in c.lower() for term in terms):
            return c
    return None

def offline_plan(question: str, columns: list[str]):
    q = question.lower()
    complaint = _find_col(columns, ["complaint type", "complaint"])
    borough = _find_col(columns, ["borough"])
    agency = _find_col(columns, ["agency"])
    created = _find_col(columns, ["created date", "created", "date"])

    group = None
    if "complaint" in q and ("type" in q or "common" in q):
        group = complaint
    elif "borough" in q:
        group = borough
    elif "agency" in q:
        group = agency

    grain = None
    if any(x in q for x in ["over time", "trend", "by month", "monthly"]):
        grain = "month"
    elif "by year" in q or "yearly" in q:
        grain = "year"
    elif "by day" in q or "daily" in q:
        grain = "day"

    filters = []
    if borough:
        for name in ["brooklyn", "bronx", "queens", "manhattan", "staten island"]:
            if name in q:
                filters.append({"column": borough, "op": "contains", "value": name, "value2": None})

    if grain and created:
        return {"filters": filters, "group_by": [], "time_column": created, "time_grain": grain,
                "metric": {"column": "*", "agg": "count"}, "sort": "asc", "limit": 100, "chart": "line"}

    if group:
        top = 10
        m = re.search(r"top\s+(\d+)", q)
        if m:
            top = min(int(m.group(1)), 50)
        return {"filters": filters, "group_by": [group], "time_column": None, "time_grain": None,
                "metric": {"column": "*", "agg": "count"}, "sort": "desc", "limit": top, "chart": "bar"}
    return None

def llm_plan(question: str, profile) -> dict:
    client, model = _client()
    schema = {"columns": profile.column_types, "sample_values": profile.sample_values}
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "SCHEMA:\n" + json.dumps(schema) + "\n\nQUESTION:\n" + question},
        ],
    )
    return json.loads(response.choices[0].message.content)

def _resolve_column(name, columns):
    if name in columns:
        return name
    matches = get_close_matches(str(name), columns, n=1, cutoff=0.82)
    if matches:
        return matches[0]
    raise ValueError(f"Unknown column requested: {name}")

def validate_plan(plan: dict, columns: list[str]) -> dict:
    safe = {"filters": [], "group_by": [], "time_column": None, "time_grain": None,
            "metric": {"column": "*", "agg": "count"}, "sort": "desc", "limit": 10, "chart": "auto"}

    for f in plan.get("filters", [])[:8]:
        if f.get("op") not in ALLOWED_OPS:
            raise ValueError("Unsafe or unsupported filter operation.")
        safe["filters"].append({"column": _resolve_column(f.get("column"), columns), "op": f["op"],
                                "value": f.get("value"), "value2": f.get("value2")})

    safe["group_by"] = [_resolve_column(c, columns) for c in plan.get("group_by", [])[:2]]
    if plan.get("time_column"):
        safe["time_column"] = _resolve_column(plan["time_column"], columns)

    grain = plan.get("time_grain")
    if grain not in ALLOWED_GRAINS:
        raise ValueError("Unsupported time grain.")
    safe["time_grain"] = grain

    metric = plan.get("metric") or {}
    agg = metric.get("agg", "count")
    if agg not in ALLOWED_AGGS:
        raise ValueError("Unsupported aggregation.")
    col = metric.get("column", "*")
    if col != "*":
        col = _resolve_column(col, columns)
    safe["metric"] = {"column": col, "agg": agg}

    safe["sort"] = "asc" if plan.get("sort") == "asc" else "desc"
    safe["limit"] = max(1, min(int(plan.get("limit", 10)), 500))
    chart = plan.get("chart", "auto")
    safe["chart"] = chart if chart in ALLOWED_CHARTS else "auto"
    return safe

def make_plan(question: str, profile, columns: list[str]) -> tuple[dict, str]:
    heuristic = offline_plan(question, columns)
    if heuristic:
        return validate_plan(heuristic, columns), "offline rules"
    return validate_plan(llm_plan(question, profile), columns), "LLM structured planner"
