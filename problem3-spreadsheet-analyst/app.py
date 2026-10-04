from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from charting import build_chart
from data_loader import load_csv, profile_dataframe
from executor import execute_plan, make_insight
from planner import make_plan

st.set_page_config(page_title="Natural Language Spreadsheet Analyst", page_icon="📊", layout="wide")
st.title("📊 Natural Language Spreadsheet Analyst")
st.caption("Upload a messy CSV, ask a question in English, and get an insight + table + chart.")

with st.sidebar:
    st.header("Safety model")
    st.markdown("✅ No arbitrary Python execution")
    st.markdown("✅ No SQL execution")
    st.markdown("✅ Allow-listed operations only")
    st.markdown("✅ Columns validated against schema")
    st.markdown("✅ Output limits enforced")
    st.divider()
    st.caption("Common NYC 311 questions also work with offline rules.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
sample_path = Path("data/nyc311_sample.csv")

if uploaded is not None:
    source = uploaded
elif sample_path.exists():
    source = sample_path
    st.info("Using local NYC 311 sample. Upload another CSV to replace it.")
else:
    st.warning("Upload a CSV, or run: python download_sample.py --rows 50000")
    st.stop()

try:
    df = load_csv(source)
except Exception as e:
    st.error(f"Could not load CSV safely: {e}")
    st.stop()

profile = profile_dataframe(df)
c1, c2, c3 = st.columns(3)
c1.metric("Rows loaded", f"{profile.rows:,}")
c2.metric("Columns", profile.columns)
c3.metric("Missing cells", f"{df.isna().sum().sum():,}")

with st.expander("Data quality & inferred schema"):
    quality = pd.DataFrame({
        "column": df.columns,
        "dtype": [profile.column_types[c] for c in df.columns],
        "null_%": [profile.null_pct[c] for c in df.columns],
        "example": [", ".join(profile.sample_values[c][:2]) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

question = st.text_input("Ask a question", placeholder="Example: What are the top 10 complaint types in Brooklyn?")

if st.button("Analyze", type="primary", disabled=not question.strip()):
    started = time.perf_counter()
    try:
        plan, planner_source = make_plan(question, profile, list(df.columns))
        result = execute_plan(df, plan)
        insight = make_insight(result)
        chart = build_chart(result, plan, question)
        elapsed = (time.perf_counter() - started) * 1000

        st.success(insight)
        a, b = st.columns(2)
        a.metric("Planner", planner_source)
        b.metric("Analysis time", f"{elapsed:.0f} ms")

        st.subheader("Result table")
        st.dataframe(result, use_container_width=True, hide_index=True)

        st.subheader("Chart")
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("A chart is not useful for a single-value result.")

        with st.expander("Safe query plan"):
            st.json(plan)
            st.caption("The LLM returns only a validated query plan; executable code is never accepted.")
    except Exception as e:
        st.error(f"Could not answer safely: {e}")
