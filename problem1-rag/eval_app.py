import json
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="FinanceBench Evaluation",page_icon="📊",layout="wide")
st.title("📊 FinanceBench Evaluation Dashboard")
st.caption("Measured on PatronusAI/financebench labels: gold answers + evidence pages.")

base=Path("data/eval")
cards=[
    ("Retrieval",base/"retrieval_summary.json"),
    ("Full QA",base/"full_summary.json"),
    ("Refusal",base/"refusal_summary.json"),
]
found=False
for title,path in cards:
    if path.exists():
        found=True
        data=json.loads(path.read_text())
        st.subheader(title)
        cols=st.columns(len(data))
        for col,(k,v) in zip(cols,data.items()):
            if isinstance(v,float):
                col.metric(k.replace("_"," ").title(),f"{v*100:.1f}%")
            else:
                col.metric(k.replace("_"," ").title(),v)
if not found:
    st.info("Run evaluation first, e.g. python evaluate_financebench.py --mode retrieval --limit 20")
