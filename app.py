import streamlit as st
from rag.config import UNKNOWN
from rag.pipeline import answer_question

st.set_page_config(page_title="Trustworthy Finance RAG",page_icon="🔎",layout="wide")
st.title("🔎 Trustworthy Finance RAG")
st.caption("Answers only from indexed documents. No evidence = no guess.")
q=st.text_input("Ask a question about your documents",placeholder="What was the company's capital expenditure in FY2018?")
if st.button("Ask",type="primary",disabled=not q.strip()):
    try:
        with st.spinner("Searching evidence and verifying the answer..."): r=answer_question(q.strip())
        if r["answer"]==UNKNOWN: st.warning(UNKNOWN)
        else:
            st.subheader("Answer"); st.write(r["answer"]); st.subheader("Sources")
            for c in r["citations"]: st.markdown(f"**{c['source']}** — page {c['page']}, chunk {c['chunk']} (similarity {c['score']:.3f})")
        with st.expander("Retrieved evidence"):
            for x in r["retrieved"]:
                st.markdown(f"**[{x['source_id']}] {x['source']} — page {x['page']}, chunk {x['chunk_index']} — {x['score']:.3f}**")
                st.write(x["text"])
    except Exception as e:
        st.error(str(e)); st.info("Run ingestion first and confirm your API key is set in .env.")
