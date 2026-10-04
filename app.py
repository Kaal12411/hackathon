import streamlit as st
from rag.config import UNKNOWN
from rag.pipeline import answer_question

st.set_page_config(
    page_title="Trustworthy Finance RAG",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Trustworthy Finance RAG")
st.caption("Evidence first. Verified citations. No evidence = no guess.")

with st.sidebar:
    st.header("Trust controls")
    st.markdown("✅ Local FAISS retrieval")
    st.markdown("✅ Hybrid semantic + keyword reranking")
    st.markdown("✅ Grounding-only generation")
    st.markdown("✅ Second-pass evidence verifier")
    st.markdown("✅ Fail-closed citation validation")
    st.divider()
    st.caption('Required fallback:')
    st.code("I don't know based on the provided documents")

q = st.text_input(
    "Ask a question about the indexed financial documents",
    placeholder="e.g. What was the company's capital expenditure in FY2018?",
)

col1, col2 = st.columns([1, 5])
ask = col1.button("Ask", type="primary", disabled=not q.strip())
col2.caption("Try an answerable filing question, then an unrelated question to test refusal.")

if ask:
    try:
        with st.spinner("Retrieving evidence → generating → verifying grounding..."):
            r = answer_question(q.strip())

        best_score = r["retrieved"][0]["score"] if r["retrieved"] else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Decision", "VERIFIED" if r["answer"] != UNKNOWN else "REFUSED")
        m2.metric("Best retrieval score", f"{best_score:.3f}")
        m3.metric("Citations", len(r["citations"]))

        st.subheader("Answer")
        if r["answer"] == UNKNOWN:
            st.warning(UNKNOWN)
            st.caption(f"Guardrail decision: {r.get('decision','refused')}")
        else:
            st.success(r["answer"])
            st.caption("This answer passed the independent grounding verifier.")

            st.subheader("Verified sources")
            for c in r["citations"]:
                with st.container(border=True):
                    st.markdown(
                        f"**[{c['source_id']}] {c['source']}**  
"
                        f"Page **{c['page']}** · Chunk **{c['chunk']}** · "
                        f"Retrieval score **{c['score']:.3f}**"
                    )
                    st.caption(c["excerpt"])

        with st.expander("🔍 Retrieved evidence / audit trail"):
            for x in r["retrieved"]:
                st.markdown(
                    f"**[{x['source_id']}] {x['source']} — page {x['page']}, "
                    f"chunk {x['chunk_index']} — hybrid {x['score']:.3f} "
                    f"(semantic {x.get('semantic_score',0):.3f}, lexical {x.get('lexical_score',0):.3f})**"
                )
                st.write(x["text"])
    except Exception as e:
        st.error(str(e))
        st.info("Run ingestion first and confirm your API key is set in .env.")
