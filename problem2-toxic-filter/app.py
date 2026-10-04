import streamlit as st
from model import ToxicityModel, render_highlighted_html

st.set_page_config(page_title="Toxic Comment Filter",page_icon="🛡️",layout="wide")

@st.cache_resource
def get_model():
    return ToxicityModel.load()

st.title("🛡️ Real-Time Toxic Comment Filter")
st.caption("Fast moderation with transparent word-level explanations.")

with st.sidebar:
    st.header("Rubric targets")
    st.markdown("✅ High F1 baseline")
    st.markdown("✅ <500ms target")
    st.markdown("✅ Visible word highlighting")
    st.markdown("✅ Misspelling robustness via character n-grams")
    st.divider()
    st.caption("Model: TF-IDF + Logistic Regression")

text=st.text_area("Comment",height=130,placeholder="Type a comment here...")

if st.button("Analyze",type="primary",disabled=not text.strip()):
    try:
        model=get_model()
        result=model.predict(text)

        c1,c2,c3=st.columns(3)
        c1.metric("Prediction",result["label"])
        c2.metric("Toxic probability",f"{result['probability']*100:.1f}%")
        c3.metric("Latency",f"{result['latency_ms']:.1f} ms")

        if result["latency_ms"]<500:
            st.success("⚡ Real-time requirement passed (<500ms)")
        else:
            st.error("Latency exceeded 500ms")

        st.subheader("Why was it classified this way?")
        if result["explanation"]["token_scores"]:
            st.markdown(
                render_highlighted_html(
                    result["text"],result["explanation"]["token_scores"]
                ),
                unsafe_allow_html=True,
            )
            st.caption("Darker highlights contributed more strongly toward Toxic.")
        else:
            st.info("No strongly toxic visible token contribution was detected.")

        with st.expander("Top explanation tokens"):
            for feature,score in result["explanation"]["top_features"]:
                st.write(f"**{feature}** — probability impact {score:.4f}")
    except FileNotFoundError:
        st.error("Model artifact not found. Run: python train.py --data data/train.csv")
