import streamlit as st

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


@st.cache_resource(show_spinner=False)
def load_reranker():
    if CrossEncoder is None:
        return None

    try:
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        st.warning(f"Could not load reranker: {e}")
        return None