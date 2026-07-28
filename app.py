"""
Streamlit demo UI for the RAG pipeline: ask a question, see the retrieved
sources and Claude's answer grounded in them.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from answer import answer_question

st.set_page_config(page_title="RAG Eval Lab", page_icon="🔎")
st.title("🔎 RAG Eval Lab")
st.caption("Ask a question about ML/AI concepts. Answers are grounded in a "
           "retrieved Wikipedia corpus, not the model's parametric memory.")

use_reranker = st.sidebar.checkbox("Use cross-encoder reranker", value=True)
k = st.sidebar.slider("Chunks to use (k)", 1, 10, 5)

question = st.text_input("Your question", placeholder="What is backpropagation?")

if st.button("Ask") and question:
    with st.spinner("Retrieving + generating..."):
        result = answer_question(question, k=k, use_reranker=use_reranker)

    st.markdown("### Answer")
    st.write(result["answer"])

    st.markdown("### Sources retrieved")
    for s in result["sources"]:
        st.markdown(f"- **{s['title']}** (`{s['chunk_id']}`)")
