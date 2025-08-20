import os
import streamlit as st
from pathlib import Path
from config import settings
from modules.rag_pipeline import build_index, answer_query,load_langsmith_cred



st.set_page_config(page_title="PDF RAG App (LangChain Modular)", page_icon="📚", layout="wide")

st.title("📚 PDF RAG — Streamlit + LangChain (Modular)")
st.caption("Upload PDFs, build an index, and query using OpenAI via LangChain.")

# Sidebar settings
st.sidebar.header("Settings")
open_api_key = st.sidebar.text_input("OpenAI API Key", value=settings.openai_api_key, type="password")
langchain_api_key = st.sidebar.text_input("Langchain API Key", value=settings.langchain_api_key, type="password")
chat_model = st.sidebar.text_input("Chat Model", value=settings.chat_model)
embed_model = st.sidebar.text_input("Embedding Model", value=settings.embedding_model)
chunk_size = st.sidebar.number_input("Chunk Size", min_value=200, max_value=4000, value=settings.chunk_size, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", min_value=0, max_value=1000, value=settings.chunk_overlap, step=50)
index_dir = st.sidebar.text_input("Index Directory", value=settings.index_dir)

load_langsmith_cred(langchain_api_key)
    
# Upload PDFs
st.subheader("1) Upload PDFs and Build Index")
uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if "tmp_pdf_dir" not in st.session_state:
    st.session_state.tmp_pdf_dir = str(Path("tmp_uploads").absolute())
Path(st.session_state.tmp_pdf_dir).mkdir(parents=True, exist_ok=True)

if uploaded and st.button("Build Index", type="primary"):
    if not open_api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
    else:
        saved_paths = []
        for f in uploaded:
            p = Path(st.session_state.tmp_pdf_dir) / f.name
            with open(p, "wb") as out:
                out.write(f.read())
            saved_paths.append(str(p))

        with st.spinner("Building index with LangChain..."):
            try:
                total_chunks = build_index(
                    pdf_path=saved_paths,
                    index_dir=index_dir,
                    api_key=open_api_key,
                    embed_model=embed_model,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                )
                st.success(f"Index built with {total_chunks} chunks. Saved to {index_dir}.")
            except Exception as e:
                st.exception(e)

# Query section
st.subheader("2) Ask Questions")
query = st.text_input("Enter your question:")
top_k = st.number_input("Top-K Retrieval", min_value=1, max_value=20, value=5)

if st.button("Ask") and query.strip():
    if not open_api_key:
        st.error("Please provide your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Querying with LangChain RAG pipeline..."):
            try:
                result = answer_query(
                    query=query,
                    index_dir=index_dir,
                    api_key=open_api_key,
                    embed_model=embed_model,
                    chat_model=chat_model,
                    k=int(top_k),
                )
                st.markdown("### Answer")
                st.write(result["response"])

                # st.markdown("### Sources")
                # for s in result["sources"]:
                #     with st.expander(f"{s['source']} — page {s['page']}"):
                #         st.write(s["text"])
            except Exception as e:
                st.exception(e)
