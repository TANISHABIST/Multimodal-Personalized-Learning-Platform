import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.embeddings import EmbeddingManager
from src.ingestion import process_single_pdf, split_documents
from src.rag_chain import get_rag_stream
from src.registry import (
    file_hash, load_registry, save_registry,
    is_processed, register_file, remove_file,
)
from src.retriever import RAGRetriever
from src.vectorstore import VectorStore

load_dotenv()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")


# ---------- cached resources (only load once per session, not on every rerun) ----------

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_manager():
    return EmbeddingManager()


@st.cache_resource(show_spinner="Connecting to vector store...")
def get_vectorstore():
    return VectorStore()


@st.cache_resource(show_spinner="Connecting to Groq...")
def get_llm(temperature: float):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY not found. Create a `.env` file (see `.env.example`) "
            "with your key, then restart the app."
        )
        st.stop()
    return ChatGroq(api_key=api_key, model_name="openai/gpt-oss-20b", temperature=temperature, max_tokens=1024)


embedding_manager = get_embedding_manager()
vectorstore = get_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "registry" not in st.session_state:
    st.session_state.registry = load_registry()


# ---------------------------- sidebar ----------------------------

with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
        help="Already-processed files are detected by content and skipped automatically.",
    )

    if uploaded_files:
        for uploaded in uploaded_files:
            file_bytes = uploaded.getvalue()
            h = file_hash(file_bytes)

            if is_processed(h, st.session_state.registry):
                st.info(f"'{uploaded.name}' already processed — skipping.")
                continue

            save_path = os.path.join(UPLOAD_DIR, uploaded.name)
            with open(save_path, "wb") as f:
                f.write(file_bytes)

            with st.status(f"Processing {uploaded.name}...", expanded=True) as status:
                st.write("Reading PDF...")
                documents = process_single_pdf(save_path)

                st.write("Splitting into chunks...")
                chunks = split_documents(documents)

                if not chunks:
                    status.update(label=f"No usable text found in {uploaded.name}", state="error")
                    continue

                st.write(f"Embedding {len(chunks)} chunks...")
                texts = [c.page_content for c in chunks]
                embeddings = embedding_manager.generate_embeddings(texts)

                st.write("Storing in vector database...")
                vectorstore.add_documents(chunks, embeddings)

                st.session_state.registry = register_file(uploaded.name, h, len(chunks), st.session_state.registry)
                save_registry(st.session_state.registry)

                status.update(label=f"'{uploaded.name}' ready ({len(chunks)} chunks)", state="complete")

    st.divider()
    st.subheader("Processed files")
    if not st.session_state.registry:
        st.caption("No files processed yet.")
    else:
        for fname, info in list(st.session_state.registry.items()):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{fname}**  \n:gray[{info['chunk_count']} chunks · {info['processed_at']}]")
            if col2.button("🗑️", key=f"del_{fname}", help=f"Remove {fname}"):
                vectorstore.delete_by_source(fname)
                st.session_state.registry = remove_file(fname, st.session_state.registry)
                save_registry(st.session_state.registry)
                uploaded_path = Path(UPLOAD_DIR) / fname
                if uploaded_path.exists():
                    uploaded_path.unlink()
                st.rerun()

    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 5)
    min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.2, 0.05)
    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.1, 0.05)

    st.divider()
    if st.button("🧹 Clear chat history"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------- main chat area ----------------------------

st.title("📚 RAG Chat")
st.caption("Ask questions about the documents you've uploaded.")

if not st.session_state.registry:
    st.warning("Upload at least one PDF from the sidebar to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 Sources (confidence: {msg.get('confidence', 0):.0%})"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — page {s['page']} — score `{s['score']:.2f}`")
                    st.caption(s["preview"])

query = st.chat_input("Ask a question about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    llm = get_llm(temperature)
    retriever = RAGRetriever(vectorstore, embedding_manager)

    with st.chat_message("assistant"):
        token_stream, sources, confidence = get_rag_stream(
            query, retriever, llm, top_k=top_k, min_score=min_score
        )
        answer = st.write_stream(token_stream)

        if sources:
            with st.expander(f"📄 Sources (confidence: {confidence:.0%})"):
                for s in sources:
                    st.markdown(f"**{s['source']}** — page {s['page']} — score `{s['score']:.2f}`")
                    st.caption(s["preview"])

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources, "confidence": confidence,
    })