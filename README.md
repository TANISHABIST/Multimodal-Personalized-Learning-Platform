# RAG Chat

A PDF question-answering app: upload PDFs, they get chunked + embedded +
stored in a local ChromaDB vector store, then you ask questions in a chat UI
and get answers grounded in your documents (with source citations).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your Groq API key
streamlit run app.py
```

Get a free Groq API key at https://console.groq.com/keys

**Important:** if you were previously using an API key that was committed to
a notebook or repo, revoke it in the Groq console and generate a new one —
treat any key that touched version control as compromised.

## How it works

- `src/ingestion.py` — loads PDFs and splits them into chunks
- `src/embeddings.py` — generates embeddings with sentence-transformers
- `src/vectorstore.py` — stores/retrieves chunks in ChromaDB
- `src/retriever.py` — similarity search over the vector store
- `src/rag_chain.py` — combines retrieval + Groq LLM into an answer
- `src/registry.py` — tracks which files were already processed (by content
  hash) so re-uploading the same PDF doesn't re-run embedding
- `app.py` — the Streamlit UI

## Features

- Upload PDFs from the browser; duplicate files (same content, even if
  renamed) are detected and skipped automatically
- Live progress while a new file is being chunked/embedded/stored
- Chat-style interface with conversation history
- Streamed answers (tokens appear as they're generated)
- Expandable source citations with page number and similarity score per answer
- Sidebar file manager — see processed files and delete one (removes its
  chunks from the vector store too)
- Adjustable retrieval settings (top_k, similarity threshold) and LLM
  temperature, live from the sidebar
- Cached models/connections so the embedding model and vector store aren't
  reloaded on every interaction

## Ideas for extending further

- Support .docx/.txt uploads (swap in other langchain loaders)
- Add a cross-encoder re-ranking step after retrieval for higher answer quality
- Hybrid search (BM25 + vector) for queries with exact keywords/names
- Per-message thumbs up/down feedback logging
- Docker Compose setup for one-command deployment