"""
ChromaDB vector store wrapper. 
Added: delete_by_source() and list_sources() so the UI can let a user
remove a previously-uploaded file's chunks.
"""
import os
import uuid
from typing import Any, List

import numpy as np
import chromadb


class VectorStore:
    """Manage document embeddings in a ChromaDB vector store."""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embedding for RAG",
                    "hnsw:space": "cosine",
                },
            )
            print(f"Vector store initialized Collection: {self.collection_name}")
            print(f"Existing documents in collection:{self.collection.count()}")

        except Exception as e:
            print(f"error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text,
            )
            print(f"successfully added {len(documents)} documents to vector store")
            print(f"total documents in collection {self.collection.count()}")
        except Exception as e:
            print(f"Error adding document to vector store: {e}")
            raise

    def delete_by_source(self, source_file: str):
        """Remove every chunk that came from a given uploaded filename.
        Used when the user deletes a file from the sidebar."""
        try:
            self.collection.delete(where={"source_file": source_file})
            print(f"Deleted all chunks for source_file={source_file}")
        except Exception as e:
            print(f"Error deleting {source_file}: {e}")
            raise

    def list_sources(self):
        """Return the set of distinct source_file values currently stored.
        Handy as a sanity check against the JSON registry."""
        if self.collection.count() == 0:
            return set()
        data = self.collection.get(include=["metadatas"])
        return {m.get("source_file", "unknown") for m in data["metadatas"]}