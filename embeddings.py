"""
Embedding generation
"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Handle document embedding generation using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embedding for {len(texts)} texts..")
        embeddings = self.model.encode(texts)
        print(f"Generated embedding with shape: {embeddings.shape}")
        return embeddings