from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CACHE_DIR = Path("data/processed/embeddings")
PASSAGE_CACHE = CACHE_DIR / "passages.npy"
QUERY_CACHE = CACHE_DIR / "queries.npy"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode_passages(self, texts):
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_queries(self, texts):
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    @staticmethod
    def save(embeddings: np.ndarray, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    @staticmethod
    def load(path: Path):
        return np.load(path)
