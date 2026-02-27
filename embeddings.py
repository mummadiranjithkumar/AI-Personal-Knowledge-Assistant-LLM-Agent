from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
EMBEDDINGS_CACHE_PATH = DATA_DIR / "embeddings_cache.pkl"


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer with a simple on-disk cache.

    The cache is keyed by a SHA256 hash of the input text to avoid
    recomputing embeddings for identical chunks across runs.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._cache: Dict[str, np.ndarray] = {}

        os.makedirs(DATA_DIR, exist_ok=True)
        self._load_cache()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_cache(self) -> None:
        if EMBEDDINGS_CACHE_PATH.exists():
            try:
                with EMBEDDINGS_CACHE_PATH.open("rb") as f:
                    cached = pickle.load(f)
                # Ensure numpy arrays
                self._cache = {k: np.asarray(v) for k, v in cached.items()}
            except Exception:
                # Corrupt cache: start fresh
                self._cache = {}

    def _save_cache(self) -> None:
        try:
            with EMBEDDINGS_CACHE_PATH.open("wb") as f:
                pickle.dump(self._cache, f)
        except Exception:
            # Best-effort cache persistence; failures should not break the app.
            pass

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """
        Embed a list of texts, using the cache whenever possible.
        """
        texts_list: List[str] = list(texts)
        keys = [self._hash_text(t) for t in texts_list]

        # Determine which need to be computed
        to_compute_indices = [i for i, k in enumerate(keys) if k not in self._cache]
        if to_compute_indices:
            to_compute_texts = [texts_list[i] for i in to_compute_indices]
            new_vectors = self.model.encode(to_compute_texts)
            for j, original_idx in enumerate(to_compute_indices):
                key = keys[original_idx]
                self._cache[key] = np.asarray(new_vectors[j])
            self._save_cache()

        # Assemble result in original order
        vectors = np.stack([self._cache[k] for k in keys], axis=0)
        return vectors

