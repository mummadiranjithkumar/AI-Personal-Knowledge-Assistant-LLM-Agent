from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np

from embeddings import EmbeddingModel


DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "vector_store.faiss"
METADATA_PATH = DATA_DIR / "vector_store_metadata.pkl"


@dataclass
class DocumentChunk:
    id: int
    doc_id: str
    source: str
    text: str


class VectorStore:
    """
    Simple FAISS-based vector store with on-disk persistence.
    """

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        self.embedding_model = embedding_model
        self.index: faiss.Index | None = None
        self.chunks: List[DocumentChunk] = []
        self._doc_ids: set[str] = set()

        self._load()

    @property
    def dimension(self) -> int:
        # Lazily infer from model if index does not exist yet
        dummy = self.embedding_model.embed_texts(["dimension-probe"])
        return int(dummy.shape[1])

    def _create_empty_index(self) -> faiss.Index:
        index = faiss.IndexFlatL2(self.dimension)
        return index

    def _load(self) -> None:
        if INDEX_PATH.exists() and METADATA_PATH.exists():
            try:
                self.index = faiss.read_index(str(INDEX_PATH))
                with METADATA_PATH.open("rb") as f:
                    data = pickle.load(f)
                self.chunks = data.get("chunks", [])
                self._doc_ids = set(data.get("doc_ids", []))
            except Exception:
                # Corrupt index or metadata: start from scratch
                self.index = self._create_empty_index()
                self.chunks = []
                self._doc_ids = set()
        else:
            self.index = self._create_empty_index()
            self.chunks = []
            self._doc_ids = set()

    def _save(self) -> None:
        if self.index is None:
            return
        try:
            faiss.write_index(self.index, str(INDEX_PATH))
            with METADATA_PATH.open("wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "doc_ids": list(self._doc_ids),
                    },
                    f,
                )
        except Exception:
            # Best-effort persistence
            pass

    def has_document(self, doc_id: str) -> bool:
        return doc_id in self._doc_ids

    def add_document(self, doc_id: str, source: str, chunk_texts: List[str]) -> None:
        """
        Add all chunks for a document, embedding them and updating FAISS.
        """
        if self.index is None:
            self.index = self._create_empty_index()

        if self.has_document(doc_id):
            return

        vectors = self.embedding_model.embed_texts(chunk_texts)
        start_id = len(self.chunks)

        for i, text in enumerate(chunk_texts):
            chunk = DocumentChunk(
                id=start_id + i,
                doc_id=doc_id,
                source=source,
                text=text,
            )
            self.chunks.append(chunk)

        self.index.add(vectors.astype("float32"))
        self._doc_ids.add(doc_id)
        self._save()

    def search(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Perform a semantic search over the stored chunks.
        """
        if self.index is None or self.index.ntotal == 0:
            return [], []

        query_vec = self.embedding_model.embed_texts([query]).astype("float32")
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        retrieved_chunks: List[DocumentChunk] = []
        retrieved_scores: List[float] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            retrieved_chunks.append(self.chunks[int(idx)])
            retrieved_scores.append(float(dist))

        return retrieved_chunks, retrieved_scores

