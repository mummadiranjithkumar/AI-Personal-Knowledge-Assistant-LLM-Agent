from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from llm import OllamaClient, LLMMessage
from vector_store import VectorStore, DocumentChunk


@dataclass
class SearchResult:
    chunks: List[DocumentChunk]
    scores: List[float]


def semantic_search(
    query: str, vector_store: VectorStore, top_k: int = 5
) -> SearchResult:
    chunks, scores = vector_store.search(query=query, top_k=top_k)
    return SearchResult(chunks=chunks, scores=scores)


def summarize_context(
    question: str, chunks: List[DocumentChunk], llm: OllamaClient
) -> str:
    context_text = "\n\n".join(
        f"[Source: {c.source}] {c.text.strip()}" for c in chunks if c.text.strip()
    )
    if not context_text:
        return ""

    system = (
        "You summarize retrieved document chunks to help answer a user question. "
        "Return a concise summary focusing only on information relevant to the question."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        "Provide a concise summary that will help answer the question."
    )

    return llm.complete(prompt=user, system_prompt=system, temperature=0.1)

