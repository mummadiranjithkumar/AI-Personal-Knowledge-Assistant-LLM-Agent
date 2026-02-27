from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict, Any

from llm import OllamaClient, LLMMessage
from tools import semantic_search, summarize_context, SearchResult
from vector_store import VectorStore, DocumentChunk


ToolName = Literal["none", "semantic_search", "summarize_context"]


@dataclass
class AgentStepTrace:
    tool_used: ToolName
    tool_input: Optional[Dict[str, Any]] = None
    retrieved_chunks: List[DocumentChunk] = field(default_factory=list)


@dataclass
class AgentResponse:
    answer: str
    steps: List[AgentStepTrace]
    reached_max_iterations: bool


class PersonalKnowledgeAgent:
    """
    LLM-based agent that decides when to retrieve from the vector store
    and when to answer directly, with a simple max-iteration guardrail.
    """

    def __init__(
        self,
        llm: OllamaClient,
        vector_store: VectorStore,
        max_iterations: int = 4,
    ) -> None:
        self.llm = llm
        self.vector_store = vector_store
        self.max_iterations = max_iterations

    def _select_action(
        self, question: str, chat_history: List[LLMMessage]
    ) -> Literal["retrieve", "answer_direct"]:
        """
        Ask the LLM whether to perform retrieval or answer directly.
        """
        system_prompt = (
            "You are a routing assistant that decides whether a question should be "
            "answered directly or requires retrieval from a user's personal documents.\n\n"
            "Respond with strict JSON only, no markdown, of the form:\n"
            '{"action": "retrieve" | "answer_direct", "reason": "..."}'
        )

        history_snippets = "\n".join(
            f"{m.role}: {m.content}" for m in chat_history[-4:]
        )
        user_prompt = (
            f"Recent chat history:\n{history_snippets}\n\n"
            f"Current question:\n{question}\n"
        )

        raw = self.llm.complete(
            prompt=user_prompt, system_prompt=system_prompt, temperature=0.1
        )
        try:
            data = json.loads(raw)
            action = data.get("action", "retrieve")
            if action not in {"retrieve", "answer_direct"}:
                return "retrieve"
            return action  # type: ignore[return-value]
        except Exception:
            # On parsing failure, default to retrieval for safety.
            return "retrieve"

    def run(
        self, question: str, chat_history: List[LLMMessage]
    ) -> AgentResponse:
        """
        Main agent loop.

        High-level logic:
        1. Ask LLM whether to retrieve from the vector store or answer directly.
        2. If answer_direct: have LLM respond using chat history only.
        3. If retrieve:
           - Run semantic search.
           - Optionally summarize retrieved chunks.
           - Ask LLM to answer using retrieved context (and summary).
        4. A simple max-iteration guardrail prevents runaway loops.
        """
        steps: List[AgentStepTrace] = []
        reached_max = False

        action = self._select_action(question, chat_history)

        # Iteration 1: either direct answer or retrieval
        if action == "answer_direct":
            messages = chat_history + [LLMMessage(role="user", content=question)]
            answer = self.llm.chat(messages)
            steps.append(AgentStepTrace(tool_used="none"))
            return AgentResponse(
                answer=answer, steps=steps, reached_max_iterations=reached_max
            )

        # Retrieval path
        search_result: SearchResult = semantic_search(
            query=question, vector_store=self.vector_store, top_k=5
        )
        steps.append(
            AgentStepTrace(
                tool_used="semantic_search",
                tool_input={"top_k": 5},
                retrieved_chunks=search_result.chunks,
            )
        )

        if not search_result.chunks:
            # Nothing to retrieve; fall back to generic answer.
            messages = chat_history + [LLMMessage(role="user", content=question)]
            answer = self.llm.chat(messages)
            return AgentResponse(
                answer=answer, steps=steps, reached_max_iterations=reached_max
            )

        # Iteration 2: optional summarization + final answer
        if self.max_iterations <= 1:
            reached_max = True

        summary = ""
        if len(search_result.chunks) > 2 and self.max_iterations >= 2:
            summary = summarize_context(
                question=question,
                chunks=search_result.chunks,
                llm=self.llm,
            )
            steps.append(
                AgentStepTrace(
                    tool_used="summarize_context",
                    tool_input={"num_chunks": len(search_result.chunks)},
                    retrieved_chunks=search_result.chunks,
                )
            )

        if len(steps) >= self.max_iterations:
            reached_max = True

        context_text = "\n\n".join(
            f"[Source: {c.source}] {c.text.strip()}"
            for c in search_result.chunks
            if c.text.strip()
        )

        final_system = (
            "You are a helpful AI personal knowledge assistant. "
            "Use ONLY the provided context from the user's documents when possible. "
            "If the answer is not in the context, say you do not know instead of "
            "hallucinating.\n"
        )

        final_user = (
            f"Question:\n{question}\n\n"
            f"Context from user's documents:\n{context_text}\n\n"
        )
        if summary:
            final_user += f"Summary of context:\n{summary}\n\n"

        answer = self.llm.complete(
            prompt=final_user, system_prompt=final_system, temperature=0.2
        )

        return AgentResponse(
            answer=answer, steps=steps, reached_max_iterations=reached_max
        )

