from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests


@dataclass
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class OllamaClient:
    """
    Minimal HTTP client for the local Ollama chat API.
    """

    def __init__(
        self,
        model: str = "llama3:8b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }

        response = requests.post(
            f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        # Ollama chat API returns the latest message as "message"
        return data["message"]["content"]

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        messages: List[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))
        return self.chat(messages, temperature=temperature)

