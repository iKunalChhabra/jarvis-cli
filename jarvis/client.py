"""
High‑level Ollama client
-----------------------------------
Implements a thin *Facade* over the official Ollama Python SDK so the rest
of the application never touches HTTP directly.
"""
from __future__ import annotations

import logging
import time
from typing import List, Dict, Iterator

import ollama                                # type: ignore[import]
from requests.exceptions import ConnectionError, ReadTimeout

LOGGER = logging.getLogger(__name__)
MODEL_NAME: str = "gemma3:4b"
MAX_RETRIES: int = 3
BACKOFF_SEC: float = 1.5


class JarvisClient:
    """Wrapper around ``ollama.chat`` with streaming support and retries."""

    def __init__(self, model: str = MODEL_NAME) -> None:
        self._model = model
        self._history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are **Jarvis**, Tony‑Stark‑style AI assistant. "
                    "Respond concisely with helpful explanations. "
                    "Format code inside triple‑backtick blocks."
                ),
            }
        ]

    # ---------- Public API -------------------------------------------------

    def reset(self) -> None:
        """
        Forget the current chat except for the system prompt.

        Time‑complexity : O(1)
        """
        self._history = self._history[:1]          # keep index‑0 (system)

    def ask(self, prompt: str) -> Iterator[str]:
        """
        Stream the assistant's response for a given *user* prompt.

        Time‑complexity: **O(L)** where *L* is the number of streamed chunks.
        Memory‑complexity: **O(H)** where *H* is history size.

        Raises
        ------
        RuntimeError
            If the Ollama backend is unreachable after ``MAX_RETRIES``.
        """
        self._history.append({"role": "user", "content": prompt})
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                stream = ollama.chat(
                    model=self._model,
                    messages=self._history,
                    stream=True,              # enables token streaming
                )
                for chunk in stream:
                    yield chunk["message"]["content"]
                # complete answer captured – append to history
                self._history.append(
                    {"role": "assistant", "content": self._history[-1]["content"]}
                )
                return
            except (ConnectionError, ReadTimeout) as exc:
                LOGGER.warning(
                    "Ollama unavailable (attempt %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
                time.sleep(BACKOFF_SEC * attempt)
        raise RuntimeError("Could not reach the Ollama server after several attempts.")