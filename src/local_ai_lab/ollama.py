from __future__ import annotations

import asyncio
from typing import Any

import httpx


class OllamaError(Exception):
    def __init__(self, message: str, status_code: int | None = None, response_text: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_retry_delays = [1.0, 2.0, 4.0]

    async def list_models(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            response = await client.get("/api/tags")
            self._raise_for_status(response, action="list models")
            payload = response.json()
            return payload.get("models", [])

    async def show_model(self, model: str) -> dict[str, Any]:
        body = {"model": model}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            response = await client.post("/api/show", json=body)
            self._raise_for_status(response, action=f"inspect model '{model}'")
            return response.json()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(base_url=self.base_url, timeout=300.0) as client:
            attempts = len(self.chat_retry_delays) + 1
            for attempt in range(attempts):
                try:
                    response = await client.post("/api/chat", json=body)
                    self._raise_for_status(response, action=f"chat with model '{model}'")
                    return response.json()
                except OllamaError as exc:
                    if exc.status_code != 503 or attempt >= attempts - 1:
                        raise
                    await asyncio.sleep(self.chat_retry_delays[attempt])
                except httpx.ConnectError as exc:
                    if attempt >= attempts - 1:
                        raise OllamaError(
                            message=f"Could not connect to Ollama while chatting with model '{model}': {exc}"
                        ) from exc
                    await asyncio.sleep(self.chat_retry_delays[attempt])

        raise OllamaError(message=f"Ollama did not return a response for model '{model}'.")

    @staticmethod
    def _raise_for_status(response: httpx.Response, action: str) -> None:
        if response.is_success:
            return

        response_text = response.text.strip()
        detail = response_text or response.reason_phrase or "Unknown Ollama error"
        raise OllamaError(
            message=f"Ollama failed to {action} ({response.status_code}): {detail}",
            status_code=response.status_code,
            response_text=response_text,
        )
