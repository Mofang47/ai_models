from __future__ import annotations

from typing import Any

import httpx


class OllamaClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def list_models(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60.0) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
            return payload.get("models", [])

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
            response = await client.post("/api/chat", json=body)
            response.raise_for_status()
            return response.json()
