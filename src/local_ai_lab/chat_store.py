from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any


class ChatStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def list_chats(self) -> list[dict[str, Any]]:
        with self._lock:
            payload = self._read_payload()
            chats = payload.get("chats", [])
            if not isinstance(chats, list):
                return []
            return chats

    def upsert_chat(self, chat: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            payload = self._read_payload()
            chats = payload.get("chats", [])
            chat_id = chat.get("id")

            replaced = False
            for index, existing in enumerate(chats):
                if existing.get("id") == chat_id:
                    chats[index] = chat
                    replaced = True
                    break

            if not replaced:
                chats.append(chat)

            payload["chats"] = chats
            self._write_payload(payload)
            return chat

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock:
            payload = self._read_payload()
            chats = payload.get("chats", [])
            filtered = [chat for chat in chats if chat.get("id") != chat_id]
            if len(filtered) == len(chats):
                return False

            payload["chats"] = filtered
            self._write_payload(payload)
            return True

    def _read_payload(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"chats": []}

        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"chats": []}

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
