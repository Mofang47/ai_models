from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    default_ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3:4b")
    default_hf_model: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    chat_store_path: Path = Path(os.getenv("CHAT_STORE_PATH", "data/chats.json"))


settings = Settings()
