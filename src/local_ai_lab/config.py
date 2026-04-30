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
    pdf_folder_path: Path = Path(os.getenv("PDF_FOLDER_PATH", str(Path.home() / "26_code/___Books")))
    pdf_index_path: Path = Path(os.getenv("PDF_INDEX_PATH", "data/pdf_index.json"))
    pdf_embedding_model: str = os.getenv("PDF_EMBEDDING_MODEL", "BAAI/bge-m3")
    pdf_reranker_model: str = os.getenv("PDF_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


settings = Settings()
