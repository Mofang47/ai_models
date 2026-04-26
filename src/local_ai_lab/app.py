from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .config import settings
from .ollama import OllamaClient


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(title="Local AI Lab")
ollama = OllamaClient(settings.ollama_base_url)


class ChatRequest(BaseModel):
    model: str = Field(default_factory=lambda: settings.default_ollama_model)
    system_prompt: str = "You are a helpful local AI assistant."
    message: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    temperature: float = 0.7


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def models() -> dict[str, object]:
    try:
        model_items = await ollama.list_models()
    except Exception as exc:  # pragma: no cover - network/runtime issue
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {exc}") from exc
    return {"models": model_items}


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> dict[str, object]:
    messages = [{"role": "system", "content": payload.system_prompt}]
    messages.extend(payload.messages)
    if payload.message.strip():
        messages.append({"role": "user", "content": payload.message})
    try:
        result = await ollama.chat(
            model=payload.model,
            messages=messages,
            temperature=payload.temperature,
        )
    except Exception as exc:  # pragma: no cover - network/runtime issue
        raise HTTPException(status_code=502, detail=f"Chat request failed: {exc}") from exc
    return result


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"default_model": settings.default_ollama_model},
    )
