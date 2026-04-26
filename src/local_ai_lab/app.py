from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .config import settings
from .ollama import OllamaClient
from .web_search import WebSearchClient


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(title="Local AI Lab")
ollama = OllamaClient(settings.ollama_base_url)
web_search = WebSearchClient()


class ChatMessage(BaseModel):
    role: str
    content: str


class SourceItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    excerpt: str = ""


class ChatRequest(BaseModel):
    model: str = Field(default_factory=lambda: settings.default_ollama_model)
    system_prompt: str = "You are a helpful local AI assistant."
    message: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float = 0.7
    use_web_search: bool = False
    web_search_query: str = ""
    web_result_count: int = 4


class ChatResponse(BaseModel):
    message: dict[str, object]
    sources: list[SourceItem] = Field(default_factory=list)


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
async def chat(payload: ChatRequest) -> ChatResponse:
    messages = [{"role": "system", "content": payload.system_prompt}]
    sources: list[dict[str, object]] = []

    if payload.use_web_search:
        query = payload.web_search_query.strip() or payload.message.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Web search needs a query or message.")
        try:
            grounding, sources = await web_search.build_grounding(
                query=query,
                max_results=max(1, min(payload.web_result_count, 8)),
            )
        except Exception as exc:  # pragma: no cover - network/runtime issue
            raise HTTPException(status_code=502, detail=f"Web search failed: {exc}") from exc

        if grounding:
            messages[0]["content"] = (
                f"{payload.system_prompt}\n\n"
                "You can use live web research that was retrieved for this request.\n"
                f"{grounding}"
            )

    messages.extend(message.model_dump() for message in payload.messages)
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
    return ChatResponse(message=result.get("message", result), sources=[SourceItem(**item) for item in sources])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"default_model": settings.default_ollama_model},
    )
