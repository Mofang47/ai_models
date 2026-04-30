from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .chat_store import ChatStore
from .config import settings
from .generate import generate_text
from .inspect_weights import inspect_model
from .ollama import OllamaClient, OllamaError
from .pdf_search import PdfSearchClient
from .web_search import WebSearchClient


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI(title="Local AI Lab")
ollama = OllamaClient(settings.ollama_base_url)
web_search = WebSearchClient()
chat_store = ChatStore(settings.chat_store_path)
pdf_search = PdfSearchClient(
    folder_path=settings.pdf_folder_path,
    index_path=settings.pdf_index_path,
    embedding_model=settings.pdf_embedding_model,
    reranker_model=settings.pdf_reranker_model,
)


class ChatMessage(BaseModel):
    role: str
    content: str


class SourceItem(BaseModel):
    title: str
    url: str
    snippet: str = ""
    excerpt: str = ""


class StoredMessage(BaseModel):
    role: str
    content: str
    sources: list[SourceItem] = Field(default_factory=list)


class StoredChat(BaseModel):
    id: str
    title: str = "New chat"
    model: str = Field(default_factory=lambda: settings.default_ollama_model)
    systemPrompt: str = "You are a helpful local AI assistant."
    temperature: float = 0.7
    useWebSearch: bool = False
    webSearchQuery: str = ""
    usePdfSearch: bool = False
    messages: list[StoredMessage] = Field(default_factory=list)
    createdAt: str
    updatedAt: str


class ChatRequest(BaseModel):
    model: str = Field(default_factory=lambda: settings.default_ollama_model)
    system_prompt: str = "You are a helpful local AI assistant."
    message: str = ""
    messages: list[ChatMessage] = Field(default_factory=list)
    temperature: float = 0.7
    use_web_search: bool = False
    web_search_query: str = ""
    web_result_count: int = 4
    use_pdf_search: bool = False
    pdf_result_count: int = 6


class ChatResponse(BaseModel):
    message: dict[str, object]
    sources: list[SourceItem] = Field(default_factory=list)


class ChatListResponse(BaseModel):
    chats: list[StoredChat] = Field(default_factory=list)


class PdfLibraryStatus(BaseModel):
    folder: str
    index: str
    exists: bool
    file_count: int
    chunk_count: int
    updated_at: str = ""
    needs_rebuild: bool
    embedding_model: str
    reranker_model: str
    mode: str


class ExplorerPlaygroundRequest(BaseModel):
    source: str = "ollama"
    model: str
    prompt: str
    system_prompt: str = "You are a helpful local AI assistant."
    temperature: float = 0.7
    max_new_tokens: int = 160


class ExplorerPlaygroundResponse(BaseModel):
    source: str
    model: str
    output: str
    prompt: str


class ExplorerModelSummary(BaseModel):
    id: str
    source: str
    title: str
    family: str = ""
    size_label: str = ""
    parameter_size: str = ""
    quantization: str = ""
    context_window: str = ""
    modified_at: str = ""


class ExplorerModelDetail(BaseModel):
    summary: ExplorerModelSummary
    raw: dict[str, object] = Field(default_factory=dict)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
async def models() -> dict[str, object]:
    try:
        model_items = await ollama.list_models()
    except OllamaError as exc:  # pragma: no cover - network/runtime issue
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/runtime issue
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {exc}") from exc
    return {"models": model_items}


@app.get("/api/chats")
async def list_chats() -> ChatListResponse:
    chats = [StoredChat.model_validate(item) for item in chat_store.list_chats()]
    chats.sort(key=lambda chat: chat.updatedAt, reverse=True)
    return ChatListResponse(chats=chats)


@app.get("/api/pdf-library")
async def pdf_library_status() -> PdfLibraryStatus:
    return PdfLibraryStatus(**pdf_search.status())


@app.post("/api/pdf-library/rebuild")
async def rebuild_pdf_library() -> PdfLibraryStatus:
    try:
        pdf_search.rebuild_index()
    except Exception as exc:  # pragma: no cover - local parser/runtime issue
        raise HTTPException(status_code=502, detail=f"PDF indexing failed: {exc}") from exc
    return PdfLibraryStatus(**pdf_search.status())


@app.get("/api/explorer/models")
async def explorer_models() -> dict[str, object]:
    try:
        ollama_models = await ollama.list_models()
    except OllamaError as exc:  # pragma: no cover - network/runtime issue
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - network/runtime issue
        raise HTTPException(status_code=502, detail=f"Could not reach Ollama: {exc}") from exc

    items: list[ExplorerModelSummary] = []
    for entry in ollama_models:
        details = entry.get("details", {})
        items.append(
            ExplorerModelSummary(
                id=entry.get("name", ""),
                source="ollama",
                title=entry.get("name", ""),
                family=str(details.get("family", "")),
                size_label=format_bytes(entry.get("size")),
                parameter_size=str(details.get("parameter_size", "")),
                quantization=str(details.get("quantization_level", "")),
                context_window="",
                modified_at=str(entry.get("modified_at", "")),
            )
        )

    items.append(
        ExplorerModelSummary(
            id=settings.default_hf_model,
            source="huggingface",
            title=settings.default_hf_model,
            family="transformers",
            size_label="",
            parameter_size="",
            quantization="",
            context_window="",
            modified_at="",
        )
    )
    return {"models": [item.model_dump() for item in items]}


@app.get("/api/explorer/models/{source}/{model_id:path}")
async def explorer_model_detail(source: str, model_id: str) -> ExplorerModelDetail:
    if source == "ollama":
        try:
            raw = await ollama.show_model(model_id)
        except OllamaError as exc:  # pragma: no cover - network/runtime issue
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - network/runtime issue
            raise HTTPException(status_code=502, detail=f"Could not inspect Ollama model: {exc}") from exc

        details = raw.get("details", {})
        summary = ExplorerModelSummary(
            id=model_id,
            source="ollama",
            title=model_id,
            family=str(details.get("family", "")),
            size_label="",
            parameter_size=str(details.get("parameter_size", "")),
            quantization=str(details.get("quantization_level", "")),
            context_window=str(raw.get("context_length", "")),
            modified_at="",
        )
        return ExplorerModelDetail(summary=summary, raw=raw)

    if source == "huggingface":
        try:
            raw = inspect_model(model_id)
        except Exception as exc:  # pragma: no cover - network/runtime issue
            raise HTTPException(status_code=502, detail=f"Could not inspect Hugging Face model: {exc}") from exc

        summary = ExplorerModelSummary(
            id=model_id,
            source="huggingface",
            title=model_id,
            family=str(raw.get("model_type", "")),
            size_label="",
            parameter_size="",
            quantization="",
            context_window=str(raw.get("max_position_embeddings", "")),
            modified_at="",
        )
        return ExplorerModelDetail(summary=summary, raw=raw)

    raise HTTPException(status_code=404, detail="Unknown model source.")


@app.post("/api/explorer/playground")
async def explorer_playground(payload: ExplorerPlaygroundRequest) -> ExplorerPlaygroundResponse:
    if payload.source == "ollama":
        try:
            result = await ollama.chat(
                model=payload.model,
                messages=[
                    {"role": "system", "content": payload.system_prompt},
                    {"role": "user", "content": payload.prompt},
                ],
                temperature=payload.temperature,
            )
        except OllamaError as exc:  # pragma: no cover - network/runtime issue
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - network/runtime issue
            raise HTTPException(status_code=502, detail=f"Playground request failed: {exc}") from exc
        return ExplorerPlaygroundResponse(
            source=payload.source,
            model=payload.model,
            prompt=payload.prompt,
            output=str(result.get("message", {}).get("content", "")),
        )

    if payload.source == "huggingface":
        try:
            output = generate_text(
                model_name=payload.model,
                prompt=payload.prompt,
                max_new_tokens=payload.max_new_tokens,
            )
        except Exception as exc:  # pragma: no cover - runtime/model issue
            raise HTTPException(status_code=502, detail=f"Transformers generation failed: {exc}") from exc
        return ExplorerPlaygroundResponse(
            source=payload.source,
            model=payload.model,
            prompt=payload.prompt,
            output=output,
        )

    raise HTTPException(status_code=404, detail="Unknown model source.")


@app.put("/api/chats/{chat_id}")
async def save_chat(chat_id: str, payload: StoredChat) -> StoredChat:
    if payload.id != chat_id:
        raise HTTPException(status_code=400, detail="Chat id mismatch.")
    return StoredChat.model_validate(chat_store.upsert_chat(payload.model_dump()))


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str) -> dict[str, bool]:
    deleted = chat_store.delete_chat(chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"deleted": True}


@app.post("/api/chat")
async def chat(payload: ChatRequest) -> ChatResponse:
    messages = [{"role": "system", "content": payload.system_prompt}]
    sources: list[dict[str, object]] = []
    grounding_sections: list[str] = []

    if payload.use_web_search:
        query = payload.web_search_query.strip() or payload.message.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Web search needs a query or message.")
        try:
            grounding, web_sources = await web_search.build_grounding(
                query=query,
                max_results=max(1, min(payload.web_result_count, 8)),
            )
        except Exception as exc:  # pragma: no cover - network/runtime issue
            raise HTTPException(status_code=502, detail=f"Web search failed: {exc}") from exc

        if grounding:
            grounding_sections.append(
                "You can use live web research that was retrieved for this request.\n" + grounding
            )
            sources.extend(web_sources)

    if payload.use_pdf_search:
        query = payload.message.strip()
        if not query:
            raise HTTPException(status_code=400, detail="PDF library search needs a message.")
        try:
            grounding, pdf_sources = pdf_search.build_grounding(
                query=query,
                max_results=max(1, min(payload.pdf_result_count, 10)),
            )
        except Exception as exc:  # pragma: no cover - local parser/runtime issue
            raise HTTPException(status_code=502, detail=f"PDF library search failed: {exc}") from exc

        if grounding:
            grounding_sections.append(grounding)
            sources.extend(pdf_sources)

    if grounding_sections:
        messages[0]["content"] = f"{payload.system_prompt}\n\n" + "\n\n".join(grounding_sections)

    messages.extend(message.model_dump() for message in payload.messages)
    if payload.message.strip():
        messages.append({"role": "user", "content": payload.message})
    try:
        result = await ollama.chat(
            model=payload.model,
            messages=messages,
            temperature=payload.temperature,
        )
    except OllamaError as exc:  # pragma: no cover - network/runtime issue
        raise HTTPException(status_code=502, detail=str(exc)) from exc
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


@app.get("/explore", response_class=HTMLResponse)
async def explore(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "explore.html",
        {"default_model": settings.default_ollama_model, "default_hf_model": settings.default_hf_model},
    )


def format_bytes(value: object) -> str:
    if not isinstance(value, int) or value <= 0:
        return ""

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return ""
