# Session Notes

## Repo

- GitHub: `https://github.com/Mofang47/ai_models`
- Branch: `main`

## What We Built

- Set up the local Python environment with `uv`
- Fixed the FastAPI template rendering bug caused by the newer `TemplateResponse` signature
- Improved the chat UI with:
  - model selection
  - multi-turn conversation continuity
  - local chat history in the browser
  - better response formatting
- Added optional live web research mode:
  - web search and source fetching
  - grounded answers with source cards in the UI
- Added backend chat persistence:
  - chat history now saves to `data/chats.json`
  - UI syncs chats with backend endpoints
- Improved app startup behavior:
  - explicit socket-managed server startup
  - clearer port-in-use error handling

## Important Files

- [README.md](/Users/yan/26_code/ai_models/README.md)
- [src/local_ai_lab/app.py](/Users/yan/26_code/ai_models/src/local_ai_lab/app.py)
- [src/local_ai_lab/cli.py](/Users/yan/26_code/ai_models/src/local_ai_lab/cli.py)
- [src/local_ai_lab/chat_store.py](/Users/yan/26_code/ai_models/src/local_ai_lab/chat_store.py)
- [src/local_ai_lab/web_search.py](/Users/yan/26_code/ai_models/src/local_ai_lab/web_search.py)
- [src/local_ai_lab/templates/index.html](/Users/yan/26_code/ai_models/src/local_ai_lab/templates/index.html)

## Current Behavior

- App chat history is persisted in `data/chats.json`
- Browser `localStorage` is still used as a fallback cache
- Web research mode is optional per chat
- Source links are shown under grounded answers

## Recent Commits

- `84dae11` Build local AI lab chat app
- `f3ad577` Add web-grounded chat and safer server startup

## Useful Commands

```zsh
cd /Users/yan/26_code/ai_models
uv sync
uv run local-ai-lab app --port 8003
```

## Next Good Steps

- Commit and push the backend chat persistence changes
- Live-test the web research flow with real network access
- Optionally add:
  - private/public repo toggle guidance
  - favicon
  - export/import chats
  - server-side search across saved chats
