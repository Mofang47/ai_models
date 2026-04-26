# Local AI Lab

This repo is set up for two parallel goals:

1. Build a small local AI app you can use every day
2. Learn how open models work by inspecting tokenization, weights, and inference

## Why this layout

There are two different ways to "play with a model":

- App mode: use a practical local runtime like Ollama to chat quickly
- Learning mode: load an open model directly in Python so you can inspect weights and inference internals

This repo supports both.

## Recommended models for this machine

This Mac has an Apple M4 with 16 GB RAM, so the comfortable range is:

- Default app model: `qwen3:4b` via Ollama
- Current multimodal model to try: `gemma4:e2b`
- Small fast baseline: `llama3.2:3b`
- Inspection model: `Qwen/Qwen2.5-0.5B-Instruct` or a similar small Hugging Face model

The small inspection model keeps iteration fast while still letting you explore real transformer weights.

## Project layout

- `src/local_ai_lab/app.py`: FastAPI app with a tiny local chat UI
- `src/local_ai_lab/ollama.py`: thin wrapper around Ollama's local HTTP API
- `src/local_ai_lab/inspect_weights.py`: inspect model weights, architecture, and tokenization
- `src/local_ai_lab/generate.py`: direct local generation from a Hugging Face model
- `src/local_ai_lab/cli.py`: command entrypoints

## Install `uv`

This machine does not currently have `uv` installed.

If you want me to do it, I can install it next. Otherwise, a common path is:

```bash
brew install uv
```

## Create the environment

After `uv` is installed:

```bash
uv venv
uv sync
```

## Run the local app

First make sure Ollama is installed and running, and pull at least one chat model:

```bash
ollama pull qwen3:4b
ollama serve
```

Then run:

```bash
uv run local-ai-lab app
```

Open `http://127.0.0.1:8000`.

## Inspect a model directly

Print architecture and a sample of the weights:

```bash
uv run local-ai-lab inspect --model Qwen/Qwen2.5-0.5B-Instruct
```

Generate text directly from the Hugging Face model path:

```bash
uv run local-ai-lab generate --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain attention simply."
```

## Suggested learning path

1. Use the local chat app to feel how model size changes speed and quality
2. Run `inspect` on a small model and look at tensor names and shapes
3. Run `generate` to compare direct inference with app-mode inference
4. Inspect logits, token ids, and decoding behavior
5. Later, add notebooks or experiments for attention, KV cache, and quantization

## Notes

- Direct Hugging Face model loading will download weights on first use
- Larger models are possible, but not all of them will be pleasant on a 16 GB MacBook Air
- The app layer and learning layer are intentionally separate so you can move fast without losing transparency
