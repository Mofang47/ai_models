from __future__ import annotations

import asyncio
import json
import socket
from contextlib import closing

import typer
import uvicorn

from .app import app as fastapi_app
from .config import settings
from .generate import generate_text
from .inspect_weights import inspect_model


app = typer.Typer(no_args_is_help=True)


def _create_server_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        sock.bind((host, port))
        sock.listen(socket.SOMAXCONN)
        sock.setblocking(False)
    except OSError:
        sock.close()
        raise

    return sock


@app.command()
def app_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the local FastAPI app."""
    config = uvicorn.Config(fastapi_app, host=host, port=port)
    server = uvicorn.Server(config)

    try:
        with closing(_create_server_socket(host, port)) as sock:
            asyncio.run(server.serve(sockets=[sock]))
    except OSError as exc:
        if exc.errno in {48, 98}:
            raise typer.BadParameter(
                f"Port {port} is already in use on {host}. Stop the old server or choose another port."
            ) from exc
        raise


@app.command("inspect")
def inspect_command(
    model: str = typer.Option(settings.default_hf_model, help="Hugging Face model id"),
    limit: int = typer.Option(12, help="How many tensors to print"),
) -> None:
    """Inspect architecture and sample tensor metadata."""
    result = inspect_model(model_name=model, limit=limit)
    print(json.dumps(result, indent=2))


@app.command("generate")
def generate_command(
    model: str = typer.Option(settings.default_hf_model, help="Hugging Face model id"),
    prompt: str = typer.Option(..., help="Prompt to generate from"),
    max_new_tokens: int = typer.Option(120, help="Maximum new tokens"),
) -> None:
    """Run direct local generation through Transformers."""
    print(generate_text(model_name=model, prompt=prompt, max_new_tokens=max_new_tokens))


@app.command("app")
def app_command(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Alias for starting the app."""
    app_server(host=host, port=port)
