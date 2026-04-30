from __future__ import annotations

from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


def inspect_model(model_name: str, limit: int = 12) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    api = HfApi()
    files = api.list_repo_files(model_name)
    safetensor_files = [name for name in files if name.endswith(".safetensors")]

    tensor_samples: list[dict[str, Any]] = []
    if safetensor_files:
        local_path = hf_hub_download(repo_id=model_name, filename=safetensor_files[0])
        with safe_open(local_path, framework="pt") as handle:
            for index, key in enumerate(handle.keys()):
                if index >= limit:
                    break
                tensor = handle.get_tensor(key)
                tensor_samples.append(
                    {
                        "name": key,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                )

    return {
        "model": model_name,
        "architectures": getattr(config, "architectures", []),
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "special_tokens": tokenizer.special_tokens_map,
        "safetensors": safetensor_files,
        "tensor_samples": tensor_samples,
    }
