from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_text(model_name: str, prompt: str, max_new_tokens: int = 120) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
