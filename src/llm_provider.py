from __future__ import annotations

import os
from typing import Any

import requests

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"


def get_setting(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def get_provider() -> str:
    provider = get_setting("GEOSCOPE_PROVIDER", "ollama").strip().lower()
    if provider not in {"ollama", "openai"}:
        raise ValueError("GEOSCOPE_PROVIDER must be 'ollama' or 'openai'.")
    return provider


def get_generation_model() -> str:
    if get_provider() == "openai":
        return get_setting("OPENAI_GENERATION_MODEL", DEFAULT_OPENAI_MODEL)
    return get_setting("OLLAMA_GENERATION_MODEL", DEFAULT_OLLAMA_MODEL)


def get_judge_model() -> str:
    if get_provider() == "openai":
        return get_setting("OPENAI_JUDGE_MODEL", DEFAULT_OPENAI_MODEL)
    return get_setting("OLLAMA_JUDGE_MODEL", "llama3.1:8b")


def generate_text(*, instructions: str, prompt: str, model: str | None = None, json_output: bool = False) -> str:
    provider = get_provider()

    if provider == "openai":
        from openai import OpenAI
        api_key = get_setting("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured. See OPENAI_REVIEWER_SETUP.md.")
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model or get_generation_model(),
            instructions=instructions,
            input=prompt,
            store=False,
        )
        return response.output_text

    payload: dict[str, Any] = {
        "model": model or get_generation_model(),
        "prompt": f"INSTRUCTIONS:\n{instructions}\n\nUSER REQUEST:\n{prompt}",
        "stream": False,
        "options": {"temperature": 0 if json_output else 0.1},
    }
    if json_output:
        payload["format"] = "json"

    response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("response", "")
