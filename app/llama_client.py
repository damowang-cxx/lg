# app/llama_client.py
from __future__ import annotations

import requests
from typing import Dict, List

PLANNER_URL = "http://127.0.0.1:8081/v1/chat/completions"
CHAT_URL    = "http://127.0.0.1:8082/v1/chat/completions"

def llama_chat(url: str, messages: List[Dict[str, str]], *, max_tokens: int = 800, temperature: float = 0.2, timeout: int = 180) -> str:
    r = requests.post(
        url,
        json={
            "model": "qwen2.5-7b-instruct",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def planner_json(messages: List[Dict[str, str]], *, max_tokens: int = 900, temperature: float = 0.2) -> str:
    return llama_chat(PLANNER_URL, messages, max_tokens=max_tokens, temperature=temperature)

def chat_text(messages: List[Dict[str, str]], *, max_tokens: int = 512, temperature: float = 0.7) -> str:
    return llama_chat(CHAT_URL, messages, max_tokens=max_tokens, temperature=temperature)