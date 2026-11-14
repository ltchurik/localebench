"""Minimal client for OpenAI-compatible /chat/completions. Standard library only.

Works with OpenAI, OpenRouter, Ollama, vLLM, LM Studio, llama.cpp server and anything
else that speaks the same protocol.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request


class ApiError(RuntimeError):
    pass


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Drop <think>…</think> blocks that some reasoning models put into the answer."""
    return _THINK_RE.sub("", text).strip()


def resolve_api_key(api_key: str | None = None) -> str:
    return api_key or os.environ.get("LOCALEBENCH_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


def chat(
    prompt: str,
    *,
    model: str,
    base_url: str,
    api_key: str | None = None,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 300,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Send one prompt, return the answer text. Retries transient errors with exponential backoff."""
    url = base_url.rstrip("/") + "/chat/completions"
    key = resolve_api_key(api_key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    ).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return strip_reasoning(data["choices"][0]["message"]["content"] or "")
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            last_err = ApiError(f"HTTP {e.code}: {msg}")
            if e.code in (400, 401, 403, 404):
                break
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
            last_err = ApiError(str(e))
        if attempt < retries - 1:
            time.sleep(backoff**attempt)
    raise last_err or ApiError("unknown error")
