"""Thin Gemini wrapper shared by every agent: disk cache, rate-limit spacing, 429 backoff, usage accounting.

Cache key = sha256(model + system prompt + input + config), so any prompt change is a cache miss.
Call `set_cache(False)` (wired to each CLI's --no-cache flag) for fresh timing runs.

Uses `client.models.generate_content` rather than the newer Interactions API: when the free-tier quota
is exhausted, Interactions requests hung indefinitely, while generate_content returns 429 immediately.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

CACHE_DIR = Path(".cache")
RETRYABLE = {429, 500, 503}  # rate limit, transient server errors
MAX_ATTEMPTS = 6
REQUEST_TIMEOUT_S = 120
MAX_SERVER_RETRY_DELAY_S = 300  # server says "retry in 17h" => daily quota is gone; fail fast instead of waiting

_client: genai.Client | None = None
_cache_enabled = True
_last_call = 0.0


class QuotaExhausted(RuntimeError):
    """The model's (daily) quota is used up; retrying now is pointless."""


def set_cache(enabled: bool) -> None:
    global _cache_enabled
    _cache_enabled = enabled


def chat_model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


def embed_model() -> str:
    return os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    thought_tokens: int
    latency_s: float  # latency of the original API call, even when served from cache
    cached: bool


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000))  # ms
    return _client


def _key(kind: str, payload: Any) -> str:
    return hashlib.sha256(f"{kind}:{json.dumps(payload, sort_keys=True)}".encode()).hexdigest()


def _cache_read(kind: str, key: str) -> dict | None:
    path = CACHE_DIR / kind / f"{key}.json"
    if _cache_enabled and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _cache_write(kind: str, key: str, value: dict) -> None:
    path = CACHE_DIR / kind / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _server_retry_delay(exc: Exception) -> float | None:
    """Parse the server's suggested wait from a 429 body, e.g. retryDelay: '64651s'."""
    m = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", str(exc))
    return float(m.group(1)) if m else None


def _call_with_backoff(fn, *args, **kwargs) -> tuple[Any, float]:
    """Space calls to stay under GEMINI_RPM, and retry 429/5xx/timeouts with exponential backoff.

    Returns (response, seconds taken by the successful attempt), so our own waiting isn't counted as latency.
    """
    global _last_call
    min_gap = 60.0 / float(os.environ.get("GEMINI_RPM", "12"))  # flash-lite free tier: 15 RPM; keep headroom
    for attempt in range(MAX_ATTEMPTS):
        wait = _last_call + min_gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            start = time.perf_counter()
            response = fn(*args, **kwargs)
            return response, time.perf_counter() - start
        except Exception as exc:  # SDK errors expose the HTTP status as .code or .status_code
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            timed_out = "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()
            if status == 429 and (server_delay := _server_retry_delay(exc) or 0) > MAX_SERVER_RETRY_DELAY_S:
                raise QuotaExhausted(f"quota exhausted for {kwargs.get('model')}; server says retry in "
                                     f"{server_delay / 3600:.1f}h. Wait, or set GEMINI_MODEL in .env.") from exc
            if (status not in RETRYABLE and not timed_out) or attempt == MAX_ATTEMPTS - 1:
                raise
            delay = max(min(2 ** attempt * 5, 60), _server_retry_delay(exc) or 0)
            reason = "timeout" if timed_out else f"HTTP {status}"
            print(f"  [llm] {reason}, retrying in {delay:.0f}s (attempt {attempt + 1}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(delay)


def pdf_part(pdf_bytes: bytes) -> dict:
    """An input part carrying a PDF document."""
    return {"type": "document", "mime_type": "application/pdf",
            "data": base64.b64encode(pdf_bytes).decode("ascii")}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def _to_sdk_part(p: dict) -> types.Part:
    if p["type"] == "text":
        return types.Part(text=p["text"])
    return types.Part.from_bytes(data=base64.b64decode(p["data"]), mime_type=p["mime_type"])


def generate(parts: list[dict], *, system: str | None = None, schema: dict | None = None,
             thinking_level: str | None = None, temperature: float | None = None,
             model: str | None = None) -> LLMResult:
    """One Gemini call. `schema` (a JSON schema) switches on native structured JSON output."""
    model = model or chat_model()
    config: dict[str, Any] = {}
    if thinking_level:
        config["thinking_level"] = thinking_level
    if temperature is not None:
        config["temperature"] = temperature

    # Hash PDFs by content digest so cache keys stay small.
    key_parts = [{**p, "data": hashlib.sha256(p["data"].encode()).hexdigest()} if "data" in p else p for p in parts]
    key = _key("chat", {"model": model, "system": system, "parts": key_parts, "schema": schema, "config": config})
    if (hit := _cache_read("chat", key)) is not None:
        return LLMResult(**{**hit, "cached": True})

    sdk_config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None,
        response_mime_type="application/json" if schema else None,
        response_json_schema=schema,
    )
    response, latency = _call_with_backoff(
        _get_client().models.generate_content, model=model,
        contents=[types.Content(role="user", parts=[_to_sdk_part(p) for p in parts])], config=sdk_config)
    usage = response.usage_metadata
    result = LLMResult(
        text=response.text or "",
        model=model,
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
        thought_tokens=usage.thoughts_token_count or 0,
        latency_s=round(latency, 3),
        cached=False,
    )
    _cache_write("chat", key, {k: v for k, v in asdict(result).items() if k != "cached"})
    return result


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed texts (one cached call per text)."""
    model = model or embed_model()
    vectors = []
    for text in texts:
        key = _key("embed", {"model": model, "text": text})
        if (hit := _cache_read("embed", key)) is None:
            result, _ = _call_with_backoff(_get_client().models.embed_content, model=model, contents=text)
            hit = {"vector": list(result.embeddings[0].values)}
            _cache_write("embed", key, hit)
        vectors.append(hit["vector"])
    return vectors
