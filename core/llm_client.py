"""Unified LLM wrapper — Anthropic, OpenAI, or Gemini via LLM_PROVIDER env var."""

from __future__ import annotations

import functools
import hashlib
import os
from typing import Optional


class LLMError(Exception):
    """Raised when the LLM provider is misconfigured or the API call fails."""


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower().strip()


def _require_key(provider: str) -> str:
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    if provider not in env_map:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{provider}'. Use anthropic, openai, or gemini."
        )
    key = os.getenv(env_map[provider], "").strip()
    if not key:
        raise LLMError(
            f"{env_map[provider]} is not set. Add it to your .env file."
        )
    return key


def _llm_temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0"))
    except ValueError:
        return 0.0


def _cache_key(provider: str, system: str, prompt: str, json_mode: bool) -> str:
    temp = _llm_temperature()
    blob = f"{provider}|{json_mode}|{temp}|{system}|{prompt}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=256)
def _cached_complete(
    _key: str, provider: str, system: str, prompt: str, json_mode: bool
) -> str:
    if provider == "anthropic":
        return _complete_anthropic(system, prompt)
    if provider == "openai":
        return _complete_openai(system, prompt, json_mode)
    if provider == "gemini":
        return _complete_gemini(system, prompt, json_mode)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}'.")


def _complete_anthropic(system: str, prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_require_key("anthropic"))
    response = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        temperature=_llm_temperature(),
        system=system or "You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _complete_openai(system: str, prompt: str, json_mode: bool) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_require_key("openai"))
    kwargs: dict = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": _llm_temperature(),
        "messages": [
            {"role": "system", "content": system or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise LLMError("OpenAI returned an empty response.")
    return content


def _complete_gemini(system: str, prompt: str, json_mode: bool) -> str:
    import google.generativeai as genai

    genai.configure(api_key=_require_key("gemini"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(
        model_name,
        system_instruction=system or None,
    )
    generation_config = genai.GenerationConfig(temperature=_llm_temperature())
    if json_mode:
        generation_config = genai.GenerationConfig(
            temperature=_llm_temperature(),
            response_mime_type="application/json",
        )
    response = model.generate_content(prompt, generation_config=generation_config)
    text = getattr(response, "text", None) or ""
    if not text.strip():
        raise LLMError("Gemini returned an empty response.")
    return text


class LLMClient:
    """Swappable LLM client for scoring, chat, and interview generation."""

    def __init__(self, provider: Optional[str] = None) -> None:
        self.provider = (provider or _provider()).lower().strip()

    def complete(self, prompt: str, system: str = "", json_mode: bool = False) -> str:
        if json_mode and self.provider == "anthropic":
            prompt = (
                f"{prompt}\n\nReturn ONLY valid JSON. No preamble, no markdown fences."
            )
        try:
            key = _cache_key(self.provider, system, prompt, json_mode)
            return _cached_complete(key, self.provider, system, prompt, json_mode)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                "AI service is temporarily unavailable. Check your API key and try again."
            ) from exc

    @staticmethod
    def clear_cache() -> None:
        _cached_complete.cache_clear()


def get_client() -> LLMClient:
    return LLMClient()
