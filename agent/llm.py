"""
Minimal OpenAI-compatible chat client (Groq by default, matching the existing
verify-document function) with a timeout and a circuit breaker. If the model is
slow, down, misconfigured or rate-limited, callers fall back to the template
brief; nothing in the incident pipeline depends on this succeeding.
"""
import json
import os
import re
import threading
import time

import httpx


PROVIDER_URLS = {"groq": "https://api.groq.com/openai/v1",
                 "gemini": "https://generativelanguage.googleapis.com/v1beta/openai"}


DEFAULT_MODELS = {"groq": "llama-3.3-70b-versatile", "gemini": "gemini-flash-latest"}


class LLMUnavailable(Exception):
    pass


class LLMClient:
    def __init__(self, api_key=None, base_url=None, model=None, timeout=20.0,
                 fail_threshold=3, cooldown_s=120):
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")
        # LLM_PROVIDER=groq|gemini picks the OpenAI-compatible endpoint (Gemini exposes one). LLM_BASE_URL overrides.
        provider = (os.environ.get("LLM_PROVIDER") or "groq").lower()
        if provider == "gemini" and not os.environ.get("LLM_API_KEY"):
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or self.api_key
        self.provider = provider
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or PROVIDER_URLS.get(provider, PROVIDER_URLS["groq"])).rstrip("/")
        # LLM_MODEL overrides. Unset now falls back to a sensible default per provider instead of silently turning the
        # AI summary OFF (which left SOS incidents with only the plain template text). Model slugs rotate: if the
        # provider retires one, set LLM_MODEL on Render rather than editing code.
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider)
        self.timeout = timeout
        self._fails = 0
        self._open_until = 0.0
        self._fail_threshold = fail_threshold
        self._cooldown = cooldown_s
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return bool(self.api_key and self.model)

    def generate_json(self, system, user):
        if not self.enabled:
            raise LLMUnavailable("not configured")
        with self._lock:
            if time.monotonic() < self._open_until:
                raise LLMUnavailable("circuit open")
        try:
            body = {"model": self.model, "temperature": 0.2, "max_tokens": 1200,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            url, hdr = f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.api_key}"}
            r = httpx.post(url, headers=hdr, json=body, timeout=self.timeout)
            if r.status_code == 400:                      # some models reject JSON mode: retry; the parser tolerates fences
                body.pop("response_format")
                r = httpx.post(url, headers=hdr, json=body, timeout=self.timeout)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            out = _extract_json(text)
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._fails += 1
                if self._fails >= self._fail_threshold:
                    self._open_until = time.monotonic() + self._cooldown
                    self._fails = 0
            raise LLMUnavailable(f"{type(e).__name__}") from e
        with self._lock:
            self._fails = 0
        return out


def _extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)
