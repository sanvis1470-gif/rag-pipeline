"""One small interface for the answer / judge LLM.

Ollama, OpenAI, Groq, Gemini, OpenRouter, vLLM ... all expose an OpenAI-compatible chat endpoint,
so a single client class covers them; only base_url, model and key differ."""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol

from .config import Config


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Return the model's text reply. json_mode asks for a JSON object where the API supports it."""


def parse_json(text: str) -> dict | None:
    """Lenient JSON object parser: small local models often wrap JSON in prose or code fences."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    for candidate in (t, (re.search(r"\{.*\}", t, flags=re.S) or [None])[0]):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


class OpenAICompatibleLLM:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 700, timeout: float = 180.0, hint: str = ""):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self.model, self.base_url, self.hint = model, base_url or "the OpenAI API", hint
        self.temperature, self.max_tokens = temperature, max_tokens

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        import openai
        kwargs = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        try:
            if json_mode:
                try:
                    resp = self.client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
                except openai.BadRequestError:          # endpoint without JSON mode: parse leniently instead
                    resp = self.client.chat.completions.create(**kwargs)
            else:
                resp = self.client.chat.completions.create(**kwargs)
        except openai.APIConnectionError as e:
            raise LLMError(f"Could not reach the LLM at {self.base_url}. {self.hint}") from e
        except openai.NotFoundError as e:
            raise LLMError(f"Model '{self.model}' was not found. {self.hint}") from e
        except openai.AuthenticationError as e:
            raise LLMError(f"The LLM rejected the API key. {self.hint}") from e
        return (resp.choices[0].message.content or "").strip()


class FakeLLM:
    """Scripted LLM for tests: handler(system, user, json_mode) -> reply. Records every call."""

    def __init__(self, handler: Callable[[str, str, bool], str]):
        self.handler = handler
        self.calls: list[tuple[str, str, bool]] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append((system, user, json_mode))
        return self.handler(system, user, json_mode)


def make_llm(cfg: Config) -> LLM:
    common = dict(model=cfg.llm_model, temperature=cfg.llm_temperature, max_tokens=cfg.llm_max_tokens)
    if cfg.llm_provider == "ollama":
        return OpenAICompatibleLLM(base_url=cfg.llm_base_url or "http://localhost:11434/v1", api_key="ollama",
                                   hint=f"Is Ollama running? Install it from ollama.com, then run: ollama pull {cfg.llm_model}",
                                   **common)
    if cfg.llm_provider == "openai":
        return OpenAICompatibleLLM(base_url=cfg.llm_base_url, api_key=None,       # key comes from OPENAI_API_KEY
                                   hint="Set the OPENAI_API_KEY environment variable.", **common)
    if not cfg.llm_base_url or not cfg.llm_model:
        raise LLMError("--llm-provider custom needs --llm-base-url and --llm-model "
                       "(and the key in the LLM_API_KEY environment variable).")
    return OpenAICompatibleLLM(base_url=cfg.llm_base_url, api_key=os.environ.get("LLM_API_KEY", "not-needed"),
                               hint="Check --llm-base-url, --llm-model and LLM_API_KEY.", **common)
