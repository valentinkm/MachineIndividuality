# adapter.py
"""
Model adapter: prompt building, response cleaning, and rating parsing.

GenericAdapter is the single configurable adapter used by all models.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Optional, Tuple

from openai import APIStatusError, NotFoundError

from .prompt_templates import PROMPT_TEMPLATES

NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

RequestKwargsFn = Callable[[str], Dict[str, Any]]
QueryFn = Callable[["GenericAdapter", Any, str, str, str], Tuple[str, str]]
CleanTextFn = Callable[[str], str]


# ──────────────────────────────────────────────────────────────────────
# Base adapter (rarely used directly)
# ──────────────────────────────────────────────────────────────────────

class ModelAdapter:
    """Base class; override selectively in per-model adapters."""

    MODEL_ID = None

    def model_identifier(self, backend_name: str) -> str:
        if self.MODEL_ID is None:
            raise NotImplementedError("MODEL_ID must be set or model_identifier() overridden")
        return self.MODEL_ID

    def build_messages(self, norm: str, word: str):
        prompt = PROMPT_TEMPLATES[norm]["template"].format(word=word)
        return [{"role": "user", "content": prompt}]

    def request_kwargs(self, backend_name: str) -> Dict[str, Any]:
        return dict(temperature=0.0, max_tokens=16)

    def clean_text(self, raw: str) -> str:
        return (raw or "").strip()

    def parse_rating(self, cleaned: str) -> str:
        if not isinstance(cleaned, str):
            return "PARSE_ERROR"
        m = NUM_RE.search(cleaned)
        return m.group(0) if m else "NO_NUMBER_FOUND"


# ──────────────────────────────────────────────────────────────────────
# Generic adapter (used by all registry entries)
# ──────────────────────────────────────────────────────────────────────

class GenericAdapter(ModelAdapter):
    """
    Single configurable adapter that emulates all model-specific adapters.

    Parameters
    ----------
    model_id_hf : str
        Identifier for HuggingFace-hosted backends.
    model_id_vllm : str
        Identifier for vLLM / OpenAI-compatible backends.
    stop_newline : bool
        If True, default stop token is ``["\\n\\n"]``.
    request_kwargs : dict | callable
        Per-backend request kwargs.
    clean_text_fn : callable
        Pre-processing applied before base clean/strip.
    query_fn : callable
        Optional override for the query logic (e.g. retries for Gemma).
    cutoff_year : int
        If set, converts year-style AoA responses to ages.
    """

    def __init__(
        self,
        model_id_hf: str,
        model_id_vllm: Optional[str] = None,
        stop_newline: bool = False,
        *,
        request_kwargs: Optional[RequestKwargsFn | Dict[str, Any]] = None,
        clean_text_fn: Optional[CleanTextFn] = None,
        query_fn: Optional[QueryFn] = None,
        cutoff_year: Optional[int] = None,
    ):
        self._id_hf = model_id_hf
        self._id_vllm = model_id_vllm or model_id_hf
        self._stop_newline = stop_newline
        self._request_kwargs = request_kwargs
        self._clean_text_fn = clean_text_fn
        self._query_fn = query_fn
        self._cutoff_year = cutoff_year

    def model_identifier(self, backend_name: str) -> str:
        return self._id_hf if backend_name == "hf" else self._id_vllm

    def _default_request_kwargs(self) -> Dict[str, Any]:
        return dict(temperature=0.0, max_tokens=256, stop=["\n"])

    def request_kwargs(self, backend_name: str) -> Dict[str, Any]:
        if callable(self._request_kwargs):
            kw = dict(self._request_kwargs(backend_name))
        elif isinstance(self._request_kwargs, dict):
            kw = dict(self._request_kwargs)
        else:
            kw = self._default_request_kwargs()
        if self._stop_newline:
            kw.setdefault("stop", ["\n\n"])
        return kw

    def clean_text(self, raw: str) -> str:
        text = raw or ""
        if self._clean_text_fn:
            text = self._clean_text_fn(text)
        return super().clean_text(text)

    def parse_rating(self, cleaned: str) -> str:
        rating = super().parse_rating(cleaned)
        if self._cutoff_year and rating not in ("NO_NUMBER_FOUND", "PARSE_ERROR"):
            try:
                val = float(rating)
                if 1900 <= val <= 2025:
                    new_age = self._cutoff_year - val
                    if 0 <= new_age <= 100:
                        return str(int(new_age))
                if val > 99:
                    return "NO_NUMBER_FOUND"
            except ValueError:
                pass
        return rating

    def query(self, client, backend_name: str, norm: str, word: str, **kwargs):
        if self._query_fn:
            return self._query_fn(self, client, backend_name, norm, word, **kwargs)
        return self._default_query(client, backend_name, norm, word, **kwargs)

    def _default_query(self, client, backend_name: str, norm: str, word: str, **kwargs):
        messages = self.build_messages(norm, word)
        defaults = self.request_kwargs(backend_name)
        kw = {**defaults, **kwargs}

        extra_body = kw.get("extra_body", {}).copy()
        if "repetition_penalty" in kw:
            extra_body["repetition_penalty"] = kw.pop("repetition_penalty")
        if extra_body:
            kw["extra_body"] = extra_body

        try:
            resp = client.chat.completions.create(
                model=self.model_identifier(backend_name),
                messages=messages,
                **kw,
            )
            raw = resp.choices[0].message.content or ""
            cleaned = self.clean_text(raw)
            if cleaned:
                return raw, cleaned
        except (NotFoundError, APIStatusError) as e:
            if isinstance(e, APIStatusError) and e.status_code not in (404, 405):
                raise
        return "", ""


# ──────────────────────────────────────────────────────────────────────
# Helper behaviors
# ──────────────────────────────────────────────────────────────────────

def strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks (used by Qwen)."""
    return THINK_RE.sub("", text or "")


def qwen_request_kwargs(backend_name: str) -> Dict[str, Any]:
    kw: Dict[str, Any] = {"temperature": 0.0, "max_tokens": 256, "stop": ["\n"]}
    if backend_name in ("vllm", "offline_vllm"):
        kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        kw["chat_template_kwargs"] = {"enable_thinking": False}
    return kw


def mistral_request_kwargs(_: str) -> Dict[str, Any]:
    return {"temperature": 0.0, "max_tokens": 256, "stop": ["\n"]}


def make_retrying_chat_query(
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    retry_status_codes: tuple[int, ...] = (503, 429, 424),
) -> QueryFn:
    """Return a query_fn that retries on transient API errors."""

    def _query(adapter: GenericAdapter, client, backend_name: str, norm: str, word: str, **kwargs):
        messages = adapter.build_messages(norm, word)
        backoff = initial_backoff
        defaults = adapter.request_kwargs(backend_name)
        kw = {**defaults, **kwargs}

        extra_body = kw.get("extra_body", {}).copy()
        if "repetition_penalty" in kw:
            extra_body["repetition_penalty"] = kw.pop("repetition_penalty")
        if extra_body:
            kw["extra_body"] = extra_body

        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=adapter.model_identifier(backend_name),
                    messages=messages,
                    **kw,
                )
                raw = resp.choices[0].message.content or ""
                return raw, adapter.clean_text(raw)
            except APIStatusError as exc:
                if exc.status_code in retry_status_codes and attempt < max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    raise
        raise RuntimeError(f"Query failed after {max_retries} retries for norm={norm}, word={word}")

    return _query


__all__ = [
    "ModelAdapter",
    "GenericAdapter",
    "strip_think_tags",
    "qwen_request_kwargs",
    "mistral_request_kwargs",
    "make_retrying_chat_query",
]
