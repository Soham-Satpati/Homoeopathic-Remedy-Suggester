"""The engine's only contact with a language model.

The engine never imports a model library. It talks to an ``LLMBackend`` (text in, text
out) and an optional ``Transcriber`` (audio in, English text out). Step 3 adds local
backends (llama.cpp, Whisper); until then ``NullLLM`` stands in, and every AI job
degrades the same way the demo did when Gemini failed:

* categorise  → unavailable; the doctor uses structured entry (``engine.finalise``)
* narrate     → the report is built from dataset evidence alone
* comparative → empty summary
* followups   → no extra questions
* transcribe  → unavailable; the patient types instead
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable


class LLMUnavailable(RuntimeError):
    """No language model is configured, or it failed to load."""


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, json_output: bool = False, schema: dict | None = None) -> str:
        """Return the model's reply. ``json_output``/``schema`` ask the backend to constrain output
        (grammar-constrained decoding) when it can; callers still parse defensively."""
        ...


@runtime_checkable
class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        ...


class NullLLM:
    name = "none"

    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, json_output: bool = False, schema: dict | None = None) -> str:
        raise LLMUnavailable("No offline language model is installed.")


class NullTranscriber:
    name = "none"

    def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        raise LLMUnavailable("No offline speech model is installed.")


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_json_reply(text: str) -> Any:
    """Parse a model reply that should be JSON.

    The demo stripped fences with ``str.lstrip("```json")``, which strips a *character set*
    and could eat the start of a reply. Small local models also tend to add a sentence
    before or after the JSON. This strips fences properly, then falls back to the outermost
    ``{...}`` / ``[...]`` span. Raises ``ValueError`` if nothing parses.
    """
    raw = _FENCE.sub("", (text or "").strip()).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = raw.find(open_c), raw.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("Model reply was not valid JSON.")
