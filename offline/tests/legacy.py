"""Load the demo app's engine functions for side-by-side comparison.

``horus3_1_streamlit.py`` imports Streamlit, Gemini, GitHub and sentence-transformers at
the top and renders UI on import. This loader installs stand-ins for those four modules,
then executes the file only up to the "ROLE SWITCH" section (where the UI begins), so every
engine function is defined exactly as written in the demo — nothing is copied or edited.

Gemini is replaced by ``FakeGemini``: tests queue the reply text it should return, or let
it raise (as it does without an API key).
"""

from __future__ import annotations

import sys
import types
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEMO_FILE = REPO / "horus3_1_streamlit.py"
EMB_DIM = 384


# ── deterministic stand-in for all-MiniLM-L6-v2 ──────────────────────────────
_FAKE_CACHE: dict = {}      # text → vector; the shipped books have ~133k symptoms, re-encoded per engine


def fake_encode(texts) -> np.ndarray:
    """Hashed character-trigram vectors, L2-normalised. Related phrases score high,
    unrelated ones low, so the semantic matching path is genuinely exercised."""
    out = np.zeros((len(texts), EMB_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        hit = _FAKE_CACHE.get(t)
        if hit is not None:
            out[i] = hit
            continue
        for w in str(t).lower().split():
            w = f"#{w}#"
            for j in range(len(w) - 2):
                out[i, zlib.crc32(w[j:j + 3].encode()) % EMB_DIM] += 1.0
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
        _FAKE_CACHE[t] = out[i].copy()
    return out


class FakeSentenceTransformer:
    """Mimics the subset of SentenceTransformer the demo calls."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
        return fake_encode(list(texts))


class FakeEmbedder:
    """The same vectors through the new engine's ``Embedder`` interface."""

    model_id = "fake-trigram"

    def encode(self, texts, batch_size=32):
        return fake_encode(list(texts))


# ── Gemini stand-in ──────────────────────────────────────────────────────────
class FakeGemini:
    replies: list = []          # queued reply texts; empty → raise like a missing key
    calls: list = []            # (system_instruction, contents, generation_config)

    class GenerationConfig:
        def __init__(self, **kw):
            self.kw = kw

    class GenerativeModel:
        def __init__(self, model_name=None, system_instruction=None):
            self.system = system_instruction

        def generate_content(self, contents, generation_config=None):
            FakeGemini.calls.append((self.system, contents, generation_config))
            if not FakeGemini.replies:
                raise RuntimeError("Gemini unavailable (no API key)")
            return types.SimpleNamespace(text=FakeGemini.replies.pop(0))


# ── Streamlit stand-in ───────────────────────────────────────────────────────
class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


class _Secrets(dict):
    def __getitem__(self, k):
        raise KeyError(k)


def _cache(*args, **kwargs):
    """Pass-through for @st.cache_resource / @st.cache_data, with or without arguments."""
    if args and callable(args[0]) and not kwargs:
        return args[0]
    return lambda fn: fn


class _StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _State()
        self.secrets = _Secrets()
        self.query_params = {}
        self.cache_resource = _cache
        self.cache_data = _cache

    def __getattr__(self, name):        # any other st.* call is a no-op
        return lambda *a, **k: None


def _install_stubs() -> dict:
    saved = {k: sys.modules.get(k) for k in
             ("streamlit", "streamlit.components", "streamlit.components.v1", "google",
              "google.generativeai", "github", "sentence_transformers")}
    st = _StreamlitStub()
    comps = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda *a, **k: None
    comps.v1 = v1
    st.components = comps

    genai = types.ModuleType("google.generativeai")
    genai.configure = lambda **k: None
    genai.GenerativeModel = FakeGemini.GenerativeModel
    genai.GenerationConfig = FakeGemini.GenerationConfig
    google = sys.modules.get("google") or types.ModuleType("google")
    google.generativeai = genai

    gh = types.ModuleType("github")

    class GithubException(Exception):
        def __init__(self, status=0, data=None):
            self.status, self.data = status, data

    gh.Github = lambda *a, **k: None
    gh.GithubException = GithubException

    stm = types.ModuleType("sentence_transformers")
    stm.SentenceTransformer = FakeSentenceTransformer
    stm.util = types.SimpleNamespace()

    sys.modules.update({"streamlit": st, "streamlit.components": comps, "streamlit.components.v1": v1,
                        "google": google, "google.generativeai": genai, "github": gh,
                        "sentence_transformers": stm})
    return saved


def _restore(saved: dict) -> None:
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def load_legacy() -> types.ModuleType:
    """Execute the demo's engine section and return it as a module."""
    lines = DEMO_FILE.read_text(encoding="utf-8").splitlines()
    cut = next(i for i, l in enumerate(lines) if "ROLE SWITCH" in l)
    while cut > 0 and not lines[cut - 1].startswith("# ─"):
        cut -= 1
    source = "\n".join(lines[:cut - 1]) + "\n"
    assert "def build_pdf(" in source and "def rank_remedies_from_dataset(" in source

    saved = _install_stubs()
    try:
        mod = types.ModuleType("horus_demo_engine")
        mod.__file__ = str(DEMO_FILE)
        exec(compile(source, str(DEMO_FILE), "exec"), mod.__dict__)
    finally:
        _restore(saved)
    return mod


def reset_legacy_dataset(mod, patients: dict) -> None:
    """Point the demo at a fixed set of saved cases and clear its module-level dataset."""
    mod.load_patients = lambda: patients
    mod._s2r_unified = {}
    mod._known_symptoms = set()
    mod._dataset_source = "none"
    mod._live_case_count = 0
    mod._ensure_dataset()
