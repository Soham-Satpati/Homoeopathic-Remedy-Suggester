"""Sentence embeddings for symptom matching — loaded only from a local folder.

The demo called ``SentenceTransformer("all-MiniLM-L6-v2")``, which downloads the model
from Hugging Face on first run. Here the model must already be in
``<resources>/models/all-MiniLM-L6-v2`` (put there by ``scripts/fetch_models.py`` when the
installer is built), and Hugging Face's offline switches are forced on so a missing file
fails fast instead of attempting a download.

If the model is missing or fails to load, ``load_embedder`` returns ``None`` and matching
falls back to the demo's lexical (word-stem) similarity — exactly as the demo did when
its model was unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 vectors."""

    model_id: str

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_dir: Path):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer   # heavy import, only when used

        self._model = SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)
        self.model_id = f"st:{Path(model_dir).name}"

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        emb = self._model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def load_embedder(model_dir: Path, prefer: str = "auto") -> Embedder | None:
    """ONNX Runtime first (what the installed app ships), then sentence-transformers."""
    if not (Path(model_dir) / "config.json").exists():
        log.warning("Embedding model not found at %s — using lexical matching", model_dir)
        return None
    if prefer in ("auto", "onnx"):
        e = load_onnx_embedder(model_dir)
        if e is not None or prefer == "onnx":
            return e
    try:
        return SentenceTransformerEmbedder(model_dir)
    except Exception:
        log.exception("Embedding model failed to load — using lexical matching")
        return None


class DatasetEmbeddingCache:
    """Embeddings of every dataset symptom, computed once and kept on disk.

    The demo re-encoded the whole dataset on every app start. Keyed by (model, dataset
    keys), so a dataset or model change produces a new file.
    """

    def __init__(self, embedder: Embedder, cache_dir: Path | None):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self._mem: dict[str, np.ndarray] = {}

    def get(self, keys: tuple) -> np.ndarray | None:
        digest = hashlib.md5("\n".join(keys).encode("utf-8")).hexdigest()
        tag = hashlib.md5(f"{self.embedder.model_id}|{digest}".encode()).hexdigest()[:16]
        if tag in self._mem:
            return self._mem[tag]
        path = self.cache_dir / f"dataset_emb_{tag}.npy" if self.cache_dir else None
        emb = None
        if path is not None and path.exists():
            try:
                emb = np.load(path)
                if emb.shape[0] != len(keys):
                    emb = None
            except Exception:
                emb = None
        if emb is None:
            try:
                emb = self.embedder.encode(list(keys), batch_size=128)
            except Exception:
                log.exception("Could not embed dataset — using lexical matching")
                return None
            if path is not None:
                try:
                    tmp = path.with_suffix(".tmp.npy")
                    np.save(tmp, emb)
                    os.replace(tmp, path)
                except OSError:
                    log.warning("Could not write embedding cache %s", path)
        self._mem[tag] = emb
        return emb


class OnnxEmbedder:
    """all-MiniLM-L6-v2 through ONNX Runtime: the same network and pooling as
    sentence-transformers (mean of token vectors, then L2-normalised) without PyTorch.

    Needs ``onnx/model.onnx`` and ``tokenizer.json`` in the model folder.
    """

    def __init__(self, model_dir: Path, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.model_id = f"onnx:{model_dir.name}"
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tok.enable_truncation(max_length=_max_seq_len(model_dir))
        tok.no_padding()
        self._tok = tok
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._sess = ort.InferenceSession(str(model_dir / "onnx" / "model.onnx"), opts,
                                          providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self._tok.encode_batch(list(texts[i:i + batch_size]))
            width = max(len(e.ids) for e in enc)
            ids = np.zeros((len(enc), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for j, e in enumerate(enc):
                ids[j, :len(e.ids)] = e.ids
                mask[j, :len(e.ids)] = 1
                types[j, :len(e.type_ids)] = e.type_ids
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feed["token_type_ids"] = types
            tokens = self._sess.run(None, feed)[0]                       # (batch, seq, 384)
            m = mask[..., None].astype(np.float32)
            pooled = (tokens * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            out.append(pooled.astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


def _max_seq_len(model_dir: Path) -> int:
    import json
    try:
        return int(json.loads((model_dir / "sentence_bert_config.json").read_text())["max_seq_length"])
    except (OSError, ValueError, KeyError):
        return 256


def load_onnx_embedder(model_dir: Path) -> Embedder | None:
    if not (Path(model_dir) / "onnx" / "model.onnx").exists():
        return None
    try:
        return OnnxEmbedder(model_dir)
    except Exception:
        log.exception("ONNX embedder failed to load")
        return None
