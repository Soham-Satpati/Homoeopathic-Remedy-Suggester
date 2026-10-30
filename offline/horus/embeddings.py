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


def store_tag(model_id: str) -> str:
    return hashlib.md5(model_id.encode()).hexdigest()[:12]


def write_seed(embedder: Embedder, keys: list[str], folder: Path) -> Path:
    """Build-time: embed ``keys`` and write the seed index the app loads on first start."""
    # half precision on disk: halves the installer's share; similarities change by < 0.001
    vecs = np.asarray(embedder.encode(list(keys), batch_size=128), dtype=np.float32).astype(np.float16)
    path = Path(folder) / f"emb_seed_{store_tag(embedder.model_id)}.npz"
    np.savez(path, keys=np.array(list(keys), dtype=object).astype(str), vectors=vecs)
    return path


def seed_matches(path: Path, keys: list[str]) -> bool:
    """True if the seed index at ``path`` holds exactly ``keys`` (so it can be reused as is)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return list(z["keys"]) == list(keys) and z["vectors"].shape[0] == len(keys)
    except (OSError, ValueError, KeyError):
        return False


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
    """Embeddings of dataset symptoms, kept on disk and grown incrementally.

    One store per embedding model: every text ever embedded keeps its vector, so switching books
    or saving a new case only embeds the texts not seen before (the demo re-encoded the whole
    dataset on every start).
    """

    CHUNK = 256

    def __init__(self, embedder: Embedder, cache_dir: Path | None, seed_dir: Path | None = None):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self._index: dict[str, int] = {}
        self._mat = np.zeros((0, 0), dtype=np.float32)
        self._recent: dict = {}         # digest → matrix (small key sets only; see subset())
        self._masks: dict = {}
        self._keys_list: list | None = None
        self._lock = __import__("threading").RLock()
        self.tag = store_tag(embedder.model_id)
        self._path = cache_dir / f"emb_store_{self.tag}.npz" if cache_dir else None
        # an index shipped with the app (built at installer time for the shipped books)
        if seed_dir is not None:
            self._merge_file(Path(seed_dir) / f"emb_seed_{self.tag}.npz")
        if self._path is not None:
            self._merge_file(self._path)

    def _merge_file(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with np.load(path, allow_pickle=False) as z:
                keys, mat = list(z["keys"]), z["vectors"].astype(np.float32)
        except Exception:
            log.warning("Embedding cache %s unreadable — ignoring it", path, exc_info=True)
            return
        if len(keys) != mat.shape[0] or not len(keys):
            return
        new = [i for i, k in enumerate(keys) if k not in self._index]
        if not new:
            return
        start = len(self._index)
        add = mat[new]
        self._mat = add if self._mat.size == 0 else np.vstack([self._mat, add])
        for j, i in enumerate(new):
            self._index[keys[i]] = start + j

    def missing(self, keys) -> int:
        with self._lock:
            return sum(1 for k in dict.fromkeys(keys) if k not in self._index)

    def _save(self) -> None:
        if self._path is None:
            return
        try:
            keys = np.array(sorted(self._index, key=self._index.get), dtype=object).astype(str)
            tmp = self._path.with_name(self._path.stem + ".tmp.npz")
            np.savez(tmp, keys=keys, vectors=self._mat)
            os.replace(tmp, self._path)
        except OSError:
            log.warning("Could not write embedding cache %s", self._path)

    def subset(self, keys: tuple, progress=None):
        """(matrix, mask) without copying: ``matrix`` is the whole store and ``mask`` marks the rows of
        ``keys``. Embeds any key not seen before. With ten books the store is ~130k × 384; copying
        it per lookup (as ``get`` does) cost hundreds of megabytes. Returns (None, None) on failure.
        Row ``i`` of the store is ``self.keys[i]``."""
        with self._lock:
            missing = [k for k in dict.fromkeys(keys) if k not in self._index]
        if missing and self._embed(missing, progress) is None:
            return None, None
        with self._lock:
            sig = (id(keys), len(keys), self._mat.shape[0])
            hit = self._masks.get(sig)
            if hit is None or hit[0] is not keys:
                mask = np.zeros(self._mat.shape[0], dtype=bool)
                mask[[self._index[k] for k in keys]] = True
                if len(self._masks) >= 4:
                    self._masks.pop(next(iter(self._masks)))
                self._masks[sig] = hit = (keys, mask)
            return self._mat, hit[1]

    @property
    def keys(self) -> list:
        with self._lock:
            if self._keys_list is None or len(self._keys_list) != len(self._index):
                self._keys_list = sorted(self._index, key=self._index.get)
            return self._keys_list

    def _embed(self, missing: list, progress=None):
        parts = []
        try:
            for i in range(0, len(missing), self.CHUNK):
                if progress:
                    progress(i / len(missing), f"Indexing symptoms for matching: {i:,} of {len(missing):,}")
                parts.append(np.asarray(self.embedder.encode(missing[i:i + self.CHUNK], batch_size=128),
                                        dtype=np.float32))
        except Exception:
            log.exception("Could not embed dataset — using lexical matching")
            return None
        new = np.vstack(parts)
        with self._lock:
            # another thread may have embedded some of these meanwhile: add only what is still new
            keep = [i for i, k in enumerate(missing) if k not in self._index]
            if keep:
                add = new[keep]
                start = self._mat.shape[0] if self._mat.size else 0
                self._mat = add if self._mat.size == 0 else np.vstack([self._mat, add])
                for j, i in enumerate(keep):
                    self._index[missing[i]] = start + j
                self._save()
        if progress:
            progress(1.0, f"Indexed {len(missing):,} symptoms")
        return new

    def get(self, keys: tuple, progress=None) -> np.ndarray | None:
        """Vectors for ``keys`` (in order), embedding any not seen before.
        ``progress(fraction, message)`` is called while embedding."""
        digest = hashlib.md5("\n".join(keys).encode("utf-8")).hexdigest()
        with self._lock:
            if digest in self._recent:
                return self._recent[digest]
            missing = [k for k in dict.fromkeys(keys) if k not in self._index]
            if missing and self._embed(missing, progress) is None:
                return None
            emb = self._mat[[self._index[k] for k in keys]]
            if len(self._recent) >= 2:
                self._recent.pop(next(iter(self._recent)))
            self._recent[digest] = emb
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
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        # Batch texts of similar length together: each batch is padded to its longest text, so
        # mixing a 3-word symptom with a 120-word one wastes most of the work. Order restored below.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vecs = self._encode_sorted([texts[i] for i in order], batch_size)
        out = np.empty_like(vecs)
        out[np.asarray(order)] = vecs
        return out

    def _encode_sorted(self, texts: list[str], batch_size: int) -> np.ndarray:
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
