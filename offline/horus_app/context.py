"""Application state shared by every screen: paths, settings, store, session, engine, AI."""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from horus.config import Paths
from horus.store import Session, Store, default_db_path

log = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "auto_lock_minutes": 15,
    "ai_enabled": True,
    "ai_preload": True,
    "backup_dir": "",
    "backup_on_exit": True,
    "keep_backups": 14,
    "lan_port": 8765,
}


# ── background work ──────────────────────────────────────────────────────────
class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str, str)


class Job(QRunnable):
    """Run ``fn`` on the thread pool; ``on_done``/``on_fail`` are called back on the GUI thread."""

    def __init__(self, fn: Callable[[], Any], on_done: Callable[[Any], None] | None = None,
                 on_fail: Callable[[str, str], None] | None = None):
        super().__init__()
        self.fn = fn
        self.signals = _Signals()
        if on_done:
            self.signals.done.connect(on_done)
        if on_fail:
            self.signals.failed.connect(on_fail)
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            result = self.fn()
        except Exception as e:                              # reported to the GUI, logged in full
            log.exception("Background job failed")
            self.signals.failed.emit(f"{e}", traceback.format_exc())
            return
        self.signals.done.emit(result)


_ACTIVE: set = set()     # keep jobs alive until their signals have fired


def run_job(fn, on_done=None, on_fail=None) -> Job:
    job = Job(fn, on_done, on_fail)
    _ACTIVE.add(job)
    job.signals.done.connect(lambda *_: _ACTIVE.discard(job))
    job.signals.failed.connect(lambda *_: _ACTIVE.discard(job))
    QThreadPool.globalInstance().start(job)
    return job


# ── context ──────────────────────────────────────────────────────────────────
@dataclass
class AppContext:
    paths: Paths = field(default_factory=Paths)
    store: Store | None = None
    session: Session | None = None
    engine: Any = None                      # HorusEngine, loaded in the background
    llm: Any = None                         # LlamaServer or None
    transcriber: Any = None                 # WhisperTranscriber or None
    settings: dict = field(default_factory=lambda: dict(DEFAULT_SETTINGS))

    def __post_init__(self):
        self.paths.user.mkdir(parents=True, exist_ok=True)
        self._load_settings()
        self._discover_ai()

    # settings live next to the database (not secret: no patient data)
    @property
    def settings_path(self) -> Path:
        return self.paths.user / "settings.json"

    def _load_settings(self) -> None:
        try:
            self.settings.update(json.loads(self.settings_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    def save_settings(self) -> None:
        tmp = self.settings_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        os.replace(tmp, self.settings_path)

    @property
    def db_path(self) -> Path:
        return default_db_path(self.paths.user)

    @property
    def backup_dir(self) -> Path:
        d = self.settings.get("backup_dir") or str(self.paths.user / "backups")
        return Path(d)

    def open_store(self) -> Store | None:
        if self.store is None and self.db_path.exists():
            self.store = Store(self.db_path)
        return self.store

    # ── AI discovery ─────────────────────────────────────────────────────
    def _discover_ai(self) -> None:
        from horus.llama import find_llama
        from horus.whisper import find_whisper
        try:
            self.llm = find_llama(self.paths.models_dir)
        except Exception:
            log.exception("Language model discovery failed")
        try:
            self.transcriber = find_whisper(self.paths.models_dir)
        except Exception:
            log.exception("Speech model discovery failed")
        if self.llm:
            os.environ.setdefault("HORUS_LLAMA_LOG", str(self.paths.user / "logs" / "ai-engine.log"))

    @property
    def ai_ready(self) -> bool:
        return bool(self.llm) and bool(self.settings.get("ai_enabled", True))

    @property
    def voice_ready(self) -> bool:
        return bool(self.transcriber) and bool(self.settings.get("ai_enabled", True))

    def load_engine(self):
        """Build the engine (call from a background job)."""
        from horus.embeddings import load_embedder
        from horus.engine import HorusEngine
        embedder = load_embedder(self.paths.embedding_model_dir)
        engine = HorusEngine(self.paths, embedder=embedder,
                             llm=self.llm if self.ai_ready else None,
                             transcriber=self.transcriber if self.voice_ready else None)
        return engine

    def apply_ai_setting(self) -> None:
        """Switch AI on/off on an already-loaded engine."""
        from horus.llm import NullLLM, NullTranscriber
        if self.engine is None:
            return
        self.engine.llm = self.llm if self.ai_ready else NullLLM()
        self.engine.transcriber = self.transcriber if self.voice_ready else NullTranscriber()
        if not self.ai_ready and self.llm:
            self.llm.stop()

    def refresh_learning(self) -> None:
        """Feed every saved case into the engine (live learning, as in the demo)."""
        if self.engine is not None and self.session is not None and self.session.user.can("cases.read"):
            self.engine.learn_from(self.session.learning_records())

    def shutdown(self) -> None:
        if self.llm:
            self.llm.stop()
