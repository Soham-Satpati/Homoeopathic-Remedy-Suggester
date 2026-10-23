"""Offline speech-to-text with faster-whisper (CTranslate2), replacing Gemini transcription.

Like the demo's instruction to Gemini, speech in any language is returned as English:
Whisper's ``translate`` task does exactly that (English speech comes back as a transcript).
Voice-activity filtering drops silence so the model is not tempted to invent words.
"""

from __future__ import annotations

import io
import logging
import threading
import wave
from pathlib import Path

import numpy as np

from .llm import LLMUnavailable

log = logging.getLogger(__name__)
SAMPLE_RATE = 16000


def wav_to_float32(audio: bytes) -> np.ndarray:
    """Decode a PCM WAV (any rate, mono/stereo, 16-bit) into 16 kHz mono float32."""
    with wave.open(io.BytesIO(audio)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("Expected 16-bit PCM audio.")
    x = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE and len(x):
        n = int(round(len(x) * SAMPLE_RATE / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


class WhisperTranscriber:
    def __init__(self, model_dir: Path, threads: int = 4):
        self.model_dir = Path(model_dir)
        self.name = f"whisper:{self.model_dir.name}"
        self.threads = threads
        self._model = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.model_dir / "model.bin").exists()

    def _load(self):
        with self._lock:
            if self._model is None:
                if not self.available:
                    raise LLMUnavailable("The offline speech model is not installed.")
                from faster_whisper import WhisperModel
                self._model = WhisperModel(str(self.model_dir), device="cpu", compute_type="int8",
                                           cpu_threads=self.threads, local_files_only=True)
            return self._model

    def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        model = self._load()
        samples = wav_to_float32(audio)
        if len(samples) < SAMPLE_RATE * 0.3:
            return ""
        segments, info = model.transcribe(samples, task="translate", beam_size=5, vad_filter=True,
                                          condition_on_previous_text=False, temperature=0.0)
        text = " ".join(s.text.strip() for s in segments).strip()
        log.info("Transcribed %.1fs of %s audio", len(samples) / SAMPLE_RATE, info.language)
        return text


def find_whisper(models_dir: Path) -> WhisperTranscriber | None:
    import json
    try:
        m = json.loads((Path(models_dir) / "manifest.json").read_text(encoding="utf-8")).get("whisper")
    except (OSError, ValueError):
        return None
    if not m:
        return None
    t = WhisperTranscriber(Path(models_dir) / m["dir"])
    return t if t.available else None
