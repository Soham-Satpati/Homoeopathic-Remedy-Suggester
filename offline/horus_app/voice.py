"""Microphone capture (Qt Multimedia) → WAV bytes for the offline speech model."""

from __future__ import annotations

import io
import logging
import wave

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices
from PySide6.QtWidgets import QPushButton

from .context import run_job

log = logging.getLogger(__name__)
MAX_SECONDS = 180


class Recorder(QObject):
    level = Signal(float)           # 0..1 input level, for a meter
    stopped = Signal(bytes)         # WAV bytes
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source = None
        self._io = None
        self._buf = bytearray()
        self._fmt = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.stop)

    @staticmethod
    def available() -> bool:
        return not QMediaDevices.defaultAudioInput().isNull()

    @property
    def recording(self) -> bool:
        return self._source is not None

    def start(self) -> None:
        dev = QMediaDevices.defaultAudioInput()
        if dev.isNull():
            self.failed.emit("No microphone was found.")
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)
        if not dev.isFormatSupported(fmt):
            fmt = dev.preferredFormat()
        self._fmt = fmt
        self._buf = bytearray()
        self._source = QAudioSource(dev, fmt, self)
        self._io = self._source.start()
        if self._io is None:
            self._source = None
            self.failed.emit("The microphone could not be opened. Check Windows privacy settings → Microphone.")
            return
        self._io.readyRead.connect(self._read)
        self._timer.start(MAX_SECONDS * 1000)

    def _read(self) -> None:
        if self._io is None:
            return
        data = bytes(self._io.readAll())
        self._buf += data
        if data and self._fmt.sampleFormat() == QAudioFormat.Int16:
            x = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2")
            if len(x):
                self.level.emit(min(1.0, float(np.abs(x).max()) / 20000.0))

    def stop(self) -> None:
        if self._source is None:
            return
        self._timer.stop()
        self._read()
        self._source.stop()
        self._source = None
        self._io = None
        try:
            self.stopped.emit(self._to_wav())
        except Exception as e:
            log.exception("Could not encode recording")
            self.failed.emit(f"Recording could not be processed: {e}")

    def _to_wav(self) -> bytes:
        fmt = self._fmt
        ch, rate = fmt.channelCount(), fmt.sampleRate()
        sf = fmt.sampleFormat()
        raw = bytes(self._buf)
        if sf == QAudioFormat.Int16:
            pcm = np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2")
        elif sf == QAudioFormat.Float:
            f = np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<f4")
            pcm = (np.clip(f, -1, 1) * 32767).astype("<i2")
        elif sf == QAudioFormat.Int32:
            pcm = (np.frombuffer(raw[: len(raw) // 4 * 4], dtype="<i4") >> 16).astype("<i2")
        elif sf == QAudioFormat.UInt8:
            pcm = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).astype("<i2")
        else:
            raise ValueError("Unsupported microphone sample format")
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(pcm.tobytes())
        return out.getvalue()


class DictateButton(QPushButton):
    """Toggle button: record → transcribe offline → emit the text."""

    text_ready = Signal(str)
    busy = Signal(bool)

    def __init__(self, transcribe, idle_text: str = "🎙  Dictate", parent=None):
        super().__init__(idle_text, parent)
        self._idle = idle_text
        self._transcribe = transcribe
        self.rec = Recorder(self)
        self.rec.stopped.connect(self._got_audio)
        self.rec.failed.connect(self._failed)
        self.rec.level.connect(lambda v: self.setText(f"⏹  Stop  {'▮' * int(1 + v * 6)}"))
        self.clicked.connect(self._toggle)

    def _toggle(self):
        if self.rec.recording:
            self.rec.stop()
        else:
            self.setText("⏹  Stop")
            self.rec.start()

    def _got_audio(self, wav: bytes):
        self.setText("Transcribing…")
        self.setEnabled(False)
        self.busy.emit(True)

        def done(text):
            self._reset()
            if text:
                self.text_ready.emit(text)

        def fail(msg, _tb):
            self._reset()
            self._failed(f"Transcription failed: {msg}")

        run_job(lambda: self._transcribe(wav, "audio/wav"), done, fail)

    def _reset(self):
        self.setEnabled(True)
        self.setText(self._idle)
        self.busy.emit(False)

    def _failed(self, msg: str):
        from .widgets import error
        self._reset()
        error(self.window(), "Microphone", msg)
