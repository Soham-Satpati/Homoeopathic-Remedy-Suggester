"""Local language model via llama.cpp's ``llama-server``, running on this computer only.

The server binary and a GGUF model ship with the app. ``LlamaServer`` starts it on demand:

* bound to 127.0.0.1 on a free port, with a random per-launch API key, so neither the
  network nor other programs on the Mac can use it;
* thinking/reasoning disabled (these jobs need short structured answers, not essays);
* replies constrained by the JSON schema each job passes (grammar-constrained decoding);
* on Apple silicon the model runs on the GPU (Metal), with a CPU fallback if Metal is unavailable;
* a watchdog process ends the server if HoRUS exits or crashes.

Talks to the server with ``urllib`` over the loopback interface — nothing leaves the Mac.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .llm import LLMUnavailable

log = logging.getLogger(__name__)

SERVER_EXE = "llama-server"
# Flags that differ between llama.cpp releases; dropped (once) if the server rejects them.
OPTIONAL_ARGS = ["--reasoning", "off", "--reasoning-budget", "0", "--no-webui"]


def physical_cores() -> int:
    n = os.cpu_count() or 4
    return max(2, min(8, n // 2 if n >= 8 else n))     # hyper-threads rarely help llama.cpp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _EarlyExit(Exception):
    pass


def gpu_layers() -> int:
    """Layers to offload to the GPU. Apple silicon runs the whole model on the GPU through Metal,
    which is several times faster than its CPU cores; Intel Macs stay on the CPU.
    ``HORUS_LLAMA_GPU_LAYERS`` overrides (0 = CPU only)."""
    env = os.environ.get("HORUS_LLAMA_GPU_LAYERS")
    if env is not None and env.strip().lstrip("-").isdigit():
        return max(0, int(env))
    return 99 if sys.platform == "darwin" and platform.machine() == "arm64" else 0


def _watchdog(child_pid: int) -> subprocess.Popen | None:
    """macOS has no kill-on-close Job Objects, so a tiny shell loop ends llama-server if HoRUS
    exits or crashes without stopping it. It also ends by itself when llama-server stops."""
    script = ('while kill -0 "$1" 2>/dev/null && kill -0 "$2" 2>/dev/null; do sleep 2; done; '
              'if kill -0 "$2" 2>/dev/null; then kill "$2" 2>/dev/null; sleep 5; kill -9 "$2" 2>/dev/null; fi')
    try:
        return subprocess.Popen(["/bin/sh", "-c", script, "horus-watchdog", str(os.getpid()), str(child_pid)],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
    except OSError:
        log.warning("Could not start the llama-server watchdog", exc_info=True)
        return None


class LlamaServer:
    """An ``LLMBackend`` backed by a local llama-server process."""

    def __init__(self, server_dir: Path, model_path: Path, *, context: int = 8192,
                 threads: int | None = None, startup_timeout: float = 180.0,
                 extra_args: list[str] | None = None):
        # absolute: the server runs with its own folder as working directory
        self.server_dir = Path(server_dir).resolve()
        self.model_path = Path(model_path).resolve()
        self.name = f"llama.cpp:{self.model_path.stem}"
        self.context = context
        self.threads = threads or physical_cores()
        self.startup_timeout = startup_timeout
        self.extra_args = extra_args or []
        self._proc: subprocess.Popen | None = None
        self._port = 0
        self._key = ""
        self._lock = threading.Lock()
        self._watchdog: subprocess.Popen | None = None
        self.gpu_layers = gpu_layers()
        self._atexit = False
        self.last_stats: dict = {}

    # ── lifecycle ────────────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return (self.server_dir / SERVER_EXE).exists() and self.model_path.exists()

    def start(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return
            if not self.available:
                raise LLMUnavailable("The offline AI model is not installed.")
            # newest flags + GPU first; then without the GPU, then without the optional flags
            attempts = [(OPTIONAL_ARGS, self.gpu_layers), (OPTIONAL_ARGS, 0),
                        ([], self.gpu_layers), ([], 0)]
            attempts = list(dict.fromkeys((tuple(o), g) for o, g in attempts))
            for i, (optional, ngl) in enumerate(attempts):
                try:
                    self._launch(list(optional), ngl, retry_possible=i + 1 < len(attempts))
                    self.gpu_layers = ngl
                    return
                except _EarlyExit:
                    log.warning("llama-server stopped at once (flags %s, GPU layers %d); retrying",
                                bool(optional), ngl)

    def _launch(self, optional: list[str], ngl: int, retry_possible: bool) -> None:
        self._port, self._key = _free_port(), secrets.token_urlsafe(24)
        cmd = [str(self.server_dir / SERVER_EXE), "-m", str(self.model_path),
               "--host", "127.0.0.1", "--port", str(self._port), "--api-key", self._key,
               "-c", str(self.context), "-t", str(self.threads), "-ngl", str(ngl),
               "--jinja", "-np", "1", *optional, *self.extra_args]
        log_path = Path(os.environ.get("HORUS_LLAMA_LOG", os.devnull))
        self._log = open(log_path, "ab") if log_path.name != os.devnull else subprocess.DEVNULL
        log.info("Starting %s", " ".join(cmd[:3]))
        env = dict(os.environ)
        for var in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):        # shared libs ship beside the binary
            env[var] = os.pathsep.join(p for p in (str(self.server_dir), env.get(var, "")) if p)
        self._proc = subprocess.Popen(cmd, cwd=str(self.server_dir), stdout=self._log,
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      env=env)
        self._watchdog = _watchdog(self._proc.pid)
        if not self._atexit:
            atexit.register(self.stop)
            self._atexit = True
        started = time.time()
        deadline = started + self.startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                if retry_possible and time.time() - started < 5:
                    raise _EarlyExit()
                raise LLMUnavailable(f"The AI engine stopped while starting (code {self._proc.returncode}).")
            try:
                if self._get("/health").get("status") == "ok":
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.5)
        self.stop()
        raise LLMUnavailable("The AI engine took too long to start.")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        dog, self._watchdog = self._watchdog, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if dog is not None and dog.poll() is None:
            dog.kill()

    # ── HTTP over loopback ───────────────────────────────────────────────
    def _req(self, method: str, path: str, body: dict | None = None, timeout: float = 5.0) -> dict:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self._port}{path}", method=method,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never via a proxy
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        return self._req("GET", path)

    # ── LLMBackend ───────────────────────────────────────────────────────
    def generate(self, prompt: str, *, system: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, json_output: bool = False, schema: dict | None = None) -> str:
        self.start()
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "seed": 7, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        elif json_output:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            out = self._req("POST", "/v1/chat/completions", body, timeout=900)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"AI engine error {e.code}: {e.read()[:300]!r}") from e
        except (urllib.error.URLError, OSError) as e:
            raise LLMUnavailable(f"AI engine not reachable: {e}") from e
        usage = out.get("usage") or {}
        self.last_stats = {"seconds": round(time.time() - t0, 2), "gpu_layers": self.gpu_layers,
                           "prompt_tokens": usage.get("prompt_tokens"),
                           "completion_tokens": usage.get("completion_tokens"),
                           "timings": out.get("timings")}
        msg = (out.get("choices") or [{}])[0].get("message") or {}
        return msg.get("content") or ""


def find_llama(models_dir: Path) -> LlamaServer | None:
    """The bundled server + model described by ``models/manifest.json`` (``llm`` entry)."""
    manifest = Path(models_dir) / "manifest.json"
    try:
        m = json.loads(manifest.read_text(encoding="utf-8")).get("llm")
    except (OSError, ValueError):
        return None
    if not m:
        return None
    srv = LlamaServer(Path(models_dir) / m["server_dir"], Path(models_dir) / m["file"],
                      context=int(m.get("context", 8192)))
    return srv if srv.available else None
