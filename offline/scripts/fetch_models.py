"""Download everything the offline app bundles. Run once, with internet, at build time.

    python scripts/fetch_models.py --dest ../models [--llm qwen3.5-4b] [--no-llm] [--no-whisper]

Fetches into ``<dest>``:
  all-MiniLM-L6-v2/      symptom-matching embeddings (PyTorch + ONNX weights)
  llm/<file>.gguf        the local language model (choice settled by bench/benchmark_llm.py)
  llama-server/          llama.cpp's server binary for this platform
  whisper-small/         faster-whisper speech model (speech → English)
  manifest.json          what was fetched, from where, and SHA-256 of every file

then reloads the embedding model with networking disabled to prove the copy is complete.
The installer build runs this; end users never do.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from horus.config import EMBEDDING_MODEL_DIRNAME, EMBEDDING_MODEL_ID, Paths  # noqa: E402

EMBEDDING_REVISION = os.environ.get("HORUS_EMBEDDING_REVISION", "main")

# Candidate language models (all Apache-2.0). Files are picked by pattern from each repo so a
# renamed upload doesn't break the build; the chosen file is recorded in the manifest.
LLM_CANDIDATES = {
    "qwen3.5-4b":    ("unsloth/Qwen3.5-4B-GGUF", [r"^Qwen3\.5-4B-Q4_K_M\.gguf$"]),
    "qwen3.5-2b":    ("unsloth/Qwen3.5-2B-GGUF", [r"^Qwen3\.5-2B-Q4_K_M\.gguf$"]),
    "gemma4-e4b":    ("google/gemma-4-E4B-it-qat-q4_0-gguf", [r"(?i)^(?!mmproj|mtp).*q4_0.*\.gguf$",
                                                               r"(?i)^(?!mmproj|mtp).*\.gguf$"]),
    "gemma4-e2b":    ("google/gemma-4-E2B-it-qat-q4_0-gguf", [r"(?i)^(?!mmproj|mtp).*q4_0.*\.gguf$",
                                                               r"(?i)^(?!mmproj|mtp).*\.gguf$"]),
    "qwen3-4b-2507": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", [r"^Qwen3-4B-Instruct-2507-Q4_K_M\.gguf$"]),
}
DEFAULT_LLM = os.environ.get("HORUS_LLM", "qwen3.5-4b")
WHISPER_REPO = os.environ.get("HORUS_WHISPER_REPO", "Systran/faster-whisper-small")
LLAMA_REPO = "ggml-org/llama.cpp"
# Pinned so every installer ships the same server build. Note: llama.cpp's "latest" release can be
# a version marker without binaries; builds are the numbered bNNNN releases.
LLAMA_TAG = os.environ.get("HORUS_LLAMA_TAG", "b11669")

# llama.cpp release asset per platform: plain CPU build (no Vulkan/CUDA/SYCL), x64 or arm64.
LLAMA_ASSET = {
    "win":   r"bin-win-cpu-x64\.zip$",
    "linux": r"bin-ubuntu-x64\.(zip|tar\.gz)$",
    "mac":   r"bin-macos-arm64\.(zip|tar\.gz)$",
}


def _platform() -> str:
    return {"win32": "win", "darwin": "mac"}.get(sys.platform, "linux")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Hugging Face downloads ───────────────────────────────────────────────────
def fetch_embedding(dest: Path, fmt: str = "both") -> dict:
    """``fmt``: "onnx" (what the installed app uses, no PyTorch weights) or "both"."""
    from huggingface_hub import snapshot_download
    target = dest / EMBEDDING_MODEL_DIRNAME
    patterns = ["*.json", "*.txt", "1_Pooling/*", "onnx/model.onnx"]
    if fmt == "both":
        patterns.append("model.safetensors")
    elif (target / "model.safetensors").exists():
        (target / "model.safetensors").unlink()
    snapshot_download(repo_id=EMBEDDING_MODEL_ID, revision=EMBEDDING_REVISION, local_dir=str(target),
                      allow_patterns=patterns)
    return {"id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_REVISION, "dir": EMBEDDING_MODEL_DIRNAME,
            "format": fmt}


def fetch_llm(dest: Path, key: str) -> dict:
    from huggingface_hub import hf_hub_download, list_repo_files
    repo, patterns = LLM_CANDIDATES[key]
    files = list_repo_files(repo)
    chosen = None
    for pat in patterns:
        hits = sorted(f for f in files if "/" not in f and re.search(pat, f))
        if hits:
            chosen = hits[0]
            break
    if chosen is None:
        raise SystemExit(f"No GGUF matching {patterns} in {repo}: {files}")
    llm_dir = dest / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    for old in llm_dir.glob("*.gguf"):                      # one model per bundle
        if old.name != chosen:
            old.unlink()
    hf_hub_download(repo, chosen, local_dir=str(llm_dir))
    return {"key": key, "repo": repo, "file": f"llm/{chosen}", "server_dir": "llama-server",
            "context": 8192}


def fetch_whisper(dest: Path) -> dict:
    from huggingface_hub import snapshot_download
    target = dest / "whisper-small"
    snapshot_download(repo_id=WHISPER_REPO, local_dir=str(target),
                      allow_patterns=["model.bin", "config.json", "tokenizer.json", "vocabulary.*",
                                      "preprocessor_config.json"])
    return {"repo": WHISPER_REPO, "dir": "whisper-small"}


# ── llama.cpp server binary from GitHub releases ─────────────────────────────
def _gh(url: str) -> bytes:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "horus-build"}
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok and "api.github.com" in url:
        headers["Authorization"] = f"Bearer {tok}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=300) as r:
        return r.read()


def _llama_release(plat: str) -> tuple[dict, dict]:
    """(release, asset): the pinned tag if it has a build for this platform, else the newest that does."""
    base = f"https://api.github.com/repos/{LLAMA_REPO}/releases"
    candidates = []
    if LLAMA_TAG:
        try:
            candidates.append(json.loads(_gh(f"{base}/tags/{LLAMA_TAG}")))
        except Exception as e:
            print(f"Pinned llama.cpp {LLAMA_TAG} unavailable ({e}); searching recent releases")
    candidates += json.loads(_gh(f"{base}?per_page=40"))
    for rel in candidates:
        hits = [a for a in rel.get("assets", []) if re.search(LLAMA_ASSET[plat], a["name"])]
        if hits:
            return rel, hits[0]
    raise SystemExit(f"No llama.cpp release with a {plat} CPU build ({LLAMA_ASSET[plat]})")


def fetch_llama_server(dest: Path, plat: str) -> dict:
    rel, asset = _llama_release(plat)
    print(f"llama.cpp {rel['tag_name']}: {asset['name']}")
    blob = _gh(asset["browser_download_url"])
    target = dest / "llama-server"
    if target.exists():
        shutil.rmtree(target)
    staging = dest / "_llama_extract"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if asset["name"].endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(blob)).extractall(staging)
    else:
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            t.extractall(staging, filter="data")
    exe = "llama-server.exe" if plat == "win" else "llama-server"
    found = next(staging.rglob(exe), None)
    if found is None:
        raise SystemExit(f"{exe} not in {asset['name']}")
    shutil.copytree(found.parent, target)                   # the exe with its DLLs / .so / .dylib
    shutil.rmtree(staging)
    if plat != "win":
        (target / exe).chmod(0o755)
    # keep only what the server needs: drop the other llama.cpp tools
    for f in target.iterdir():
        if f.is_file() and f.name.startswith("llama-") and f.name not in (exe,) and \
                not f.suffix.lower() in (".dll", ".so", ".dylib"):
            f.unlink()
    return {"repo": LLAMA_REPO, "tag": rel["tag_name"], "asset": asset["name"], "dir": "llama-server"}


# ── manifest & verification ──────────────────────────────────────────────────
def write_manifest(dest: Path, parts: dict) -> dict:
    old = {}
    if (dest / "manifest.json").exists():
        old = json.loads((dest / "manifest.json").read_text())
    manifest = {**{k: v for k, v in old.items() if k != "files"}, **parts}
    for k in ("llm", "llama_server", "whisper"):
        if k in manifest and k not in parts and not (dest / manifest[k].get("dir", manifest[k].get("file", "-"))).exists():
            manifest.pop(k)
    manifest["files"] = {str(p.relative_to(dest)).replace(os.sep, "/"): sha256(p)
                         for p in sorted(dest.rglob("*"))
                         if p.is_file() and p.name not in (".gitattributes", "README.md", "manifest.json")
                         and ".cache" not in p.parts and not p.parts[len(dest.parts)].startswith("_")}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_offline(dest: Path) -> None:
    """Load the embedding model from disk in a fresh process with all network access disabled."""
    code = (
        "import sys, pathlib; sys.path.insert(0, %r)\n"
        "from horus.embeddings import load_embedder\n"
        "d = pathlib.Path(%r)\n"
        "for pref in ('onnx', 'st'):\n"
        "    e = load_embedder(d, prefer=pref)\n"
        "    if e is None: print(pref, 'not available'); continue\n"
        "    v = e.encode(['worse after midnight'])\n"
        "    assert v.shape == (1, 384), v.shape\n"
        "    print(pref, 'offline load OK', e.model_id)\n" % (str(HERE.parent), str(dest / EMBEDDING_MODEL_DIRNAME))
    )
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Paths().models_dir)
    ap.add_argument("--llm", default=DEFAULT_LLM, choices=sorted(LLM_CANDIDATES))
    ap.add_argument("--no-llm", action="store_true", help="skip the language model and llama-server")
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--platform", default=_platform(), choices=sorted(LLAMA_ASSET))
    ap.add_argument("--embedding", default="both", choices=["both", "onnx"])
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    parts = {"embedding": fetch_embedding(args.dest, args.embedding)}
    if not args.no_llm:
        parts["llm"] = fetch_llm(args.dest, args.llm)
        parts["llama_server"] = fetch_llama_server(args.dest, args.platform)
    if not args.no_whisper:
        parts["whisper"] = fetch_whisper(args.dest)
    manifest = write_manifest(args.dest, parts)
    total = sum(p.stat().st_size for p in args.dest.rglob("*") if p.is_file()) / 2**20
    print(f"Wrote {len(manifest['files'])} files ({total:.0f} MB) to {args.dest}  [{platform.platform()}]")
    verify_offline(args.dest)


if __name__ == "__main__":
    main()
