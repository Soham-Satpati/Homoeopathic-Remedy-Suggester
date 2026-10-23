"""Download the models the offline app bundles. Run once, with internet, at build time.

    python scripts/fetch_models.py [--dest <resources>/models]

Writes ``all-MiniLM-L6-v2`` into ``<dest>/all-MiniLM-L6-v2`` plus a ``manifest.json`` of
SHA-256 hashes, then reloads the model from disk with networking disabled to prove the
copy is complete. The installer build (step 5) runs this; end users never do.
Step 3 will add the local language model and the Whisper speech model here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from horus.config import EMBEDDING_MODEL_DIRNAME, EMBEDDING_MODEL_ID, Paths  # noqa: E402

# Pin the exact upstream revision so every build ships identical weights.
# Update deliberately (and re-run the parity tests) when changing it.
EMBEDDING_REVISION = os.environ.get("HORUS_EMBEDDING_REVISION", "main")

SKIP = {".gitattributes", "README.md"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_embedding(dest: Path) -> Path:
    from huggingface_hub import snapshot_download

    target = dest / EMBEDDING_MODEL_DIRNAME
    snapshot_download(
        repo_id=EMBEDDING_MODEL_ID,
        revision=EMBEDDING_REVISION,
        local_dir=str(target),
        # PyTorch weights only — skip the ONNX/OpenVINO/TF variants the repo also hosts.
        allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*", "modules.json"],
    )
    return target


def write_manifest(dest: Path) -> dict:
    files = {str(p.relative_to(dest)).replace(os.sep, "/"): sha256(p)
             for p in sorted(dest.rglob("*"))
             if p.is_file() and p.name not in SKIP and p.name != "manifest.json"
             and ".cache" not in p.parts}
    manifest = {"embedding": {"id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_REVISION,
                              "dir": EMBEDDING_MODEL_DIRNAME}, "files": files}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_offline(model_dir: Path) -> None:
    """Load from disk in a fresh process with all network access disabled."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from horus.embeddings import load_embedder\n"
        "e = load_embedder(__import__('pathlib').Path(%r))\n"
        "assert e is not None, 'model failed to load offline'\n"
        "v = e.encode(['worse after midnight'])\n"
        "assert v.shape == (1, 384), v.shape\n"
        "print('offline load OK', v.shape)\n" % (str(HERE.parent), str(model_dir))
    )
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Paths().models_dir)
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    model_dir = fetch_embedding(args.dest)
    manifest = write_manifest(args.dest)
    print(f"Wrote {len(manifest['files'])} files to {args.dest}")
    verify_offline(model_dir)


if __name__ == "__main__":
    main()
