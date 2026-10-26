"""Build the HoRUS desktop app folder with PyInstaller.

    python offline/packaging/build_app.py [--models models]

Produces ``dist/HoRUS/`` containing ``HoRUS.exe`` and ``_internal/resources/`` with the remedy
dataset and every bundled model (exactly the files listed in ``models/manifest.json``, so
download caches never ship). Run on the target OS. The installer (horus.iss) packs this folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import metadata
from pathlib import Path

import PyInstaller.__main__

OFFLINE = Path(__file__).resolve().parents[1]
REPO = OFFLINE.parent
SEP = ";" if sys.platform == "win32" else ":"
NAME = "HoRUS"

def data_files() -> list[str]:
    """books.json plus every file it names (book JSON, pickled index, co-occurrence table)."""
    books = json.loads((REPO / "books.json").read_text(encoding="utf-8"))["books"]
    files = ["books.json"]
    for b in books:
        files += [b[k] for k in ("json", "pkl", "cooccurrence") if b.get(k)]
    return files


DATA_FILES = data_files()

# Third-party packages whose licence texts ship with the app (THIRD-PARTY-NOTICES.txt).
NOTICE_PACKAGES = ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6", "numpy", "pandas", "reportlab",
                   "scikit-learn", "scipy", "joblib", "threadpoolctl",
                   "cryptography", "onnxruntime", "tokenizers", "faster-whisper", "ctranslate2", "av",
                   "huggingface-hub", "qrcode", "pyinstaller"]


def stage_models(models: Path, staging: Path) -> dict:
    manifest = json.loads((models / "manifest.json").read_text())
    if staging.exists():
        shutil.rmtree(staging)
    for rel in manifest["files"]:
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(models / rel, dst)
    shutil.copy2(models / "manifest.json", staging / "manifest.json")
    return manifest


# Visual C++ runtime, deployed app-locally (Microsoft-supported) so llama-server and the speech
# engine start on PCs without the VC++ redistributable — installing that would need admin rights.
VC_RUNTIME = ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
              "vcomp140.dll", "concrt140.dll"]


def vc_runtime_dlls() -> list[Path]:
    if sys.platform != "win32":
        return []
    import os
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    return [sys32 / n for n in VC_RUNTIME if (sys32 / n).exists()]


def build_seed_index(staging: Path) -> None:
    """Embed every shipped book's symptoms now, so the installed app doesn't spend minutes doing it
    on first start. The app merges this with its own cache (same model, same keys)."""
    import time
    sys.path.insert(0, str(OFFLINE))
    from horus.books import discover_books
    from horus.config import Paths
    from horus.dataset import load_base_dataset
    from horus.embeddings import OnnxEmbedder, write_seed
    paths = Paths(resources=REPO, user=REPO / "build" / "seed-user")
    keys = sorted(load_base_dataset(paths, discover_books(paths)).s2r)
    t = time.time()
    out = write_seed(OnnxEmbedder(staging / "all-MiniLM-L6-v2"), keys, staging)
    print(f"Seed index: {len(keys):,} symptoms in {time.time() - t:.0f}s → {out.name} "
          f"({out.stat().st_size / 2**20:.0f} MB)")


def third_party_notices(dest: Path, manifest: dict) -> None:
    lines = ["HoRUS bundles the following third-party software and models. Each remains under its own",
             "licence, reproduced or referenced below.", ""]
    for pkg in NOTICE_PACKAGES:
        try:
            md = metadata.metadata(pkg)
        except metadata.PackageNotFoundError:
            continue
        lic = md.get("License-Expression") or md.get("License") or ""
        if len(lic) > 200:
            lic = lic[:200] + "…"
        lines.append(f"- {pkg} {md.get('Version')} — {lic or 'see package'} — {md.get('Home-page') or ''}")
    lines += ["", "Models:"]
    for key in ("embedding", "llm", "whisper", "llama_server"):
        m = manifest.get(key)
        if m:
            lines.append(f"- {key}: {m.get('repo') or m.get('id')} {m.get('tag', '')} {m.get('file', '')}".rstrip())
    lines += ["", "Language model and speech model licences: Apache-2.0 / MIT, see the model pages on",
              "huggingface.co. llama.cpp: MIT (github.com/ggml-org/llama.cpp). Qt for Python: LGPL-3.0;",
              "Qt libraries are dynamically linked and can be replaced in the _internal folder."]
    dest.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=REPO / "models")
    args = ap.parse_args()

    missing = [f for f in DATA_FILES if not (REPO / f).exists()]
    if missing:
        sys.exit(f"Missing data files: {missing}")
    staging = REPO / "build" / "models-staging"
    manifest = stage_models(args.models, staging)
    build_seed_index(staging)
    vc = vc_runtime_dlls()
    if (staging / "llama-server").exists():
        for dll in vc:
            if not (staging / "llama-server" / dll.name).exists():
                shutil.copy2(dll, staging / "llama-server" / dll.name)
    print("VC++ runtime bundled:", [d.name for d in vc])
    icon = OFFLINE / "packaging" / "horus.ico"

    cmd = [
        str(OFFLINE / "HoRUS.py"),
        "--name", NAME, "--onedir", "--windowed", "--noconfirm", "--clean",
        "--icon", str(icon),
        "--paths", str(OFFLINE),
        "--distpath", str(REPO / "dist"), "--workpath", str(REPO / "build" / "pyi"),
        "--specpath", str(REPO / "build"),
        "--add-data", f"{staging}{SEP}resources/models",
        "--add-data", f"{icon}{SEP}resources",
        "--collect-data", "faster_whisper",            # Silero VAD model used to skip silence
        "--collect-binaries", "ctranslate2",
        "--collect-binaries", "onnxruntime",
        "--hidden-import", "PySide6.QtSvgWidgets",
        "--hidden-import", "PySide6.QtMultimedia",
        "--hidden-import", "horus_app.lan",
        "--add-data", f"{OFFLINE / 'horus_app' / 'assets'}{SEP}horus_app/assets",
        "--hidden-import", "sklearn.cluster._hdbscan.hdbscan",      # building clusters for added books
        "--collect-submodules", "sklearn.cluster",
    ]
    for f in DATA_FILES:
        cmd += ["--add-data", f"{REPO / f}{SEP}resources"]
    for dll in vc:
        cmd += ["--add-binary", f"{dll}{SEP}."]
    for mod in ("tkinter", "matplotlib", "torch", "torchvision", "sentence_transformers", "transformers",
                "IPython", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
                "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtCharts",
                "PySide6.QtDataVisualization", "PySide6.QtPdf"):
        cmd += ["--exclude-module", mod]
    PyInstaller.__main__.run(cmd)

    out = REPO / "dist" / NAME
    third_party_notices(out / "THIRD-PARTY-NOTICES.txt", manifest)
    shutil.copy2(OFFLINE / "packaging" / "README.txt", out / "README.txt")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**30
    print(f"Built {out} ({size:.2f} GB)")


if __name__ == "__main__":
    main()
