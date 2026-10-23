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

DATA_FILES = [
    "case_studies_model.pkl", "rheumatic_model.pkl",
    "rheumatic.json", "Case_studies_combined.json",
    "rheumatic_conditional_probabilities.csv",
    "Case_studies_combined_conditional_probabilities.csv",
]

# Third-party packages whose licence texts ship with the app (THIRD-PARTY-NOTICES.txt).
NOTICE_PACKAGES = ["PySide6", "PySide6-Essentials", "PySide6-Addons", "shiboken6", "numpy", "pandas", "reportlab",
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
    ]
    for f in DATA_FILES:
        cmd += ["--add-data", f"{REPO / f}{SEP}resources"]
    for mod in ("tkinter", "matplotlib", "torch", "torchvision", "sentence_transformers", "transformers",
                "IPython", "pytest", "scipy", "sklearn", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
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
