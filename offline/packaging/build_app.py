"""Build the HoRUS macOS app bundle with PyInstaller.

    python offline/packaging/build_app.py [--models models] [--version 1.5.0] [--sign IDENTITY]

Produces ``dist/HoRUS.app``. Inside it, ``Contents/Resources/resources/`` holds the remedy dataset
and every bundled model (exactly the files listed in ``models/manifest.json``, so download caches
never ship). Run on a Mac of the target architecture (Apple silicon builds Apple silicon).

The bundle is signed afterwards (``sign_app.sh``): ad hoc by default, or with a Developer ID
identity for distribution. ``build_dmg.py`` then packs it into a disk image.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import PyInstaller.__main__

OFFLINE = Path(__file__).resolve().parents[1]
REPO = OFFLINE.parent
SEP = ":"
NAME = "HoRUS"
BUNDLE_ID = "com.sohamsatpati.horus"
MIN_MACOS = "13.0"

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


def build_seed_index(staging: Path, models: Path | None = None) -> None:
    """Embed every shipped book's symptoms now, so the installed app doesn't spend minutes doing it
    on first start. The app merges this with its own cache (same model, same keys). A seed already
    built for the same books (``scripts/build_seed.py`` in CI) is copied instead of re-embedded."""
    import time
    sys.path.insert(0, str(OFFLINE))
    from horus.embeddings import OnnxEmbedder, seed_matches, store_tag, write_seed
    keys = shipped_symptom_keys()
    embedder = OnnxEmbedder(staging / "all-MiniLM-L6-v2")
    name = f"emb_seed_{store_tag(embedder.model_id)}.npz"
    if models is not None and seed_matches(models / name, keys):
        shutil.copy2(models / name, staging / name)
        print(f"Seed index: {len(keys):,} symptoms, reused {models / name}")
        return
    t = time.time()
    out = write_seed(embedder, keys, staging)
    print(f"Seed index: {len(keys):,} symptoms in {time.time() - t:.0f}s → {out.name} "
          f"({out.stat().st_size / 2**20:.0f} MB)")


def shipped_symptom_keys() -> list[str]:
    sys.path.insert(0, str(OFFLINE))
    from horus.books import discover_books
    from horus.config import Paths
    from horus.dataset import load_base_dataset
    paths = Paths(resources=REPO, user=REPO / "build" / "seed-user")
    return sorted(load_base_dataset(paths, discover_books(paths)).s2r)


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
              "Qt libraries are dynamically linked and can be replaced in HoRUS.app/Contents/Frameworks."]
    dest.write_text("\n".join(lines), encoding="utf-8")


def app_version() -> str:
    m = re.search(r'^APP_VERSION = "(.*)"', (OFFLINE / "horus_app" / "app.py").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "0.0.0"


def write_info_plist(app: Path, version: str) -> None:
    """Keys PyInstaller does not set from the command line. Without NSMicrophoneUsageDescription
    macOS ends the app the moment dictation opens the microphone."""
    plist = app / "Contents" / "Info.plist"
    info = plistlib.loads(plist.read_bytes())
    info.update({
        "CFBundleName": NAME,
        "CFBundleDisplayName": NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "LSMinimumSystemVersion": MIN_MACOS,
        "LSApplicationCategoryType": "public.app-category.medical",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": True,         # the clinic theme is light-only
        "NSHumanReadableCopyright": "© Soham Satpati. For clinical reference only.",
        "NSMicrophoneUsageDescription":
            "HoRUS transcribes dictated case notes and patients' spoken answers on this Mac. "
            "Audio never leaves the computer.",
        "NSLocalNetworkUsageDescription":
            "In the Clinic LAN edition, tablets on the clinic Wi-Fi send intake questionnaires to HoRUS.",
    })
    plist.write_bytes(plistlib.dumps(info))


def ensure_executable(app: Path) -> None:
    """The llama.cpp server is shipped as data; make sure it kept its executable bit."""
    for srv in app.rglob("llama-server"):
        if srv.is_file() and not srv.is_symlink():
            srv.chmod(0o755)


def sign(app: Path, identity: str) -> None:
    subprocess.run(["bash", str(OFFLINE / "packaging" / "sign_app.sh"), str(app), identity], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--version", default=app_version())
    ap.add_argument("--sign", default=os.environ.get("HORUS_SIGN_IDENTITY", "-"),
                    help='codesign identity; "-" (default) signs ad hoc')
    args = ap.parse_args()
    if sys.platform != "darwin":
        sys.exit("Build the macOS app on a Mac.")

    missing = [f for f in DATA_FILES if not (REPO / f).exists()]
    if missing:
        sys.exit(f"Missing data files: {missing}")
    staging = REPO / "build" / "models-staging"
    manifest = stage_models(args.models, staging)
    build_seed_index(staging, args.models)
    icns = OFFLINE / "packaging" / "horus.icns"
    png = OFFLINE / "packaging" / "horus.png"

    cmd = [
        str(OFFLINE / "HoRUS.py"),
        "--name", NAME, "--onedir", "--windowed", "--noconfirm", "--clean",
        "--icon", str(icns),
        "--osx-bundle-identifier", BUNDLE_ID,
        "--target-arch", "arm64" if os.uname().machine == "arm64" else "x86_64",
        "--paths", str(OFFLINE),
        "--distpath", str(REPO / "dist"), "--workpath", str(REPO / "build" / "pyi"),
        "--specpath", str(REPO / "build"),
        "--add-data", f"{staging}{SEP}resources/models",
        "--add-data", f"{png}{SEP}resources",
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
    for mod in ("tkinter", "matplotlib", "torch", "torchvision", "sentence_transformers", "transformers",
                "IPython", "pytest", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
                "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtCharts",
                "PySide6.QtDataVisualization", "PySide6.QtPdf"):
        cmd += ["--exclude-module", mod]
    PyInstaller.__main__.run(cmd)

    app = REPO / "dist" / f"{NAME}.app"
    shutil.rmtree(REPO / "dist" / NAME, ignore_errors=True)      # the plain folder; the .app is the product
    write_info_plist(app, args.version)
    ensure_executable(app)
    res = app / "Contents" / "Resources"
    third_party_notices(res / "THIRD-PARTY-NOTICES.txt", manifest)
    sign(app, args.sign)
    size = sum(p.stat().st_size for p in app.rglob("*") if p.is_file() and not p.is_symlink()) / 2**30
    print(f"Built {app} ({size:.2f} GB, version {args.version}, signed {'ad hoc' if args.sign == '-' else 'Developer ID'})")


if __name__ == "__main__":
    main()
