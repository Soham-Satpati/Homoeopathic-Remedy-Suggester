"""Package the engine test console with PyInstaller.

    python offline/packaging/build_test_console.py

Produces ``dist/HoRUS-Offline-Test/`` (a folder with ``HoRUS-Offline-Test.exe``) containing
the engine, the remedy dataset and the embedding model, so it runs with no Python install
and no internet. Run on the target OS (Windows builds Windows). The build pipeline runs it,
then the .exe's own ``--selftest`` / ``--gui-smoke`` checks, before publishing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import PyInstaller.__main__

OFFLINE = Path(__file__).resolve().parents[1]
REPO = OFFLINE.parent
SEP = ";" if sys.platform == "win32" else ":"
NAME = "HoRUS-Offline-Test"

# Everything the engine reads (see horus/config.py). Only the first conditional-probability
# file found is used; both are shipped so the choice matches a source checkout.
DATA_FILES = [
    "case_studies_model.pkl", "rheumatic_model.pkl",
    "rheumatic.json", "Case_studies_combined.json",
    "rheumatic_conditional_probabilities.csv",
    "Case_studies_combined_conditional_probabilities.csv",
]
MODEL_DIR = REPO / "models" / "all-MiniLM-L6-v2"

# transformers checks installed package versions at import time, so their metadata must ship.
METADATA = ["transformers", "sentence-transformers", "torch", "tokenizers", "huggingface-hub",
            "safetensors", "tqdm", "regex", "numpy", "packaging", "filelock", "pyyaml", "requests",
            "scikit-learn", "scipy", "pillow"]


def main() -> None:
    missing = [f for f in DATA_FILES if not (REPO / f).exists()]
    if missing:
        sys.exit(f"Missing data files: {missing}")
    if not (MODEL_DIR / "config.json").exists():
        sys.exit(f"Embedding model missing at {MODEL_DIR} — run offline/scripts/fetch_models.py first")

    args = [
        str(OFFLINE / "testconsole" / "horus_test_console.py"),
        "--name", NAME,
        "--onedir", "--windowed", "--noconfirm", "--clean",
        "--paths", str(OFFLINE),
        "--distpath", str(REPO / "dist"),
        "--workpath", str(REPO / "build"),
        "--specpath", str(REPO / "build"),
        "--hidden-import", "horus.engine",
        "--add-data", f"{MODEL_DIR}{SEP}resources/models/all-MiniLM-L6-v2",
        # trim tooling the app never uses
        "--exclude-module", "matplotlib", "--exclude-module", "IPython",
        "--exclude-module", "pytest", "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio", "--exclude-module", "tensorboard",
    ]
    for f in DATA_FILES:
        args += ["--add-data", f"{REPO / f}{SEP}resources"]
    for pkg in METADATA:
        args += ["--copy-metadata", pkg]
    PyInstaller.__main__.run(args)


if __name__ == "__main__":
    main()
