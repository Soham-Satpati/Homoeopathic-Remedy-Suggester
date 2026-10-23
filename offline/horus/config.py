"""Where the engine finds its files.

Two kinds of location:

* **resources** – read-only data shipped with the app: the remedy datasets, pickled
  models, cluster CSVs, conditional probabilities, the miasm rubric and the bundled
  embedding model. In a packaged build this is the installer's resource folder; in a
  source checkout it is the repository root (the same files the demo app uses).
* **user data** – the patient database, caches and logs, kept in the OS user-data folder.

Both can be overridden with ``HORUS_RESOURCE_DIR`` / ``HORUS_USER_DIR``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "HoRUS"

# File names exactly as produced by symptom_clustering_pro.py / used by the demo app.
MODEL_PICKLES = ("case_studies_model.pkl", "rheumatic_model.pkl")
DATASET_JSONS = ("rheumatic.json", "Case_studies_combined.json")
CONDITIONAL_PROB_CANDIDATES = (
    "symptom_conditional_probabilities.csv",
    "rheumatic_conditional_probabilities.csv",
    "Case_studies_combined_conditional_probabilities.csv",
)
CLUSTER_CSV_NAMES = ("remedy_modalities", "remedy_area_modalities", "remedy_area")
MIASM_RUBRIC_FILE = "miasm_rubric.json"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_DIRNAME = "all-MiniLM-L6-v2"


def _default_resource_dir() -> Path:
    env = os.environ.get("HORUS_RESOURCE_DIR")
    if env:
        return Path(env)
    bundle = getattr(sys, "_MEIPASS", None)          # PyInstaller one-folder/one-file builds
    if bundle:
        return Path(bundle) / "resources"
    # source checkout: <repo>/offline/horus/config.py -> <repo>
    return Path(__file__).resolve().parents[2]


def _default_user_dir() -> Path:
    env = os.environ.get("HORUS_USER_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_NAME


@dataclass
class Paths:
    resources: Path = field(default_factory=_default_resource_dir)
    user: Path = field(default_factory=_default_user_dir)

    def resource(self, name: str) -> Path:
        return self.resources / name

    @property
    def models_dir(self) -> Path:
        return self.resources / "models"

    @property
    def embedding_model_dir(self) -> Path:
        return self.models_dir / EMBEDDING_MODEL_DIRNAME

    @property
    def cache_dir(self) -> Path:
        d = self.user / "cache"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def miasm_rubric_path(self) -> Path | None:
        """A rubric in the user folder (doctor's edits) wins over the shipped one."""
        for p in (self.user / MIASM_RUBRIC_FILE, self.resources / MIASM_RUBRIC_FILE):
            if p.exists():
                return p
        return None
