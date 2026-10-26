"""Books (source datasets) and the decision settings that weight them.

A *book* is a remedy dataset in the HoRUS JSON format::

    {remedy: {chapter: [{"symptom": "...", "labels": {"Area": .., "Modalities": .., ...}}, ...]}}

with an optional pickled index (``*_model.pkl`` from ``symptom_clustering_pro.py``) and an optional
co-occurrence table (``*_conditional_probabilities.csv[.gz]``) used for symptom suggestions.

Shipped books are listed in ``<resources>/books.json``. Books a clinic adds itself live in
``<user>/books/<id>/book.json`` (see ``import_book``).

``DecisionConfig`` says which books are used and how strongly each source and each part of the case
counts. ``DecisionConfig.classical(...)`` with the two original books reproduces the online demo
exactly; every weight at 1.0 means "no bias".
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

from .config import Paths

BOOKS_MANIFEST = "books.json"
USER_BOOKS_DIR = "books"

# The two books the online demo used — kept as the reference set for parity tests.
DEMO_BOOK_IDS = ("rheumatic", "case_studies")

_FALLBACK_BOOKS = [
    {"id": "rheumatic", "title": "Rheumatic remedies", "json": "rheumatic.json",
     "pkl": "rheumatic_model.pkl", "cooccurrence": "rheumatic_conditional_probabilities.csv"},
    {"id": "case_studies", "title": "Case studies (combined)", "json": "Case_studies_combined.json",
     "pkl": "case_studies_model.pkl", "cooccurrence": "Case_studies_combined_conditional_probabilities.csv"},
]

CLASSICAL_CATEGORY_WEIGHTS = {"psychological": 2.0, "general": 1.5, "physical": 1.0, "causation": 0.0}
CLASSICAL_MODALITY_WEIGHT = 0.6


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    json_path: Path
    pkl_path: Path | None = None
    cooccurrence_path: Path | None = None
    source: str = "shipped"               # "shipped" | "user"
    description: str = ""
    exclude_chapters: tuple = ()          # chapters that are not symptoms (e.g. remedy relationships)

    @property
    def available(self) -> bool:
        return self.json_path.exists()

    def to_public(self) -> dict:
        return {"id": self.id, "title": self.title, "source": self.source, "description": self.description,
                "has_cooccurrence": bool(self.cooccurrence_path and self.cooccurrence_path.exists())}


def _resolve(folder: Path, name: str | None) -> Path | None:
    return (folder / name) if name else None


def discover_books(paths: Paths) -> list[Book]:
    """Shipped books (books.json, or the two original books) followed by clinic-added books."""
    books: list[Book] = []
    manifest = paths.resource(BOOKS_MANIFEST)
    entries = _FALLBACK_BOOKS
    if manifest.exists():
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))["books"]
        except (OSError, ValueError, KeyError):
            entries = _FALLBACK_BOOKS
    for e in entries:
        b = Book(e["id"], e["title"], paths.resource(e["json"]), _resolve(paths.resources, e.get("pkl")),
                 _resolve(paths.resources, e.get("cooccurrence")), "shipped", e.get("description", ""),
                 tuple(c.lower() for c in e.get("exclude_chapters", ())))
        if b.available:
            books.append(b)
    user_dir = paths.user / USER_BOOKS_DIR
    if user_dir.exists():
        for meta in sorted(user_dir.glob("*/book.json")):
            try:
                e = json.loads(meta.read_text(encoding="utf-8"))
                b = Book(e["id"], e["title"], meta.parent / e["json"], None,
                         _resolve(meta.parent, e.get("cooccurrence")), "user", e.get("description", ""))
                if b.available and all(x.id != b.id for x in books):
                    books.append(b)
            except (OSError, ValueError, KeyError):
                continue
    return books


# ── decision settings ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DecisionConfig:
    books: tuple = ()                      # enabled book ids, in order; () → every available book
    book_weights: tuple = ()               # ((id, weight), ...); missing → 1.0
    learned_weight: float = 1.0            # how strongly this clinic's saved cases count (0 = ignore)
    category_weights: tuple = tuple(sorted(CLASSICAL_CATEGORY_WEIGHTS.items()))
    modality_weight: float = CLASSICAL_MODALITY_WEIGHT

    # construction helpers
    @classmethod
    def classical(cls, books: tuple | list = ()) -> "DecisionConfig":
        return cls(books=tuple(books))

    def weight_of(self, book_id: str) -> float:
        return dict(self.book_weights).get(book_id, 1.0)

    @property
    def categories(self) -> dict:
        return dict(self.category_weights)

    def resolve(self, available: list[Book]) -> "DecisionConfig":
        """Drop books that are not installed; empty selection → all available."""
        ids = [b.id for b in available]
        chosen = [b for b in self.books if b in ids] if self.books else ids
        return replace(self, books=tuple(chosen))

    @property
    def is_unbiased(self) -> bool:
        return (all(w == 1.0 for _, w in self.book_weights) and self.learned_weight == 1.0
                and self.categories == CLASSICAL_CATEGORY_WEIGHTS
                and self.modality_weight == CLASSICAL_MODALITY_WEIGHT)

    # persistence
    def to_dict(self) -> dict:
        return {"books": list(self.books), "book_weights": dict(self.book_weights),
                "learned_weight": self.learned_weight, "category_weights": dict(self.category_weights),
                "modality_weight": self.modality_weight}

    @classmethod
    def from_dict(cls, d: dict | None) -> "DecisionConfig":
        if not d:
            return cls()
        cw = {**CLASSICAL_CATEGORY_WEIGHTS, **{k: float(v) for k, v in (d.get("category_weights") or {}).items()
                                               if k in CLASSICAL_CATEGORY_WEIGHTS}}
        return cls(books=tuple(d.get("books") or ()),
                   book_weights=tuple(sorted((str(k), _clamp(v)) for k, v in (d.get("book_weights") or {}).items())),
                   learned_weight=_clamp(d.get("learned_weight", 1.0), 0.0, 5.0),
                   category_weights=tuple(sorted((k, _clamp(v, 0.0, 5.0)) for k, v in cw.items())),
                   modality_weight=_clamp(d.get("modality_weight", CLASSICAL_MODALITY_WEIGHT), 0.0, 5.0))

    def describe(self, books: list[Book]) -> str:
        """One line for reports and PDFs."""
        titles = {b.id: b.title for b in books}
        used = [f"{titles.get(b, b)}" + (f" ×{self.weight_of(b):g}" if self.weight_of(b) != 1.0 else "")
                for b in self.books]
        c = self.categories
        bits = [f"Books: {', '.join(used) or 'none'}"]
        bits.append("saved clinic cases " + ("not used" if self.learned_weight == 0 else
                                             f"×{self.learned_weight:g}" if self.learned_weight != 1 else "used"))
        if self.categories != CLASSICAL_CATEGORY_WEIGHTS or self.modality_weight != CLASSICAL_MODALITY_WEIGHT:
            bits.append(f"weights mind {c['psychological']:g} · generals {c['general']:g} · particulars "
                        f"{c['physical']:g} · modalities ×{self.modality_weight:g} · causation {c['causation']:g}")
        else:
            bits.append("classical weights")
        return "; ".join(bits)


def _clamp(v, lo: float = 0.0, hi: float = 5.0) -> float:
    try:
        return round(min(hi, max(lo, float(v))), 3)
    except (TypeError, ValueError):
        return 1.0


# ── importing a book ─────────────────────────────────────────────────────────
class BookFormatError(ValueError):
    pass


def validate_book_data(data) -> dict:
    """Check the HoRUS book format; return counts. Raises BookFormatError with a readable message."""
    if not isinstance(data, dict) or not data:
        raise BookFormatError("The file must contain a JSON object of remedies.")
    remedies = entries = 0
    for rem, chapters in data.items():
        if not isinstance(chapters, dict):
            raise BookFormatError(f"Remedy “{rem}” must map chapter names to lists of symptoms.")
        n = 0
        for chap, items in chapters.items():
            if not isinstance(items, list):
                raise BookFormatError(f"“{rem}” → “{chap}” must be a list of symptoms.")
            for it in items:
                if isinstance(it, dict) and str(it.get("symptom", "")).strip():
                    n += 1
        if n:
            remedies += 1
            entries += n
    if not entries:
        raise BookFormatError("No symptoms found. Each entry needs a “symptom” field.")
    return {"remedies": remedies, "entries": entries}


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or "book"


def import_book(paths: Paths, source: Path, title: str, progress=None) -> Book:
    """Copy a book into the clinic's book folder and build its co-occurrence clusters."""
    from .clusters import build_cooccurrence
    data = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    stats = validate_book_data(data)
    base = paths.user / USER_BOOKS_DIR
    book_id = "user_" + slugify(title)
    n = 2
    while (base / book_id).exists():
        book_id = f"user_{slugify(title)}_{n}"
        n += 1
    folder = base / book_id
    tmp = base / (book_id + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        shutil.copy2(source, tmp / "data.json")
        cp = build_cooccurrence(data, progress=progress)
        cp.to_csv(tmp / "cooccurrence.csv.gz", index=False, compression="gzip")
        meta = {"id": book_id, "title": title.strip() or Path(source).stem, "json": "data.json",
                "cooccurrence": "cooccurrence.csv.gz", "description": f"{stats['remedies']} remedies, "
                f"{stats['entries']:,} symptoms; added from {Path(source).name}", **stats}
        (tmp / "book.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.rename(folder)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return next(b for b in discover_books(paths) if b.id == book_id)


def remove_book(paths: Paths, book_id: str) -> None:
    folder = paths.user / USER_BOOKS_DIR / book_id
    if not book_id.startswith("user_") or not folder.exists():
        raise ValueError("Only books added in this clinic can be removed.")
    shutil.rmtree(folder)
