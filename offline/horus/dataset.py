"""Remedy dataset: base symptom→remedy index plus what is learned from saved cases.

Ported from ``load_all_dataset_symptoms``, ``_learning_payload`` and
``build_live_dataset`` in the demo. The only change is where inputs come from: files are
read from ``Paths.resources`` instead of the working directory, and saved cases are passed
in as patient records instead of being fetched from GitHub.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from .config import DATASET_JSONS, MODEL_PICKLES, Paths

log = logging.getLogger(__name__)

# Each saved case adds its symptoms → top remedies (rank-weighted) to the remedy index,
# and its symptom pairs to the co-occurrence clusters.
LEARN_RANK_WEIGHTS = (0.30, 0.15, 0.10)   # weight added for remedy rank 1, 2, 3
LIVE_PRIOR_STRENGTH = 10.0                # pseudo-count: how strongly base clusters resist new cases

LearningPayload = tuple  # ((sorted symptom tuple), (top-3 remedies tuple)), ...


@dataclass(frozen=True)
class Dataset:
    known: frozenset            # every dataset symptom, lower-cased
    s2r: dict                   # symptom (lower) -> {remedy: score}
    source: str                 # human-readable list of what was loaded
    live_cases: int = 0         # saved cases merged in
    origin: dict | None = None  # symptom -> {remedy: frozenset(book ids)}; for book weights
    books: tuple = ()           # ids of the books loaded
    merged_names: bool = False  # remedy names unified across capitalisations

    @property
    def remedy_universe(self) -> set:
        return {rem for rems in self.s2r.values() for rem in rems}


def _legacy_books(paths: Paths) -> list:
    """The two original books, by file name (used when no book list is given)."""
    from .books import discover_books
    return [b for b in discover_books(paths) if b.id in ("rheumatic", "case_studies")]


def remedy_key(name: str) -> str:
    return " ".join(str(name).split()).casefold()


class _RemedyNames:
    """One display name per remedy, whatever capitalisation each book uses
    ("Gelsemium" / "gelsemium", "Natrum Muriaticum" / "natrum muriaticum")."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.names: dict = {}

    def __call__(self, name: str) -> str:
        if not self.enabled:
            return name
        k = remedy_key(name)
        cur = self.names.get(k)
        clean = " ".join(str(name).split())
        # prefer a capitalised spelling ("Gelsemium" over "gelsemium"), else the first seen
        if cur is None or (not cur[:1].isupper() and clean[:1].isupper()):
            self.names[k] = clean if clean[:1].isupper() else (cur or clean[:1].upper() + clean[1:])
        return self.names[k]


def _merge_names(s2r: dict, origin: dict, names: _RemedyNames) -> tuple:
    """Re-key remedies by their final display name (a capitalised spelling may be met late)."""
    out, org = {}, {}
    for sym, rems in s2r.items():
        d, o = {}, {}
        for rem, score in rems.items():
            n = names.names.get(remedy_key(rem), rem)
            d[n] = max(d.get(n, 0), score)
            o.setdefault(n, set()).update(origin.get(sym, {}).get(rem, ()))
        out[sym], org[sym] = d, o
    return out, org


def load_base_dataset(paths: Paths, books: list | None = None, progress=None,
                      merge_remedy_names: bool = False) -> Dataset:
    """Merge pkl models + raw JSON of the chosen books; max score wins on collision.

    With the two original books this is exactly the demo's ``load_all_dataset_symptoms``
    (including its cross-book symptom counts in the JSON layer, and its file order).
    """
    books = _legacy_books(paths) if books is None else list(books)
    known: set = set()
    s2r: dict = defaultdict(dict)
    origin: dict = defaultdict(lambda: defaultdict(set))
    sources: list = []
    canon = _RemedyNames(merge_remedy_names)

    # Layer 1: pkl files, in the demo's order (case_studies, then rheumatic, then any other)
    order = {name: i for i, name in enumerate(MODEL_PICKLES)}
    for b in sorted((b for b in books if b.pkl_path), key=lambda b: order.get(b.pkl_path.name, 99)):
        if not b.pkl_path.exists():
            continue
        try:
            with open(b.pkl_path, "rb") as f:
                d = pickle.load(f)   # trusted: shipped with the app, never user-supplied
            for sym, rem_dict in d.get("symptom_to_remedies", {}).items():
                sl = sym.strip().lower()
                known.add(sl)
                for rem, score in rem_dict.items():
                    rem = canon(rem)
                    s2r[sl][rem] = max(s2r[sl].get(rem, 0), score)
                    origin[sl][rem].add(b.id)
            sources.append(b.pkl_path.name)
        except Exception:
            log.exception("Could not read %s", b.pkl_path)

    # Layer 2: raw JSON (always parsed — fills any pkl gaps), demo order: rheumatic, case studies, others
    jorder = {name: i for i, name in enumerate(DATASET_JSONS)}
    all_entries: list = []
    ordered = sorted(books, key=lambda b: jorder.get(b.json_path.name, 99))
    for n, b in enumerate(ordered):
        if progress:
            progress(n / max(1, len(ordered)), f"Loading {b.title}…")
        try:
            with open(b.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for remedy, sections in data.items():
                for chapter, section_list in sections.items():
                    if str(chapter).lower() in b.exclude_chapters:
                        continue
                    for item in section_list:
                        if isinstance(item, dict):
                            sym = item.get("symptom", "").strip().lower()
                            if sym:
                                known.add(sym)
                                all_entries.append((sym, canon(remedy), b.id))
            sources.append(b.json_path.name)
        except Exception:
            log.exception("Could not read %s", b.json_path)

    counts = Counter(sym for sym, _, _ in all_entries)
    for sym, rem, bid in all_entries:
        score = round(1.0 / max(1, counts[sym]), 6)
        s2r[sym][rem] = max(s2r[sym].get(rem, 0), score)
        origin[sym][rem].add(bid)

    if merge_remedy_names:
        s2r, origin = _merge_names(s2r, origin, canon)
    source_label = ", ".join(dict.fromkeys(sources)) if sources else "none"
    shared: dict = {}          # one frozenset object per distinct set of books (most are one book)
    frozen_origin = {s: {r: shared.setdefault(frozenset(v), frozenset(v)) for r, v in rems.items()}
                     for s, rems in origin.items()}
    return Dataset(frozenset(known), dict(s2r), source_label, 0, frozen_origin, tuple(b.id for b in books),
                   merge_remedy_names)


def learning_payload(patients: Mapping) -> LearningPayload:
    """Compact, hashable snapshot of every saved case.

    ``patients`` uses the demo's record shape: ``{pid: {"details": {}, "cases": [...]}}``
    (a bare list of cases is also accepted, as in very old records).
    """
    out = []
    for rec in patients.values():
        cases = rec if isinstance(rec, list) else rec.get("cases", [])
        for c in cases:
            cat = c.get("categorised") or {}
            syms = sorted({
                s.get("symptom", "").strip().lower()
                for k in ("physical", "psychological", "general")
                for s in cat.get(k, [])
                if isinstance(s, dict) and s.get("symptom", "").strip()
            })
            rems = tuple(r for r in (c.get("top_remedies") or [])[:3] if r)
            if syms:
                out.append((tuple(syms), rems))
    return tuple(out)


def build_live_dataset(base: Dataset, payload: Iterable, weight: float = 1.0) -> Dataset:
    """Base dataset merged with everything learned from saved cases. ``weight`` scales how much the
    clinic's own cases count (1.0 = as in the demo, 0 = not at all)."""
    payload = tuple(payload)
    if not payload or weight == 0:
        return base
    known = set(base.known)
    s2r = dict(base.s2r)          # shallow copy; touched entries are copied below
    spelling = {remedy_key(r): r for r in base.remedy_universe} if base.merged_names else {}
    touched = set()
    for syms, rems in payload:
        for sym in syms:
            known.add(sym)
            if sym not in touched:
                s2r[sym] = dict(base.s2r.get(sym, {}))
                touched.add(sym)
            for rank, rem in enumerate(rems):
                rem = spelling.get(remedy_key(rem), rem)
                w = LEARN_RANK_WEIGHTS[rank] if rank < len(LEARN_RANK_WEIGHTS) else 0.05
                s2r[sym][rem] = round(s2r[sym].get(rem, 0.0) + w * weight, 6)
    label = f"{base.source} + {len(payload)} saved case(s)"
    return Dataset(frozenset(known), s2r, label, len(payload), base.origin, base.books, base.merged_names)


def check_symptoms_against_dataset(dataset: Dataset, symptom_list: list) -> tuple:
    """Split into (found, missing) using the known-symptom set."""
    found, missing = [], []
    for s in symptom_list:
        (found if s.strip().lower() in dataset.known else missing).append(s)
    return found, missing
