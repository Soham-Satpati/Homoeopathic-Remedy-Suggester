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

    @property
    def remedy_universe(self) -> set:
        return {rem for rems in self.s2r.values() for rem in rems}


def load_base_dataset(paths: Paths) -> Dataset:
    """Merge pkl models + raw JSON; max score wins on collision."""
    known: set = set()
    s2r: dict = defaultdict(dict)
    sources: list = []

    # Layer 1: pkl files (case_studies + rheumatic)
    for fname in MODEL_PICKLES:
        fpath = paths.resource(fname)
        if not fpath.exists():
            continue
        try:
            with open(fpath, "rb") as f:
                d = pickle.load(f)   # trusted: shipped with the app, never user-supplied
            for sym, rem_dict in d.get("symptom_to_remedies", {}).items():
                sl = sym.strip().lower()
                known.add(sl)
                for rem, score in rem_dict.items():
                    s2r[sl][rem] = max(s2r[sl].get(rem, 0), score)
            sources.append(fname)
        except Exception:
            log.exception("Could not read %s", fpath)

    # Layer 2: raw JSON (always parsed — fills any pkl gaps)
    all_entries: list = []
    for fname in DATASET_JSONS:
        fpath = paths.resource(fname)
        if not fpath.exists():
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for remedy, sections in data.items():
                for section_list in sections.values():
                    for item in section_list:
                        if isinstance(item, dict):
                            sym = item.get("symptom", "").strip().lower()
                            if sym:
                                known.add(sym)
                                all_entries.append((sym, remedy))
            sources.append(fname)
        except Exception:
            log.exception("Could not read %s", fpath)

    counts = Counter(sym for sym, _ in all_entries)
    for sym, rem in all_entries:
        score = round(1.0 / max(1, counts[sym]), 6)
        s2r[sym][rem] = max(s2r[sym].get(rem, 0), score)

    source_label = ", ".join(dict.fromkeys(sources)) if sources else "none"
    return Dataset(frozenset(known), dict(s2r), source_label, 0)


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


def build_live_dataset(base: Dataset, payload: Iterable) -> Dataset:
    """Base dataset merged with everything learned from saved cases."""
    payload = tuple(payload)
    if not payload:
        return base
    known = set(base.known)
    s2r = dict(base.s2r)          # shallow copy; touched entries are copied below
    touched = set()
    for syms, rems in payload:
        for sym in syms:
            known.add(sym)
            if sym not in touched:
                s2r[sym] = dict(base.s2r.get(sym, {}))
                touched.add(sym)
            for rank, rem in enumerate(rems):
                w = LEARN_RANK_WEIGHTS[rank] if rank < len(LEARN_RANK_WEIGHTS) else 0.05
                s2r[sym][rem] = round(s2r[sym].get(rem, 0.0) + w, 6)
    label = f"{base.source} + {len(payload)} saved case(s)"
    return Dataset(frozenset(known), s2r, label, len(payload))


def check_symptoms_against_dataset(dataset: Dataset, symptom_list: list) -> tuple:
    """Split into (found, missing) using the known-symptom set."""
    found, missing = [], []
    for s in symptom_list:
        (found if s.strip().lower() in dataset.known else missing).append(s)
    return found, missing
