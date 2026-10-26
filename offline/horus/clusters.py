"""Build a book's symptom co-occurrence table — the data behind "Symptoms that often occur with this case".

Same method as the shipped tables (``rheumatic_conditional_probabilities.csv``):

1. Cluster the book's symptom entries with HDBSCAN on one-hot remedy + modality features
   (min_cluster_size=6, Hamming distance, leaf selection — as in ``symptom_clustering_pro.py``).
2. Within each cluster, for two symptoms A and B that share at least one remedy:
       P(B | A) = shared remedies / remedies listing A      (rounded to 4 places)
   with remedy counts taken over the whole book.

Large books make enormous clusters (most entries have no modalities, so a cluster is close to
"one remedy's symptoms"). Since the app shows at most 10 suggestions per symptom, each symptom
keeps its ``top_k`` strongest partners; ties prefer the same chapter, then shared modalities.
Output columns match the shipped tables, one row per unordered pair.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

COLUMNS = ["Symptom1", "Symptom2", "Remedies_S1", "Remedies_S2", "Shared_Remedies", "P_B_given_A", "P_A_given_B"]


def clean_modalities(text) -> list[str]:
    """As in symptom_clustering_pro.py."""
    if not text:
        return []
    text = str(text).lower().strip()
    if text in ("", "none", "no", "-", "nil", "null"):
        return []
    items = re.split(r"[,;\n]|\band\b|\bor\b|\bwith\b", text)
    return [i.strip() for i in items if len(i.strip()) > 2]


def _entries(data: dict, exclude_chapters=()) -> list[tuple]:
    rows = []
    for rem, chapters in data.items():
        if not isinstance(chapters, dict):
            continue
        for chap, items in chapters.items():
            if str(chap).lower() in exclude_chapters:
                continue
            for it in items if isinstance(items, list) else []:
                if not isinstance(it, dict):
                    continue
                sym = str(it.get("symptom") or "").strip()
                if sym:
                    mods = clean_modalities((it.get("labels") or {}).get("Modalities"))
                    rows.append((sym, sym.lower(), str(rem), str(chap), mods))
    return rows


def cluster_labels(rows: list[tuple]) -> np.ndarray:
    from sklearn.cluster import HDBSCAN
    from sklearn.preprocessing import MultiLabelBinarizer
    rem = MultiLabelBinarizer().fit_transform([[r[2]] for r in rows])
    mod = MultiLabelBinarizer().fit_transform([r[4] for r in rows])
    X = np.hstack([rem, mod]).astype(bool) if mod.shape[1] else rem.astype(bool)
    return HDBSCAN(min_cluster_size=6, metric="hamming", cluster_selection_method="leaf",
                   algorithm="ball_tree", copy=True).fit_predict(X)


def build_cooccurrence(data: dict, top_k: int = 15, progress=None, exclude_chapters=()) -> pd.DataFrame:
    rows = _entries(data, tuple(c.lower() for c in exclude_chapters))
    if len(rows) < 6:
        return pd.DataFrame(columns=COLUMNS)
    if progress:
        progress("Clustering symptoms…")
    labels = cluster_labels(rows)

    remedies = defaultdict(set)          # symptom (lower) → remedies listing it anywhere in the book
    display, chapters, mods = {}, defaultdict(set), defaultdict(set)
    for disp, low, rem, chap, m in rows:
        remedies[low].add(rem)
        display.setdefault(low, disp)
        chapters[low].add(chap)
        mods[low].update(m)

    members = defaultdict(set)
    for (disp, low, rem, chap, m), lab in zip(rows, labels):
        if lab != -1:
            members[int(lab)].add(low)
    if progress:
        progress(f"Linking symptoms in {len(members)} clusters…")

    kept: dict[tuple, None] = {}
    for cid, syms in members.items():
        by_rem = defaultdict(set)
        for s in syms:
            for r in remedies[s]:
                by_rem[r].add(s)
        for a in syms:
            ra = remedies[a]
            cands = set().union(*(by_rem[r] for r in ra)) - {a}
            if not cands:
                continue
            scored = []
            for b in cands:
                shared = len(ra & remedies[b])
                p = round(shared / len(ra), 4)
                scored.append((-p, -bool(chapters[a] & chapters[b]), -len(mods[a] & mods[b]), b))
            scored.sort()
            for *_k, b in scored[:top_k]:
                kept[(a, b) if a < b else (b, a)] = None

    out = []
    for a, b in kept:
        ra, rb = remedies[a], remedies[b]
        shared = len(ra & rb)
        out.append((display[a], display[b], len(ra), len(rb), shared,
                    round(shared / len(ra), 4), round(shared / len(rb), 4)))
    df = pd.DataFrame(out, columns=COLUMNS)
    return df.sort_values(["Symptom1", "Symptom2"], kind="stable").reset_index(drop=True)
