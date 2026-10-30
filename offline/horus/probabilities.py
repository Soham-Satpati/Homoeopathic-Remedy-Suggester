"""Symptom co-occurrence (conditional probabilities) and symptom suggestions.

Ported from ``load_conditional_probabilities``, ``build_live_conditional_probabilities``,
``get_suggestions_per_symptom`` and ``get_top_suggestions``.

Performance change, same results: the demo filtered the whole table (hundreds of
thousands of rows) once per patient symptom. ``SymptomIndex`` groups row positions by
symptom once, keeping the table's row order, so every lookup visits exactly the rows the
demo visited, in the same order.
"""

from __future__ import annotations

import re
from itertools import combinations

import pandas as pd

from .config import CONDITIONAL_PROB_CANDIDATES, Paths
from .dataset import LIVE_PRIOR_STRENGTH

CP_COLS = ["Symptom1", "Symptom2", "P_B_given_A", "P_A_given_B"]


def _read_table(path) -> pd.DataFrame:
    """Read .csv or .csv.gz. Gzip is decompressed in one call: pandas' streaming gzip reader
    takes the interpreter lock for every small chunk and crawls while the UI thread is busy."""
    import gzip
    import io
    from pathlib import Path
    path = Path(path)
    if path.suffix == ".gz":
        return pd.read_csv(io.BytesIO(gzip.decompress(path.read_bytes())))
    return pd.read_csv(path)


def load_book_cooccurrence(books: list) -> pd.DataFrame | None:
    """Co-occurrence tables of the chosen books, combined (.csv or .csv.gz). A pair that appears
    in several books keeps one row per book; suggestions take the strongest."""
    frames = []
    for b in books:
        p = getattr(b, "cooccurrence_path", None)
        if p is None or not p.exists():
            continue
        df = _read_table(p)
        df.columns = [c.strip() for c in df.columns]
        if all(c in df.columns for c in CP_COLS):
            frames.append(df[CP_COLS])
    if not frames:
        return None
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    # strings as categories: one copy of each symptom text instead of one per row
    return df.astype({"Symptom1": "category", "Symptom2": "category"})


def load_conditional_probabilities(paths: Paths) -> pd.DataFrame | None:
    """The demo's loader: the first table found (kept for reference and parity tests)."""
    for fname in CONDITIONAL_PROB_CANDIDATES:
        fpath = paths.resource(fname)
        if fpath.exists():
            df = pd.read_csv(fpath)
            df.columns = [c.strip() for c in df.columns]
            return df
    return None


def _lower_col(col: pd.Series) -> pd.Series:
    """Lower-cased, stripped symptom column; for categories each distinct text is lowered once."""
    import numpy as np
    if isinstance(col.dtype, pd.CategoricalDtype):
        cats = np.array([str(c).strip().lower() for c in col.cat.categories] + [""], dtype=object)
        return pd.Series(cats[col.cat.codes.to_numpy()], index=col.index)
    return col.astype(str).str.strip().str.lower()


def build_live_conditional_probabilities(base: pd.DataFrame | None, payload,
                                        weight: float = 1.0) -> pd.DataFrame | None:
    """Base co-occurrence clusters, updated with every saved case.

    For a symptom A seen in n_A saved cases, of which n_AB also had symptom B:
        P(B|A) = (P_base(B|A) * k + n_AB) / (k + n_A)       k = LIVE_PRIOR_STRENGTH
    New pairs that only appear in saved cases enter with P_base = 0.
    ``weight`` scales the saved-case counts (1.0 = as in the demo; 0 = book values only).
    """
    n_a, n_ab = {}, {}
    if weight == 0:
        return base
    for syms, _ in payload:
        for s in syms:
            n_a[s] = n_a.get(s, 0) + weight
        for a, b in combinations(syms, 2):         # syms are sorted → a < b
            n_ab[(a, b)] = n_ab.get((a, b), 0) + weight
    if not n_a:
        return base

    k = LIVE_PRIOR_STRENGTH

    def cond(prior: float, a: str, b: str) -> float:   # P(b | a)
        key = (a, b) if a < b else (b, a)
        return round((prior * k + n_ab.get(key, 0)) / (k + n_a[a]), 6)

    if base is not None and all(c in base.columns for c in CP_COLS):
        df = base[CP_COLS].copy()
        s1, s2 = _lower_col(df["Symptom1"]), _lower_col(df["Symptom2"])
        for i in df.index[s1.isin(n_a.keys()) | s2.isin(n_a.keys())]:
            a, b = s1[i], s2[i]
            if a in n_a:
                df.at[i, "P_B_given_A"] = cond(float(df.at[i, "P_B_given_A"]), a, b)
            if b in n_a:
                df.at[i, "P_A_given_B"] = cond(float(df.at[i, "P_A_given_B"]), b, a)
        # only pairs between symptoms of saved cases can be new
        both = s1.isin(n_a.keys()) & s2.isin(n_a.keys())
        existing = set(zip(s1[both], s2[both])) | set(zip(s2[both], s1[both]))
    else:
        df = pd.DataFrame(columns=CP_COLS)
        existing = set()

    new_rows = [
        {"Symptom1": a, "Symptom2": b,
         "P_B_given_A": round(c / (k + n_a[a]), 6),
         "P_A_given_B": round(c / (k + n_a[b]), 6)}
        for (a, b), c in n_ab.items() if (a, b) not in existing
    ]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df if len(df) else None


class _Side:
    """Rows of one direction (Symptom1 → Symptom2 or back), grouped by symptom, in table order.
    Read-only mapping: ``side[symptom] -> [(other_lower, other_display, prob), ...]``."""

    def __init__(self, key, other, other_disp, prob, lower: list, display: list):
        import numpy as np
        order = np.argsort(key, kind="stable")          # stable: keeps the table's row order per symptom
        self._key, self._other = key[order], other[order]
        self._disp, self._prob = other_disp[order], prob[order]
        self._lower, self._display = lower, display
        uniq, starts = np.unique(self._key, return_index=True)
        ends = list(starts[1:]) + [len(self._key)]
        self._span = {lower[int(k)]: (int(a), int(b)) for k, a, b in zip(uniq, starts, ends)}

    def __contains__(self, k) -> bool:
        return k in self._span

    def __iter__(self):
        return iter(self._span)

    def __len__(self) -> int:
        return len(self._span)

    def get(self, k, default=None):
        return self[k] if k in self._span else default

    def __getitem__(self, k) -> list:
        a, b = self._span[k]
        lo, dp = self._lower, self._display
        return [(lo[o], dp[d], float(p)) for o, d, p in
                zip(self._other[a:b].tolist(), self._disp[a:b].tolist(), self._prob[a:b].tolist())]


class SymptomIndex:
    """Row lookup by lower-cased Symptom1 / Symptom2, preserving table order.

    Stored as integer-coded numpy arrays (a few bytes per row): the ten shipped books have ~1.6
    million rows, which as Python tuples took over a gigabyte."""

    def __init__(self, cp_df: pd.DataFrame):
        import numpy as np
        d1 = cp_df["Symptom1"].astype(str)
        d2 = cp_df["Symptom2"].astype(str)
        disp_codes, display = pd.factorize(pd.concat([d1, d2], ignore_index=True), sort=False)
        display = [str(x) for x in display]
        low_of_disp = [x.strip().lower() for x in display]
        low_codes, lower = pd.factorize(pd.Series(low_of_disp), sort=False)
        lower = [str(x) for x in lower]
        n = len(cp_df)
        dc = disp_codes.astype(np.int32)
        lc = np.asarray(low_codes, dtype=np.int32)[dc]
        s1d, s2d, s1, s2 = dc[:n], dc[n:], lc[:n], lc[n:]
        pba = cp_df["P_B_given_A"].astype(float).to_numpy()
        pab = cp_df["P_A_given_B"].astype(float).to_numpy()
        # forward[ps] = [(other_lower, other_display, P(other|ps)), ...] in row order
        self.forward = _Side(s1, s2, s2d, pba, lower, display)
        self.reverse = _Side(s2, s1, s1d, pab, lower, display)

    def rows_for(self, ps: str):
        """Forward rows (ps is Symptom1) then reverse rows (ps is Symptom2) — the demo's order."""
        return self.forward.get(ps, []) + self.reverse.get(ps, [])

    def has(self, ps: str) -> bool:
        return ps in self.forward or ps in self.reverse

    @property
    def keys(self) -> tuple:
        """Every symptom that has co-occurrence rows, sorted."""
        if not hasattr(self, "_keys"):
            self._keys = tuple(sorted(set(self.forward) | set(self.reverse)))
        return self._keys


def _expanded_rows(index: SymptomIndex, ps: str, resolve):
    """Rows for one patient symptom: [(other, display, score, via)], plus the table symptoms used.

    Without ``resolve`` (the demo) only the exact symptom text is looked up. With it, each patient
    symptom is mapped to table symptoms (``resolve(ps) -> [(table_symptom, weight)]``: itself when
    the tables contain it verbatim, else its closest table symptoms) and their rows are used, each
    probability multiplied by the weight; ``via`` names the table symptom when it isn't ``ps``."""
    if resolve is None:
        return [(o, d, p, None) for o, d, p in index.rows_for(ps)], set()
    anchors = resolve(ps)
    out = []
    for key, sim, n_key in anchors:
        via = None if key == ps else key
        for o, d, p in index.rows_for(key):
            if looks_like_symptom(d):
                shared = p * n_key               # remedies listing both symptoms
                out.append((o, d, p * sim * shared / (shared + SHARED_PRIOR), via))
    return out, {a[0] for a in anchors}


# P(B|A) = 100% when A is listed under one remedy only: every other symptom of that remedy gets it.
# Links are therefore scaled by shared / (shared + 2), shared = remedies listing both symptoms.
SHARED_PRIOR = 2.0


_JUNK = re.compile(r"[\^■□•|\\{}~_]|\.{3,}|\b[b-hj-z]\b.*\b[b-hj-z]\b.*\b[b-hj-z]\b")


_FRAGMENT = re.compile(r"(and|or|in|at|on|about|across|as|with|from|to|of|a|an|the|but|also|then|which|"
                       r"who|when|while|after|before|during|for|by|into|under|over|up|set)\b")


_CONJ = re.compile(r"(and|or|but|also|then|which|who|set)\b")


def looks_like_symptom(text: str) -> bool:
    """Hide scanning debris ("a girl. ^enuresis", "m n'gfit: restless") from suggestions."""
    t = str(text).strip()
    if len(t) < 4 or _JUNK.search(t.lower().replace(" a ", " ")):
        return False
    # sub-entries cut from their heading ("in stomach", "and creeping under l. eye", "a man, set");
    # longer entries starting "when…", "during…", "after…" are real symptoms
    low = t.lower()
    if _FRAGMENT.match(low) and not low.startswith("as if") and (len(low.split()) <= 4 or _CONJ.match(low)):
        return False
    letters = sum(c.isalnum() or c.isspace() or c in ".,;:()'\"-<>!?/" for c in t)
    return letters / len(t) >= 0.9


def get_suggestions_per_symptom(patient_symptoms: list, index: SymptomIndex | None,
                                top_n: int = 10, resolve=None) -> dict:
    """{patient_symptom: [top_n {symptom, prob}]} — each patient symptom gets its own list.
    ``resolve``: see ``_expanded_rows`` (None = exact wording only, as in the demo)."""
    if index is None or not patient_symptoms:
        return {}
    patient_syms_lower = {s.strip().lower() for s in patient_symptoms if s.strip()}
    result = {}
    for ps_orig in patient_symptoms:
        ps = ps_orig.strip().lower()
        if not ps:
            continue
        scores = {}         # candidate_lower → (prob, display_str, via)
        rows, anchors = _expanded_rows(index, ps, resolve)
        for other, disp, p, via in rows:
            if other not in patient_syms_lower and other not in anchors:
                prev = scores.get(other, (0, disp, via))
                # demo behaviour: max probability, display text of the latest row seen
                scores[other] = (max(prev[0], p), disp, via if p >= prev[0] else prev[2])
        top = sorted(scores.values(), key=lambda x: -x[0])[:top_n]
        result[ps_orig] = [{"symptom": disp, "prob": round(prob, 4), **({"via": via} if via else {})}
                           for prob, disp, via in top]
    return result


def get_top_suggestions(patient_symptoms: list, index: SymptomIndex | None,
                        top_n: int = 10, resolve=None) -> list:
    """Aggregate across all patient symptoms → top_n globally, with the strongest trigger."""
    if index is None or not patient_symptoms:
        return []
    patient_syms_lower = {s.strip().lower() for s in patient_symptoms if s.strip()}
    agg = {}
    expanded = {}
    for ps_orig in patient_symptoms:
        ps = ps_orig.strip().lower()
        if ps and ps not in expanded:
            expanded[ps] = _expanded_rows(index, ps, resolve)
    all_anchors = set().union(*(a for _, a in expanded.values())) if expanded else set()
    for ps_orig in patient_symptoms:
        ps = ps_orig.strip().lower()
        if not ps:
            continue
        for other, disp, p, via in expanded[ps][0]:
            if other in patient_syms_lower or other in all_anchors:
                continue
            if other not in agg:
                agg[other] = {"probs": [], "display": disp, "best_trigger": ps_orig, "best_prob": p,
                              "via": via}
            agg[other]["probs"].append(p)
            if p > agg[other]["best_prob"]:
                agg[other]["best_prob"] = p
                agg[other]["best_trigger"] = ps_orig
                agg[other]["via"] = via

    averaged = [
        {
            "symptom": v["display"],
            "avg_prob": round(sum(v["probs"]) / len(v["probs"]), 4),
            "support": len(v["probs"]),
            "triggered_by": v["best_trigger"],
            **({"via": v["via"]} if v["via"] else {}),
        }
        for v in agg.values()
    ]
    if resolve is None:
        averaged.sort(key=lambda x: (-x["avg_prob"], -x["support"]))
    else:
        # mapped symptoms bring many weak, single-remedy pairs (P = 1.0 when a symptom belongs to one
        # remedy); a suggestion linked to more of the case comes first
        averaged.sort(key=lambda x: (-(x["avg_prob"] * x["support"] ** 0.5), -x["support"], x["symptom"]))
    return averaged[:top_n]
