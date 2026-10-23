"""Symptom co-occurrence (conditional probabilities) and symptom suggestions.

Ported from ``load_conditional_probabilities``, ``build_live_conditional_probabilities``,
``get_suggestions_per_symptom`` and ``get_top_suggestions``.

Performance change, same results: the demo filtered the whole table (hundreds of
thousands of rows) once per patient symptom. ``SymptomIndex`` groups row positions by
symptom once, keeping the table's row order, so every lookup visits exactly the rows the
demo visited, in the same order.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from .config import CONDITIONAL_PROB_CANDIDATES, Paths
from .dataset import LIVE_PRIOR_STRENGTH

CP_COLS = ["Symptom1", "Symptom2", "P_B_given_A", "P_A_given_B"]


def load_conditional_probabilities(paths: Paths) -> pd.DataFrame | None:
    for fname in CONDITIONAL_PROB_CANDIDATES:
        fpath = paths.resource(fname)
        if fpath.exists():
            df = pd.read_csv(fpath)
            df.columns = [c.strip() for c in df.columns]
            return df
    return None


def build_live_conditional_probabilities(base: pd.DataFrame | None, payload) -> pd.DataFrame | None:
    """Base co-occurrence clusters, updated with every saved case.

    For a symptom A seen in n_A saved cases, of which n_AB also had symptom B:
        P(B|A) = (P_base(B|A) * k + n_AB) / (k + n_A)       k = LIVE_PRIOR_STRENGTH
    New pairs that only appear in saved cases enter with P_base = 0.
    """
    n_a, n_ab = {}, {}
    for syms, _ in payload:
        for s in syms:
            n_a[s] = n_a.get(s, 0) + 1
        for a, b in combinations(syms, 2):         # syms are sorted → a < b
            n_ab[(a, b)] = n_ab.get((a, b), 0) + 1
    if not n_a:
        return base

    k = LIVE_PRIOR_STRENGTH

    def cond(prior: float, a: str, b: str) -> float:   # P(b | a)
        key = (a, b) if a < b else (b, a)
        return round((prior * k + n_ab.get(key, 0)) / (k + n_a[a]), 6)

    if base is not None and all(c in base.columns for c in CP_COLS):
        df = base.copy()
        s1 = df["Symptom1"].astype(str).str.strip().str.lower()
        s2 = df["Symptom2"].astype(str).str.strip().str.lower()
        for i in df.index[s1.isin(n_a.keys()) | s2.isin(n_a.keys())]:
            a, b = s1[i], s2[i]
            if a in n_a:
                df.at[i, "P_B_given_A"] = cond(float(df.at[i, "P_B_given_A"]), a, b)
            if b in n_a:
                df.at[i, "P_A_given_B"] = cond(float(df.at[i, "P_A_given_B"]), b, a)
        existing = set(zip(s1, s2)) | set(zip(s2, s1))
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


class SymptomIndex:
    """Row lookup by lower-cased Symptom1 / Symptom2, preserving table order."""

    def __init__(self, cp_df: pd.DataFrame):
        s1_disp = cp_df["Symptom1"].tolist()
        s2_disp = cp_df["Symptom2"].tolist()
        s1 = cp_df["Symptom1"].str.strip().str.lower().tolist()
        s2 = cp_df["Symptom2"].str.strip().str.lower().tolist()
        pba = cp_df["P_B_given_A"].astype(float).tolist()
        pab = cp_df["P_A_given_B"].astype(float).tolist()
        # forward[ps] = [(other_lower, other_display, P(other|ps)), ...] in row order
        self.forward: dict = {}
        self.reverse: dict = {}
        for i in range(len(s1)):
            self.forward.setdefault(s1[i], []).append((s2[i], s2_disp[i], pba[i]))
            self.reverse.setdefault(s2[i], []).append((s1[i], s1_disp[i], pab[i]))

    def rows_for(self, ps: str):
        """Forward rows (ps is Symptom1) then reverse rows (ps is Symptom2) — the demo's order."""
        return self.forward.get(ps, []) + self.reverse.get(ps, [])


def get_suggestions_per_symptom(patient_symptoms: list, index: SymptomIndex | None,
                                top_n: int = 10) -> dict:
    """{patient_symptom: [top_n {symptom, prob}]} — each patient symptom gets its own list."""
    if index is None or not patient_symptoms:
        return {}
    patient_syms_lower = {s.strip().lower() for s in patient_symptoms if s.strip()}
    result = {}
    for ps_orig in patient_symptoms:
        ps = ps_orig.strip().lower()
        if not ps:
            continue
        scores = {}         # candidate_lower → (prob, display_str)
        for other, disp, p in index.rows_for(ps):
            if other not in patient_syms_lower:
                prev = scores.get(other, (0, disp))
                # demo behaviour: max probability, display text of the latest row seen
                scores[other] = (max(prev[0], p), disp)
        top = sorted(scores.values(), key=lambda x: -x[0])[:top_n]
        result[ps_orig] = [{"symptom": disp, "prob": round(prob, 4)} for prob, disp in top]
    return result


def get_top_suggestions(patient_symptoms: list, index: SymptomIndex | None,
                        top_n: int = 10) -> list:
    """Aggregate across all patient symptoms → top_n globally, with the strongest trigger."""
    if index is None or not patient_symptoms:
        return []
    patient_syms_lower = {s.strip().lower() for s in patient_symptoms if s.strip()}
    agg = {}
    for ps_orig in patient_symptoms:
        ps = ps_orig.strip().lower()
        if not ps:
            continue
        for other, disp, p in index.rows_for(ps):
            if other in patient_syms_lower:
                continue
            if other not in agg:
                agg[other] = {"probs": [], "display": disp, "best_trigger": ps_orig, "best_prob": p}
            agg[other]["probs"].append(p)
            if p > agg[other]["best_prob"]:
                agg[other]["best_prob"] = p
                agg[other]["best_trigger"] = ps_orig

    averaged = [
        {
            "symptom": v["display"],
            "avg_prob": round(sum(v["probs"]) / len(v["probs"]), 4),
            "support": len(v["probs"]),
            "triggered_by": v["best_trigger"],
        }
        for v in agg.values()
    ]
    averaged.sort(key=lambda x: (-x["avg_prob"], -x["support"]))
    return averaged[:top_n]
