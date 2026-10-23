"""Deterministic miasm rubric. Ported from ``assess_miasm``.

Keyword rubric after Hahnemann / J.H. Allen / Banerjea. Override it by placing a
``miasm_rubric.json`` (``{"psora": ["kw", ...], ...}``) in the user-data folder or the
resource folder (see ``Paths.miasm_rubric_path``).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MIASM_RUBRIC = {
    "psora": ["anxiety", "anxious", "fear", "apprehens", "palpitat", "itch", "burning", "chilly",
              "sensitive to cold", "craving for sweets", "desire for sweets", "sweets",
              "weeping", "sad", "functional", "hunger", "dry skin", "worry"],
    "sycosis": ["secretive", "jealous", "suspicious", "wart", "growth", "tumour", "tumor", "cyst",
                "catarrh", "worse damp", "worse wet", "damp weather", "swelling", "stiffness",
                "obesity", "fixed idea", "hurried", "excess"],
    "syphilis": ["worse at night", "worse night", "after midnight", "sunset", "ulcer", "destruct",
                 "suicid", "despair", "alone in the world", "isolation", "forsaken", "bone pain",
                 "fissure", "deformit", "self-destruct"],
    "tubercular": ["dissatisf", "change", "travel", "restless", "better open air", "better in open air",
                   "desire open air", "worse closed room", "worse stuffy", "worse warmth",
                   "worse warm room", "cold drinks", "cold milk", "smoked meat", "emaciat",
                   "catches cold", "night sweat", "tired easily", "allerg"],
}
HISTORY_HINTS = r"family history|past history|history of|since childhood|mother had|father had|" \
                r"runs in (the )?family|previous(ly)? (treated|had)|vaccinat|recurrent|chronic"

_NAMES = {"psora": "Psora", "sycosis": "Sycosis", "syphilis": "Syphilis", "tubercular": "Tubercular"}


def load_miasm_rubric(path: Path | None) -> dict:
    if path is not None and Path(path).exists():
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            return {k.lower(): [str(x).lower() for x in v] for k, v in d.items() if isinstance(v, list)}
        except Exception:
            log.exception("Invalid miasm rubric %s — using the default", path)
    return DEFAULT_MIASM_RUBRIC


def assess_miasm(cat: dict, raw_text: str = "", rubric: dict | None = None) -> dict:
    rubric = rubric if rubric is not None else DEFAULT_MIASM_RUBRIC
    items, seen = [], set()

    def _add(label, text):
        key = text.strip().lower()
        if key and key not in seen:          # a modality shared by many symptoms counts ONCE
            seen.add(key)
            items.append((label, key))

    for k in ("physical", "psychological", "general"):
        for s in cat.get(k, []):
            if s.get("unverified"):
                continue
            _add(s["symptom"], s["symptom"])
            for w in s.get("worse", []):
                _add(f"worse {w}", f"worse {w}")
            for b in s.get("better", []):
                _add(f"better {b}", f"better {b}")
    for pol in ("worse", "better"):
        for m in (cat.get("case_modalities") or {}).get(pol, []):
            _add(f"{pol} {m}", f"{pol} {m}")
    for c in cat.get("causation", []):
        _add(f"ailments from {c}", c)

    evidence = {m: [] for m in rubric}
    for label, text in items:
        for miasm, kws in rubric.items():
            for kw in kws:
                if re.search(rf"(?<![a-z]){re.escape(kw)}", text):   # word-start: "itch" ≠ "stitching"
                    evidence[miasm].append(f"{label}  [{kw}]")
                    break
    scores = {m: len(v) for m, v in evidence.items()}
    ordered = sorted(scores, key=lambda m: -scores[m])
    has_history = bool(re.search(HISTORY_HINTS, (raw_text or "").lower()))

    nm = lambda m: _NAMES.get(m, m.title())
    if not ordered or scores[ordered[0]] == 0:
        label = "Undetermined"
    else:
        top = scores[ordered[0]]
        tied = [m for m in ordered if scores[m] == top]
        secondary = [m for m in ordered if m not in tied and scores[m] >= max(2, top * 0.6)]
        if len(tied) > 1:
            label = "Mixed " + " / ".join(nm(t) for t in tied)
        else:
            label = f"{nm(tied[0])} (predominant)"
        if secondary:
            label += ", with " + " and ".join(nm(m) for m in secondary) + " features"

    text = (f"Provisional: {label} — confirm from chronic, past and family history."
            if not has_history else f"{label}.")
    if has_history is False:
        label = f"Provisional: {label}"
    return {"label": label, "scores": scores, "evidence": evidence, "provisional": not has_history,
            "text": text}
