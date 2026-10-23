"""Code-side verification of parsed cases. Ported from ``finalise_categorised`` and friends.

Whatever produced the structured case — Gemini in the demo, a local model or the doctor's
structured entry here — it passes through ``finalise_categorised`` before it can affect a
ranking: unsupported modalities and causes are dropped, unsupported symptoms are marked
``unverified``, grief/shock/fright causes are never lost, and miasm + red flags are attached.
"""

from __future__ import annotations

import re

from .miasm import assess_miasm
from .safety import detect_red_flags

STOP = set("""a an the and or of to in on at for with from by as is are was were be been being it its this
that these those she he they her his their them i my me you your we our has have had having not no
very more most less least some any all each when then than so such into out up down over under
again also only just can will would could should may might feel feels feeling felt sensation
symptom symptoms patient complaint""".split())

# words the parser may legitimately use that won't appear verbatim in the patient's text
PARSER_SYNONYMS = {
    "palpi": {"pound", "racin", "heart", "flutt", "thump", "beati"},
    "grief": {"loss", "death", "bereav", "mourn", "heart", "died", "passe"},
    "chill": {"cold", "chill", "cover", "warm"},
    "thirs": {"drink", "water", "thirs"},
    "anxie": {"anxio", "worry", "panic", "fear", "nervo"},
    "insom": {"sleep", "awake", "wake"},
    "restl": {"restl", "cannot", "still"},
}

SYMPTOM_GROUPS = ("physical", "psychological", "general")


def content_stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in STOP}


def is_supported(item_text: str, raw_stems: set) -> bool:
    stems = content_stems(item_text)
    if not stems:
        return True
    for s in stems:
        if s in raw_stems:
            return True
        if raw_stems & PARSER_SYNONYMS.get(s, set()):
            return True
    return False


def finalise_categorised(cat: dict, raw_text: str, rubric: dict | None = None) -> dict:
    """Verify parser output against the patient's own words and attach code-made decisions.

    Note: like the demo, this mutates and returns ``cat``.
    """
    raw_stems = content_stems(raw_text)
    raw_l = (raw_text or "").lower()
    unverified = []
    for k in SYMPTOM_GROUPS:
        clean = []
        for s in cat.get(k, []) or []:
            if not isinstance(s, dict) or not str(s.get("symptom", "")).strip():
                continue
            s = {
                "symptom": str(s["symptom"]).strip(),
                "worse":  [m for m in (s.get("worse") or [])  if is_supported(m, raw_stems)],
                "better": [m for m in (s.get("better") or []) if is_supported(m, raw_stems)],
            }
            if not is_supported(s["symptom"], raw_stems):
                s["unverified"] = True
                unverified.append(s["symptom"])
            clean.append(s)
        cat[k] = clean

    # case-wide modalities: validate, then pull out any modality the parser copied onto
    # most symptoms (it describes the whole case, not each symptom)
    cm = cat.get("case_modalities") or {}
    case_mod = {pol: [m for m in (cm.get(pol) or []) if isinstance(m, str) and is_supported(m, raw_stems)]
                for pol in ("worse", "better")}
    syms_all = [s for k in SYMPTOM_GROUPS for s in cat[k]]
    if len(syms_all) >= 3:
        for pol in ("worse", "better"):
            counts = {}
            for s in syms_all:
                for m in s[pol]:
                    counts[m.strip().lower()] = counts.get(m.strip().lower(), 0) + 1
            spread = {m for m, n in counts.items() if n >= 3 and n >= 0.5 * len(syms_all)}
            # sorted() for a stable order; the demo iterated the set directly
            for m in sorted(spread):
                if m not in [x.lower() for x in case_mod[pol]]:
                    case_mod[pol].append(m)
            for s in syms_all:
                s[pol] = [m for m in s[pol] if m.strip().lower() not in spread]
    cat["case_modalities"] = case_mod

    causes = [c for c in (cat.get("causation") or []) if is_supported(c, raw_stems)]
    if re.search(r"\bgrie(f|ving)|\bloss\b|bereave|death of|died|passed away|mourning|heartbreak", raw_l) \
            and not any("grief" in c.lower() for c in causes):
        causes.insert(0, "grief from loss")
    if re.search(r"\bshock\b", raw_l) and not any("shock" in c.lower() for c in causes):
        causes.append("emotional shock")
    if re.search(r"\bfright|\bscare[d]?\b|\bterror", raw_l) and not any("fright" in c.lower() for c in causes):
        causes.append("fright")
    causes = list(dict.fromkeys(c.strip() for c in causes if c.strip()))
    low = [c.lower() for c in causes]
    cat["causation"] = [c for c, l in zip(causes, low)
                        if not any(l != o and re.search(rf"\b{re.escape(l)}\b", o) for o in low)]
    cat["concomitants"] = [c for c in (cat.get("concomitants") or []) if is_supported(c, raw_stems)]
    cat["unverified"] = unverified
    cat.pop("miasm_detail", None)

    m = assess_miasm(cat, raw_text, rubric)
    cat["miasm"] = m["label"]
    cat["miasm_detail"] = m
    cat["red_flags"] = detect_red_flags(raw_text, cat)
    cat["_raw_text"] = raw_text
    return cat


def refresh_derived(cat: dict, rubric: dict | None = None) -> dict:
    """Recompute code-made decisions (miasm, red flags) after the doctor edits symptoms."""
    cat = dict(cat)
    raw = cat.get("_raw_text", "")
    m = assess_miasm(cat, raw, rubric)
    cat["miasm"], cat["miasm_detail"] = m["label"], m
    cat["red_flags"] = detect_red_flags(raw, cat)
    return cat


def empty_case() -> dict:
    """Skeleton for structured (no-AI) entry — same shape the parser returns."""
    return {"physical": [], "psychological": [], "general": [],
            "case_modalities": {"worse": [], "better": []},
            "causation": [], "concomitants": [], "clinical_summary": ""}
