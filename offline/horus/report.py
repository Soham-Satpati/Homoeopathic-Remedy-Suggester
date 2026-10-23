"""Report assembly and AI-text filtering. Ported from the demo's section 5.

The ranking is final before any language model sees it. Model text is only slotted in
after three filters: sentences naming a remedy outside the ranking are dropped,
potency/dose wording is dropped, and a rationale may only cite patient symptoms that the
remedy actually matched in the dataset.
"""

from __future__ import annotations

import re

from .miasm import assess_miasm
from .parsing import content_stems

COMMON_REMEDY_NAMES = {
    "aconite", "aconitum", "arsenicum", "pulsatilla", "phosphorus", "sulphur", "lycopodium", "sepia",
    "ignatia", "natrum", "lachesis", "belladonna", "bryonia", "calcarea", "nux vomica", "rhus tox",
    "tuberculinum", "medorrhinum", "syphilinum", "psorinum", "thuja", "silicea", "graphites",
    "causticum", "kali", "carcinosin", "staphysagria", "gelsemium", "cactus", "spigelia", "latrodectus",
    "digitalis", "aurum", "platina", "chamomilla", "cimicifuga", "mercurius", "hepar", "apis",
    "arnica", "hypericum", "opium", "stramonium", "hyoscyamus", "veratrum", "china", "ferrum",
    "baryta", "conium", "nitric acid", "phosphoric acid", "argentum", "anacardium", "carbo veg",
}
POTENCY_RE = re.compile(r"\b\d+\s?(c|ch|x|m|lm|k)\b|\bq\b potency|\bpotenc|\bdos(e|age|ing)\b|"
                        r"\bmother tincture\b|\brepeat(ed)? (every|after)|\bdrops?\b|\bpills?\b|\bglobules?\b",
                        re.I)
# generic words that appear in symptoms but don't identify one (ignored by the evidence check)
EVIDENCE_IGNORE = {"cover", "worse", "bette", "telli", "since", "makes", "sense", "ailme", "recen",
                   "sympt", "sudde", "marke", "tende", "occas", "gener", "overa", "inten", "sharp"}
METHOD_NOTE = ("Remedies, order, scores and miasm are computed from the dataset and rubric; "
               "text explanations are AI-written and filtered.")


def remedy_name_patterns(allowed: list, universe: set) -> list:
    allowed_l = {a.lower() for a in allowed}
    allowed_tokens = {t for a in allowed_l for t in a.split()}
    pats = set()
    for name in universe | COMMON_REMEDY_NAMES:
        n = name.lower().strip()
        if not n or n in allowed_l or n in allowed_tokens:
            continue
        if any(a.startswith(n + " ") or a == n for a in allowed_l):
            continue                     # short form of an allowed remedy, e.g. "ignatia"
        first = n.split()[0]
        if n.count(" ") and first in allowed_tokens:
            continue                     # e.g. "kali carbonicum" vs allowed "kali iodatum"
        if len(n) >= 5:
            pats.add(n)
        if len(first) >= 6 and first not in allowed_tokens:
            pats.add(first)
    return sorted(pats, key=len, reverse=True)


def sanitise_text(text: str, banned_patterns: list, removed: list) -> str:
    """Drop sentences that name a non-listed remedy or give potency/dosage advice."""
    if not text:
        return text
    keep = []
    for sent in re.split(r"(?<=[.!?])\s+", str(text)):
        low = sent.lower()
        bad = next((p for p in banned_patterns if re.search(rf"\b{re.escape(p)}\b", low)), None)
        if bad:
            removed.append(f"mentioned '{bad}', which is not in the dataset ranking")
            continue
        if POTENCY_RE.search(sent):
            removed.append("potency/dosage wording")
            continue
        keep.append(sent)
    return " ".join(keep).strip()


def patient_vocab(categorised: dict) -> set:
    words = []
    for c in categorised.get("causation", []):
        words.append(c)
    for pol in ("worse", "better"):
        words += (categorised.get("case_modalities") or {}).get(pol, [])
    for k in ("physical", "psychological", "general"):
        for s in categorised.get(k, []):
            if not s.get("unverified"):
                words += [s["symptom"]] + s.get("worse", []) + s.get("better", [])
    return content_stems(" ".join(words))


def keep_evidenced(text: str, evidence: list, vocab: set, removed: list) -> str:
    """Drop sentences citing a patient symptom this remedy did NOT match in the dataset."""
    if not text:
        return text
    ev = content_stems(" ".join(f"{e['patient']} {e['dataset']}" for e in evidence))
    keep = []
    for sent in re.split(r"(?<=[.!?])\s+", str(text)):
        cited = (content_stems(sent) & vocab) - EVIDENCE_IGNORE
        if cited - ev:
            removed.append("claims not supported by the remedy's dataset matches")
            continue
        keep.append(sent)
    return " ".join(keep).strip()


def title_case(name: str) -> str:
    return " ".join(w if w.isupper() else w.capitalize() for w in str(name).split())


def narration_inputs(categorised: dict, ranked: list) -> tuple[list[str], str]:
    """The symptom lines and remedy/evidence block the narrator is shown."""
    lines = []
    for c in categorised.get("causation", []):
        lines.append(f"[Causation] ailments from {c}")
    cm = categorised.get("case_modalities") or {}
    if cm.get("worse") or cm.get("better"):
        lines.append(f"[Whole case] worse: {', '.join(cm.get('worse', [])) or '—'} | "
                     f"better: {', '.join(cm.get('better', [])) or '—'}")
    for cat in ("psychological", "general", "physical"):
        for s in categorised.get(cat, []):
            if s.get("unverified"):
                continue
            worse = ", ".join(s.get("worse", [])) or "—"
            better = ", ".join(s.get("better", [])) or "—"
            lines.append(f"[{cat.title()}] {s['symptom']} | worse: {worse} | better: {better}")

    blocks = []
    for i, r in enumerate(ranked, 1):
        ev = "; ".join(f"\"{e['patient']}\" ≈ dataset \"{e['dataset']}\" (sim {e['similarity']:.2f})"
                       for e in r["evidence"][:6])
        blocks.append(f"{i}. {r['name']} — covers {r['coverage']}/{r['total_items']} patient symptoms; "
                      f"evidence: {ev}")
    return lines, "\n".join(blocks)


def assemble_report(categorised: dict, ranked: list, g: dict, universe: set,
                    rubric: dict | None = None) -> dict:
    """Build the report from the dataset ranking; model text is slotted in after validation."""
    banned = remedy_name_patterns([r["name"] for r in ranked], universe)
    removed: list = []

    vocab = patient_vocab(categorised)
    by_rank = {}
    for item in g.get("rationales", []) or []:
        try:
            by_rank[int(item.get("rank"))] = str(item.get("rationale", ""))
        except (TypeError, ValueError, AttributeError):
            continue

    top10 = []
    for i, r in enumerate(ranked, 1):
        auto = "Matched in dataset: " + "; ".join(
            f"{e['patient']} ≈ “{e['dataset']}”" for e in r["evidence"][:3]) + "."
        text = keep_evidenced(sanitise_text(by_rank.get(i, ""), banned, removed),
                              r["evidence"], vocab, removed) or auto
        top10.append({
            "rank": i,
            "name": title_case(r["name"]),
            "role": f"covers {r['coverage']}/{r['total_items']}",
            "rationale": text,
            "keySymptoms": [e["patient"] for e in r["evidence"][:4]],
            "evidence": r["evidence"],
            "datasetScore": r["score"],
            "coverage": r["coverage"],
            "totalItems": r["total_items"],
        })

    pr = ranked[0]
    primary_why = keep_evidenced(sanitise_text(g.get("primaryWhy", ""), banned, removed),
                                 pr["evidence"], vocab, removed) or top10[0]["rationale"]
    mon = [sanitise_text(m, banned, removed) for m in (g.get("monitoringPoints") or [])]
    mon = [m for m in mon if m and not re.search(r"heart rate|blood pressure|ecg|pulse|bpm|test", m, re.I)]

    miasm = categorised.get("miasm_detail") or assess_miasm(categorised, rubric=rubric)
    return {
        "primaryRemedy": {
            "name": title_case(pr["name"]),
            "why": primary_why,
            "keyIndications": [f"{e['patient']} → dataset: “{e['dataset']}”" for e in pr["evidence"][:5]],
        },
        "top10Remedies": top10,
        "miasmaticAnalysis": miasm["text"],
        "miasmLabel": miasm["label"],
        "caseEssence": sanitise_text(g.get("caseEssence", ""), banned, removed)
                       or categorised.get("clinical_summary", ""),
        "remedyRelationships": sanitise_text(g.get("differentiation", ""), banned, removed),
        "monitoringPoints": mon,
        "redFlags": categorised.get("red_flags", []),
        "unverifiedSymptoms": categorised.get("unverified", []),
        "_removed": sorted(set(removed)),
        "_method": METHOD_NOTE,
    }


def no_match_report(categorised: dict, rubric: dict | None = None) -> dict:
    miasm = categorised.get("miasm_detail") or assess_miasm(categorised, rubric=rubric)
    return {
        "top10Remedies": [],
        "primaryRemedy": {
            "name": "No dataset match",
            "why": "No remedies in the dataset matched these symptoms closely enough. "
                   "Rephrase symptoms closer to repertory language or expand the dataset.",
            "keyIndications": [],
        },
        "miasmaticAnalysis": miasm["text"],
        "caseEssence": categorised.get("clinical_summary", ""),
        "remedyRelationships": "",
        "monitoringPoints": [],
        "redFlags": categorised.get("red_flags", []),
        "_warning": "No remedies matched in dataset.",
    }
