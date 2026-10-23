"""Deterministic red-flag screen. Ported unchanged from ``detect_red_flags``."""

from __future__ import annotations

import re

RED_FLAG_RULES = [
    {
        "title": "Possible heart emergency",
        "all": [r"\bchest\b|\bprecordi|\bheart\b|\bcardiac\b"],
        "any": [r"\bpain|\bpressure|\btight|\bsqueez|\bcrush|\bstitch|\bheav"],
        "plus_any": [r"left arm|\barm\b|\bjaw\b|\bradiat|\bspread|\bsweat|breathless|short(ness)? of breath"
                     r"|\bfaint|\bpalpitat|\bpounding|fear (that )?(she|he|they|i) (will|would|might) die"
                     r"|fear of death|fear of dying"],
        "advice": "Chest pain or pressure with spreading to the arm/jaw, sweating, breathlessness, "
                  "palpitations or a sense of impending death can be a heart attack. Arrange urgent "
                  "medical assessment (ECG and cardiac blood tests) now — before or alongside any "
                  "homeopathic treatment.",
    },
    {
        "title": "Possible stroke",
        "any": [r"face (droop|drooping)|facial droop|slurred speech|sudden(ly)? (weakness|numbness)"
                r"|one side of (the )?body|cannot (speak|lift)|sudden (loss of )?vision|worst headache"],
        "advice": "Sudden one-sided weakness/numbness, facial droop, speech or vision loss, or a sudden "
                  "worst-ever headache needs emergency care immediately.",
    },
    {
        "title": "Severe breathing difficulty",
        "any": [r"can'?t breathe|cannot breathe|gasping|blue lips|lips turn(ing)? blue|severe breathless"
                r"|struggling to breathe|throat (is )?(closing|swelling)"],
        "advice": "Severe breathlessness, blue lips or throat swelling needs emergency care immediately.",
    },
    {
        "title": "Risk of self-harm",
        "any": [r"suicid|kill (myself|herself|himself)|end (my|her|his) life|self[- ]harm|want(s)? to die"
                r"|no reason to live"],
        "advice": "Thoughts of suicide or self-harm need same-day support from a mental-health "
                  "professional or crisis line. Do not leave the patient without support.",
    },
    {
        "title": "Serious bleeding",
        "any": [r"vomit(ing)? blood|blood in (vomit|stool)|black (tarry )?stool|coughing (up )?blood"
                r"|heavy bleeding|bleeding (in|during) pregnan"],
        "advice": "Vomiting or coughing blood, black stools or heavy bleeding needs urgent medical care.",
    },
    {
        "title": "Possible meningitis / serious infection",
        "all": [r"\bfever"],
        "any": [r"stiff neck|neck stiffness|confus|rash that does not fade|non-blanching|seizure|convuls"
                r"|unconscious|unresponsive"],
        "advice": "Fever with neck stiffness, confusion, a non-fading rash or seizures needs emergency care.",
    },
    {
        "title": "Loss of consciousness / seizure",
        "any": [r"\bseizure|\bconvuls|lost consciousness|loss of consciousness|passed out|\bfaint(ed|ing)\b"],
        "advice": "Fainting, loss of consciousness or seizures need prompt medical evaluation.",
    },
]


def detect_red_flags(raw_text: str, categorised: dict | None = None) -> list:
    """Return [{title, advice, trigger}] for red-flag patterns in the case. Pure code."""
    parts = [raw_text or ""]
    if categorised:
        for k in ("physical", "psychological", "general"):
            for s in categorised.get(k, []):
                parts.append(" ".join([s.get("symptom", "")] + s.get("worse", []) + s.get("better", [])))
        parts.extend(categorised.get("causation", []))
    text = " ".join(parts).lower()

    flags = []
    for rule in RED_FLAG_RULES:
        ok = all(re.search(p, text) for p in rule.get("all", []))
        ok = ok and all(re.search(p, text) for p in rule.get("any", []))
        ok = ok and all(re.search(p, text) for p in rule.get("plus_any", []))
        if ok:
            hit = None
            for p in rule.get("plus_any", []) + rule.get("any", []):
                m = re.search(p, text)
                if m:
                    hit = m.group(0)
                    break
            flags.append({"title": rule["title"], "advice": rule["advice"], "trigger": hit or ""})
    return flags
