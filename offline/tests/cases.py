"""Test inputs: the real saved cases plus synthetic cases aimed at the tricky code paths."""

from __future__ import annotations

import copy
import json

from legacy import REPO

PATIENT_FILES = ("data/horus3_patients.json", "Horus_V2/patients.json")


def saved_patients() -> dict:
    """Every patient record in the repo's sample patient files (4 real cases)."""
    out = {}
    for f in PATIENT_FILES:
        out.update(json.loads((REPO / f).read_text(encoding="utf-8")))
    return out


def _parser_view(cat: dict) -> dict:
    """What the parser itself returned: drop the fields finalise_categorised adds."""
    keep = ("physical", "psychological", "general", "case_modalities", "causation",
            "concomitants", "clinical_summary")
    return {k: copy.deepcopy(cat[k]) for k in keep if k in cat}


def _s(sym, worse=(), better=()):
    return {"symptom": sym, "worse": list(worse), "better": list(better)}


SYNTHETIC = [
    {   # grief + shock, history present, unsupported symptom, modality spread across symptoms
        "id": "grief-shock-spread",
        "raw": ("Since her mother died last spring she has been weeping alone, sighing often, with a lump "
                "in the throat. The shock left her sleepless. Headache is worse in the evening and the "
                "sighing is worse in the evening; the weeping is worse in the evening too. Family history "
                "of asthma. She craves salt and is chilly."),
        "parsed": {
            "physical": [_s("headache", ["evening"]), _s("lump in the throat", ["evening"])],
            "psychological": [_s("weeping alone", ["evening"]), _s("sighing", ["evening"]),
                              _s("fear of robbers")],
            "general": [_s("craving for salt"), _s("chilly patient"), _s("sleepless")],
            "case_modalities": {"worse": [], "better": []},
            "causation": ["grief", "grief from loss of mother"],
            "concomitants": [],
            "clinical_summary": "Grief after mother's death with weeping and sighing.",
        },
    },
    {   # left-sided chest pain → red flag; right-sided dataset rows must be rejected
        "id": "left-chest-red-flag",
        "raw": ("Stitching pain in the left side of the chest going to the left arm with sweating, "
                "worse lying on the left side, better sitting up. Anxious, fears she might die. "
                "Thirsty for cold water."),
        "parsed": {
            "physical": [_s("stitching pain left chest radiating to left arm", ["lying on left side"],
                            ["sitting up"]), _s("sweating with chest pain")],
            "psychological": [_s("anxiety with fear of death")],
            "general": [_s("thirst for cold water")],
            "case_modalities": {"worse": [], "better": []},
            "causation": [],
            "concomitants": ["sweating with chest pain"],
            "clinical_summary": "Left chest pain to arm with fear of death.",
        },
    },
    {   # fright cause, right-sided complaint, unsupported modality, case-wide modalities
        "id": "fright-right-knee",
        "raw": ("After a fright in a road accident, the right knee became swollen and stiff. Pain is "
                "worse from first motion and in damp weather, better from continued motion and warmth. "
                "All complaints are worse at night. Restless and irritable."),
        "parsed": {
            "physical": [_s("right knee swollen and stiff", ["first motion", "damp weather", "full moon"],
                            ["continued motion", "warmth"])],
            "psychological": [_s("restless"), _s("irritable")],
            "general": [],
            "case_modalities": {"worse": ["at night", "thunderstorms"], "better": []},
            "causation": ["fright", "road accident"],
            "concomitants": [],
            "clinical_summary": "Right knee stiffness after fright, worse first motion.",
        },
    },
    {   # nothing in the dataset should match → "No dataset match" path
        "id": "no-match",
        "raw": "Zzyzx qwerty plorf.",
        "parsed": {
            "physical": [_s("zzyzx qwerty")], "psychological": [], "general": [],
            "case_modalities": {"worse": [], "better": []}, "causation": [], "concomitants": [],
            "clinical_summary": "",
        },
    },
    {   # suicidal ideation + fever with stiff neck → two red flags; tubercular keywords
        "id": "multi-flag",
        "raw": ("High fever with a stiff neck and confusion since yesterday. He says he wants to die. "
                "Desires open air, worse in a closed room, craves cold milk, catches cold easily, "
                "recurrent chronic coughs since childhood."),
        "parsed": {
            "physical": [_s("high fever with stiff neck"), _s("chronic cough")],
            "psychological": [_s("wants to die"), _s("confusion")],
            "general": [_s("desire for open air"), _s("craving cold milk"), _s("catches cold easily")],
            "case_modalities": {"worse": ["closed room"], "better": ["open air"]},
            "causation": [],
            "concomitants": [],
            "clinical_summary": "Fever, stiff neck, confusion and suicidal words.",
        },
    },
]


def all_cases() -> list[dict]:
    """[{id, raw, parsed}] — real saved cases first, then synthetic."""
    out = []
    for pid, rec in saved_patients().items():
        for i, c in enumerate(rec.get("cases", []) if isinstance(rec, dict) else rec):
            out.append({"id": f"{pid}#{i}", "raw": c["raw_symptoms"], "parsed": _parser_view(c["categorised"])})
    return out + copy.deepcopy(SYNTHETIC)


RED_FLAG_TEXTS = [
    "crushing chest pressure spreading to the jaw with sweating",
    "sudden weakness on one side of the body and slurred speech",
    "she is gasping, lips turning blue",
    "has been thinking of suicide",
    "vomiting blood since morning",
    "fever with neck stiffness and a non-blanching rash",
    "fainted twice at work",
    "mild headache in the morning, better after tea",
    "chest feels heavy, palpitations at night",
    "heart pounding with fear she will die",
    "itching eruptions between the fingers",
    "",
]

# Canned model replies, shared by the demo (via FakeGemini) and the new engine (via FakeLLM).
NARRATION_REPLY = json.dumps({
    "caseEssence": "Grief and weeping are central. Give Ignatia 200C twice daily.",
    "primaryWhy": "It matched the weeping and the evening aggravation. Natrum muriaticum is also similar.",
    "rationales": [
        {"rank": 1, "rationale": "Covers weeping alone. Its blood pressure effect is notable."},
        {"rank": 2, "rationale": "Matches the headache. Take 6 pills of it."},
        {"rank": "x", "rationale": "ignored"},
        {"rank": 3, "rationale": "Covers craving for salt and chilly patient strongly."},
    ],
    "differentiation": "Rank 1 covers emotions best; Sepia would be an alternative.",
    "monitoringPoints": ["weeping alone", "heart rate", "sleeplessness", "repeat every 2 hours"],
})


def parser_reply(parsed: dict) -> str:
    return json.dumps(parsed, ensure_ascii=False)
