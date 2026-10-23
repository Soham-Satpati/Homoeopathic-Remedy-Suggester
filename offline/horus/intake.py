"""Patient intake questionnaire. Ported from the demo's patient side (data only, no UI)."""

from __future__ import annotations

PATIENT_QUESTIONS = [
    {"id": "complaint", "label": "Main complaint",
     "q": "What is the main problem that is troubling you?",
     "hint": "Describe it in your own words — there are no wrong answers."},
    {"id": "onset", "label": "Onset",
     "q": "When did it start, and was anything happening in your life around that time?",
     "hint": "For example: after an illness, grief, stress, an injury, a change in weather or diet."},
    {"id": "sensation", "label": "Location & sensation",
     "q": "Where exactly do you feel it, and what does it feel like?",
     "hint": "Burning, throbbing, stitching, aching, pressing, numb… Does it move or spread anywhere?"},
    {"id": "worse", "label": "Worse from",
     "q": "What makes it worse?",
     "hint": "Time of day or night, heat or cold, movement, rest, touch, food, weather, emotions…"},
    {"id": "better", "label": "Better from",
     "q": "What makes it better?",
     "hint": "Warmth, cold, pressure, rest, open air, lying down, eating…"},
    {"id": "mind", "label": "Mind & emotions",
     "q": "How have you been feeling emotionally lately?",
     "hint": "Any fears, worries, irritability, sadness, restlessness? Do you prefer company or being alone?"},
    {"id": "sleep", "label": "Sleep",
     "q": "How is your sleep?",
     "hint": "Trouble falling or staying asleep, waking at a certain hour, dreams, sleeping position."},
    {"id": "appetite", "label": "Appetite, thirst & cravings",
     "q": "Tell us about your appetite, thirst, and any food cravings or dislikes.",
     "hint": "Hot or cold drinks, sweet, salty, spicy, sour foods…"},
    {"id": "thermal", "label": "Temperature",
     "q": "Are you generally a warm person or a chilly person?",
     "hint": "How do you react to hot weather, cold weather, fans, blankets?"},
    {"id": "other", "label": "Other information",
     "q": "Is there anything else you would like the doctor to know?",
     "hint": "Other complaints, medicines you take, past illnesses, family history."},
]


def compose_patient_narrative(questions: list, answers: dict) -> str:
    parts = []
    for q in questions:
        a = answers.get(q["id"], "").strip()
        if a:
            parts.append(f"{q['label']}: {a}")
    return "\n".join(parts)


def intake_record(questions: list, answers: dict, intake_id: str, timestamp: str) -> dict:
    """The intake entry the demo stored under a patient's ``intakes`` list."""
    return {
        "id": intake_id,
        "timestamp": timestamp,
        "qa": [{"question": q["q"], "label": q["label"], "answer": answers.get(q["id"], "").strip()}
               for q in questions],
        "raw_text": compose_patient_narrative(questions, answers),
        "status": "pending",
    }
