"""Benchmark cases with reference parses, written to the demo's parser rules (prompts.py).

These complement the 4 saved cases (whose reference is Gemini's own parse). Each narrative
contains at least one trap the rules call out: polarity ("chilly even in mild weather" is a
general, not a modality), whole-case modalities that must not be copied onto every symptom,
stated causes that must not be dropped, negations, and medication mentions that are not
symptoms.
"""

from __future__ import annotations


def S(sym, worse=(), better=()):
    return {"symptom": sym, "worse": list(worse), "better": list(better)}


def case(id, raw, physical=(), psychological=(), general=(), case_worse=(), case_better=(),
         causation=(), concomitants=()):
    return {"id": id, "raw": raw, "reference": {
        "physical": list(physical), "psychological": list(psychological), "general": list(general),
        "case_modalities": {"worse": list(case_worse), "better": list(case_better)},
        "causation": list(causation), "concomitants": list(concomitants), "clinical_summary": ""}}


REFERENCE_CASES = [
    case("child-teething",
         "My 14-month-old son is teething. He is very irritable and wants to be carried all the time; "
         "he quietens only when carried. One cheek is red and hot, the other pale. He has green, "
         "slimy stools that smell like rotten eggs. Everything is worse in the evening and at night. "
         "He is very thirsty for cold water.",
         physical=[S("one cheek red and hot, the other pale"), S("green slimy stools smelling like rotten eggs")],
         psychological=[S("irritable, wants to be carried", better=["being carried"])],
         general=[S("thirsty for cold water")],
         case_worse=["evening", "night"],
         causation=["teething"]),
    case("menses-grief",
         "Since her father passed away six months ago, her periods have become scanty and late. She "
         "weeps easily but cannot cry in front of others and dislikes consolation. Headache like a nail "
         "driven into the head, better from lying on it. She sighs a lot. She craves salt.",
         physical=[S("menses scanty and late"), S("headache like a nail driven into the head",
                                                  better=["lying on the painful side"])],
         psychological=[S("weeps easily but not in front of others"), S("dislikes consolation"),
                        S("sighs a lot")],
         general=[S("craving for salt")],
         causation=["grief from loss of father"]),
    case("eczema-chilly",
         "Itchy eczema on both hands for three years with thick cracked skin that oozes a sticky "
         "honey-like discharge. The itching is worse from warmth of the bed and better from cold "
         "water. He is chilly even in mild weather and constipated, no urge for days. Uses a steroid "
         "cream sometimes.",
         physical=[S("itchy eczema on hands with thick cracked skin oozing sticky honey-like discharge",
                     worse=["warmth of the bed"], better=["cold water"]),
                   S("constipation without urge for days")],
         general=[S("chilly even in mild weather")]),
    case("acidity-anger",
         "Burning in the stomach and sour belching for two months, started after a big quarrel at work. "
         "Worse after eating spicy food and coffee, better from warm milk. Very irritable, impatient, "
         "and wakes at 3 a.m. thinking about business and cannot sleep again. Sensitive to noise and "
         "light. Chilly.",
         physical=[S("burning in the stomach with sour belching", worse=["spicy food", "coffee"],
                     better=["warm milk"])],
         psychological=[S("irritable and impatient"), S("sensitive to noise and light")],
         general=[S("wakes at 3 a.m. thinking of business and cannot sleep again"), S("chilly")],
         causation=["anger from quarrel at work"]),
    case("asthma-damp",
         "Wheezing and breathlessness that comes every year in the rainy season. Worse in damp weather "
         "and between 2 and 4 in the morning; she has to sit up and lean forward to breathe. Better "
         "in dry weather. Anxious and restless during attacks. Wants frequent small sips of warm water.",
         physical=[S("wheezing and breathlessness", worse=["damp weather", "2 to 4 a.m."],
                     better=["sitting up and leaning forward", "dry weather"])],
         psychological=[S("anxious and restless during attacks")],
         general=[S("frequent small sips of warm water")]),
    case("ankle-injury",
         "Twisted his right ankle playing football yesterday. The ankle is swollen and bruised and "
         "feels sore as if beaten. He says he is fine and does not want anyone to touch him. Pain is "
         "worse from walking and better from lying still.",
         physical=[S("right ankle swollen and bruised, sore as if beaten",
                     worse=["walking", "touch"], better=["lying still"])],
         psychological=[S("says he is fine, does not want to be touched")],
         causation=["injury"]),
    case("fever-sudden",
         "High fever came on suddenly last night after being out in a cold dry wind. Face is red, "
         "skin dry and burning, very restless and frightened, says she is going to die. Great thirst "
         "for cold water. No sweat.",
         physical=[S("high fever with red face and dry burning skin")],
         psychological=[S("restless and frightened, says she is going to die")],
         general=[S("great thirst for cold water"), S("no sweat")],
         causation=["exposure to cold dry wind"],
         concomitants=[]),
    case("migraine-sun",
         "Throbbing right-sided headache that begins at sunrise, peaks at noon and goes by sunset. "
         "Worse from the sun and noise, better in a dark room and from pressure. Vomiting with the "
         "headache. Craves sour things. Hates being in a crowd.",
         physical=[S("throbbing right-sided headache from sunrise to sunset, peaking at noon",
                     worse=["sun", "noise"], better=["dark room", "pressure"]),
                   S("vomiting with the headache")],
         psychological=[S("aversion to crowds")],
         general=[S("craving for sour things")],
         concomitants=["vomiting with the headache"]),
    case("insomnia-fright",
         "Since a road accident she witnessed last month, she has nightmares and wakes with a start. "
         "Afraid to go out alone, palpitations when she hears a car horn. Trembling of hands. Appetite "
         "is poor. All her complaints are worse at night and better in company.",
         physical=[S("trembling of hands"), S("palpitations on hearing a car horn")],
         psychological=[S("afraid to go out alone"), S("nightmares, wakes with a start")],
         general=[S("poor appetite")],
         case_worse=["night"], case_better=["company"],
         causation=["fright from witnessing a road accident"]),
    case("back-stiff",
         "Lower back stiffness for years, worst on first getting up in the morning and after rest, "
         "eases after moving about for a while, and is worse in cold damp weather. Hot water bath "
         "helps a lot. Restless at night, keeps turning in bed. Takes paracetamol when bad. "
         "Thirst for cold milk.",
         physical=[S("lower back stiffness", worse=["first motion after rest", "morning on rising",
                                                     "cold damp weather"],
                     better=["continued motion", "hot bath"])],
         psychological=[S("restless at night, keeps turning in bed")],
         general=[S("thirst for cold milk")]),
]
