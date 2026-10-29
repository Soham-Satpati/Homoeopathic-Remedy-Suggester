"""Production matching fixes (1.4): one name per remedy, time-of-day modalities, and symptom
suggestions for symptoms not worded exactly like the books."""

from __future__ import annotations

import pytest
from conftest import make_engine

from horus.config import Paths
from horus.engine import HorusEngine
from horus.ranking import time_conflict

HEADACHE_CASE = {
    "physical": [{"symptom": s, "worse": [], "better": []}
                 for s in ("recurrent headache in the forehead", "heaviness of head", "occasional dizziness")],
    "psychological": [], "general": [],
    "case_modalities": {"worse": ["afternoon", "prolonged screen time"], "better": ["resting in a quiet room"]},
}


def _remedy_spellings(ds):
    seen = {}
    for rems in ds.s2r.values():
        for r in rems:
            seen.setdefault(r.casefold(), set()).add(r)
    return seen


def test_one_name_per_remedy_across_books(paths):
    demo = make_engine(paths, semantic=False)                    # the demo keeps both spellings
    assert any(len(v) > 1 for v in _remedy_spellings(demo.dataset).values())
    e = HorusEngine(paths)
    spell = _remedy_spellings(e.dataset)
    assert all(len(v) == 1 for v in spell.values())
    assert "Gelsemium" in spell["gelsemium"]
    # a rubric listing both spellings keeps the higher score under one name
    for sym, rems in demo.dataset.s2r.items():
        low = {}
        for r, sc in rems.items():
            low.setdefault(r.casefold(), []).append(sc)
        dup = {k: max(v) for k, v in low.items() if len(v) > 1}
        if dup:
            merged = {r.casefold(): sc for r, sc in e.dataset.s2r[sym].items()}
            assert all(abs(merged[k] - v) < 1e-9 for k, v in dup.items())
            break
    # a remedy is never listed twice in a ranking
    names = [r["name"].casefold() for r in e.rank(HEADACHE_CASE, top_n=30)]
    assert len(names) == len(set(names))


def test_saved_cases_use_the_dataset_spelling(paths):
    e = HorusEngine(paths)
    e.learn_from({"P1": {"details": {}, "cases": [{
        "categorised": {"physical": [{"symptom": "zz unique learned symptom"}]},
        "top_remedies": ["gelsemium", "NATRUM MURIATICUM"]}]}})
    rems = e.dataset.s2r["zz unique learned symptom"]
    assert set(rems) == {"Gelsemium", next(r for r in e.dataset.remedy_universe if r.casefold() == "natrum muriaticum")}


@pytest.mark.parametrize("p,d,bad", [
    ("worse afternoon", "worse mornings", True),
    ("worse afternoon", "worse in the afternoon, 4 to 8 p.m.", False),
    ("worse at night", "worse evening and night", False),
    ("worse noon", "worse afternoon", True),
    ("worse afternoon", "worse from motion", False),
    ("worse in the morning on waking", "worse evening", True),
])
def test_time_of_day_conflict(p, d, bad):
    assert time_conflict(p, d) is bad


def test_afternoon_never_matches_another_time(paths):
    e = HorusEngine(paths)
    for r in e.rank(HEADACHE_CASE, top_n=30):
        for ev in r["evidence"]:
            if ev["patient"].startswith("Worse afternoon"):
                assert not time_conflict("afternoon", ev["dataset"]), ev


def test_suggestions_for_free_text_symptoms(paths):
    demo = make_engine(paths, semantic=False)
    per, top = demo.suggestions(HEADACHE_CASE)
    assert not top                                              # the demo needs exact wording
    e = make_engine(paths, semantic=False, books=("rheumatic", "case_studies", "guiding_symptoms_1"),
                    demo_parity=False)
    per, top = e.suggestions(HEADACHE_CASE)
    assert top, "free-text symptoms should be mapped to the books' symptoms"
    patient = {s["symptom"].lower() for s in HEADACHE_CASE["physical"]}
    vias = {x["via"] for x in top if "via" in x}
    assert vias and all(x["symptom"].lower() not in patient | vias for x in top)
    assert all(x["triggered_by"] in {s["symptom"] for s in HEADACHE_CASE["physical"]} for x in top)
    # exact wording still behaves exactly as before (no "via")
    exact_sym = next(iter(e._suggestion_index().forward))
    cat = {"physical": [{"symptom": exact_sym}], "psychological": [], "general": []}
    per, top = e.suggestions(cat)
    assert top and per[exact_sym]
    assert all("remedies" in x for x in top)
    # a suggestion pointing to one of the case's leading remedies is ranked before an equal one that doesn't
    leaders = {r["name"] for r in e.rank(HEADACHE_CASE, top_n=5)}
    _, top = e.suggestions(HEADACHE_CASE)
    assert all(set(x["remedies"]) <= leaders for x in top)


def _real_model_dir():
    d = Paths().embedding_model_dir
    return d if (d / "config.json").exists() else None


@pytest.mark.skipif(_real_model_dir() is None, reason="embedding model not fetched (run scripts/fetch_models.py)")
def test_headache_case_with_real_model(paths, capsys):
    """The case from the 1.3 report, with the shipped embedding model."""
    from horus.embeddings import load_embedder
    e = HorusEngine(paths, embedder=load_embedder(_real_model_dir()), use_disk_cache=False)
    per, top = e.suggestions(HEADACHE_CASE)
    ranked = e.rank(HEADACHE_CASE)
    with capsys.disabled():
        print("\n::notice title=headache suggestions::" + " | ".join(
            f"{x['symptom'][:60]} ({x['avg_prob']:.0%}, {x['triggered_by']} via {x.get('via', '=')[:40]})"
            for x in top))
        print("::notice title=headache ranking::" + " | ".join(
            f"{r['name']} {r['coverage']}/{r['total_items']}: " + "; ".join(
                f"{ev['patient']}≈{ev['dataset'][:40]} {ev['similarity']}" for ev in r["evidence"])
            for r in ranked))
    assert top and len(top) >= 5
    names = [r["name"].casefold() for r in ranked]
    assert len(names) == len(set(names))
