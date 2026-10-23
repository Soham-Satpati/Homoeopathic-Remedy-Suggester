"""The new engine must make exactly the decisions the demo makes.

Every test feeds identical inputs to the demo's own functions (loaded unedited from
horus3_1_streamlit.py) and to the new engine, and requires identical output: parsing
checks, red flags, miasm, dataset loading, live learning, remedy ranking in both matching
modes, report assembly with and without AI text, symptom suggestions and the PDF.
"""

from __future__ import annotations

import copy
import io
from datetime import datetime

import pandas as pd
import pytest
from cases import (NARRATION_REPLY, RED_FLAG_TEXTS, all_cases, parser_reply, saved_patients)
from conftest import make_engine
from legacy import FakeSentenceTransformer, reset_legacy_dataset

from horus import prompts
from horus.engine import HorusEngine
from horus.dataset import learning_payload
from horus.miasm import assess_miasm
from horus.parsing import finalise_categorised
from horus.safety import detect_red_flags

CASES = all_cases()
CASE_IDS = [c["id"] for c in CASES]
LEARNING = {"no-saved-cases": {}, "with-saved-cases": saved_patients()}


def _norm_case(cat: dict) -> dict:
    """The demo appends spread modalities in set-iteration order (varies between runs);
    the engine sorts them. Compare case_modalities order-insensitively, all else exactly."""
    out = copy.deepcopy(cat)
    for pol in ("worse", "better"):
        out["case_modalities"][pol] = sorted(out["case_modalities"][pol])
    return out


def _set_mode(legacy, semantic: bool):
    legacy.system_data = {"model": FakeSentenceTransformer(), "clusters": {}} if semantic else None


@pytest.fixture(params=list(LEARNING), ids=list(LEARNING))
def learned(request, legacy, paths):
    """Demo and engine primed with the same saved cases."""
    patients = LEARNING[request.param]
    reset_legacy_dataset(legacy, patients)
    engines = {}

    def engine(semantic: bool, llm=None):
        key = (semantic, id(llm))
        if key not in engines:
            e = make_engine(paths, semantic, llm)
            e.learn_from(patients)
            engines[key] = e
        return engines[key]

    return patients, engine


# ── safety / parsing / miasm ─────────────────────────────────────────────────
@pytest.mark.parametrize("text", RED_FLAG_TEXTS)
def test_red_flags_text_only(legacy, text):
    assert detect_red_flags(text) == legacy.detect_red_flags(text)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_finalise_matches_demo(legacy, case):
    old = legacy.finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    new = finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    assert _norm_case(new) == _norm_case(old)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_miasm_and_flags_on_parsed_case(legacy, case):
    cat = finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    assert assess_miasm(cat, case["raw"]) == legacy.assess_miasm(cat, case["raw"])
    assert detect_red_flags(case["raw"], cat) == legacy.detect_red_flags(case["raw"], cat)


def test_synthetic_cases_hit_the_tricky_paths(paths):
    """Guard against the inputs silently going soft: each synthetic case must exercise its path."""
    e = make_engine(paths, semantic=False)
    by_id = {c["id"]: e.finalise(copy.deepcopy(c["parsed"]), c["raw"]) for c in CASES}
    spread = by_id["grief-shock-spread"]
    assert "evening" in spread["case_modalities"]["worse"]            # pulled up from 4 symptoms
    assert "fear of robbers" in spread["unverified"]                  # not in the patient's words
    assert spread["causation"][0].startswith("grief") and "emotional shock" in spread["causation"]
    assert not spread["miasm"].startswith("Provisional")              # family history present
    assert [f["title"] for f in by_id["left-chest-red-flag"]["red_flags"]] == ["Possible heart emergency"]
    knee = by_id["fright-right-knee"]
    assert "full moon" not in knee["physical"][0]["worse"]            # unsupported modality dropped
    assert "thunderstorms" not in knee["case_modalities"]["worse"]
    assert "fright" in knee["causation"]
    assert len(by_id["multi-flag"]["red_flags"]) >= 2
    assert e.report("T", by_id["no-match"])["primaryRemedy"]["name"] == "No dataset match"


# ── dataset & live learning ──────────────────────────────────────────────────
def test_base_dataset_matches_demo(legacy, paths):
    from horus.dataset import load_base_dataset
    known, s2r, source = legacy.load_all_dataset_symptoms()
    ds = load_base_dataset(paths)
    assert ds.s2r == s2r and set(ds.known) == known and ds.source == source


@pytest.mark.parametrize("which", list(LEARNING))
def test_live_dataset_matches_demo(legacy, paths, which):
    from horus.dataset import build_live_dataset, load_base_dataset
    patients = LEARNING[which]
    legacy.load_patients = lambda: patients
    payload = legacy._learning_payload()
    assert learning_payload(patients) == payload
    k, s2r, label, n = legacy.build_live_dataset(payload)
    ds = build_live_dataset(load_base_dataset(paths), payload)
    assert (ds.s2r, set(ds.known), ds.source, ds.live_cases) == (s2r, k, label, n)


# ── ranking & reports ────────────────────────────────────────────────────────
@pytest.mark.parametrize("semantic", [False, True], ids=["lexical", "semantic"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_ranking_matches_demo(legacy, learned, case, semantic):
    _, engine = learned
    _set_mode(legacy, semantic)
    cat = finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    old = legacy.rank_remedies_from_dataset(copy.deepcopy(cat), legacy._s2r_unified, top_n=10)
    new = engine(semantic).rank(copy.deepcopy(cat), top_n=10)
    assert new == old


def test_semantic_mode_is_really_exercised(legacy, learned):
    """The stand-in embedder must produce real matches, or the semantic tests prove nothing."""
    _, engine = learned
    cat = finalise_categorised(copy.deepcopy(CASES[0]["parsed"]), CASES[0]["raw"])
    ranked = engine(True).rank(cat)
    assert ranked and ranked[0]["coverage"] >= 2


@pytest.mark.parametrize("semantic", [False, True], ids=["lexical", "semantic"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_report_without_ai_matches_demo(legacy, learned, gemini, case, semantic):
    """No model: the demo's Gemini call fails and it falls back to dataset-only text."""
    _, engine = learned
    _set_mode(legacy, semantic)
    cat = finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    old = legacy.gemini_report("PT-X", copy.deepcopy(cat))
    new = engine(semantic).report("PT-X", copy.deepcopy(cat))
    assert new == old


@pytest.mark.parametrize("semantic", [False, True], ids=["lexical", "semantic"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_report_with_ai_text_matches_demo(legacy, learned, gemini, fake_llm, case, semantic):
    """Same narration reply → same filtering (banned remedies, potency, unsupported claims)."""
    _, engine = learned
    _set_mode(legacy, semantic)
    cat = finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    gemini.replies = [NARRATION_REPLY]
    old = legacy.gemini_report("PT-X", copy.deepcopy(cat), extra_label="[BASE]")
    eng = engine(semantic, fake_llm)
    fake_llm.replies = [NARRATION_REPLY]
    new = eng.report("PT-X", copy.deepcopy(cat), extra_label="[BASE]")
    assert new == old
    if new.get("top10Remedies"):
        # the model saw exactly the prompt the demo sent Gemini
        g_system, g_prompt, g_cfg = gemini.calls[-1]
        call = fake_llm.calls[-1]
        assert (call["system"], call["prompt"]) == (g_system, g_prompt)
        assert (call["temperature"], call["max_tokens"]) == (
            g_cfg.kw["temperature"], g_cfg.kw["max_output_tokens"])
        assert new["_removed"], "canned reply should have triggered the filters"


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_categorise_matches_demo(legacy, gemini, fake_llm, paths, case):
    """Same parser reply → same verified case; same prompt and settings sent to the model."""
    reply = parser_reply(case["parsed"])
    gemini.replies = [reply]
    old = legacy.gemini_categorise(case["raw"], mode="case_studies")
    e = make_engine(paths, semantic=False, llm=fake_llm)
    fake_llm.replies = [reply]
    new = e.categorise(case["raw"], mode="case_studies")
    assert _norm_case(new) == _norm_case(old)
    g_system, g_prompt, g_cfg = gemini.calls[-1]
    call = fake_llm.calls[-1]
    assert call["system"] == g_system and call["prompt"] == g_prompt
    assert (call["temperature"], call["max_tokens"]) == (0.0, 3000) == (
        g_cfg.kw["temperature"], g_cfg.kw["max_output_tokens"])


def test_prompts_identical_to_demo(legacy):
    for mode in ("case_studies", "keynote"):
        FakeGeminiCapture = []
        legacy_g = legacy.genai.GenerativeModel
        try:
            class Capture:
                def __init__(self, model_name=None, system_instruction=None):
                    FakeGeminiCapture.append(system_instruction)

                def generate_content(self, *a, **k):
                    raise RuntimeError("stop")
            legacy.genai.GenerativeModel = Capture
            with pytest.raises(RuntimeError):
                legacy.gemini_categorise("x", mode=mode)
        finally:
            legacy.genai.GenerativeModel = legacy_g
        assert prompts.categorise_system(mode) == FakeGeminiCapture[0]
    assert prompts.AGENT_SAFEGUARDS == legacy.AGENT_SAFEGUARDS


def test_comparative_and_followups_match_demo(legacy, gemini, fake_llm, paths):
    e = make_engine(paths, semantic=False, llm=fake_llm)
    cat = e.finalise(copy.deepcopy(CASES[0]["parsed"]), CASES[0]["raw"])
    base = e.report("P", cat)
    enh = e.report("P", cat)
    gemini.replies = ["  Both plans agree.  "]
    fake_llm.replies = ["  Both plans agree.  "]
    assert e.comparative_summary(base, enh) == legacy.gemini_comparative_summary(base, enh)
    assert fake_llm.calls[-1]["system"] == gemini.calls[-1][0]
    assert fake_llm.calls[-1]["prompt"] == gemini.calls[-1][1]

    reply = '["When is it worse?", " ", "Does warmth help?", "Q3", "Q4"]'
    gemini.replies = [reply]
    fake_llm.replies = [reply]
    assert e.followups("Main complaint: headache") == legacy.gemini_followups("Main complaint: headache")


# ── co-occurrence suggestions ────────────────────────────────────────────────
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_suggestions_match_demo(legacy, learned, case):
    patients, engine = learned
    legacy.load_patients = lambda: patients
    cp_old = legacy.build_live_conditional_probabilities(legacy._learning_payload())
    cat = finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    syms = [s["symptom"] for k in ("physical", "psychological", "general") for s in cat.get(k, [])]
    old_per = legacy.get_suggestions_per_symptom(syms, cp_old.copy(), top_n=10)
    old_top = legacy.get_top_suggestions(syms, cp_old.copy(), top_n=10)
    new_per, new_top = engine(False).suggestions(cat, top_n=10)
    assert (new_per, new_top) == (old_per, old_top)


def test_suggestions_on_dataset_symptoms(legacy, learned):
    """Real cases rarely use exact dataset wording, so also query with dataset symptoms
    that are guaranteed to have co-occurrence rows."""
    patients, engine = learned
    legacy.load_patients = lambda: patients
    cp_old = legacy.build_live_conditional_probabilities(legacy._learning_payload())
    syms = list(pd.unique(cp_old["Symptom1"]))[:6] + list(pd.unique(cp_old["Symptom2"]))[-3:]
    cat = {"physical": [{"symptom": s, "worse": [], "better": []} for s in syms]}
    old = (legacy.get_suggestions_per_symptom(syms, cp_old.copy(), 10),
           legacy.get_top_suggestions(syms, cp_old.copy(), 10))
    new = engine(False).suggestions(cat, 10)
    assert new == old
    assert any(new[0].values()) and new[1], "query should produce suggestions"


def test_live_conditional_probabilities_match_demo(legacy, paths):
    from horus.probabilities import build_live_conditional_probabilities, load_conditional_probabilities
    patients = saved_patients()
    legacy.load_patients = lambda: patients
    payload = legacy._learning_payload()
    old = legacy.build_live_conditional_probabilities(payload)
    new = build_live_conditional_probabilities(load_conditional_probabilities(paths), payload)
    pd.testing.assert_frame_equal(new, old)


# ── enhanced case + PDF ──────────────────────────────────────────────────────
def test_pdf_identical_to_demo(legacy, learned, gemini):
    """Byte-identical PDFs (reportlab invariant mode, fixed date), including the enhanced
    sections B–D the demo adds after symptom suggestions."""
    import reportlab.rl_config as rl
    _, engine = learned
    e = engine(False)
    case = CASES[0]
    cat = e.finalise(copy.deepcopy(case["parsed"]), case["raw"])
    rpt = e.report("PT-1", cat)
    selected = {"weakness of memory": {"worse": ["evening"], "better": [], "avg_prob": 0.42,
                                       "triggered_by": "palpitations"}}
    enh_cat, added = e.enhanced_case(cat, selected)
    enh_rpt = e.report("PT-1", enh_cat)
    details = {"name": "Test Patient", "age": "40", "gender": "Female"}
    fixed = datetime(2026, 10, 23, 12, 0)

    class FrozenDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    old_inv, old_dt = rl.invariant, legacy.datetime
    rl.invariant = 1
    legacy.datetime = FrozenDT
    try:
        for kwargs in ({}, {"report_enhanced": enh_rpt, "added_symptoms": added,
                            "comparative_summary": "Both plans agree."}):
            old = legacy.build_pdf("PT-1", cat, rpt, patient_details=details, **kwargs)
            new = e.pdf("PT-1", cat, rpt, patient_details=details, generated_on=fixed, **kwargs)
            assert new == old
    finally:
        rl.invariant, legacy.datetime = old_inv, old_dt

    from pypdf import PdfReader
    text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(new)).pages)
    assert "Test Patient" in text and "D. Comparative Clinical Summary" in text


# ── with the real all-MiniLM-L6-v2 (runs when scripts/fetch_models.py has been run) ──
def _real_model_dir():
    from horus.config import Paths
    d = Paths().embedding_model_dir
    return d if (d / "config.json").exists() else None


@pytest.mark.skipif(_real_model_dir() is None, reason="embedding model not fetched (run scripts/fetch_models.py)")
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_ranking_with_real_model_matches_demo(legacy, paths, case):
    """The demo's SentenceTransformer and the engine's local-only loader, same weights."""
    from sentence_transformers import SentenceTransformer

    from horus.embeddings import load_embedder
    patients = saved_patients()
    reset_legacy_dataset(legacy, patients)
    # The engine pins the CPU so every clinic machine ranks identically. Run the demo on the CPU
    # too: on Apple Silicon it would otherwise pick the GPU (MPS), whose arithmetic differs slightly.
    legacy.system_data = {"model": SentenceTransformer(str(_real_model_dir()), device="cpu"), "clusters": {}}
    e = HorusEngine(paths, embedder=load_embedder(_real_model_dir()), use_disk_cache=False)
    e.learn_from(patients)
    cat = finalise_categorised(copy.deepcopy(case["parsed"]), case["raw"])
    old = legacy.rank_remedies_from_dataset(copy.deepcopy(cat), legacy._s2r_unified, top_n=10)
    new = e.rank(copy.deepcopy(cat))
    assert new == old
    assert e.matching_mode == "semantic"
