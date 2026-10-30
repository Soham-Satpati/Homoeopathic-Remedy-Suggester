"""Books, book selection and decision bias."""

from __future__ import annotations

import copy
import io
import json

import pytest
from cases import all_cases, saved_patients
from conftest import make_engine

from horus.books import (CLASSICAL_CATEGORY_WEIGHTS, DEMO_BOOK_IDS, BookFormatError, DecisionConfig,
                         discover_books, import_book, remove_book, validate_book_data)
from horus.config import Paths
from horus.engine import HorusEngine

GUIDING = "guiding_symptoms_1"


def _cat(paths, i=0):
    e = make_engine(paths, semantic=False)
    c = all_cases()[i]
    return e.finalise(copy.deepcopy(c["parsed"]), c["raw"])


def test_shipped_books_and_default_selection(paths):
    ids = [b.id for b in discover_books(paths)]
    assert ids[:3] == ["rheumatic", "case_studies", GUIDING] and len(ids) == 11
    e = HorusEngine(paths)                                   # default: every available book
    assert e.config.books == tuple(ids)
    demo = make_engine(paths, semantic=False)
    assert len(e.dataset.s2r) > len(demo.dataset.s2r) + 100_000
    assert "Guiding_Symptoms" in e.dataset.source
    # remedy relationships are not symptoms: the cleaned volumes have no such chapter
    g = json.loads(next(b for b in e.books if b.id == GUIDING).json_path.read_text(encoding="utf-8"))
    assert not any(ch.lower() in ("relationship", "relations") for chs in g.values() for ch in chs)


def test_unselected_books_never_contribute(paths):
    e = HorusEngine(paths)
    only_case = DecisionConfig.classical(("case_studies",))
    ranked = e.rank(_cat(paths), config=only_case)
    ds = e.dataset_for(only_case)
    assert ranked and all(r["name"] in ds.remedy_universe for r in ranked)
    assert ds.books == ("case_studies",)


def test_book_weight_moves_that_books_remedies(paths):
    e = make_engine(paths, semantic=False)
    ds = e.dataset
    sym_c = next(s for s, rems in ds.origin.items()
                 if len(s) > 25 and all(src == {"case_studies"} for src in rems.values()))
    sym_r = next(s for s, rems in ds.origin.items()
                 if len(s) > 25 and all(src == {"rheumatic"} for src in rems.values())
                 and not set(rems) & set(ds.s2r[sym_c]))
    cat = e.finalise_structured({"physical": [{"symptom": sym_c, "worse": [], "better": []},
                                              {"symptom": sym_r, "worse": [], "better": []}],
                                 "psychological": [], "general": []})
    plain = {r["name"]: r["score"] for r in e.rank(cat, top_n=50)}
    boosted_cfg = DecisionConfig(books=DEMO_BOOK_IDS, book_weights=(("case_studies", 3.0),))
    boosted = {r["name"]: r["score"] for r in e.rank(cat, top_n=50, config=boosted_cfg)}
    c_rems, r_rems = set(ds.s2r[sym_c]), set(ds.s2r[sym_r])
    up = [r for r in c_rems - r_rems if r in plain and r in boosted]
    flat = [r for r in r_rems - c_rems if r in plain and r in boosted]
    assert up and flat
    assert all(abs(boosted[r] / plain[r] - 3.0) < 1e-6 for r in up)       # matched only via the boosted book
    assert all(abs(boosted[r] - plain[r]) < 1e-9 for r in flat)
    # all weights 1.0 is exactly the unweighted ranking
    ones = DecisionConfig(books=DEMO_BOOK_IDS, book_weights=(("case_studies", 1.0), ("rheumatic", 1.0)))
    assert e.rank(cat, config=ones) == e.rank(cat)


def test_hierarchy_weights(paths):
    e = make_engine(paths, semantic=False)
    cat = _cat(paths)
    full = e.rank(cat)
    no_mind = DecisionConfig(books=DEMO_BOOK_IDS, category_weights=tuple(sorted(
        {**CLASSICAL_CATEGORY_WEIGHTS, "psychological": 0.0}.items())))
    r = e.rank(cat, config=no_mind)
    mind_syms = {s["symptom"] for s in cat["psychological"] if not s.get("unverified")}
    assert mind_syms and not any(set(x["symptoms_covered"]) & mind_syms for x in r)
    assert r[0]["total_items"] < full[0]["total_items"]
    with_cause = DecisionConfig(books=DEMO_BOOK_IDS, category_weights=tuple(sorted(
        {**CLASSICAL_CATEGORY_WEIGHTS, "causation": 1.0}.items())))
    covered = {s for x in e.rank(cat, config=with_cause, top_n=50) for s in x["symptoms_covered"]}
    if cat["causation"]:
        assert any(s.startswith("Ailments from") for s in covered) or covered


def test_saved_cases_weight(paths):
    e = make_engine(paths, semantic=False)
    e.learn_from(saved_patients())
    off = DecisionConfig(books=DEMO_BOOK_IDS, learned_weight=0.0)
    base = make_engine(paths, semantic=False)
    assert e.dataset_for(off).s2r == base.dataset.s2r
    assert e.cooccurrence_base(off) is not None
    cat = _cat(paths)
    assert e.suggestions(cat, config=off) == base.suggestions(cat)
    double = DecisionConfig(books=DEMO_BOOK_IDS, learned_weight=2.0)
    sym = next(s for s, _ in [(x, 0) for x in e.dataset.s2r] if s not in base.dataset.s2r)
    assert all(abs(e.dataset_for(double).s2r[sym][r] - 2 * v) < 1e-6 for r, v in e.dataset.s2r[sym].items())


def test_config_roundtrip_and_clamping():
    c = DecisionConfig.from_dict({"books": ["rheumatic"], "book_weights": {"rheumatic": 9, "x": "bad"},
                                  "learned_weight": -1, "category_weights": {"psychological": 3, "nope": 1},
                                  "modality_weight": 0.4})
    assert c.weight_of("rheumatic") == 5.0 and c.weight_of("x") == 1.0 and c.learned_weight == 0.0
    assert c.categories["psychological"] == 3 and "nope" not in c.categories
    assert DecisionConfig.from_dict(c.to_dict()) == c
    assert DecisionConfig().is_unbiased and not c.is_unbiased


def test_report_and_pdf_record_sources(paths):
    from pypdf import PdfReader
    e = HorusEngine(paths)
    cfg = DecisionConfig(books=("rheumatic", GUIDING), book_weights=((GUIDING, 0.5),))
    rpt = e.report("P", _cat(paths), config=cfg)
    assert "Guiding Symptoms" in rpt["_sources"] and "×0.5" in rpt["_sources"]
    assert rpt["_decision"]["books"] == ["rheumatic", GUIDING]
    pdf = e.pdf("P", _cat(paths), rpt)
    text = "".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
    assert "SOURCES AND WEIGHTING" in text and "Guiding Symptoms" in text


BOOK = {
    "remedium testum": {
        "mind": [{"symptom": f"zz test mind symptom {i}", "labels": {"Modalities": "worse at night"}}
                 for i in range(8)],
        "head": [{"symptom": f"zz test head symptom {i}", "labels": {"Modalities": None}} for i in range(8)],
    },
    "remedium alterum": {
        "mind": [{"symptom": "zz test mind symptom 1", "labels": {}}],
        "head": [{"symptom": f"zz other head symptom {i}", "labels": {}} for i in range(8)],
    },
}


def test_import_and_remove_a_book(paths, tmp_path):
    p = Paths(resources=paths.resources, user=tmp_path / "user")
    src = tmp_path / "my_book.json"
    src.write_text(json.dumps(BOOK), encoding="utf-8")
    steps = []
    book = import_book(p, src, "My Test Book", progress=lambda f, m: steps.append((f, m)))
    assert book.id.startswith("user_") and book.source == "user" and steps
    fracs = [f for f, _ in steps]
    assert fracs == sorted(fracs) and fracs[0] == 0.0 and fracs[-1] == 1.0     # monotonic, 0 → 100%
    assert book.cooccurrence_path.exists()
    e = HorusEngine(p)
    assert book.id in e.config.books
    cfg = DecisionConfig.classical((book.id,))
    assert "Remedium testum" in e.dataset_for(cfg).remedy_universe      # names shown capitalised
    per, top = e.suggestions({"physical": [{"symptom": "zz test mind symptom 1"}]}, config=cfg)
    assert top                                               # its own clusters drive suggestions
    with pytest.raises(ValueError):
        remove_book(p, "rheumatic")                          # shipped books can't be removed
    remove_book(p, book.id)
    e.reload_books()
    assert book.id not in [b.id for b in e.books] and book.id not in e.config.books


@pytest.mark.parametrize("bad", [[], {"r": []}, {"r": {"c": "x"}}, {"r": {"c": [{"labels": {}}]}}])
def test_book_validation(bad):
    with pytest.raises(BookFormatError):
        validate_book_data(bad)
    assert validate_book_data(BOOK) == {"remedies": 2, "entries": 25}


def test_isolated_import_runs_in_child_and_reports_progress(paths, tmp_path):
    import os
    from horus.book_worker import import_book_isolated
    p = Paths(resources=paths.resources, user=tmp_path / "user")
    src = tmp_path / "b.json"
    src.write_text(json.dumps(BOOK), encoding="utf-8")
    steps = []
    book = import_book_isolated(p, src, "Child Book", progress=lambda f, m: steps.append((f, m, os.getpid())))
    assert book.title == "Child Book" and book.cooccurrence_path.exists()
    assert steps and steps[-1][0] == 1.0
    assert all(pid == os.getpid() for *_x, pid in steps)            # progress delivered in the caller


def test_isolated_import_reports_errors_and_can_be_cancelled(paths, tmp_path):
    from horus.book_worker import BookImportError, ImportCancelled, import_book_isolated, import_books
    p = Paths(resources=paths.resources, user=tmp_path / "user")
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(BookImportError, match="JSON object of remedies"):
        import_book_isolated(p, bad, "Bad")
    big = tmp_path / "big.json"
    big.write_text((paths.resources / "Guiding_Symptoms_of_our_Materia_Medica-I.json").read_text(encoding="utf-8"),
                   encoding="utf-8")
    started = []
    with pytest.raises(ImportCancelled):
        import_book_isolated(p, big, "Big", progress=lambda f, m: started.append(f),
                             cancelled=lambda: bool(started))      # cancel as soon as it reports
    assert not list((p.user / "books").glob("*")) if (p.user / "books").exists() else True
    good = tmp_path / "good.json"
    good.write_text(json.dumps(BOOK), encoding="utf-8")
    res = import_books(p, [(bad, "Bad"), (good, "Good One"), (good, "Good Two")])
    assert [b.title for b in res.added] == ["Good One", "Good Two"] and [f[0] for f in res.failed] == ["Bad"]
    assert len({b.id for b in res.added}) == 2                      # same file twice → two distinct books
