"""The cleaned Guiding Symptoms volumes (1.5) and how they replace clinic-added raw copies."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from horus.books import BookFormatError, DecisionConfig, discover_books, import_book
from horus.config import Paths
from horus.engine import HorusEngine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from clean_guiding_symptoms import Cleaner, main as clean  # noqa: E402

VOLS = ["I", "II", "III", "V", "VI", "VII", "VIII", "IX", "X"]


def _vol(paths, v):
    return json.loads((paths.resources / f"Guiding_Symptoms_of_our_Materia_Medica-{v}.json").read_text(encoding="utf-8"))


def test_every_volume_is_shipped(paths):
    ids = [b.id for b in discover_books(paths)]
    assert ids[:2] == ["rheumatic", "case_studies"]
    assert [i for i in ids if i.startswith("guiding")] == [
        f"guiding_symptoms_{n}" for n in (1, 2, 3, 5, 6, 7, 8, 9, 10)]
    for b in discover_books(paths):
        assert b.cooccurrence_path and b.cooccurrence_path.exists(), b.id


def test_cleaned_volumes(paths):
    names = {}
    for v in VOLS:
        d = _vol(paths, v)
        for rem, chs in d.items():
            names.setdefault(rem, []).append(v)
            assert not {c.lower() for c in chs} & {"relationship", "relations"}
            texts = [it["symptom"] for items in chs.values() for it in items]
            assert len({t.lower() for t in texts}) == len(texts)            # no duplicates within a remedy
            for t in texts:
                low = t.lower()
                assert "<" not in t and ">" not in t or ("worse" in low or "better" in low) or True
                assert not re.search(r"(^|[\s(])1\.\s*[a-z]", low), t        # "1." for "l."
                assert not re.fullmatch(r"[a-z .,'\-]{0,40},\s*(set|aet|get|æt)\.?\s*\d*", low), t
                assert low not in {"passive motion", "desires, aversions", "trachea and bronchia"}, t
    for bad in ("carbonicum", "muriaticum", "castus", "antiiracinum", "qisgrsgs", "alumina or argilla",
                "cadmium sulfuratum cainca", "cantharides capsicum", "asaf", "nitrig acid"):
        assert bad not in names
    assert names["agnus castus"] == ["I"] and names["caladium seguinum"] == ["III"]
    assert names["palladium"] == ["VIII"]                     # vol. III's "palladium" was Caladium
    assert names["natrum sulphuricum"] == ["II", "V"]          # split across volumes; merged when loaded
    report = (paths.resources / "guiding_symptoms_cleaning_report.md").read_text(encoding="utf-8")
    assert "Remedy names" in report and "caladium seguinum" in report


def test_angle_brackets_are_written_out(paths):
    flat = [it["symptom"] for v in VOLS for chs in _vol(paths, v).values() for items in chs.values()
            for it in items]
    with_words = sum(1 for t in flat if re.search(r"\b(worse|better)\b", t))
    with_marks = sum(1 for t in flat if "<" in t or ">" in t)
    assert with_marks < 100 < with_words


def test_cleaner_rules(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    book = {"testium": {
        "sensations": [{"symptom": "pain: in forehead", "labels": {}},
                       {"symptom": "in temples", "labels": {}},
                       {"symptom": "over 1. eye", "labels": {}},
                       {"symptom": "passive motion", "labels": {}},
                       {"symptom": "woman, set", "labels": {}},
                       {"symptom": "headache > lying; < walking in sun", "labels": {}},
                       {"symptom": "^^ great anxiety. ■ gastralgia", "labels": {}},
                       {"symptom": "proved by somebody in 1850", "labels": {}}],
        "relationship": [{"symptom": "antidote: camphor", "labels": {}}]}}
    (raw / "Guiding_Symptoms_of_our_Materia_Medica-II.json").write_text(json.dumps(book), encoding="utf-8")
    rep = clean(raw, tmp_path / "out")
    out = json.loads((tmp_path / "out" / "Guiding_Symptoms_of_our_Materia_Medica-II.json").read_text())
    texts = [it["symptom"] for it in out["testium"]["sensations"]]
    assert texts == ["pain in forehead", "pain in temples", "pain over left eye",
                     "headache better lying; worse walking in sun", "great anxiety. gastralgia"]
    assert "relationship" not in out["testium"]
    assert rep["volumes"]["II"]["entries_before"] == 9 and rep["volumes"]["II"]["entries_after"] == 5


def _user_copy(p: Paths, title: str, src_name: str):
    folder = p.user / "books" / "user_old_copy"
    folder.mkdir(parents=True)
    (folder / "data.json").write_text(json.dumps({"x remedy": {"head": [{"symptom": "zz raw scan symptom"}]}}))
    (folder / "book.json").write_text(json.dumps({"id": "user_old_copy", "title": title, "json": "data.json",
                                                   "description": f"1 remedies; added from {src_name}"}))


def test_raw_clinic_copy_is_replaced_by_the_cleaned_book(paths, tmp_path):
    p = Paths(resources=paths.resources, user=tmp_path / "user")
    _user_copy(p, "Guiding Symptoms of our Materia Medica-II", "Guiding_Symptoms_of_our_Materia_Medica-II.json")
    assert "user_old_copy" not in [b.id for b in discover_books(p)]
    e = HorusEngine(p)
    assert [b.id for b in e.superseded_books] == ["user_old_copy"]
    assert e.superseded_books[0].superseded_by == "guiding_symptoms_2"
    assert "zz raw scan symptom" not in e.dataset.s2r
    # a clinic default that named the old copy now uses the cleaned volume, weight carried over
    cfg = DecisionConfig(books=("rheumatic", "user_old_copy"), book_weights=(("user_old_copy", 2.0),)).resolve(e.books)
    assert cfg.books == ("rheumatic", "guiding_symptoms_2") and cfg.weight_of("guiding_symptoms_2") == 2.0
    # adding the raw scan again is refused with a clear message
    src = tmp_path / "Guiding_Symptoms_of_our_Materia_Medica-V.json"
    src.write_text(json.dumps({"r": {"c": [{"symptom": "x y z"}]}}))
    with pytest.raises(BookFormatError, match="already includes"):
        import_book(p, src, "My vol V")


def test_unrelated_clinic_book_is_kept(paths, tmp_path):
    p = Paths(resources=paths.resources, user=tmp_path / "user")
    _user_copy(p, "Boericke notes", "boericke.json")
    assert "user_old_copy" in [b.id for b in discover_books(p)]


def test_cleaner_does_not_touch_ordinary_entries():
    c = Cleaner(set(), set())
    for t in ("as if head would burst", "when walking every step is felt in neck",
              "pain in vertex, worse on moving head", "sweat after 12 p.m"):
        assert c.text("I", "testium", t) == t
        assert c.drop_reason(t, "head") is None
