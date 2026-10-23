"""Guarantees that are new in the offline engine (not ported behaviour)."""

from __future__ import annotations

import ast
import copy
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from cases import all_cases
from conftest import make_engine

from horus.engine import HorusEngine
from horus.llm import LLMUnavailable, parse_json_reply

PKG = Path(__file__).resolve().parents[1] / "horus"
ONLINE_MODULES = {"streamlit", "google", "github", "requests", "httpx", "urllib3", "openai",
                  "anthropic", "aiohttp"}


def test_package_imports_no_online_libraries():
    for py in PKG.glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in ONLINE_MODULES, f"{py.name} imports {n}"


def test_full_workflow_with_network_blocked(paths, monkeypatch):
    """Load, learn, parse, rank, report, suggest and print a PDF with sockets disabled."""
    def no_network(*a, **k):
        raise AssertionError("engine attempted a network connection")
    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)

    from cases import saved_patients
    e = HorusEngine.from_defaults(paths)          # real loader: model folder absent → lexical
    e.learn_from(saved_patients())
    case = all_cases()[0]
    cat = e.finalise(copy.deepcopy(case["parsed"]), case["raw"])
    rpt = e.report("PT-1", cat)
    assert rpt["top10Remedies"]
    e.suggestions(cat)
    assert e.pdf("PT-1", cat, rpt)[:4] == b"%PDF"


def test_import_in_fresh_process_has_no_heavy_or_online_deps():
    """Importing the engine must not pull in torch/transformers (only loaded with the model)."""
    code = ("import sys; sys.path.insert(0, %r); import horus, horus.engine; "
            "bad = [m for m in ('torch','transformers','sentence_transformers','streamlit','github')"
            " if m in sys.modules]; print(bad); assert not bad" % str(PKG.parent))
    subprocess.run([sys.executable, "-c", code], check=True)


# ── no AI installed ──────────────────────────────────────────────────────────
def test_degrades_cleanly_without_a_model(paths):
    e = make_engine(paths, semantic=False)            # NullLLM
    assert not e.ai_available
    case = all_cases()[0]
    with pytest.raises(LLMUnavailable):
        e.categorise(case["raw"])
    cat = e.finalise(copy.deepcopy(case["parsed"]), case["raw"])
    rpt = e.report("PT-1", cat)
    assert rpt["top10Remedies"] and rpt["top10Remedies"][0]["rationale"].startswith("Matched in dataset")
    assert e.comparative_summary(rpt, rpt) == ""
    assert e.followups("Main complaint: headache") == []
    with pytest.raises(LLMUnavailable):
        e.transcribe(b"RIFF....", "audio/wav")


def test_structured_entry_without_free_text(paths):
    """Doctor fills fields directly: entries verify against themselves, nothing is unverified."""
    e = make_engine(paths, semantic=False)
    cat = e.empty_case()
    cat["psychological"].append({"symptom": "weeping when telling symptoms", "worse": ["consolation"],
                                 "better": []})
    cat["general"].append({"symptom": "chilly", "worse": [], "better": ["warmth"]})
    cat["causation"].append("grief")
    out = e.finalise_structured(cat)
    assert out["unverified"] == []
    assert out["psychological"][0]["worse"] == ["consolation"]
    assert e.report("PT-2", out)["top10Remedies"]


def test_bad_model_reply_does_not_break_report(paths, fake_llm):
    e = make_engine(paths, semantic=False, llm=fake_llm)
    case = all_cases()[0]
    cat = e.finalise(copy.deepcopy(case["parsed"]), case["raw"])
    fake_llm.replies = ["Sorry, I cannot help with that."]
    rpt = e.report("PT-1", cat)
    assert rpt["top10Remedies"][0]["rationale"].startswith("Matched in dataset")
    fake_llm.replies = ["not json"]
    with pytest.raises(ValueError):
        e.categorise(case["raw"])


@pytest.mark.parametrize("reply,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('```\n[1, 2]\n```', [1, 2]),
    ('Here is the JSON:\n{"a": {"b": [1]}}\nHope that helps!', {"a": {"b": [1]}}),
    ('json{"s": "starts with j-s-o-n letters"}', {"s": "starts with j-s-o-n letters"}),
    ('Questions: ["When?", "Where?"] done', ["When?", "Where?"]),
])
def test_parse_json_reply(reply, expected):
    assert parse_json_reply(reply) == expected


def test_parse_json_reply_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json_reply("no json here")


# ── embedding cache ──────────────────────────────────────────────────────────
def test_embedding_cache_on_disk_gives_identical_results(paths, tmp_path):
    from horus.config import Paths
    paths = Paths(resources=paths.resources, user=tmp_path)     # own cache folder
    case = all_cases()[0]
    first = make_engine(paths, semantic=True, disk_cache=True)
    cat = first.finalise(copy.deepcopy(case["parsed"]), case["raw"])
    ranked_fresh = first.rank(copy.deepcopy(cat))
    files = list(paths.cache_dir.glob("dataset_emb_*.npy"))
    assert len(files) == 1
    second = make_engine(paths, semantic=True, disk_cache=True)   # loads from disk
    assert second.rank(copy.deepcopy(cat)) == ranked_fresh


def test_missing_embedding_model_falls_back_to_lexical(tmp_path):
    from horus.embeddings import load_embedder
    assert load_embedder(tmp_path / "nope") is None


def test_learning_updates_only_when_cases_change(paths):
    from cases import saved_patients
    e = make_engine(paths, semantic=False)
    base_size = len(e.dataset.s2r)
    pts = saved_patients()
    e.learn_from(pts)
    live = e.dataset
    assert live.live_cases == 4 and len(live.s2r) >= base_size
    e.learn_from(copy.deepcopy(pts))
    assert e.dataset is live                       # unchanged input → no rebuild
