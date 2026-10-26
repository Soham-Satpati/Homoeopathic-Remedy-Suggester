"""Live learning: saved cases update the remedy index and the co-occurrence clusters, and deleting a
case (or a patient) removes exactly its contribution."""

from __future__ import annotations

import pytest
from conftest import make_engine

import horus.crypto as crypto
from horus.store import Store


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(crypto, "SCRYPT_N", 2 ** 12)
    store, _ = Store.create(tmp_path / "h.db", edition="clinic", clinic_name="L", admin_username="admin",
                            admin_password="password-123")
    return store.login("admin", "password-123")


def _case(syms, remedies):
    return {"categorised": {"physical": [{"symptom": s, "worse": [], "better": []} for s in syms],
                            "psychological": [], "general": []},
            "top_remedies": remedies, "raw_symptoms": ""}


NEW_A, NEW_B = "zz learned symptom alpha", "zz learned symptom beta"


def test_cases_update_and_deletion_reverts(paths, session):
    e = make_engine(paths, semantic=False)
    base_s2r = e.dataset.s2r
    base_sug = e.suggestions({"physical": [{"symptom": NEW_A}]})
    assert base_sug == ({NEW_A: []}, [])                     # unknown to the books

    pid = session.create_patient({"name": "P"})
    c1 = session.add_case(pid, _case([NEW_A, NEW_B], ["sepia", "pulsatilla"]))
    e.learn_from(session.learning_records())
    assert e.learned_cases == 1
    assert e.dataset.s2r[NEW_A] == {"sepia": 0.3, "pulsatilla": 0.15}          # rank-weighted remedies
    per, top = e.suggestions({"physical": [{"symptom": NEW_A}]})
    assert [x["symptom"] for x in top] == [NEW_B]                               # new co-occurrence pair
    p1 = top[0]["avg_prob"]

    c2 = session.add_case(pid, _case([NEW_A, NEW_B], ["sepia"]))
    e.learn_from(session.learning_records())
    assert e.learned_cases == 2 and e.dataset.s2r[NEW_A]["sepia"] == 0.6
    p2 = e.suggestions({"physical": [{"symptom": NEW_A}]})[1][0]["avg_prob"]
    assert p2 > p1                                                              # stronger with more evidence

    session.delete_case(c2)
    e.learn_from(session.learning_records())
    assert e.learned_cases == 1 and e.dataset.s2r[NEW_A]["sepia"] == 0.3
    assert e.suggestions({"physical": [{"symptom": NEW_A}]})[1][0]["avg_prob"] == p1

    session.delete_case(c1)
    e.learn_from(session.learning_records())
    assert e.learned_cases == 0
    assert e.dataset.s2r == base_s2r                                            # exactly back to the books
    assert e.suggestions({"physical": [{"symptom": NEW_A}]}) == base_sug


def test_learning_from_existing_book_symptom_updates_its_probabilities(paths, session):
    """A saved case pairing two book symptoms shifts their P(B|A) towards the clinic's evidence;
    deleting the patient restores the book value."""
    e = make_engine(paths, semantic=False)
    idx = e._suggestion_index()
    a = next(iter(idx.forward))
    b, _disp, p_book = idx.forward[a][0]
    pid = session.create_patient({"name": "Q"})
    for _ in range(3):
        session.add_case(pid, _case([a, b], ["sepia"]))
    e.learn_from(session.learning_records())
    p_live = dict((o, p) for o, _d, p in e._suggestion_index().forward[a])[b]
    assert p_live == round((p_book * 10 + 3) / (10 + 3), 6) and p_live >= p_book
    session.delete_patient(pid)
    e.learn_from(session.learning_records())
    assert dict((o, p) for o, _d, p in e._suggestion_index().forward[a])[b] == p_book


def test_deleting_cases_needs_permission(session):
    from horus.store import PermissionDenied
    pid = session.create_patient({"name": "R"})
    cid = session.add_case(pid, _case(["x"], ["sepia"]))
    session.add_user("desk", "Desk", "staff", "temp-pass-1")
    staff = session.store.login("desk", "temp-pass-1")
    with pytest.raises(PermissionDenied):
        staff.delete_case(cid)
    assert session.delete_case(cid) == pid and session.delete_case(cid) is None
    assert "case.deleted" in [x["action"] for x in session.audit_entries()]
