"""Encrypted store: security properties, roles, audit chain, backup/restore, demo import."""

from __future__ import annotations

import json
import sqlite3

import pytest
from cases import saved_patients

import horus.crypto as crypto
from horus.store import AuthError, PermissionDenied, Store, StoreError

PW = "correct horse 1"


@pytest.fixture(autouse=True)
def fast_kdf(monkeypatch):
    """scrypt at production cost makes each login ~0.2 s; tests use a cheaper setting.
    (Cost parameters are stored with every wrap, so this cannot leak into real databases.)"""
    monkeypatch.setattr(crypto, "SCRYPT_N", 2 ** 12)


@pytest.fixture
def clinic(tmp_path):
    store, recovery = Store.create(tmp_path / "horus.db", edition="clinic", clinic_name="Test Clinic",
                                   admin_username="admin", admin_password=PW, admin_display_name="Dr Admin")
    return store, recovery, store.login("admin", PW)


def _add_case(s, pid, name="A"):
    return s.add_case(pid, {"raw_symptoms": f"secret words {name}", "top_remedies": ["sepia"],
                            "categorised": {"physical": [{"symptom": "headache", "worse": [], "better": []}]}})


def test_nothing_identifying_is_stored_in_clear(clinic, tmp_path):
    store, _, s = clinic
    pid = s.create_patient({"name": "Anjali Sengupta", "contact": "9830012345", "age": "41"})
    _add_case(s, pid, "Anjali")
    s.add_intake(pid, {"raw_text": "pain since my divorce", "qa": []})
    store.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("horus.db*"))
    for secret in (b"Anjali", b"Sengupta", b"9830012345", b"secret words", b"divorce", b"headache"):
        assert secret not in raw, secret


def test_login_wrong_password_and_lockout(clinic):
    store, _, _ = clinic
    for _ in range(4):
        with pytest.raises(AuthError, match="Wrong username or password"):
            store.login("admin", "nope-nope")
    with pytest.raises(AuthError):
        store.login("admin", "nope-nope")              # 5th → locks
    with pytest.raises(AuthError, match="Too many failed attempts"):
        store.login("admin", PW)                       # even the right password waits


def test_records_bound_to_their_row(clinic):
    """Swapping sealed blobs between patients must fail, not silently show the wrong data."""
    store, _, s = clinic
    a = s.create_patient({"name": "A"})
    b = s.create_patient({"name": "B"})
    con = sqlite3.connect(store.path)
    blob = con.execute("SELECT details FROM patients WHERE id = ?", (a,)).fetchone()[0]
    con.execute("UPDATE patients SET details = ? WHERE id = ?", (blob, b))
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="integrity"):
        s.get_patient(b)


def test_roles(clinic):
    store, _, admin = clinic
    admin.add_user("reception", "Front desk", "staff", "temp-pass-1")
    admin.add_user("drb", "Dr B", "doctor", "temp-pass-2")
    pid = admin.create_patient({"name": "P"})
    _add_case(admin, pid)

    staff = store.login("reception", "temp-pass-1")
    assert staff.user.must_change_password
    assert staff.list_patients()[0]["case_count"] == 0          # staff sees no case data
    with pytest.raises(PermissionDenied):
        staff.list_cases(pid)
    with pytest.raises(PermissionDenied):
        staff.delete_patient(pid)
    with pytest.raises(PermissionDenied):
        staff.backup(store.path.parent / "x.horusbak")
    staff.add_intake(pid, {"raw_text": "intake", "qa": []})      # staff can take intakes

    doc = store.login("drb", "temp-pass-2")
    assert len(doc.list_cases(pid)) == 1
    with pytest.raises(PermissionDenied):
        doc.add_user("x1", "x", "staff", "temp-pass-3")
    with pytest.raises(PermissionDenied):
        doc.audit_entries()


def test_every_user_reads_the_same_data(clinic):
    store, _, admin = clinic
    admin.add_user("drb", "Dr B", "doctor", "temp-pass-2")
    pid = admin.create_patient({"name": "Shared"})
    doc = store.login("drb", "temp-pass-2")
    doc.change_password("temp-pass-2", "new-pass-22")
    assert store.login("drb", "new-pass-22").get_patient(pid)["details"]["name"] == "Shared"
    with pytest.raises(AuthError):
        store.login("drb", "temp-pass-2")


def test_solo_edition_has_one_account(tmp_path):
    store, _ = Store.create(tmp_path / "s.db", edition="solo", clinic_name="Solo", admin_username="dr",
                            admin_password=PW)
    s = store.login("dr", PW)
    with pytest.raises(StoreError, match="Solo"):
        s.add_user("x1", "x", "doctor", "temp-pass-3")
    s.set_edition("clinic")
    s.add_user("x1", "x", "doctor", "temp-pass-3")
    with pytest.raises(StoreError):
        s.set_edition("solo")                          # two active accounts now


def test_recovery_key_resets_admin_password(clinic):
    store, recovery, s = clinic
    pid = s.create_patient({"name": "Kept"})
    with pytest.raises(AuthError):
        store.recover("AAAA-BBBB-CCCC-DDDD-EEEE-FFFF-GGGG-HHHH", "admin", "brand-new-pw")
    s2 = store.recover(recovery.lower().replace("-", " "), "admin", "brand-new-pw")
    assert s2.get_patient(pid)["details"]["name"] == "Kept"
    with pytest.raises(AuthError):
        store.login("admin", PW)


def test_last_admin_cannot_be_removed(clinic):
    _, _, s = clinic
    with pytest.raises(StoreError):
        s.set_user_active(s.user.id, False)
    with pytest.raises(StoreError):
        s.set_user_role(s.user.id, "doctor")


def test_audit_chain_detects_tampering(clinic):
    store, _, s = clinic
    pid = s.create_patient({"name": "P"})
    s.get_patient(pid)
    assert store.verify_audit_chain() == (True, None)
    actions = [e["action"] for e in s.audit_entries()]
    assert {"database.created", "login", "patient.created", "patient.viewed"} <= set(actions)
    s.get_patient(pid)                              # so the victim is not the last entry
    con = sqlite3.connect(store.path)
    victim = con.execute("SELECT id FROM audit WHERE action = 'patient.viewed'").fetchone()[0]
    con.execute("DELETE FROM audit WHERE id = ?", (victim,))
    con.commit()
    con.close()
    ok, broken = store.verify_audit_chain()
    assert not ok and broken == victim + 1


def test_audit_truncation_detected_with_key(clinic):
    """Cutting entries off the end keeps the plain chain valid; the keyed checkpoint catches it."""
    store, _, s = clinic
    pid = s.create_patient({"name": "P"})
    s.get_patient(pid)
    assert s.verify_audit() == (True, None)
    con = sqlite3.connect(store.path)
    con.execute("DELETE FROM audit WHERE id = (SELECT MAX(id) FROM audit)")
    con.commit()
    con.close()
    assert store.verify_audit_chain()[0]            # chain alone can't tell
    assert s.verify_audit()[0] is False             # checkpoint can


def test_audit_never_contains_patient_details(clinic):
    store, _, s = clinic
    pid = s.create_patient({"name": "Rahul Dey", "contact": "9000000001"})
    _add_case(s, pid)
    dump = json.dumps(s.audit_entries())
    assert "Rahul" not in dump and "9000000001" not in dump and "secret words" not in dump


def test_backup_and_restore(clinic, tmp_path):
    store, recovery, s = clinic
    pid = s.create_patient({"name": "Backed Up"})
    _add_case(s, pid)
    bk = s.backup(tmp_path / "b" / "clinic.horusbak")
    raw = bk.read_bytes()
    assert b"Backed Up" not in raw and b"SQLite format" not in raw     # whole file sealed

    with pytest.raises(StoreError, match="already exists"):
        Store.restore(bk, store.path, "admin", PW)
    with pytest.raises(Exception):
        Store.restore(bk, tmp_path / "r0.db", "admin", "wrong-password")

    restored = Store.restore(bk, tmp_path / "r1.db", "admin", PW)
    r = restored.login("admin", PW)
    assert r.get_patient(pid)["details"]["name"] == "Backed Up"
    assert len(r.list_cases(pid)) == 1
    assert restored.verify_audit_chain()[0]

    via_recovery = Store.restore(bk, tmp_path / "r2.db", "admin", recovery, use_recovery_key=True)
    assert via_recovery.login("admin", PW).get_patient(pid)


def test_tampered_backup_is_rejected(clinic, tmp_path):
    _, _, s = clinic
    s.create_patient({"name": "X"})
    bk = s.backup(tmp_path / "t.horusbak")
    data = bytearray(bk.read_bytes())
    data[-5] ^= 0xFF
    bk.write_bytes(bytes(data))
    with pytest.raises(Exception):
        Store.restore(bk, tmp_path / "r.db", "admin", PW)


def test_import_demo_patients_and_learning(clinic):
    store, _, s = clinic
    demo = saved_patients()
    demo["PT-2025-009"] = {"details": {"name": "Intake Only"}, "cases": [], "intakes": [
        {"id": "abc", "timestamp": "2026-01-02 10:00", "qa": [], "raw_text": "pain", "status": "pending"}]}
    stats = s.import_demo_patients(demo)
    assert stats["patients_new"] == 5 and stats["cases"] == 4 and stats["intakes"] == 1
    again = s.import_demo_patients(demo)                       # idempotent for cases
    assert again["cases"] == 0 and again["skipped"] == 4
    assert len(s.pending_intakes()) >= 1

    # learning payload is identical to what the demo derived from the JSON file
    from horus.dataset import learning_payload
    assert learning_payload(s.learning_records()) == learning_payload(saved_patients())

    case = s.list_cases("PT-2026-001")[0]
    assert case["timestamp"] == "2026-10-12T22:06:00"
    assert case["categorised"] == saved_patients()["PT-2026-001"]["cases"][0]["categorised"]


def test_patient_ids_follow_demo_scheme(clinic):
    _, _, s = clinic
    s.create_patient({"name": "a"}, patient_id="PT-2026-007")
    assert s.new_patient_id(2026) == "PT-2026-008"
    assert s.new_patient_id(2031) == "PT-2031-001"


def test_logout_drops_the_key(clinic):
    _, _, s = clinic
    s.logout()
    with pytest.raises(AuthError):
        s.list_patients()


def test_search_and_intake_merge(clinic):
    _, _, s = clinic
    pid = s.create_patient({"name": "Mitra Bose", "contact": "98300"})
    s.create_patient({"name": "Other"})
    assert [p["id"] for p in s.list_patients("bose")] == [pid]
    assert [p["id"] for p in s.list_patients("98300")] == [pid]
    s.add_intake(pid, {"raw_text": "x", "qa": [], "details": {"age": "33", "name": ""}})
    d = s.get_patient(pid)["details"]
    assert d["age"] == "33" and d["name"] == "Mitra Bose"           # blanks never overwrite
    assert s.list_patients("bose")[0]["pending_intakes"] == 1


def test_concurrent_writers_never_lose_or_mix_writes(clinic):
    """Window, background jobs and the tablet server share one connection. A transaction that
    fails in one thread must not roll back writes made by another thread meanwhile."""
    import threading
    from horus.store import StoreError
    store, _, s = clinic
    pid = s.create_patient({"name": "Busy"})
    errors, done = [], []

    def intakes(n):
        try:
            for i in range(n):
                s.add_intake(pid, {"raw_text": f"t{i}", "qa": []}, source="lan")
            done.append(n)
        except Exception as e:           # pragma: no cover - reported below
            errors.append(e)

    def failing_tx(n):
        for _ in range(n):
            try:
                with s._tx():
                    s._con.execute("UPDATE patients SET updated_at = 'x' WHERE id = ?", (pid,))
                    raise StoreError("boom")      # forces ROLLBACK
            except StoreError:
                pass

    def cases(n):
        try:
            for i in range(n):
                _add_case(s, pid, str(i))
            done.append(n)
        except Exception as e:           # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=intakes, args=(40,)), threading.Thread(target=intakes, args=(40,)),
          threading.Thread(target=failing_tx, args=(60,)), threading.Thread(target=cases, args=(30,))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors
    assert len(s.patient_intakes(pid)) == 80
    assert len(s.list_cases(pid)) == 30
    assert store.verify_audit_chain()[0]
