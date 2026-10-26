"""Headless UI flows: setup → kiosk intake → inbox → case from intake → patients → settings → lock."""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PySide6 = pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import horus.crypto as crypto  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def no_leftover_jobs(qapp):
    yield
    from PySide6.QtCore import QEvent, QThreadPool
    assert QThreadPool.globalInstance().waitForDone(30000), "background jobs still running after the test"
    for w in QApplication.topLevelWidgets():                # one app per process, like the real thing
        w.close()
        w.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()


@pytest.fixture(autouse=True)
def fast_matching(monkeypatch):
    """Semantic matching with the stand-in embedder (numpy) instead of word overlap over 20k symptoms."""
    import horus.embeddings
    from legacy import FakeEmbedder
    monkeypatch.setattr(horus.embeddings, "load_embedder", lambda *a, **k: FakeEmbedder())


@pytest.fixture(autouse=True)
def dialogs(monkeypatch):
    """Modal message boxes would block a headless test: record them instead."""
    shown = []
    import horus_app.admin, horus_app.case_flow, horus_app.kiosk, horus_app.login, horus_app.main_window, \
        horus_app.patients, horus_app.setup_wizard, horus_app.voice, horus_app.widgets
    for mod in (horus_app.admin, horus_app.case_flow, horus_app.kiosk, horus_app.login, horus_app.main_window,
                horus_app.patients, horus_app.setup_wizard, horus_app.voice, horus_app.widgets):
        for name in ("error", "info"):
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, lambda *a, _n=name, **k: shown.append((_n, a[1:3])))
        if hasattr(mod, "confirm"):
            monkeypatch.setattr(mod, "confirm", lambda *a, **k: True)
    return shown


@pytest.fixture
def ctx(tmp_path, monkeypatch, qapp):
    monkeypatch.setattr(crypto, "SCRYPT_N", 2 ** 12)
    from horus.config import Paths
    from horus_app.context import AppContext
    paths = Paths(resources=Paths().resources, user=tmp_path / "user")
    c = AppContext(paths)
    c.llm = c.transcriber = None                    # AI off: the structured paths must work alone
    c.settings.update(ai_enabled=False, backup_on_exit=False)
    return c


def _pump(ms: int) -> None:
    """Run the Qt event loop for ``ms``. (QTest.qWait keeps Python's interpreter lock while it waits,
    which starves the app's background threads; a real event loop releases it, like the app does.)"""
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def wait(cond, timeout=120):
    t = time.time()
    while not cond():
        _pump(50)
        if time.time() - t > timeout:
            raise TimeoutError
    return True


def test_setup_wizard_creates_clinic_and_shows_recovery_key(ctx):
    from horus_app.setup_wizard import SetupWizard
    wiz = SetupWizard(ctx)
    wiz.edition_group.button(2).setChecked(True)            # Clinic LAN
    wiz.clinic.setText("Test Clinic")
    wiz.display.setText("Dr Test")
    wiz.username.setText("drtest")
    wiz.pw1.setText("password-123")
    wiz.pw2.setText("password-123")
    wiz._create()
    assert wiz.stack.currentIndex() == 3
    key = wiz.key_label.text()
    assert len(key) == 39 and key.count("-") == 7
    assert ctx.store.edition == "clinic_lan" and ctx.store.clinic_name == "Test Clinic"
    assert wiz.session.user.role == "admin"
    assert wiz.pw1.text() == ""                             # password not left in the widget


def _signed_in(ctx):
    from horus.store import Store
    ctx.store, _ = Store.create(ctx.db_path, edition="clinic", clinic_name="UI", admin_username="admin",
                                admin_password="password-123")
    ctx.session = ctx.store.login("admin", "password-123")
    from horus_app.main_window import MainWindow
    win = MainWindow(ctx)
    wait(lambda: ctx.engine is not None)
    return win


def test_kiosk_intake_to_case(ctx, dialogs):
    win = _signed_in(ctx)
    s = ctx.session

    # patient fills the questionnaire on the clinic PC
    win.go("kiosk")
    k = win.kiosk
    assert not win.nav.isVisible()
    k.d_name.setText("Kiosk Patient")
    k.d_age.setText("52")
    k._start()
    answers = {"complaint": "Burning pain in the stomach", "worse": "after midnight",
               "better": "warm drinks", "thermal": "very chilly, wants to be covered"}
    for _ in range(len(k.questions)):
        q = k.questions[k.idx]
        k.answer.setPlainText(answers.get(q["id"], ""))
        k._next()
    assert k.stack.currentIndex() == 2 and "PT-" in k.thanks_id.text()
    assert len(s.pending_intakes()) == 1

    # leaving kiosk needs the password
    assert not s.verify_password("wrong-password", "kiosk.exit")

    # doctor opens the inbox and starts a case from it
    win._leave_kiosk()
    win.go("intakes")
    assert win.intakes.list.count() == 1
    intake = win.intakes.items[0]
    win._case_from_intake(intake)
    cf = win.case
    assert "Burning pain" in cf.words.toPlainText()
    assert cf.pt_existing.currentData() == intake["patient_id"]
    cf.tabs.setCurrentIndex(1)
    cf.st_table.add("physical", {"symptom": "burning pain in the stomach", "worse": ["after midnight"],
                                 "better": ["warm drinks"]})
    cf.st_table.add("general", {"symptom": "very chilly", "worse": [], "better": []})
    cf.st_table.add("psychological", {"symptom": "fear of ghosts", "worse": [], "better": []})
    cf._read_case()
    # verified against the intake words: the invented symptom is flagged, the others are not
    from horus_app.widgets import cell_checked
    rows = {cf.rv_table.item(r, 2).text(): cell_checked(cf.rv_table, r) for r in range(cf.rv_table.rowCount())}
    assert rows["fear of ghosts"] is False and rows["very chilly"] is True
    cf._analyse()
    wait(lambda: cf.steps.currentIndex() == 2)
    assert cf.rp_saved.text().startswith("✓")
    assert "fear of ghosts" in cf.rp_view.toPlainText()          # listed under "Not used"
    assert s.pending_intakes() == []                             # intake marked reviewed
    case = s.list_cases(intake["patient_id"])[0]
    assert case["report"]["top10Remedies"] and case["raw_symptoms"].startswith("Main complaint")

    # patients page shows it
    win.go("patients")
    assert win.patients.table.rowCount() == 1
    assert win.patients.table.item(0, 6).text() == "1"

    # suggestions: either the table opens, or the doctor is told there are none (never a hang/crash)
    cf.steps.setCurrentIndex(2)
    cf._open_suggestions()
    assert cf.steps.currentIndex() == 3 or any(t == ("No suggestions",) or t[0] == "No suggestions"
                                               for _, t in dialogs)
    win.close()


def test_settings_tabs_follow_role_and_backup(ctx, tmp_path):
    win = _signed_in(ctx)
    tabs = [win.settings.tabs.tabText(i) for i in range(win.settings.tabs.count())]
    assert tabs == ["My account", "Backups", "Users", "Books && weighting", "Clinic && AI", "Audit log"]
    from horus_app.admin import make_backup
    ctx.settings["backup_dir"] = str(tmp_path / "bk")
    p = make_backup(ctx)
    assert p.exists() and p.suffix == ".horusbak"

    ctx.session.add_user("frontdesk", "Front desk", "staff", "temp-pass-1")
    ctx.session = ctx.store.login("frontdesk", "temp-pass-1")
    win.on_signed_in()
    tabs = [win.settings.tabs.tabText(i) for i in range(win.settings.tabs.count())]
    assert tabs == ["My account"]
    assert not win.nav_btns["case"][0].isVisibleTo(win) and not win.nav_btns["intakes"][0].isVisibleTo(win)
    win.close()


def test_unlock_dialog(ctx):
    win = _signed_in(ctx)
    from horus_app.login import UnlockDialog
    d = UnlockDialog(ctx.session)
    d.pw.setText("nope-nope")
    d._unlock()
    assert d.result_kind is None and d.msg.isVisibleTo(d)
    d.pw.setText("password-123")
    d._unlock()
    assert d.result_kind == "unlocked"
    actions = [e["action"] for e in ctx.session.audit_entries()]
    assert "screen.unlock.failed" in actions and "screen.unlock" in actions
    win.close()


def test_report_appears_instantly_then_ai_explanations_fill_in(ctx):
    """Dataset ranking is shown and saved at once; AI text arrives later and updates the saved case."""
    import threading
    from cases import NARRATION_REPLY
    gate = threading.Event()

    class SlowLLM:
        name = "slow-fake"
        _proc = None

        def generate(self, prompt, **kw):
            gate.wait(20)                              # explanations are slow on a CPU
            return NARRATION_REPLY

        def stop(self):
            pass

    win = _signed_in(ctx)
    ctx.llm = ctx.engine.llm = SlowLLM()
    ctx.settings["ai_enabled"] = True
    cf = win.case
    cf.reset()
    cf.pt_name.setText("Async Patient")
    cf.tabs.setCurrentIndex(1)
    cf.st_table.add("psychological", {"symptom": "weeping alone", "worse": ["evening"], "better": []})
    cf.st_table.add("general", {"symptom": "craving for salt", "worse": [], "better": []})
    cf._read_case()
    cf._analyse()
    wait(lambda: cf.steps.currentIndex() == 2)
    first = cf.report["top10Remedies"][0]["rationale"]
    assert first.startswith("Matched in dataset")              # instant, dataset-only
    assert cf.rp_saved.text().startswith("✓") and cf.rp_busy.isVisibleTo(cf)
    pid = cf.pt_existing.currentData()
    gate.set()
    wait(lambda: "explanations added" in cf.rp_saved.text())
    saved = ctx.session.list_cases(pid)[0]["report"]
    assert saved["_removed"]                                    # filters ran on the AI text
    assert saved["top10Remedies"][0]["name"] == cf.report["top10Remedies"][0]["name"]   # ranking unchanged
    win.close()


def test_suggestion_checkboxes_are_real_and_drive_comparison(ctx):
    """Regression: the Confirmed column had no visible checkboxes on Windows."""
    from PySide6.QtWidgets import QCheckBox
    from horus_app.widgets import cell_checked, set_cell_checked
    win = _signed_in(ctx)
    cf = win.case
    cf.reset()
    cf.pt_name.setText("Throat Patient")
    cf.tabs.setCurrentIndex(1)
    cf.st_table.add("physical", {"symptom": "sore throat", "worse": [], "better": []})
    cf._read_case()
    # the review table uses real checkboxes too
    assert isinstance(cf.rv_table.cellWidget(0, 0).checkbox, QCheckBox) and cell_checked(cf.rv_table, 0)
    cf._analyse()
    wait(lambda: cf.steps.currentIndex() == 2)
    cf._open_suggestions()
    assert cf.steps.currentIndex() == 3 and cf.sg_table.rowCount() > 0
    for r in range(cf.sg_table.rowCount()):
        cb = cf.sg_table.cellWidget(r, 0).checkbox
        assert isinstance(cb, QCheckBox) and cb.isVisibleTo(cf) and not cb.isChecked()
    assert not cf.cmp_go.isEnabled()                       # nothing ticked → can't compare
    set_cell_checked(cf.sg_table, 0, True)
    cf.sg_table.item(0, 4).setText("night, cold")
    assert cf.cmp_go.isEnabled() and "1 symptom" in cf.cmp_count.text()
    cf._compare()
    wait(lambda: cf.cmp_pdf.isEnabled())
    added = cf._cmp[1]
    assert len(added) == 1 and added[0]["worse"] == ["night", "cold"]
    assert "B. With confirmed suggestions" in cf.cmp_enh.toPlainText()
    win.close()


def test_reanalysis_updates_same_case_and_delete_case_relearns(ctx):
    win = _signed_in(ctx)
    cf = win.case
    cf.reset()
    cf.pt_name.setText("Twice Patient")
    cf.tabs.setCurrentIndex(1)
    cf.st_table.add("physical", {"symptom": "zz unique learning symptom", "worse": [], "better": []})
    cf._read_case()
    cf._analyse()
    wait(lambda: cf.steps.currentIndex() == 2)
    pid = cf.current_pid
    cf.steps.setCurrentIndex(1)                                  # "← Edit symptoms" and analyse again
    cf.rv_table.add("general", {"symptom": "chilly", "worse": [], "better": []})
    cf._analyse()
    wait(lambda: cf.rp_saved.text().startswith("✓ Updated"))
    cases = ctx.session.list_cases(pid)
    assert len(cases) == 1 and ctx.engine.learned_cases == 1      # one consultation counted once
    assert "chilly" in [s["symptom"] for s in cases[0]["categorised"]["general"]]

    from horus_app.patients import PatientDialog
    dlg = PatientDialog(ctx, pid)
    dlg._delete_case()                                           # confirm() is auto-accepted in tests
    assert ctx.session.list_cases(pid) == [] and ctx.engine.learned_cases == 0
    assert "zz unique learning symptom" not in ctx.engine.dataset.s2r
    win.close()


def test_admin_deletes_own_account_and_sole_user_deletes_clinic(ctx, monkeypatch):
    import horus_app.main_window as mw
    win = _signed_in(ctx)
    s = ctx.session
    pid = s.create_patient({"name": "Stays"})
    # with another admin: only the account goes
    s.add_user("other", "Other Admin", "admin", "temp-pass-1")
    win.on_signed_in()
    assert ctx.session.account_deletion_plan() == "account"

    class FakeLogin:                                      # the next person signs in
        def __init__(self, store, *a, **k):
            self.session = store.login("other", "temp-pass-1")

        def exec(self):
            return mw.QDialog.Accepted
    monkeypatch.setattr(mw, "LoginDialog", FakeLogin)
    ctx.session.delete_own_account("password-123")
    win._account_deleted("account", "", False)
    assert ctx.session.user.username == "other" and ctx.session.get_patient(pid)
    assert "admin" not in ctx.store.usernames()

    # the last user deletes the clinic → app restarts into setup
    ctx.session.set_user_active(ctx.session.list_users()[0]["id"], True)  # no-op for "other"
    assert ctx.session.account_deletion_plan() == "clinic"
    ctx.settings["backup_dir"] = str(ctx.paths.user / "bk")
    from horus_app.admin import make_backup
    bk = make_backup(ctx)
    db = ctx.db_path
    win._account_deleted("clinic", "temp-pass-1", True)
    assert win.restart_setup and not db.exists() and not bk.exists()
    assert ctx.store is None and ctx.session is None


def test_books_tab_fills_in_when_books_finish_loading(ctx, monkeypatch, tmp_path):
    """Regression: the tab was built at sign-in while books were loading and stayed on
    'still loading' forever. It must show progress, then the editor, without being reopened."""
    import json
    from PySide6.QtWidgets import QProgressBar
    import horus_app.decision as dec
    from horus.store import Store
    ctx.store, _ = Store.create(ctx.db_path, edition="clinic", clinic_name="UI", admin_username="admin",
                                admin_password="password-123")
    ctx.session = ctx.store.login("admin", "password-123")
    from horus_app.main_window import MainWindow
    win = MainWindow(ctx)                                   # settings tabs are built before books are ready
    tab = next(win.settings.tabs.widget(i) for i in range(win.settings.tabs.count())
               if win.settings.tabs.tabText(i).startswith("Books"))
    assert tab.prog_card.parent() is not None and not hasattr(tab, "editor")
    seen = []
    ctx.events.prep_progress.connect(lambda f, m: seen.append(f))
    wait(lambda: ctx.engine_ready)
    _pump(100)
    assert hasattr(tab, "editor") and tab.prog_card.parent() is None          # filled in by itself
    assert seen and seen[-1] == 1.0 and not win.st_bar.isVisible()
    assert [b.id for b in tab.editor.books] == ["rheumatic", "case_studies", "guiding_symptoms_1"]

    # add several books at once (one is not a valid book): overall progress, the good one is added,
    # the bad one is reported, and the import runs in a separate process
    src = tmp_path / "extra_book.json"
    src.write_text(json.dumps({"remedium novum": {"mind": [
        {"symptom": f"novel symptom {i}", "labels": {"Modalities": "worse at night"}} for i in range(12)]}}))
    bad = tmp_path / "not_a_book.json"
    bad.write_text(json.dumps(["just", "a", "list"]))
    monkeypatch.setattr(dec.QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(src), str(bad)], ""))
    monkeypatch.setattr(dec.AddBooksDialog, "exec", lambda self: dec.QDialog.Accepted)
    monkeypatch.setattr(dec.AddBooksDialog, "items", lambda self: [(src, "Extra Book"), (bad, "Broken")])
    shown = []
    monkeypatch.setattr(dec, "info", lambda *a, **k: shown.append(a[2]))
    values = []
    tab.prog_bar.valueChanged.connect(values.append)
    tab._add()
    assert tab.adding and tab.prog_card.parent() is not None and tab.cancel_btn.isVisibleTo(tab)
    wait(lambda: not tab.adding and ctx.engine_ready and hasattr(tab, "editor")
         and any(b.title == "Extra Book" for b in tab.editor.books))
    assert shown and "Extra Book" in shown[-1] and "Broken" in shown[-1] and "Not added" in shown[-1]
    adding = values[:values.index(100) + 1]                # the import: 0 → 100 …
    assert adding == sorted(adding) and adding[0] == 0
    reindex = values[len(adding):]                         # … then indexing the new book: 0 → 100
    assert reindex and reindex == sorted(reindex) and reindex[-1] == 100
    extra = next(b for b in ctx.engine.books if b.title == "Extra Book")
    assert extra.id in ctx.engine.config.books

    # delete it (clinic book → removed from disk)
    tab._delete(extra)
    wait(lambda: ctx.engine_ready and hasattr(tab, "editor") and all(b.id != extra.id for b in tab.editor.books))
    assert not extra.json_path.exists() and extra.id not in ctx.engine.config.books

    # delete a shipped book → hidden from this clinic, then restored
    tab._delete(next(b for b in ctx.engine.books if b.id == "guiding_symptoms_1"))
    wait(lambda: ctx.engine_ready and "guiding_symptoms_1" not in [b.id for b in tab.editor.books])
    assert ctx.session.hidden_books() == ["guiding_symptoms_1"]
    assert "guiding_symptoms_1" not in ctx.engine.config.books
    tab._restore()
    wait(lambda: ctx.engine_ready and "guiding_symptoms_1" in [b.id for b in tab.editor.books])
    assert ctx.session.hidden_books() == [] and "guiding_symptoms_1" in ctx.engine.config.books
    win.close()
