"""Main window: navigation, status, auto-lock, engine/AI start-up, LAN server, restore."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QStackedWidget, QStatusBar, QVBoxLayout, QWidget)

from horus.store import Store

from .admin import SettingsPage, make_backup
from .case_flow import CaseFlow
from .context import AppContext, run_job
from .kiosk import KioskPage
from .login import LoginDialog
from .patients import IntakesPage, PatientsPage
from .widgets import button, error, info, label

log = logging.getLogger(__name__)


class _ActivityFilter(QObject):
    def __init__(self, win):
        super().__init__()
        self.win = win

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.MouseButtonPress, QEvent.KeyPress, QEvent.Wheel):
            self.win.last_activity = time.time()
        return False


class MainWindow(QMainWindow):
    lan_event = Signal(str)

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.lan = None
        self.last_activity = time.time()
        self.locked = False
        self.setWindowTitle(f"HoRUS — {ctx.store.clinic_name}")
        self.resize(1360, 860)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.nav = QFrame()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(230)
        nv = QVBoxLayout(self.nav)
        nv.setContentsMargins(12, 16, 12, 12)
        nv.setSpacing(4)
        title = label("HoRUS", "h1")
        nv.addWidget(title)
        self.clinic_lbl = label(ctx.store.clinic_name, "muted")
        nv.addWidget(self.clinic_lbl)
        nv.addSpacing(14)
        self.pages = QStackedWidget()
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        self.case = CaseFlow(ctx)
        self.patients = PatientsPage(ctx)
        self.intakes = IntakesPage(ctx)
        self.kiosk = KioskPage(ctx)
        self.settings = SettingsPage(ctx)
        self.nav_btns = {}
        for key, text, widget in (("case", "New case", self.case), ("patients", "Patients", self.patients),
                                  ("intakes", "New intakes", self.intakes),
                                  ("kiosk", "Patient questionnaire", self.kiosk),
                                  ("settings", "Settings", self.settings)):
            b = QPushButton(text)
            b.setObjectName("navbtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.group.addButton(b)
            idx = self.pages.addWidget(widget)
            b.clicked.connect(lambda _=False, k=key, i=idx: self.go(k, i))
            nv.addWidget(b)
            self.nav_btns[key] = (b, idx)
        nv.addStretch(1)
        self.user_lbl = label("", "muted")
        nv.addWidget(self.user_lbl)
        nv.addWidget(button("Lock screen", slot=self.lock))
        nv.addWidget(button("Sign out", slot=self.sign_out))
        h.addWidget(self.nav)
        h.addWidget(self.pages, 1)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.st_engine = QLabel("Loading remedy dataset…")
        from PySide6.QtWidgets import QProgressBar
        self.st_bar = QProgressBar()
        self.st_bar.setRange(0, 100)
        self.st_bar.setFixedWidth(180)
        self.st_bar.setFormat("%p%")
        self.st_bar.setTextVisible(True)
        self.status.addPermanentWidget(self.st_bar)
        self.st_ai = QLabel("")
        self.st_lan = QLabel("")
        for w in (self.st_engine, self.st_ai, self.st_lan):
            self.status.addPermanentWidget(w)

        self.case.saved.connect(lambda _pid: self.patients.refresh())
        self.patients.new_case_for.connect(self._new_case_for)
        self.intakes.start_case.connect(self._case_from_intake)
        self.intakes.changed.connect(self._update_badges)
        self.kiosk.finished.connect(self._leave_kiosk)
        self.kiosk.submitted.connect(self._update_badges)
        self.settings.restore_requested.connect(self.restore)
        self.settings.lan_toggled.connect(self.set_lan)
        self.settings.books_changed.connect(self._books_changed)
        self.ctx.events.books_changed.connect(self._books_changed)
        self.settings.account_deleted.connect(self._account_deleted)
        self.restart_setup = False
        self.lan_event.connect(self._on_lan_event)

        self._filter = _ActivityFilter(self)
        QApplication.instance().installEventFilter(self._filter)
        self.idle_timer = QTimer(self)
        self.idle_timer.timeout.connect(self._check_idle)
        self.idle_timer.start(15_000)
        self.badge_timer = QTimer(self)
        self.badge_timer.timeout.connect(self._update_badges)
        self.badge_timer.start(30_000)

        self.on_signed_in()
        self._load_engine()

    # ── navigation / roles ───────────────────────────────────────────────
    def go(self, key: str, idx: int | None = None):
        btn, i = self.nav_btns[key]
        btn.setChecked(True)
        self.pages.setCurrentIndex(idx if idx is not None else i)
        if key == "patients":
            self.patients.refresh()
        elif key == "intakes":
            self.intakes.refresh()
        elif key == "kiosk":
            self._enter_kiosk()

    def on_signed_in(self):
        s = self.ctx.session
        u = s.user
        self.user_lbl.setText(f"{u.display_name}\n{u.role}")
        self.nav_btns["case"][0].setVisible(u.can("cases.write"))
        self.nav_btns["intakes"][0].setVisible(u.can("intakes.read"))
        self.settings.rebuild()
        if u.can("cases.write"):
            self.case.reset()
            self.go("case")
        else:
            self.go("patients")
        self._update_badges()
        self._update_ai_status()
        if self.ctx.engine is not None:
            self._apply_clinic_config()
            run_job(self.ctx.refresh_learning)
        if self.ctx.store.edition == "clinic_lan" and u.can("intakes.write") and self.ctx.settings.get("lan_auto"):
            self.set_lan(True)

    def _update_badges(self):
        s = self.ctx.session
        if s is None or not s.user.can("intakes.read"):
            return
        try:
            n = len(s.pending_intakes())
        except Exception:
            return
        self.nav_btns["intakes"][0].setText(f"New intakes  ({n})" if n else "New intakes")

    def _new_case_for(self, pid: str):
        self.go("case")
        self.case.reset()
        self.case.refresh_patients(select=pid)

    def _case_from_intake(self, intake: dict):
        self.go("case")
        self.case.start_from_intake(intake)

    def _enter_kiosk(self):
        self.nav.hide()
        self.kiosk.reset()

    def _leave_kiosk(self):
        self.nav.show()
        self.go("patients")

    # ── engine / AI ──────────────────────────────────────────────────────
    def _load_engine(self):
        def done(engine):
            self.ctx.engine = engine
            self._apply_clinic_config()
            self.case.refresh_capabilities()
            run_job(self.ctx.refresh_learning)
            self.prepare_books()
            if self.ctx.ai_ready and self.ctx.settings.get("ai_preload", True):
                self.st_ai.setText("AI: starting…")
                run_job(self.ctx.llm.start, lambda _r: self._update_ai_status(),
                        lambda m, _tb: self.st_ai.setText("AI: could not start (see log)"))

        def fail(msg, tb):
            self.st_engine.setText("Dataset failed to load")
            error(self, "HoRUS could not load its dataset", msg, tb)
        run_job(self.ctx.load_engine, done, fail)

    def _apply_clinic_config(self):
        from .decision import clinic_config
        if self.ctx.engine is not None and self.ctx.session is not None:
            hidden = set(self.ctx.session.hidden_books())
            if hidden != self.ctx.engine.hidden_books:
                self.ctx.engine.reload_books(hidden)
            self.ctx.engine.set_config(clinic_config(self.ctx))
            self.case.refresh_decision()

    def _update_engine_status(self):
        e = self.ctx.engine
        if e is None:
            return
        ds = e.dataset
        self.st_engine.setText(f"{len(e.config.books)} book(s) · {len(ds.s2r):,} symptoms · "
                               f"{len(ds.remedy_universe)} remedies · matching: {e.matching_mode}")

    def prepare_books(self):
        """Load and index the selected books in the background, with progress for every screen."""
        e = self.ctx.engine
        if e is None:
            return
        self.ctx.engine_ready = False
        self._on_prep(0.0, "Loading books…")
        self.st_bar.show()

        def done(_r):
            self.ctx.engine_ready = True
            self.st_bar.hide()
            self._update_engine_status()
            self.ctx.events.engine_ready.emit()

        def fail(msg, tb):
            self.ctx.engine_ready = True              # the engine still works (lexical matching fallback)
            self.st_bar.hide()
            self._update_engine_status()
            self.ctx.events.engine_ready.emit()
        run_job(lambda progress: e.warm_up(progress=progress), done, fail, self._on_prep)

    def _on_prep(self, fraction: float, message: str):
        self.ctx.prep_state = (fraction, message)
        self.st_engine.setText(message)
        self.st_bar.setValue(int(fraction * 100))
        self.ctx.events.prep_progress.emit(fraction, message)

    def _books_changed(self):
        self._update_engine_status()
        self.case.refresh_decision()
        self.prepare_books()

    def _update_ai_status(self):
        if not self.ctx.llm and not self.ctx.transcriber:
            self.st_ai.setText("AI: not installed")
        elif not self.ctx.settings.get("ai_enabled", True):
            self.st_ai.setText("AI: off")
        else:
            running = self.ctx.llm is not None and self.ctx.llm._proc is not None
            self.st_ai.setText("AI: ready" if running else "AI: starts on first use")

    # ── LAN server ───────────────────────────────────────────────────────
    def set_lan(self, on: bool):
        from .lan import IntakeServer
        if on:
            if self.lan is None:
                self.lan = IntakeServer(self._lan_submit, self.ctx.paths.user / "lan",
                                        port=int(self.ctx.settings.get("lan_port", 8765)),
                                        on_event=self.lan_event.emit)
            try:
                self.lan.start()
            except OSError as e:
                return error(self, "Tablet intake", f"Could not start on port {self.lan.port}: {e}")
            self.ctx.session.audit("lan.started")
            self.st_lan.setText(f"Tablets: on · code {self.lan.pin}")
            self.ctx.settings["lan_auto"] = True
        else:
            if self.lan:
                self.lan.stop()
                self.ctx.session.audit("lan.stopped")
            self.st_lan.setText("")
            self.ctx.settings["lan_auto"] = False
        self.ctx.save_settings()

    def _lan_submit(self, details: dict, qa: list, raw_text: str) -> str:
        """Called on the server thread. The store serialises writes with its lock."""
        s = self.ctx.session
        if s is None:
            raise RuntimeError("HoRUS is signed out")
        prev = (details.pop("prev_id", "") or "").strip().upper()
        pid = prev if prev and s.get_patient(prev) else s.create_patient(details)
        s.add_intake(pid, {"qa": qa, "raw_text": raw_text, "details": details}, source="lan")
        return pid

    def _on_lan_event(self, msg: str):
        self.status.showMessage(msg, 8000)
        self._update_badges()

    # ── lock / sign out / restore ────────────────────────────────────────
    def _check_idle(self):
        mins = int(self.ctx.settings.get("auto_lock_minutes", 15))
        in_kiosk = self.pages.currentWidget() is self.kiosk
        if not self.locked and not in_kiosk and time.time() - self.last_activity > mins * 60:
            self.lock()

    def lock(self):
        """Hide everything until the user unlocks. The tablet server keeps running meanwhile."""
        if self.locked:
            return
        from .login import UnlockDialog
        self.locked = True
        self.ctx.session.audit("screen.locked")
        self.centralWidget().hide()
        while True:
            dlg = UnlockDialog(self.ctx.session, self)
            dlg.exec()
            if dlg.result_kind == "unlocked":
                break
            if dlg.result_kind == "quit":
                self.locked = False
                self.close()
                return
            if dlg.result_kind == "switch":
                login = LoginDialog(self.ctx.store, self)
                if login.exec() == QDialog.Accepted and login.session:
                    self._end_session()
                    self.ctx.session = login.session
                    self.on_signed_in()
                    break
        self.locked = False
        self.last_activity = time.time()
        self.centralWidget().show()

    def _end_session(self):
        if self.lan and self.lan.running:
            self.set_lan(False)
        if self.ctx.session:
            try:
                self.ctx.session.logout()
            except Exception:
                pass

    def sign_out(self):
        self._backup_on_exit()
        self._end_session()
        self.hide()
        dlg = LoginDialog(self.ctx.store)
        if dlg.exec() == QDialog.Accepted and dlg.session:
            self.ctx.session = dlg.session
            self.on_signed_in()
            self.show()
        else:
            self.ctx.session = None
            self.close()

    def restore(self, backup: Path):
        """Restore over the current database (kept aside as a safety copy), then sign in again."""
        from .login import LoginDialog as _Login
        from PySide6.QtWidgets import QInputDialog, QLineEdit
        user = self.ctx.session.user.username
        pw, ok = QInputDialog.getText(self, "Restore", f"Password for {user} (as it was in the backup):",
                                      QLineEdit.Password)
        if not ok:
            return
        db = self.ctx.db_path
        aside = db.with_name(f"horus-before-restore-{datetime.now():%Y%m%d-%H%M%S}.db")
        tmp = db.with_name("horus-restoring.db")
        try:
            if tmp.exists():
                tmp.unlink()
            Store.restore(backup, tmp, user, pw).close()
        except Exception as e:
            return error(self, "Restore failed", "The backup could not be opened with that password.", str(e))
        self._end_session()
        self.ctx.store.close()
        for suffix in ("-wal", "-shm"):
            side = Path(str(db) + suffix)
            if side.exists():
                side.unlink()
        os.replace(db, aside)
        os.replace(tmp, db)
        self.ctx.store = Store(db)
        self.ctx.engine = None
        info(self, "Restored", f"The backup has been restored. The previous database was kept as {aside.name}.")
        self.hide()
        dlg = _Login(self.ctx.store)
        if dlg.exec() == QDialog.Accepted and dlg.session:
            self.ctx.session = dlg.session
            self._load_engine()
            self.on_signed_in()
            self.show()
        else:
            self.close()

    def _account_deleted(self, kind: str, password: str, delete_backups: bool):
        if kind == "account":
            info(self, "Account deleted", "Your account has been deleted.")
            self.ctx.session = None
            self.hide()
            dlg = LoginDialog(self.ctx.store)
            if dlg.exec() == QDialog.Accepted and dlg.session:
                self.ctx.session = dlg.session
                self.on_signed_in()
                self.show()
            else:
                self._close_quietly()
            return
        # the only user: delete the clinic, then start again with first-time setup
        import shutil
        if self.lan and self.lan.running:
            self.set_lan(False)
        try:
            self.ctx.session.delete_clinic(password, self.ctx.backup_dir if delete_backups else None)
        except Exception as e:
            return error(self, "Not deleted", str(e))
        for folder in ("books", "lan", "cache"):
            shutil.rmtree(self.ctx.paths.user / folder, ignore_errors=True)
        self.ctx.store = self.ctx.session = self.ctx.engine = None
        info(self, "Clinic deleted", "The clinic has been deleted from this computer. HoRUS will now start the "
                                     "first-time setup.")
        self.restart_setup = True
        self._close_quietly()

    def _close_quietly(self):
        """Close with no one signed in (so no automatic backup is attempted)."""
        self.ctx.session = None
        self.close()

    def _backup_on_exit(self):
        s = self.ctx.session
        if s and s.user.can("backup") and self.ctx.settings.get("backup_on_exit", True):
            try:
                make_backup(self.ctx, "auto")
            except Exception:
                log.exception("Automatic backup failed")

    def closeEvent(self, ev):
        from PySide6.QtCore import QThreadPool
        self._backup_on_exit()
        self.ctx.shutdown()                            # stops the AI server, so AI jobs end quickly
        QThreadPool.globalInstance().waitForDone(15000)  # never exit with background work running
        self._end_session()
        self.idle_timer.stop()
        self.badge_timer.stop()
        QApplication.instance().removeEventFilter(self._filter)
        super().closeEvent(ev)
