"""Settings: users, backups, audit log, clinic & AI settings, LAN tablets, own password."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QByteArray, Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
                               QHeaderView, QInputDialog, QLineEdit, QListWidget, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from horus.store import EDITIONS, ROLES, StoreError, iter_backups

from .context import AppContext
from .login import ChangePasswordDialog
from .widgets import button, card, confirm, error, fmt_ts, info, label, page, row

EDITION_NAMES = {"solo": "Solo (one doctor)", "clinic": "Clinic (several users)",
                 "clinic_lan": "Clinic LAN (with tablet intake)"}


def make_backup(ctx: AppContext, reason: str = "manual") -> Path:
    folder = ctx.backup_dir
    folder.mkdir(parents=True, exist_ok=True)
    name = f"HoRUS-{datetime.now():%Y%m%d-%H%M%S}-{reason}.horusbak"
    path = ctx.session.backup(folder / name)
    keep = int(ctx.settings.get("keep_backups", 14))
    for old in list(iter_backups(folder))[keep:]:
        try:
            old.unlink()
        except OSError:
            pass
    return path


class SettingsPage(QWidget):
    restore_requested = Signal(Path)
    lan_toggled = Signal(bool)

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        w, v = page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(w)
        v.addWidget(label("Settings", "h1"))
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)

    def rebuild(self):
        """Tabs depend on the signed-in user's role and the edition."""
        self.tabs.clear()
        u = self.ctx.session.user
        self.tabs.addTab(self._account_tab(), "My account")
        if u.can("backup"):
            self.tabs.addTab(self._backup_tab(), "Backups")
        if u.can("users.manage"):
            self.tabs.addTab(self._users_tab(), "Users")
        if u.can("settings"):
            self.tabs.addTab(self._clinic_tab(), "Clinic & AI")
        if u.can("intakes.write") and self.ctx.store.edition == "clinic_lan":
            self.tabs.addTab(self._lan_tab(), "Tablets (LAN)")
        if u.can("audit.read"):
            self.tabs.addTab(self._audit_tab(), "Audit log")

    # ── my account ───────────────────────────────────────────────────────
    def _account_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        u = self.ctx.session.user
        v.addWidget(card(label(u.display_name, "h2"), label(f"Username {u.username} · role: {u.role}", "muted"),
                         row(button("Change password…", slot=lambda: ChangePasswordDialog(
                             self.ctx.session, self).exec()), None)))
        v.addStretch(1)
        return w

    # ── backups ──────────────────────────────────────────────────────────
    def _backup_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Backups are sealed files: they can only be opened with a HoRUS user's password or the "
                          "recovery key. Keep copies on a USB drive or another computer.", "info"))
        self.bk_folder = QLineEdit(str(self.ctx.backup_dir))
        self.bk_folder.setReadOnly(True)
        self.bk_exit = QCheckBox("Back up automatically when HoRUS closes")
        self.bk_exit.setChecked(bool(self.ctx.settings.get("backup_on_exit", True)))
        self.bk_exit.toggled.connect(lambda on: self._set("backup_on_exit", on))
        self.bk_keep = QSpinBox()
        self.bk_keep.setRange(1, 365)
        self.bk_keep.setValue(int(self.ctx.settings.get("keep_backups", 14)))
        self.bk_keep.valueChanged.connect(lambda n: self._set("keep_backups", n))
        f = QFormLayout()
        f.addRow("Backup folder", row(self.bk_folder, button("Change…", slot=self._choose_folder)))
        f.addRow("", self.bk_exit)
        f.addRow("Keep the newest", row(self.bk_keep, label("backups"), None))
        v.addWidget(card(f))
        self.bk_list = QListWidget()
        v.addWidget(self.bk_list, 1)
        restore = button("Restore a backup…", danger=True, slot=self._restore)
        restore.setVisible(self.ctx.session.user.can("restore"))
        v.addLayout(row(button("Back up now", primary=True, slot=self._backup_now),
                        button("Open folder", slot=lambda: _open_folder(self.ctx.backup_dir)), None, restore))
        self._list_backups()
        return w

    def _list_backups(self):
        self.bk_list.clear()
        for p in iter_backups(self.ctx.backup_dir) if self.ctx.backup_dir.exists() else []:
            self.bk_list.addItem(f"{p.name}   ({p.stat().st_size / 1024:.0f} KB)")

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Backup folder", str(self.ctx.backup_dir))
        if d:
            self._set("backup_dir", d)
            self.bk_folder.setText(d)
            self._list_backups()

    def _backup_now(self):
        try:
            path = make_backup(self.ctx)
        except Exception as e:
            return error(self, "Backup failed", str(e))
        self._list_backups()
        info(self, "Backed up", f"Saved {path.name}")

    def _restore(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose backup", str(self.ctx.backup_dir),
                                              "HoRUS backup (*.horusbak)")
        if path and confirm(self, "Restore backup",
                            "Restoring replaces everything on this computer with the backup's contents. The "
                            "current database is kept aside as a safety copy. Continue?", "Restore", danger=True):
            self.restore_requested.emit(Path(path))

    # ── users ────────────────────────────────────────────────────────────
    def _users_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        if self.ctx.store.edition == "solo":
            v.addWidget(label("The Solo edition has a single account. Switch to the Clinic edition under "
                              "“Clinic & AI” to add doctors and staff.", "info"))
        self.u_table = QTableWidget(0, 5)
        self.u_table.setHorizontalHeaderLabels(["Username", "Name", "Role", "Active", "Last sign-in"])
        self.u_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.u_table.verticalHeader().setVisible(False)
        self.u_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.u_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        v.addWidget(self.u_table, 1)
        v.addWidget(label("Roles — admin: everything. doctor: patients, cases, reports, backups. staff: register "
                          "patients and take intakes; cannot see cases or reports.", "muted"))
        v.addLayout(row(button("Add user…", primary=True, slot=self._add_user), button("Reset password…",
                        slot=self._reset_pw), button("Change role…", slot=self._role),
                        button("Activate / deactivate", slot=self._toggle_active), None))
        self._list_users()
        return w

    def _list_users(self):
        self.users = self.ctx.session.list_users()
        self.u_table.setRowCount(0)
        for u in self.users:
            r = self.u_table.rowCount()
            self.u_table.insertRow(r)
            for c, val in enumerate((u["username"], u["display_name"], u["role"], "yes" if u["active"] else "no",
                                     fmt_ts(u["last_login"]))):
                self.u_table.setItem(r, c, QTableWidgetItem(str(val)))

    def _sel_user(self):
        r = self.u_table.currentRow()
        return self.users[r] if 0 <= r < len(self.users) else None

    def _add_user(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Add user")
        f = QFormLayout(dlg)
        un, dn, pw = QLineEdit(), QLineEdit(), QLineEdit()
        role = QComboBox()
        role.addItems(list(ROLES))
        role.setCurrentText("doctor")
        f.addRow("Username", un)
        f.addRow("Full name", dn)
        f.addRow("Role", role)
        f.addRow("Temporary password", pw)
        f.addRow(label("They will be asked to choose their own password at first sign-in.", "muted"))
        f.addRow(row(None, button("Cancel", slot=dlg.reject), button("Add", primary=True, slot=dlg.accept)))
        if dlg.exec() == QDialog.Accepted:
            try:
                self.ctx.session.add_user(un.text(), dn.text(), role.currentText(), pw.text())
            except StoreError as e:
                return error(self, "Not added", str(e))
            self._list_users()

    def _reset_pw(self):
        u = self._sel_user()
        if not u:
            return
        pw, ok = QInputDialog.getText(self, "Reset password", f"Temporary password for {u['username']}:",
                                      QLineEdit.Password)
        if ok:
            try:
                self.ctx.session.reset_user_password(u["id"], pw)
                info(self, "Password reset", "They will choose a new password at next sign-in.")
            except StoreError as e:
                error(self, "Not reset", str(e))

    def _role(self):
        u = self._sel_user()
        if not u:
            return
        role, ok = QInputDialog.getItem(self, "Change role", f"Role for {u['username']}:", list(ROLES),
                                        list(ROLES).index(u["role"]), False)
        if ok:
            try:
                self.ctx.session.set_user_role(u["id"], role)
            except StoreError as e:
                return error(self, "Not changed", str(e))
            self._list_users()

    def _toggle_active(self):
        u = self._sel_user()
        if not u:
            return
        try:
            self.ctx.session.set_user_active(u["id"], not u["active"])
        except StoreError as e:
            return error(self, "Not changed", str(e))
        self._list_users()

    # ── clinic & AI ──────────────────────────────────────────────────────
    def _clinic_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.cl_name = QLineEdit(self.ctx.store.clinic_name)
        self.cl_name.editingFinished.connect(lambda: self.ctx.session.set_clinic_name(self.cl_name.text()))
        self.cl_ed = QComboBox()
        for e in EDITIONS:
            self.cl_ed.addItem(EDITION_NAMES[e], e)
        self.cl_ed.setCurrentIndex(self.cl_ed.findData(self.ctx.store.edition))
        self.cl_ed.activated.connect(self._edition)
        self.lock_min = QSpinBox()
        self.lock_min.setRange(1, 240)
        self.lock_min.setValue(int(self.ctx.settings.get("auto_lock_minutes", 15)))
        self.lock_min.valueChanged.connect(lambda n: self._set("auto_lock_minutes", n))
        f.addRow("Clinic name", self.cl_name)
        f.addRow("Edition", self.cl_ed)
        f.addRow("Lock the screen after", row(self.lock_min, label("minutes without activity"), None))
        v.addWidget(card(f))

        llm, wh = self.ctx.llm, self.ctx.transcriber
        ai_txt = (f"Language model: {Path(llm.model_path).name}" if llm else "Language model: not installed") + \
                 ("\nSpeech model: Whisper (offline)" if wh else "\nSpeech model: not installed") + \
                 (f"\nThis computer has {self.ctx.ram_gb:.0f} GB of memory"
                  + (" — the AI may be slow; 8 GB or more is recommended." if self.ctx.ram_gb < 7.5 else ".")
                  if getattr(self.ctx, "ram_gb", None) else "")
        self.ai_on = QCheckBox("Use the offline AI (reading free text, explanations, voice, follow-up questions)")
        self.ai_on.setChecked(bool(self.ctx.settings.get("ai_enabled", True)))
        self.ai_on.setEnabled(bool(llm or wh))
        self.ai_on.toggled.connect(self._ai_toggle)
        self.ai_pre = QCheckBox("Start the AI in the background after sign-in (faster first use; uses ~3–4 GB RAM)")
        self.ai_pre.setChecked(bool(self.ctx.settings.get("ai_preload", True)))
        self.ai_pre.toggled.connect(lambda on: self._set("ai_preload", on))
        v.addWidget(card(label("Offline AI", "h2"), label(ai_txt, "muted"), self.ai_on, self.ai_pre,
                         label("The AI only reads text into symptoms and writes explanations. Remedies, their order, "
                               "the miasm and red flags are always decided by the dataset and fixed rules.", "muted")))

        v.addWidget(card(label("Data", "h2"),
                         row(button("Import patients.json from the online demo…", slot=self._import),
                             button("Issue a new recovery key…", slot=self._rotate), None)))
        v.addStretch(1)
        return w

    def _edition(self):
        e = self.cl_ed.currentData()
        try:
            self.ctx.session.set_edition(e)
        except StoreError as ex:
            self.cl_ed.setCurrentIndex(self.cl_ed.findData(self.ctx.store.edition))
            return error(self, "Edition", str(ex))
        info(self, "Edition changed", f"Now using {EDITION_NAMES[e]}.")
        self.rebuild()

    def _ai_toggle(self, on: bool):
        self._set("ai_enabled", on)
        self.ctx.apply_ai_setting()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose patients.json", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig") or "{}")
            stats = self.ctx.session.import_demo_patients(data)
        except Exception as e:
            return error(self, "Import failed", str(e))
        self.ctx.refresh_learning()
        info(self, "Imported", f"{stats['patients_new']} new patients ({stats['patients_merged']} merged), "
                               f"{stats['cases']} cases, {stats['intakes']} intakes. "
                               f"{stats['skipped']} cases were already here.")

    def _rotate(self):
        if not confirm(self, "New recovery key", "Issue a new recovery key? The old key will stop working."):
            return
        key = self.ctx.session.rotate_recovery_key()
        dlg = QDialog(self)
        dlg.setWindowTitle("New recovery key")
        vv = QVBoxLayout(dlg)
        vv.addWidget(label("Store this key safely, away from this computer. It is shown only once.", "warnbox"))
        k = label(key, wrap=False)
        k.setTextInteractionFlags(Qt.TextSelectableByMouse)
        k.setStyleSheet("font-family: Consolas, monospace; font-size: 16pt; padding: 12px;")
        vv.addWidget(k)
        vv.addLayout(row(None, button("Done", primary=True, slot=dlg.accept)))
        dlg.exec()

    # ── LAN tablets ──────────────────────────────────────────────────────
    def _lan_tab(self) -> QWidget:
        from PySide6.QtSvgWidgets import QSvgWidget
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Patients can fill in the intake questionnaire on a tablet or phone connected to the clinic "
                          "Wi-Fi. Nothing goes over the internet. Keep HoRUS signed in while tablets are in use.",
                          "info"))
        self.lan_state = label("", "h2")
        self.lan_urls = label("", rich=True)
        self.lan_urls.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lan_qr = QSvgWidget()
        self.lan_qr.setFixedSize(220, 220)
        self.lan_btn = button("", primary=True, slot=self._lan_toggle)
        v.addWidget(card(self.lan_state, row(self.lan_qr, self.lan_urls),
                         row(self.lan_btn, button("New pairing code", slot=self._lan_pin), None)))
        v.addWidget(label("First time on a tablet: open the address shown above. The browser warns that the "
                          "connection's certificate is not trusted — this is expected for a clinic-only server. "
                          "Check that the fingerprint matches, then continue.", "muted"))
        v.addStretch(1)
        self._lan_refresh()
        return w

    def _lan_refresh(self):
        lan = getattr(self.window(), "lan", None)
        on = bool(lan and lan.running)
        self.lan_state.setText("Tablet intake is ON" if on else "Tablet intake is off")
        self.lan_btn.setText("Stop tablet intake" if on else "Start tablet intake")
        if on:
            import qrcode
            import qrcode.image.svg
            img = qrcode.make(lan.urls[0], image_factory=qrcode.image.svg.SvgPathImage, box_size=8)
            self.lan_qr.load(QByteArray(img.to_string()))
            self.lan_qr.show()
            urls = "<br>".join(f"<b style='font-size:14pt'>{u}</b>" for u in lan.urls)
            self.lan_urls.setText(f"{urls}<br><br>Pairing code: <b style='font-size:20pt'>{lan.pin}</b>"
                                  f"<br><br><span style='color:#64748b'>Certificate fingerprint:<br>"
                                  f"{lan.fingerprint[:47]}<br>{lan.fingerprint[48:]}</span>")
        else:
            self.lan_qr.hide()
            self.lan_urls.setText("")

    def _lan_toggle(self):
        lan = getattr(self.window(), "lan", None)
        self.lan_toggled.emit(not (lan and lan.running))
        self._lan_refresh()

    def _lan_pin(self):
        lan = getattr(self.window(), "lan", None)
        if lan and lan.running:
            lan.new_pin()
            self._lan_refresh()

    # ── audit ────────────────────────────────────────────────────────────
    def _audit_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        t = QTableWidget(0, 5)
        t.setHorizontalHeaderLabels(["Time", "User", "Action", "Record", "Detail"])
        t.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for e in self.ctx.session.audit_entries(1000):
            r = t.rowCount()
            t.insertRow(r)
            for c, val in enumerate((fmt_ts(e["ts"][:19]), e["username"], e["action"], e["target"], e["detail"])):
                t.setItem(r, c, QTableWidgetItem(str(val or "")))
        v.addWidget(t, 1)
        v.addLayout(row(label("Every sign-in and every access to a patient record is logged. The log holds record "
                              "IDs only, never patient details.", "muted"), None,
                        button("Verify integrity", slot=self._verify)))
        return w

    def _verify(self):
        ok, broken = self.ctx.session.verify_audit()
        if ok:
            info(self, "Audit log", "The audit log is intact: no entries have been changed or removed.")
        else:
            error(self, "Audit log", f"The audit log has been altered (problem at entry {broken}).")

    def _set(self, key, value):
        self.ctx.settings[key] = value
        self.ctx.save_settings()


def _open_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))          # noqa: S606 — opens Explorer on the backup folder
