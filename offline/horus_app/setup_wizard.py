"""First run: create the clinic database (or restore one), show the recovery key, import data."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QFileDialog, QFormLayout, QLineEdit,
                               QRadioButton, QStackedWidget, QVBoxLayout, QWidget)

from horus.store import AuthError, Store, StoreError

from .context import AppContext
from .widgets import button, card, error, label, row

EDITION_TEXT = {
    "solo": ("Solo", "One doctor on one computer. A single password-protected account."),
    "clinic": ("Clinic", "Several people on one computer: doctors, and staff who register patients and take "
                         "intakes. Each has their own login and role."),
    "clinic_lan": ("Clinic LAN", "As Clinic, plus patients can fill in their intake questionnaire on a tablet "
                                 "over the clinic Wi-Fi. No internet is used."),
}


class SetupWizard(QDialog):
    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.session = None
        self.setWindowTitle("Set up HoRUS")
        self.setMinimumSize(640, 560)
        self.stack = QStackedWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.addWidget(self.stack)
        for build in (self._welcome, self._edition, self._account, self._recovery, self._import,
                      self._restore):
            self.stack.addWidget(build())

    # ── pages ────────────────────────────────────────────────────────────
    def _welcome(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Welcome to HoRUS", "h1"))
        v.addWidget(label("Clinical homeopathic intelligence that runs entirely on this computer. "
                          "Patient records are encrypted on this PC and nothing is sent over the internet.",
                          "muted"))
        v.addSpacing(16)
        v.addWidget(card(label("Set up a new clinic", "h2"),
                         label("Create the encrypted patient database and your administrator account.", "muted"),
                         row(None, button("Set up →", primary=True, slot=lambda: self.stack.setCurrentIndex(1)))))
        v.addWidget(card(label("Restore from a backup", "h2"),
                         label("Moving to a new computer? Restore a HoRUS backup file (.horusbak).", "muted"),
                         row(None, button("Restore…", slot=lambda: self.stack.setCurrentIndex(5)))))
        v.addStretch(1)
        return w

    def _edition(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("How will HoRUS be used?", "h1"))
        v.addWidget(label("You can change this later in Settings.", "muted"))
        self.edition_group = QButtonGroup(self)
        for i, (key, (name, desc)) in enumerate(EDITION_TEXT.items()):
            rb = QRadioButton(name)
            rb.setProperty("edition", key)
            f = QFont(rb.font())
            f.setBold(True)
            rb.setFont(f)
            rb.setChecked(key == "solo")
            self.edition_group.addButton(rb, i)
            v.addWidget(card(rb, label(desc, "muted")))
        v.addStretch(1)
        v.addLayout(row(button("← Back", slot=lambda: self.stack.setCurrentIndex(0)), None,
                        button("Next →", primary=True, slot=lambda: self.stack.setCurrentIndex(2))))
        return w

    def _account(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Clinic and administrator", "h1"))
        v.addWidget(label("The administrator manages users, backups and settings. Use a strong password "
                          "you will remember — it unlocks the encrypted records.", "muted"))
        form = QFormLayout()
        self.clinic = QLineEdit()
        self.clinic.setPlaceholderText("e.g. Satpati Homoeo Clinic")
        self.display = QLineEdit()
        self.display.setPlaceholderText("e.g. Dr. S. Satpati")
        self.username = QLineEdit()
        self.username.setPlaceholderText("e.g. doctor")
        self.pw1 = QLineEdit()
        self.pw1.setEchoMode(QLineEdit.Password)
        self.pw2 = QLineEdit()
        self.pw2.setEchoMode(QLineEdit.Password)
        for lab_, wid in (("Clinic name", self.clinic), ("Your name", self.display), ("Username", self.username),
                          ("Password (8+ characters)", self.pw1), ("Confirm password", self.pw2)):
            form.addRow(lab_, wid)
        v.addLayout(form)
        v.addStretch(1)
        self.create_btn = button("Create clinic", primary=True, slot=self._create)
        v.addLayout(row(button("← Back", slot=lambda: self.stack.setCurrentIndex(1)), None, self.create_btn))
        return w

    def _recovery(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Your recovery key", "h1"))
        v.addWidget(label("If the administrator password is forgotten, this key is the ONLY way to unlock the "
                          "patient records. HoRUS cannot recover them otherwise. Print it or save it somewhere "
                          "safe, away from this computer.", "warnbox"))
        self.key_label = label("", wrap=False)
        self.key_label.setAlignment(Qt.AlignCenter)
        self.key_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.key_label.setStyleSheet("font-family: Consolas, monospace; font-size: 17pt; padding: 18px;")
        v.addWidget(card(self.key_label))
        v.addLayout(row(button("Copy", slot=lambda: QGuiApplication.clipboard().setText(self._key)),
                        button("Save to a file…", slot=self._save_key), None))
        self.saved_ok = QCheckBox("I have stored the recovery key safely")
        v.addWidget(self.saved_ok)
        v.addStretch(1)
        nxt = button("Continue →", primary=True, slot=lambda: self.stack.setCurrentIndex(4))
        nxt.setEnabled(False)
        self.saved_ok.toggled.connect(nxt.setEnabled)
        v.addLayout(row(None, nxt))
        return w

    def _import(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Bring in existing patients (optional)", "h1"))
        v.addWidget(label("If you used the online HoRUS demo, choose its patients.json file to import every "
                          "patient, case and intake. You can also do this later from Settings.", "muted"))
        self.import_result = label("", "info")
        self.import_result.hide()
        v.addWidget(self.import_result)
        v.addLayout(row(button("Choose patients.json…", slot=self._do_import), None))
        v.addStretch(1)
        v.addLayout(row(None, button("Finish", primary=True, slot=self.accept)))
        return w

    def _restore(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Restore from a backup", "h1"))
        form = QFormLayout()
        self.bk_path = QLineEdit()
        self.bk_path.setReadOnly(True)
        pick = button("Browse…", slot=self._pick_backup)
        form.addRow("Backup file", row(self.bk_path, pick))
        self.r_user = QLineEdit()
        self.r_secret = QLineEdit()
        self.r_secret.setEchoMode(QLineEdit.Password)
        self.r_use_key = QCheckBox("Use the recovery key instead of a password")
        form.addRow("Username", self.r_user)
        form.addRow("Password", self.r_secret)
        form.addRow("", self.r_use_key)
        v.addLayout(form)
        v.addStretch(1)
        v.addLayout(row(button("← Back", slot=lambda: self.stack.setCurrentIndex(0)), None,
                        button("Restore", primary=True, slot=self._do_restore)))
        return w

    # ── actions ──────────────────────────────────────────────────────────
    def _edition_key(self) -> str:
        return self.edition_group.checkedButton().property("edition")

    def _create(self):
        if self.pw1.text() != self.pw2.text():
            return error(self, "Passwords differ", "The two passwords do not match.")
        try:
            store, key = Store.create(self.ctx.db_path, edition=self._edition_key(),
                                      clinic_name=self.clinic.text(), admin_username=self.username.text(),
                                      admin_password=self.pw1.text(), admin_display_name=self.display.text())
            self.ctx.store = store
            self.session = store.login(self.username.text(), self.pw1.text())
        except StoreError as e:
            return error(self, "Could not create the clinic", str(e))
        self._key = key
        self.key_label.setText(key)
        self.pw1.clear()
        self.pw2.clear()
        self.stack.setCurrentIndex(3)

    def _save_key(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save recovery key", "HoRUS recovery key.txt",
                                              "Text (*.txt)")
        if path:
            Path(path).write_text(
                f"HoRUS recovery key for {self.ctx.store.clinic_name}\n\n{self._key}\n\n"
                "Keep this somewhere safe and away from the clinic computer. It unlocks the encrypted "
                "patient records if the administrator password is lost.\n", encoding="utf-8")

    def _do_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose patients.json", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig") or "{}")
            if not isinstance(data, dict):
                raise ValueError("The file must contain a JSON object of patients.")
            stats = self.session.import_demo_patients(data)
        except Exception as e:
            return error(self, "Import failed", str(e))
        self.import_result.setText(f"Imported {stats['patients_new']} new patients "
                                   f"({stats['patients_merged']} merged), {stats['cases']} cases and "
                                   f"{stats['intakes']} intakes.")
        self.import_result.show()

    def _pick_backup(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose backup", "", "HoRUS backup (*.horusbak)")
        if path:
            self.bk_path.setText(path)

    def _do_restore(self):
        if not self.bk_path.text():
            return error(self, "Restore", "Choose a backup file first.")
        try:
            store = Store.restore(Path(self.bk_path.text()), self.ctx.db_path, self.r_user.text(),
                                  self.r_secret.text(), use_recovery_key=self.r_use_key.isChecked())
            self.ctx.store = store
            if self.r_use_key.isChecked():
                info_text = ("Restored. Sign in with your usual password, or use 'Forgot password' with the "
                             "recovery key to set a new one.")
                self.session = None
                from .widgets import info
                info(self, "Restored", info_text)
            else:
                self.session = store.login(self.r_user.text(), self.r_secret.text())
        except (StoreError, AuthError) as e:
            return error(self, "Restore failed", str(e))
        except Exception as e:
            return error(self, "Restore failed", "The backup could not be opened with those details.", str(e))
        self.accept()
