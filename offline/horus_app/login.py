"""Sign-in, forced password change, and password recovery."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QFormLayout, QLineEdit, QVBoxLayout

from horus.store import AuthError, Session, Store, StoreError

from .widgets import button, error, info, label, row


class LoginDialog(QDialog):
    def __init__(self, store: Store, parent=None, locked_user: str | None = None):
        super().__init__(parent)
        self.store = store
        self.session: Session | None = None
        self.setWindowTitle("HoRUS — sign in")
        self.setMinimumWidth(420)
        v = QVBoxLayout(self)
        v.setContentsMargins(28, 24, 28, 24)
        v.addWidget(label(store.clinic_name or "HoRUS", "h1"))
        v.addWidget(label("Screen locked — sign in to continue." if locked_user else "Sign in to continue.",
                          "muted"))
        form = QFormLayout()
        self.user = QComboBox()
        self.user.setEditable(True)
        self.user.addItems(store.usernames())
        if locked_user:
            self.user.setCurrentText(locked_user)
        self.pw = QLineEdit()
        self.pw.setEchoMode(QLineEdit.Password)
        self.pw.returnPressed.connect(self._login)
        form.addRow("Username", self.user)
        form.addRow("Password", self.pw)
        v.addLayout(form)
        self.msg = label("", "warnbox")
        self.msg.hide()
        v.addWidget(self.msg)
        self.go = button("Sign in", primary=True, slot=self._login)
        v.addLayout(row(button("Forgot password…", slot=self._forgot), None, self.go))
        self.pw.setFocus(Qt.OtherFocusReason)

    def _login(self):
        try:
            self.session = self.store.login(self.user.currentText(), self.pw.text())
        except AuthError as e:
            self.msg.setText(str(e))
            self.msg.show()
            self.pw.selectAll()
            return
        self.pw.clear()
        if self.session.user.must_change_password:
            dlg = ChangePasswordDialog(self.session, self, forced=True)
            if dlg.exec() != QDialog.Accepted:
                self.session.logout()
                self.session = None
                return
        self.accept()

    def _forgot(self):
        dlg = RecoveryDialog(self.store, self)
        if dlg.exec() == QDialog.Accepted:
            self.session = dlg.session
            self.accept()


class ChangePasswordDialog(QDialog):
    def __init__(self, session: Session, parent=None, forced: bool = False):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Change password")
        self.setMinimumWidth(400)
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        if forced:
            v.addWidget(label("Please choose your own password before continuing.", "info"))
        form = QFormLayout()
        self.old = QLineEdit()
        self.new1 = QLineEdit()
        self.new2 = QLineEdit()
        for w in (self.old, self.new1, self.new2):
            w.setEchoMode(QLineEdit.Password)
        form.addRow("Current password", self.old)
        form.addRow("New password", self.new1)
        form.addRow("Confirm new password", self.new2)
        v.addLayout(form)
        v.addLayout(row(None, button("Cancel", slot=self.reject), button("Change", primary=True, slot=self._go)))

    def _go(self):
        if self.new1.text() != self.new2.text():
            return error(self, "Passwords differ", "The new passwords do not match.")
        try:
            self.session.change_password(self.old.text(), self.new1.text())
        except (AuthError, StoreError) as e:
            return error(self, "Not changed", str(e))
        self.accept()


class RecoveryDialog(QDialog):
    def __init__(self, store: Store, parent=None):
        super().__init__(parent)
        self.store = store
        self.session = None
        self.setWindowTitle("Reset administrator password")
        self.setMinimumWidth(480)
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        v.addWidget(label("Enter the recovery key that was shown when HoRUS was set up. Other users can ask "
                          "an administrator to reset their password from Settings → Users.", "muted"))
        form = QFormLayout()
        self.key = QLineEdit()
        self.key.setPlaceholderText("XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX")
        self.user = QLineEdit()
        self.new1 = QLineEdit()
        self.new2 = QLineEdit()
        for w in (self.new1, self.new2):
            w.setEchoMode(QLineEdit.Password)
        form.addRow("Recovery key", self.key)
        form.addRow("Administrator username", self.user)
        form.addRow("New password", self.new1)
        form.addRow("Confirm", self.new2)
        v.addLayout(form)
        v.addLayout(row(None, button("Cancel", slot=self.reject), button("Reset", primary=True, slot=self._go)))

    def _go(self):
        if self.new1.text() != self.new2.text():
            return error(self, "Passwords differ", "The new passwords do not match.")
        try:
            self.session = self.store.recover(self.key.text(), self.user.text(), self.new1.text())
        except (AuthError, StoreError) as e:
            return error(self, "Not reset", str(e))
        info(self, "Password reset", "The password has been reset and you are signed in.")
        self.accept()
