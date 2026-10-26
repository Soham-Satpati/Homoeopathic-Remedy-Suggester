"""Patients list and record view; pending intakes inbox."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFileDialog, QFormLayout, QHeaderView,
                               QInputDialog, QLineEdit, QListWidget, QListWidgetItem, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .context import AppContext
from .widgets import button, card, confirm, error, esc, fmt_ts, label, page, report_html, row


def _item(text, data=None) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text if text is not None else ""))
    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    if data is not None:
        it.setData(Qt.UserRole, data)
    return it


class PatientsPage(QWidget):
    new_case_for = Signal(str)

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        w, v = page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(w)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by name, patient ID or phone…")
        self.search.textChanged.connect(self.refresh)
        self.add_btn = button("+ Register patient", slot=self._register)
        v.addLayout(row(label("Patients", "h1"), None, self.search, self.add_btn))
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["Patient ID", "Name", "Age", "Gender", "Phone", "Last visit", "Cases"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(lambda r, _c: self._open(self.table.item(r, 0).text()))
        v.addWidget(self.table, 1)
        self.count = label("", "muted")
        v.addLayout(row(self.count, None, button("Open", slot=self._open_selected)))

    def refresh(self):
        if not self.ctx.session:
            return
        rows = self.ctx.session.list_patients(self.search.text())
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for p in rows:
            d = p["details"]
            r = self.table.rowCount()
            self.table.insertRow(r)
            pend = f"  •{p['pending_intakes']} new intake" if p["pending_intakes"] else ""
            vals = [p["id"], d.get("name", "") + pend, d.get("age", ""), d.get("gender", ""), d.get("contact", ""),
                    fmt_ts(p["last_case"]) if p["last_case"] else "—", p["case_count"]]
            for c, val in enumerate(vals):
                self.table.setItem(r, c, _item(val))
        self.table.setSortingEnabled(True)
        self.count.setText(f"{len(rows)} patient(s)")

    def _open_selected(self):
        r = self.table.currentRow()
        if r >= 0:
            self._open(self.table.item(r, 0).text())

    def _open(self, pid: str):
        dlg = PatientDialog(self.ctx, pid, self)
        dlg.new_case.connect(self.new_case_for.emit)
        dlg.exec()
        self.refresh()

    def _register(self):
        dlg = DetailsDialog(self, {})
        if dlg.exec() == QDialog.Accepted:
            try:
                pid = self.ctx.session.create_patient(dlg.details())
            except Exception as e:
                return error(self, "Not registered", str(e))
            self.refresh()
            self._open(pid)


class DetailsDialog(QDialog):
    def __init__(self, parent, details: dict, title="Patient details"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(460)
        v = QVBoxLayout(self)
        f = QFormLayout()
        self.f = {}
        for key, lab_ in (("name", "Full name"), ("age", "Age"), ("gender", "Gender"), ("contact", "Phone"),
                          ("address", "Address"), ("notes", "Notes")):
            if key == "gender":
                w = QComboBox()
                w.addItems(["", "Female", "Male", "Other"])
                w.setCurrentText(details.get(key, ""))
            else:
                w = QLineEdit(details.get(key, ""))
            self.f[key] = w
            f.addRow(lab_, w)
        v.addLayout(f)
        v.addLayout(row(None, button("Cancel", slot=self.reject), button("Save", primary=True, slot=self._ok)))

    def _ok(self):
        if not self.f["name"].text().strip():
            return error(self, "Name needed", "Enter the patient's name.")
        self.accept()

    def details(self) -> dict:
        return {k: (w.currentText() if isinstance(w, QComboBox) else w.text()).strip() for k, w in self.f.items()}


class PatientDialog(QDialog):
    new_case = Signal(str)

    def __init__(self, ctx: AppContext, pid: str, parent=None):
        super().__init__(parent)
        self.ctx, self.pid = ctx, pid
        s = ctx.session
        self.setWindowTitle(f"Patient {pid}")
        self.resize(1000, 720)
        v = QVBoxLayout(self)
        self.p = s.get_patient(pid)
        d = self.p["details"]
        self.head = label("", "h1")
        self.sub = label("", "muted")
        self._set_head(d)
        can_cases = s.user.can("cases.write")
        nc = button("New case for this patient", primary=True, slot=self._new_case)
        nc.setVisible(can_cases)
        v.addLayout(row(self.head, None, button("Edit details", slot=self._edit), nc))
        v.addWidget(self.sub)
        tabs = QTabWidget()
        v.addWidget(tabs, 1)

        if s.user.can("cases.read"):
            self.cases = s.list_cases(pid)
            split = QSplitter()
            self.case_list = QListWidget()
            for c in reversed(self.cases):
                top = ", ".join((c.get("top_remedies") or [])[:3]) or "—"
                it = QListWidgetItem(f"{fmt_ts(c['timestamp'])}\n{top}")
                it.setData(Qt.UserRole, c["id"])
                self.case_list.addItem(it)
            self.case_view = QTextBrowser()
            split.addWidget(self.case_list)
            split.addWidget(self.case_view)
            split.setSizes([260, 700])
            self.case_list.currentItemChanged.connect(self._show_case)
            cw = QWidget()
            cl = QVBoxLayout(cw)
            cl.addWidget(split, 1)
            del_case = button("Delete this case…", danger=True, slot=self._delete_case)
            del_case.setVisible(s.user.can("patients.delete"))
            cl.addLayout(row(del_case, None, button("Save PDF of this case…", slot=self._case_pdf)))
            self.cases_tab_index = tabs.count()
            tabs.addTab(cw, f"Cases ({len(self.cases)})")
            if self.case_list.count():
                self.case_list.setCurrentRow(0)

        if s.user.can("intakes.read"):
            ib = QTextBrowser()
            parts = []
            for it in s.patient_intakes(pid):
                qa = "".join(f"<p><b>{esc(x['label'])}</b><br>{esc(x['answer']) or '—'}</p>" for x in it.get("qa", [])
                             if x.get("answer"))
                parts.append(f"<h3>{esc(fmt_ts(it['timestamp']))} · {esc(it['status'])} · {esc(it['source'])}</h3>"
                             f"{qa or '<p>' + esc(it.get('raw_text', '')) + '</p>'}<hr>")
            ib.setHtml("".join(parts) or "<p>No intake questionnaires.</p>")
            tabs.addTab(ib, "Intakes")

        bottom = []
        if s.user.can("patients.delete"):
            bottom.append(button("Delete patient…", danger=True, slot=self._delete))
        v.addLayout(row(*bottom, None, button("Close", slot=self.accept)))

    def _set_head(self, d: dict):
        self.head.setText(d.get("name") or self.pid)
        bits = [self.pid] + [x for x in (d.get("age") and f"Age {d['age']}", d.get("gender"), d.get("contact")) if x]
        self.sub.setText("  ·  ".join(bits))

    def _edit(self):
        dlg = DetailsDialog(self, self.p["details"])
        if dlg.exec() == QDialog.Accepted:
            try:
                self.ctx.session.save_patient_details(self.pid, dlg.details())
            except Exception as e:
                return error(self, "Not saved", str(e))
            self.p["details"] = dlg.details()
            self._set_head(self.p["details"])

    def _case(self):
        it = self.case_list.currentItem() if hasattr(self, "case_list") else None
        if it is None:
            return None
        return next((c for c in self.cases if c["id"] == it.data(Qt.UserRole)), None)

    def _show_case(self, *_):
        c = self._case()
        if not c:
            return
        rpt = c.get("report")
        if rpt:
            html = report_html(rpt, c.get("categorised") or {}, fmt_ts(c["timestamp"]))
        else:   # imported demo case: no stored report → show what was stored
            cat = c.get("categorised") or {}
            html = (f"<h2>{esc(fmt_ts(c['timestamp']))}</h2><p><b>Top remedies:</b> "
                    f"{esc(', '.join(c.get('top_remedies') or []))}</p>"
                    f"<p><b>Notes:</b> {esc(c.get('clinical_notes', ''))}</p>"
                    f"<p><b>Patient's words:</b> {esc(c.get('raw_symptoms', ''))}</p>"
                    f"<p class='muted'>Imported from the online demo. Miasm: {esc(cat.get('miasm', '—'))}</p>")
        if c.get("raw_symptoms") and rpt:
            html = html.replace("</body>", f"<h2>Patient's words</h2><p>{esc(c['raw_symptoms'])}</p></body>")
        self.case_view.setHtml(html)

    def _case_pdf(self):
        c = self._case()
        if not c:
            return
        if self.ctx.engine is None:
            return error(self, "Still loading", "Try again in a moment.")
        rpt = c.get("report")
        cat = c.get("categorised") or {}
        if not rpt:
            rpt = self.ctx.engine.report(self.pid, cat)
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF", f"HoRUS_{self.pid}_{c['timestamp'][:10]}.pdf",
                                              "PDF (*.pdf)")
        if path:
            try:
                Path(path).write_bytes(self.ctx.engine.pdf(self.pid, cat, rpt, patient_details=self.p["details"]))
                self.ctx.session.audit("pdf.exported", self.pid)
            except Exception as e:
                error(self, "PDF not saved", str(e))

    def _delete_case(self):
        c = self._case()
        if not c:
            return
        if not confirm(self, "Delete case",
                       f"Permanently delete the case of {fmt_ts(c['timestamp'])}?\n\nIts symptoms and remedies "
                       "are also removed from the learned clusters and remedy weights.", "Delete", danger=True):
            return
        try:
            self.ctx.session.delete_case(c["id"])
        except Exception as e:
            return error(self, "Not deleted", str(e))
        self.cases = [x for x in self.cases if x["id"] != c["id"]]
        self.case_list.takeItem(self.case_list.currentRow())
        if not self.case_list.count():
            self.case_view.setHtml("<p>No cases.</p>")
        self.ctx.refresh_learning()

    def _new_case(self):
        self.new_case.emit(self.pid)
        self.accept()

    def _delete(self):
        text, ok = QInputDialog.getText(self, "Delete patient",
                                        f"This permanently deletes {self.pid} with all cases and intakes.\n"
                                        f"Type the patient ID to confirm:")
        if ok and text.strip() == self.pid:
            try:
                self.ctx.session.delete_patient(self.pid)
            except Exception as e:
                return error(self, "Not deleted", str(e))
            self.ctx.refresh_learning()
            self.accept()


class IntakesPage(QWidget):
    start_case = Signal(dict)
    changed = Signal()

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        w, v = page()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(w)
        v.addWidget(label("New patient intakes", "h1"))
        v.addWidget(label("Questionnaires patients filled in (here in kiosk mode or on a clinic tablet).", "muted"))
        split = QSplitter()
        self.list = QListWidget()
        self.view = QTextBrowser()
        split.addWidget(self.list)
        split.addWidget(self.view)
        split.setSizes([300, 700])
        v.addWidget(split, 1)
        self.start_btn = button("Start case from this intake", primary=True, slot=self._start)
        self.dismiss_btn = button("Dismiss", slot=self._dismiss)
        v.addLayout(row(None, self.dismiss_btn, self.start_btn))
        self.list.currentItemChanged.connect(self._show)
        self.items: list = []

    def refresh(self) -> int:
        if not self.ctx.session or not self.ctx.session.user.can("intakes.read"):
            return 0
        self.items = self.ctx.session.pending_intakes()
        self.list.clear()
        for it in self.items:
            li = QListWidgetItem(f"{it['details'].get('name', '?')} · {it['patient_id']}\n"
                                 f"{fmt_ts(it['timestamp'])} · {it['source']}")
            li.setData(Qt.UserRole, it["id"])
            self.list.addItem(li)
        self.view.setHtml("<p class='muted'>No new intakes.</p>" if not self.items else "")
        if self.items:
            self.list.setCurrentRow(0)
        can_case = self.ctx.session.user.can("cases.write")
        self.start_btn.setVisible(can_case)
        return len(self.items)

    def _current(self):
        li = self.list.currentItem()
        return next((x for x in self.items if li and x["id"] == li.data(Qt.UserRole)), None)

    def _show(self, *_):
        it = self._current()
        if not it:
            return
        d = it["details"]
        qa = "".join(f"<p><b>{esc(x['label'])}</b><br>{esc(x['answer']) or '—'}</p>" for x in it.get("qa", []))
        self.view.setHtml(f"<h2>{esc(d.get('name', ''))}</h2><p class='muted'>{esc(it['patient_id'])} · "
                          f"Age {esc(d.get('age', '—'))} · {esc(d.get('gender', ''))} · {esc(d.get('contact', ''))}"
                          f"</p>{qa}")

    def _start(self):
        it = self._current()
        if it:
            self.start_case.emit(it)

    def _dismiss(self):
        it = self._current()
        if it and confirm(self, "Dismiss intake", "Mark this intake as dismissed? It stays in the patient's record."):
            self.ctx.session.mark_intake(it["id"], "dismissed")
            self.refresh()
            self.changed.emit()
