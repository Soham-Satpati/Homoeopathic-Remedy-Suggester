"""New case: words (typed or dictated) → symptoms (AI parse or structured entry) → doctor review →
dataset report (saved automatically, as in the demo) → PDF, symptom suggestions, comparative report."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout,
                               QHeaderView, QLineEdit, QPlainTextEdit, QSplitter, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser, QVBoxLayout,
                               QWidget)

from horus.parsing import SYMPTOM_GROUPS

from .context import AppContext, run_job
from .voice import DictateButton
from .widgets import Busy, button, card, error, esc, label, page, report_html, row

GROUP_LABEL = {"psychological": "Mind", "general": "General", "physical": "Physical"}
MODES = [("case_studies", "Classical totality (causation → mind → generals → particulars)"),
         ("keynote", "Keynotes / strange, rare, peculiar")]


def _split(text: str) -> list[str]:
    return [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]


class SymptomTable(QTableWidget):
    """Editable symptom list. Column 0 'Use' unticked = unverified (left out of the ranking)."""

    COLS = ["Use", "Category", "Symptom", "Worse from", "Better from"]

    def __init__(self):
        super().__init__(0, len(self.COLS))
        self.setHorizontalHeaderLabels(self.COLS)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.Interactive)
        h.setSectionResizeMode(4, QHeaderView.Interactive)
        self.setColumnWidth(3, 190)
        self.setColumnWidth(4, 190)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setWordWrap(True)

    def add(self, group: str = "physical", sym: dict | None = None):
        sym = sym or {"symptom": "", "worse": [], "better": []}
        r = self.rowCount()
        self.insertRow(r)
        use = QTableWidgetItem()
        use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        use.setCheckState(Qt.Unchecked if sym.get("unverified") else Qt.Checked)
        if sym.get("unverified"):
            use.setToolTip("Not found in the patient's words — tick to use it anyway.")
        self.setItem(r, 0, use)
        combo = QComboBox()
        for g in ("psychological", "general", "physical"):
            combo.addItem(GROUP_LABEL[g], g)
        combo.setCurrentIndex(combo.findData(group))
        self.setCellWidget(r, 1, combo)
        self.setItem(r, 2, QTableWidgetItem(sym.get("symptom", "")))
        self.setItem(r, 3, QTableWidgetItem(", ".join(sym.get("worse", []))))
        self.setItem(r, 4, QTableWidgetItem(", ".join(sym.get("better", []))))
        return r

    def load(self, cat: dict):
        self.setRowCount(0)
        for g in ("psychological", "general", "physical"):
            for s in cat.get(g, []):
                self.add(g, s)
        self.resizeRowsToContents()

    def remove_selected(self):
        for r in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(r)

    def symptoms(self) -> dict:
        out = {g: [] for g in SYMPTOM_GROUPS}
        for r in range(self.rowCount()):
            text = (self.item(r, 2).text() if self.item(r, 2) else "").strip()
            if not text:
                continue
            s = {"symptom": text, "worse": _split(self.item(r, 3).text() if self.item(r, 3) else ""),
                 "better": _split(self.item(r, 4).text() if self.item(r, 4) else "")}
            if self.item(r, 0).checkState() != Qt.Checked:
                s["unverified"] = True
            out[self.cellWidget(r, 1).currentData()].append(s)
        return out


class CaseFlow(QWidget):
    saved = Signal(str)             # patient id

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.cat: dict | None = None
        self.report: dict | None = None
        self.raw = ""
        self.intake_id: str | None = None
        self.case_saved = False
        self.case_id: int | None = None
        self.generation = 0           # bumps on every analysis; stale AI results are ignored
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.steps = QStackedWidget()
        lay.addWidget(self.steps)
        self.steps.addWidget(self._input_page())
        self.steps.addWidget(self._review_page())
        self.steps.addWidget(self._report_page())
        self.steps.addWidget(self._compare_page())

    # ════════════════════════════════════════════════════════════════════
    # 1. INPUT
    # ════════════════════════════════════════════════════════════════════
    def _input_page(self) -> QWidget:
        w, v = page()
        v.addWidget(label("New case", "h1"))

        # patient
        self.pt_existing = QComboBox()
        self.pt_existing.setEditable(True)
        self.pt_existing.setInsertPolicy(QComboBox.NoInsert)
        self.pt_existing.setMinimumWidth(320)
        self.pt_name, self.pt_age, self.pt_contact = QLineEdit(), QLineEdit(), QLineEdit()
        self.pt_age.setMaximumWidth(70)
        self.pt_gender = QComboBox()
        self.pt_gender.addItems(["", "Female", "Male", "Other"])
        self.pt_existing.currentIndexChanged.connect(self._existing_changed)
        form = QFormLayout()
        form.addRow("Patient", row(self.pt_existing, label("or leave as “New patient” and fill in:", "muted"),
                                   None))
        form.addRow("Name", row(self.pt_name, label("Age"), self.pt_age, label("Gender"), self.pt_gender,
                                label("Phone"), self.pt_contact))
        self.mode = QComboBox()
        for k, t in MODES:
            self.mode.addItem(t, k)
        form.addRow("Analysis", self.mode)
        v.addWidget(card(form))

        # words / structured
        self.tabs = QTabWidget()
        words = QWidget()
        wl = QVBoxLayout(words)
        wl.addWidget(label("The patient's own words — type, paste, or dictate. HoRUS reads them into symptoms and "
                           "then checks every symptom against these words.", "muted"))
        self.words = QPlainTextEdit()
        self.words.setPlaceholderText("e.g. Since her husband died she wakes after midnight with anxiety…")
        wl.addWidget(self.words, 1)
        self.dictate = DictateButton(lambda wav, mime: self.ctx.transcriber.transcribe(wav, mime).strip())
        self.dictate.text_ready.connect(lambda t: self.words.appendPlainText(t))
        self.ai_note = label("", "warnbox")
        wl.addWidget(self.ai_note)
        wl.addLayout(row(self.dictate, None))
        self.tabs.addTab(words, "Patient's words")

        struct = QWidget()
        sl = QVBoxLayout(struct)
        sl.addWidget(label("Enter symptoms directly. If the patient's words are also filled in, symptoms not "
                           "found in them are flagged for you to confirm.", "muted"))
        self.st_table = SymptomTable()
        sl.addWidget(self.st_table, 1)
        sl.addLayout(row(button("+ Mind", slot=lambda: self.st_table.add("psychological")),
                         button("+ General", slot=lambda: self.st_table.add("general")),
                         button("+ Physical", slot=lambda: self.st_table.add("physical")),
                         button("Remove selected", slot=self.st_table.remove_selected), None))
        f2 = QFormLayout()
        self.st_cw, self.st_cb, self.st_cause = QLineEdit(), QLineEdit(), QLineEdit()
        self.st_cw.setPlaceholderText("comma-separated, e.g. after midnight, cold")
        f2.addRow("Whole case — worse", self.st_cw)
        f2.addRow("Whole case — better", self.st_cb)
        f2.addRow("Causation (ailments from)", self.st_cause)
        sl.addLayout(f2)
        self.tabs.addTab(struct, "Structured entry")
        v.addWidget(self.tabs, 1)

        self.in_busy = Busy()
        self.read_btn = button("Read the case →", primary=True, slot=self._read_case)
        v.addLayout(row(self.in_busy, None, button("Clear", slot=self.reset), self.read_btn))
        return w

    def refresh_capabilities(self):
        ai, voice = self.ctx.ai_ready, self.ctx.voice_ready
        self.dictate.setVisible(voice)
        if ai:
            self.ai_note.hide()
        else:
            self.ai_note.setText("The offline AI is not installed or is switched off, so free text cannot be read "
                                 "into symptoms automatically. Use “Structured entry” — the words you type here "
                                 "are still used to check those symptoms.")
            self.ai_note.show()

    def refresh_patients(self, select: str | None = None):
        self.pt_existing.blockSignals(True)
        self.pt_existing.clear()
        self.pt_existing.addItem("New patient", None)
        for p in self.ctx.session.list_patients():
            d = p["details"]
            self.pt_existing.addItem(f"{p['id']} — {d.get('name', '')}", p["id"])
        self.pt_existing.blockSignals(False)
        if select:
            i = self.pt_existing.findData(select)
            if i >= 0:
                self.pt_existing.setCurrentIndex(i)
        self._existing_changed()

    def _existing_changed(self):
        pid = self.pt_existing.currentData()
        d = {}
        if pid:
            p = self.ctx.session.get_patient(pid)
            d = p["details"] if p else {}
        self.pt_name.setText(d.get("name", ""))
        self.pt_age.setText(d.get("age", ""))
        self.pt_gender.setCurrentText(d.get("gender", ""))
        self.pt_contact.setText(d.get("contact", ""))

    def start_from_intake(self, intake: dict):
        """Open a new case pre-filled from a patient's intake questionnaire."""
        self.reset()
        self.refresh_patients(select=intake["patient_id"])
        self.words.setPlainText(intake.get("raw_text", ""))
        self.intake_id = intake["id"]
        self.tabs.setCurrentIndex(0 if self.ctx.ai_ready else 1)

    def reset(self):
        self.generation += 1
        self.case_id = None
        if hasattr(self, "rp_busy"):
            self.rp_busy.stop()
        self.cat = self.report = None
        self.raw = ""
        self.intake_id = None
        self.case_saved = False
        self.words.clear()
        self.st_table.setRowCount(0)
        for e in (self.st_cw, self.st_cb, self.st_cause):
            e.clear()
        self.steps.setCurrentIndex(0)
        if self.ctx.session:
            self.refresh_patients()
        self.refresh_capabilities()
        self.tabs.setCurrentIndex(0 if self.ctx.ai_ready else 1)

    def _read_case(self):
        if self.ctx.engine is None:
            return error(self, "Still loading", "The remedy dataset is still loading — try again in a moment.")
        self.raw = self.words.toPlainText().strip()
        structured = self.tabs.currentIndex() == 1
        engine = self.ctx.engine
        if structured:
            st = self.st_table.symptoms()
            if not any(st.values()):
                return error(self, "No symptoms", "Add at least one symptom.")
            cat = engine.empty_case()
            for g in SYMPTOM_GROUPS:
                cat[g] = [{k: v for k, v in s.items() if k != "unverified"} for s in st[g]]
            cat["case_modalities"] = {"worse": _split(self.st_cw.text()), "better": _split(self.st_cb.text())}
            cat["causation"] = _split(self.st_cause.text())
            cat = engine.finalise(cat, self.raw) if self.raw else engine.finalise_structured(cat)
            self._to_review(cat)
            return
        if not self.raw:
            return error(self, "Nothing to read", "Type or dictate the patient's words first.")
        if not self.ctx.ai_ready:
            return error(self, "AI not available",
                         "Reading free text needs the offline AI. Use “Structured entry” instead.")
        self.read_btn.setEnabled(False)
        self.in_busy.start("Reading the case with the offline AI… (first use can take up to a minute)")
        mode = self.mode.currentData()

        def done(cat):
            self.in_busy.stop()
            self.read_btn.setEnabled(True)
            self._to_review(cat)

        def fail(msg, tb):
            self.in_busy.stop()
            self.read_btn.setEnabled(True)
            error(self, "Could not read the case",
                  "The offline AI could not read this case. You can enter the symptoms in “Structured "
                  "entry” instead.", tb)
        run_job(lambda: engine.categorise(self.raw, mode=mode), done, fail)

    # ════════════════════════════════════════════════════════════════════
    # 2. REVIEW
    # ════════════════════════════════════════════════════════════════════
    def _review_page(self) -> QWidget:
        w, v = page()
        v.addWidget(label("Review the symptoms", "h1"))
        v.addWidget(label("Correct anything the reading got wrong. Unticked symptoms were not found in the "
                          "patient's words and are left out of the ranking unless you tick them.", "muted"))
        self.flags = label("", "banner", rich=True)
        v.addWidget(self.flags)
        self.rv_table = SymptomTable()
        v.addWidget(self.rv_table, 1)
        v.addLayout(row(button("+ Mind", slot=lambda: self.rv_table.add("psychological")),
                        button("+ General", slot=lambda: self.rv_table.add("general")),
                        button("+ Physical", slot=lambda: self.rv_table.add("physical")),
                        button("Remove selected", slot=self.rv_table.remove_selected), None))
        f = QFormLayout()
        self.rv_cw, self.rv_cb, self.rv_cause = QLineEdit(), QLineEdit(), QLineEdit()
        self.rv_summary = QLineEdit()
        f.addRow("Whole case — worse", self.rv_cw)
        f.addRow("Whole case — better", self.rv_cb)
        f.addRow("Causation (ailments from)", self.rv_cause)
        f.addRow("Clinical summary", self.rv_summary)
        v.addWidget(card(f))
        self.rv_miasm = label("", "muted")
        v.addWidget(self.rv_miasm)
        self.rv_busy = Busy("Ranking remedies from the dataset…")
        self.analyse_btn = button("Analyse →", primary=True, slot=self._analyse)
        v.addLayout(row(button("← Back", slot=lambda: self.steps.setCurrentIndex(0)), self.rv_busy, None,
                        self.analyse_btn))
        return w

    def _to_review(self, cat: dict):
        self.cat = cat
        self.rv_table.load(cat)
        cm = cat.get("case_modalities") or {}
        self.rv_cw.setText(", ".join(cm.get("worse", [])))
        self.rv_cb.setText(", ".join(cm.get("better", [])))
        self.rv_cause.setText(", ".join(cat.get("causation", [])))
        self.rv_summary.setText(cat.get("clinical_summary", ""))
        self._show_flags(cat.get("red_flags") or [])
        self.rv_miasm.setText(f"Miasm (rubric, updates on analysis): {cat.get('miasm', '—')}")
        self.steps.setCurrentIndex(1)

    def _show_flags(self, flags: list):
        if flags:
            self.flags.setText("<b>⚠ Seek urgent medical care.</b> " + " ".join(
                f"<b>{esc(f['title'])}.</b> {esc(f['advice'])}" for f in flags))
            self.flags.show()
        else:
            self.flags.hide()

    def _collect_review(self) -> dict:
        cat = copy.deepcopy(self.cat or self.ctx.engine.empty_case())
        cat.update(self.rv_table.symptoms())
        cat["case_modalities"] = {"worse": _split(self.rv_cw.text()), "better": _split(self.rv_cb.text())}
        cat["causation"] = _split(self.rv_cause.text())
        cat["clinical_summary"] = self.rv_summary.text().strip()
        cat["unverified"] = [s["symptom"] for g in SYMPTOM_GROUPS for s in cat[g] if s.get("unverified")]
        cat.setdefault("_raw_text", self.raw)
        return cat

    def _patient_details(self) -> dict:
        return {"name": self.pt_name.text().strip(), "age": self.pt_age.text().strip(),
                "gender": self.pt_gender.currentText(), "contact": self.pt_contact.text().strip()}

    def _analyse(self):
        cat = self._collect_review()
        if not any(s for g in SYMPTOM_GROUPS for s in cat[g] if not s.get("unverified")):
            return error(self, "No symptoms in use", "Tick or add at least one symptom.")
        engine = self.ctx.engine
        pid = self.pt_existing.currentData() or "NEW"
        details = self._patient_details()
        mode = self.mode.currentData()
        self.analyse_btn.setEnabled(False)
        self.rv_busy.start("Ranking remedies from the dataset…")
        self.generation += 1
        gen = self.generation

        def done(res):
            cat2, rpt = res
            self.rv_busy.stop()
            self.analyse_btn.setEnabled(True)
            self.cat, self.report = cat2, rpt
            self._show_report()
            if self.ctx.ai_ready and rpt.get("top10Remedies"):
                self._narrate(gen, pid, details, mode)

        def fail(msg, tb):
            self.rv_busy.stop()
            self.analyse_btn.setEnabled(True)
            error(self, "Analysis failed", msg, tb)

        def work():   # instant: dataset ranking only; explanations follow in the background
            cat2 = engine.refresh(cat)
            return cat2, engine.report(pid, cat2, patient_details=details, mode=mode, narrate=False)
        run_job(work, done, fail)

    def _narrate(self, gen: int, pid: str, details: dict, mode: str):
        engine, cat = self.ctx.engine, self.cat
        self.rp_busy.start("Writing explanations with the offline AI — the ranking above is final…")

        def done(rpt):
            if gen != self.generation:
                return                                   # the doctor has moved on
            self.rp_busy.stop()
            self.report = rpt
            self.rp_view.setHtml(report_html(rpt, cat))
            if self.case_id is not None:
                try:
                    self.ctx.session.update_case(self.case_id, {"report": rpt,
                                                                "clinical_notes": rpt.get("caseEssence", "")})
                    self.rp_saved.setText(self.rp_saved.text() + " · explanations added")
                except Exception as e:
                    error(self, "Explanations not saved", str(e))

        def fail(msg, tb):
            if gen == self.generation:
                self.rp_busy.stop()
                self.rp_saved.setText(self.rp_saved.text() + " · AI explanations unavailable")
        run_job(lambda: engine.report(pid, cat, patient_details=details, mode=mode), done, fail)

    # ════════════════════════════════════════════════════════════════════
    # 3. REPORT
    # ════════════════════════════════════════════════════════════════════
    def _report_page(self) -> QWidget:
        w, v = page()
        self.rp_title = label("", "h1")
        self.rp_saved = label("", "muted")
        v.addLayout(row(self.rp_title, None, self.rp_saved))
        self.rp_busy = Busy()
        v.addWidget(self.rp_busy)
        self.rp_view = QTextBrowser()
        self.rp_view.setOpenLinks(False)
        v.addWidget(self.rp_view, 1)
        self.sug_btn = button("Suggest more symptoms…", slot=self._open_suggestions)
        v.addLayout(row(button("← Edit symptoms", slot=lambda: self.steps.setCurrentIndex(1)), None,
                        self.sug_btn, button("Save PDF…", slot=self._save_pdf),
                        button("New case", primary=True, slot=self.reset)))
        return w

    def _show_report(self):
        details = self._patient_details()
        self.rp_title.setText(f"Treatment plan — {details.get('name') or 'New patient'}")
        self.rp_view.setHtml(report_html(self.report, self.cat))
        self.steps.setCurrentIndex(2)
        self._save_case()

    def _save_case(self):
        """Saved on reaching the report, as the demo did; re-analysing saves a new version."""
        s = self.ctx.session
        details = self._patient_details()
        try:
            pid = self.pt_existing.currentData()
            if not pid:
                if not details["name"]:
                    details["name"] = "Unnamed patient"
                pid = s.create_patient(details)
            elif any(details.values()):
                current = s.get_patient(pid)["details"]
                if any(details.get(k) != current.get(k) for k in details if details.get(k)):
                    s.save_patient_details(pid, {**current, **{k: v for k, v in details.items() if v}})
            top = [r["name"] for r in self.report.get("top10Remedies", [])] or \
                  [self.report.get("primaryRemedy", {}).get("name", "")]
            self.case_id = s.add_case(pid, {
                "raw_symptoms": self.raw, "categorised": self.cat, "top_remedies": top,
                "clinical_notes": self.report.get("caseEssence", ""), "patient_details": details,
                "analysis_mode": self.mode.currentData(), "report": self.report,
                "ai_model": getattr(self.ctx.llm, "name", None) if self.ctx.ai_ready else None,
            })
            if self.intake_id:
                s.mark_intake(self.intake_id, "reviewed")
                self.intake_id = None
        except Exception as e:
            self.rp_saved.setText("⚠ Not saved")
            return error(self, "Case not saved", str(e))
        self.current_pid = pid
        self.rp_saved.setText(f"✓ Saved to {pid} · {datetime.now():%H:%M}")
        self.refresh_patients(select=pid)
        self.ctx.refresh_learning()
        self.saved.emit(pid)

    def _pdf_name(self, suffix=""):
        pid = getattr(self, "current_pid", "case")
        return f"HoRUS_{pid}_{datetime.now():%Y%m%d}{suffix}.pdf"

    def _save_pdf(self, report_enhanced=None, added=None, summary=None, suffix=""):
        path, _ = QFileDialog.getSaveFileName(self, "Save treatment plan", self._pdf_name(suffix), "PDF (*.pdf)")
        if not path:
            return
        try:
            pdf = self.ctx.engine.pdf(getattr(self, "current_pid", "—"), self.cat, self.report,
                                      patient_details=self._patient_details(),
                                      report_enhanced=report_enhanced, added_symptoms=added,
                                      comparative_summary=summary)
            Path(path).write_bytes(pdf)
            self.ctx.session.audit("pdf.exported", getattr(self, "current_pid", None))
        except Exception as e:
            error(self, "PDF not saved", str(e))

    # ════════════════════════════════════════════════════════════════════
    # 4. SUGGESTIONS + COMPARATIVE REPORT
    # ════════════════════════════════════════════════════════════════════
    def _compare_page(self) -> QWidget:
        w, v = page()
        v.addWidget(label("Symptoms that often occur with this case", "h1"))
        v.addWidget(label("From the symptom co-occurrence clusters. Ask the patient about these; tick the ones "
                          "they confirm, add their modalities, then compare the two analyses.", "muted"))
        split = QSplitter(Qt.Vertical)
        self.sg_table = QTableWidget(0, 6)
        self.sg_table.setHorizontalHeaderLabels(["Confirmed", "Symptom", "Likelihood", "Linked to",
                                                 "Worse from", "Better from"])
        h = self.sg_table.horizontalHeader()
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.sg_table.verticalHeader().setVisible(False)
        split.addWidget(self.sg_table)
        res = QWidget()
        rl = QVBoxLayout(res)
        rl.setContentsMargins(0, 0, 0, 0)
        self.cmp_summary = label("", "info")
        self.cmp_summary.hide()
        rl.addWidget(self.cmp_summary)
        two = QSplitter(Qt.Horizontal)
        self.cmp_base, self.cmp_enh = QTextBrowser(), QTextBrowser()
        two.addWidget(self.cmp_base)
        two.addWidget(self.cmp_enh)
        rl.addWidget(two, 1)
        split.addWidget(res)
        split.setSizes([260, 420])
        v.addWidget(split, 1)
        self.cmp_busy = Busy()
        self.cmp_pdf = button("Save comparative PDF…", slot=self._save_cmp_pdf)
        self.cmp_pdf.setEnabled(False)
        v.addLayout(row(button("← Back to report", slot=lambda: self.steps.setCurrentIndex(2)), self.cmp_busy,
                        None, self.cmp_pdf, button("Compare analyses", primary=True, slot=self._compare)))
        return w

    def _open_suggestions(self):
        per, top = self.ctx.engine.suggestions(self.cat, top_n=10)
        seen, rows = set(), []
        for x in top:
            rows.append((x["symptom"], x["avg_prob"], x["triggered_by"]))
            seen.add(x["symptom"].lower())
        for sym, items in per.items():
            for x in items[:3]:
                if x["symptom"].lower() not in seen:
                    rows.append((x["symptom"], x["prob"], sym))
                    seen.add(x["symptom"].lower())
        self.sg_table.setRowCount(0)
        for sym, p, trig in rows:
            r = self.sg_table.rowCount()
            self.sg_table.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Unchecked)
            self.sg_table.setItem(r, 0, chk)
            for c, val in ((1, sym), (2, f"{p:.0%}"), (3, trig)):
                it = QTableWidgetItem(str(val))
                it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                if c == 2:
                    it.setData(Qt.UserRole, float(p))
                self.sg_table.setItem(r, c, it)
            self.sg_table.setItem(r, 4, QTableWidgetItem(""))
            self.sg_table.setItem(r, 5, QTableWidgetItem(""))
        if not rows:
            error(self, "No suggestions", "No co-occurrence data matched these symptoms. Suggestions appear when "
                  "symptoms are worded like the dataset.")
            return
        self.cmp_base.clear()
        self.cmp_enh.clear()
        self.cmp_summary.hide()
        self.cmp_pdf.setEnabled(False)
        self.steps.setCurrentIndex(3)

    def _compare(self):
        selected = {}
        for r in range(self.sg_table.rowCount()):
            if self.sg_table.item(r, 0).checkState() == Qt.Checked:
                selected[self.sg_table.item(r, 1).text()] = {
                    "worse": _split(self.sg_table.item(r, 4).text()),
                    "better": _split(self.sg_table.item(r, 5).text()),
                    "avg_prob": self.sg_table.item(r, 2).data(Qt.UserRole) or 0.0,
                    "triggered_by": self.sg_table.item(r, 3).text()}
        if not selected:
            return error(self, "Nothing selected", "Tick the symptoms the patient confirmed.")
        engine = self.ctx.engine
        pid = getattr(self, "current_pid", "—")
        base_cat = self.cat
        self.cmp_busy.start("Analysing with the added symptoms…")

        def work():
            enh_cat, added = engine.enhanced_case(base_cat, selected)
            base = engine.report(pid, base_cat, extra_label="[BASE]")
            enh = engine.report(pid, enh_cat, extra_label="[ENHANCED]")
            summary = engine.comparative_summary(base, enh)
            return base, enh_cat, enh, added, summary

        def done(res):
            self.cmp_busy.stop()
            base, enh_cat, enh, added, summary = res
            self._cmp = (enh, added, summary)
            self.cmp_base.setHtml(report_html(base, base_cat, "A. Original symptoms", show_evidence=3))
            self.cmp_enh.setHtml(report_html(enh, enh_cat, "B. With confirmed suggestions", show_evidence=3))
            if summary:
                self.cmp_summary.setText(summary)
                self.cmp_summary.show()
            self.cmp_pdf.setEnabled(True)

        run_job(work, done, lambda m, tb: (self.cmp_busy.stop(), error(self, "Comparison failed", m, tb)))

    def _save_cmp_pdf(self):
        enh, added, summary = self._cmp
        self._save_pdf(report_enhanced=enh, added=added, summary=summary or None, suffix="_comparative")
