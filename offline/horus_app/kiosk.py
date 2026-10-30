"""Patient self-intake on the clinic PC ("kiosk mode"), ported from the demo's patient side.

The patient answers three questions per screen (typing or speaking — speech in Bengali, Hindi or
English is transcribed offline into English), the offline AI may ask up to three follow-up
questions, and the answers are stored as a pending intake for the doctor. Leaving kiosk mode
requires the signed-in user's password, so patients cannot browse records.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFormLayout, QFrame, QLineEdit, QPlainTextEdit, QScrollArea,
                               QStackedWidget, QVBoxLayout, QWidget)

from horus.intake import PATIENT_QUESTIONS, QUESTIONS_PER_PAGE, compose_patient_narrative

from .context import AppContext, run_job
from .voice import DictateButton
from .widgets import Busy, button, card, error, label, row

BIG = "font-size: 15pt;"


class KioskPage(QWidget):
    finished = Signal()          # staff unlocked and left kiosk mode
    submitted = Signal()

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx
        self.setObjectName("page")
        v = QVBoxLayout(self)
        v.setContentsMargins(60, 20, 60, 20)
        top = row(label("Patient questionnaire", "h1"), None, button("Staff: exit", slot=self._exit))
        v.addLayout(top)
        self.stack = QStackedWidget()
        v.addWidget(self.stack, 1)
        self.stack.addWidget(self._details_page())
        self.stack.addWidget(self._question_page())
        self.stack.addWidget(self._thanks_page())
        self.reset()

    # ── pages ────────────────────────────────────────────────────────────
    def _details_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(label("Welcome. Please answer a few questions about your health before you see the doctor. "
                          "You can type, or press the microphone and speak — in English, Bengali or Hindi.", "info"))
        f = QFormLayout()
        self.d_name, self.d_age, self.d_phone, self.d_prev = QLineEdit(), QLineEdit(), QLineEdit(), QLineEdit()
        self.d_prev.setPlaceholderText("e.g. PT-2026-004 (if you have visited before)")
        self.d_gender = QComboBox()
        self.d_gender.addItems(["", "Female", "Male", "Other"])
        for lab_, wid in (("Your full name", self.d_name), ("Age", self.d_age), ("Gender", self.d_gender),
                          ("Phone number", self.d_phone), ("Patient ID (if known)", self.d_prev)):
            wid.setStyleSheet(BIG)
            f.addRow(label(lab_), wid)
        v.addWidget(card(f))
        v.addStretch(1)
        go = button("Start questions →", primary=True, slot=self._start)
        go.setStyleSheet(BIG)
        v.addLayout(row(None, go))
        return w

    def _question_page(self) -> QWidget:
        """Three questions per screen, each with its own answer box and microphone."""
        w = QWidget()
        v = QVBoxLayout(w)
        self.q_progress = label("", "muted")
        self.q_progress.setStyleSheet("font-size: 13pt;")
        v.addWidget(self.q_progress)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        self.slots = []
        for _ in range(QUESTIONS_PER_PAGE):
            box = QFrame()
            box.setObjectName("card")
            bv = QVBoxLayout(box)
            text = label("")
            text.setWordWrap(True)
            text.setStyleSheet("font-size: 16pt; font-weight: 600;")
            hint = label("", "muted")
            hint.setWordWrap(True)
            hint.setStyleSheet("font-size: 12pt;")
            ans = QPlainTextEdit()
            ans.setStyleSheet("font-size: 14pt;")
            ans.setMinimumHeight(70)
            ans.setMaximumHeight(105)           # three questions fit on one screen
            mic = DictateButton(lambda wav, mime: self.ctx.transcriber.transcribe(wav, mime).strip(),
                                "🎙  Speak your answer")
            mic.text_ready.connect(ans.appendPlainText)
            bv.addWidget(text)
            bv.addWidget(hint)
            bv.addWidget(ans)
            bv.addLayout(row(mic, None))
            iv.addWidget(box)
            self.slots.append({"box": box, "text": text, "hint": hint, "answer": ans, "mic": mic})
        iv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        self.q_busy = Busy("Thinking of any follow-up questions…")
        v.addLayout(row(self.q_busy, None))
        self.back_btn = button("← Back", slot=self._back)
        self.next_btn = button("Next →", primary=True, slot=self._next)
        for b in (self.back_btn, self.next_btn):
            b.setStyleSheet(BIG)
        v.addLayout(row(self.back_btn, None, self.next_btn))
        return w

    @property
    def answer(self) -> QPlainTextEdit:
        """The first answer box on the screen."""
        return self.slots[0]["answer"]

    def page_questions(self) -> list:
        return self.questions[self.idx:self.idx + QUESTIONS_PER_PAGE]

    def _thanks_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addStretch(1)
        t = label("Thank you. Your answers have been given to the doctor.")
        t.setStyleSheet("font-size: 20pt; font-weight: 600;")
        t.setAlignment(Qt.AlignCenter)
        v.addWidget(t)
        self.thanks_id = label("", "muted")
        self.thanks_id.setAlignment(Qt.AlignCenter)
        self.thanks_id.setStyleSheet("font-size: 14pt;")
        v.addWidget(self.thanks_id)
        v.addStretch(1)
        nxt = button("Next patient", primary=True, slot=self.reset)
        nxt.setStyleSheet(BIG)
        v.addLayout(row(None, nxt, None))
        return w

    # ── flow ─────────────────────────────────────────────────────────────
    def reset(self):
        self.questions = [dict(q) for q in PATIENT_QUESTIONS]
        self.answers: dict = {}
        self.idx = 0
        self.followups_done = False
        for e in (self.d_name, self.d_age, self.d_phone, self.d_prev):
            e.clear()
        self.d_gender.setCurrentIndex(0)
        self.stack.setCurrentIndex(0)

    def _start(self):
        if not self.d_name.text().strip():
            return error(self, "Your name", "Please enter your name to begin.")
        self.stack.setCurrentIndex(1)
        self._show_q()

    def _show_q(self):
        qs = self.page_questions()
        n = len(self.questions)
        first, last = self.idx + 1, self.idx + len(qs)
        self.q_progress.setText(f"Questions {first}–{last} of {n}" if last > first else f"Question {first} of {n}")
        for slot, q in zip(self.slots, qs + [None] * (len(self.slots) - len(qs))):
            slot["box"].setVisible(q is not None)
            if q is None:
                continue
            slot["text"].setText(q["q"])
            slot["hint"].setText(q.get("hint", ""))
            slot["answer"].setPlainText(self.answers.get(q["id"], ""))
            slot["mic"].setVisible(self.ctx.voice_ready)
        self.slots[0]["answer"].setFocus()
        self.back_btn.setEnabled(self.idx > 0)
        self.next_btn.setText("Finish ✓" if last == n and self.followups_done else "Next →")

    def _store_answer(self):
        for slot, q in zip(self.slots, self.page_questions()):
            self.answers[q["id"]] = slot["answer"].toPlainText().strip()

    def _back(self):
        self._store_answer()
        self.idx = max(0, self.idx - QUESTIONS_PER_PAGE)
        self._show_q()

    def _next(self):
        self._store_answer()
        if self.idx + QUESTIONS_PER_PAGE < len(self.questions):
            self.idx += QUESTIONS_PER_PAGE
            self._show_q()
            return
        if not self.followups_done and self.ctx.ai_ready and self.ctx.engine is not None:
            self.followups_done = True
            self.next_btn.setEnabled(False)
            self.q_busy.start()
            text = compose_patient_narrative(self.questions, self.answers)

            def done(extra):
                self.q_busy.stop()
                self.next_btn.setEnabled(True)
                if extra:
                    self.idx = len(self.questions)          # follow-ups get a screen of their own
                    for i, q in enumerate(extra):
                        self.questions.append({"id": f"followup_{i}", "label": "Follow-up", "q": q,
                                               "hint": "Answer in your own words."})
                    self._show_q()
                else:
                    self._submit()
            run_job(lambda: self.ctx.engine.followups(text), done,
                    lambda m, tb: (self.q_busy.stop(), self.next_btn.setEnabled(True), self._submit()))
            return
        self._submit()

    def _submit(self):
        s = self.ctx.session
        details = {"name": self.d_name.text().strip(), "age": self.d_age.text().strip(),
                   "gender": self.d_gender.currentText(), "contact": self.d_phone.text().strip()}
        prev = self.d_prev.text().strip().upper()
        try:
            pid = prev if prev and s.get_patient(prev) else None
            if pid is None:
                pid = s.create_patient(details)
            qa = [{"question": q["q"], "label": q["label"], "answer": self.answers.get(q["id"], "")}
                  for q in self.questions]
            s.add_intake(pid, {"qa": qa, "raw_text": compose_patient_narrative(self.questions, self.answers),
                               "details": details}, source="kiosk")
        except Exception as e:
            return error(self, "Not saved", f"Please call a member of staff. ({e})")
        self.thanks_id.setText(f"Your patient ID is {pid}. Please keep it for your next visit.")
        self.stack.setCurrentIndex(2)
        self.submitted.emit()

    def _exit(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Staff only")
        v = QVBoxLayout(dlg)
        v.addWidget(label(f"Enter the password for {self.ctx.session.user.display_name} to leave the "
                          "patient questionnaire.", "muted"))
        pw = QLineEdit()
        pw.setEchoMode(QLineEdit.Password)
        v.addWidget(pw)
        v.addLayout(row(None, button("Cancel", slot=dlg.reject), button("Exit", primary=True, slot=dlg.accept)))
        pw.returnPressed.connect(dlg.accept)
        if dlg.exec() == QDialog.Accepted:
            if self.ctx.session.verify_password(pw.text(), "kiosk.exit"):
                self.reset()
                self.finished.emit()
            else:
                error(self, "Staff only", "That password is not correct.")
