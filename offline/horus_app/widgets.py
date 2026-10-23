"""Small shared widgets and the HTML report renderer."""

from __future__ import annotations

import html
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from .theme import REPORT_CSS


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def label(text: str = "", name: str | None = None, wrap: bool = True, rich: bool = False) -> QLabel:
    lab = QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    lab.setTextFormat(Qt.RichText if rich else Qt.PlainText)
    if rich:
        lab.setOpenExternalLinks(False)
    return lab


def button(text: str, primary: bool = False, danger: bool = False, slot=None) -> QPushButton:
    b = QPushButton(text)
    if primary:
        b.setProperty("primary", True)
    if danger:
        b.setProperty("danger", True)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


def card(*widgets, margins=(16, 14, 16, 14), spacing=8) -> QFrame:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    for w in widgets:
        if isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    return f


def row(*items, spacing=8, stretch_end=False) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setSpacing(spacing)
    for it in items:
        if it is None:
            lay.addStretch(1)
        elif isinstance(it, QWidget):
            lay.addWidget(it)
        else:
            lay.addLayout(it)
    if stretch_end:
        lay.addStretch(1)
    return lay


def page() -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    w.setObjectName("page")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(24, 20, 24, 20)
    lay.setSpacing(12)
    return w, lay


class Busy(QWidget):
    """Inline 'working…' indicator."""

    def __init__(self, text: str = "Working…"):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setMaximumWidth(140)
        self.bar.setTextVisible(False)
        self.text = label(text, "muted")
        lay.addWidget(self.bar)
        lay.addWidget(self.text, 1)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.hide()

    def start(self, text: str | None = None):
        if text:
            self.text.setText(text)
        self.show()

    def stop(self):
        self.hide()


def error(parent, title: str, text: str, detail: str = "") -> None:
    box = QMessageBox(QMessageBox.Warning, title, text, QMessageBox.Ok, parent)
    if detail:
        box.setDetailedText(detail)
    box.exec()


def info(parent, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


def confirm(parent, title: str, text: str, yes: str = "Yes", danger: bool = False) -> bool:
    box = QMessageBox(QMessageBox.Warning if danger else QMessageBox.Question, title, text, parent=parent)
    y = box.addButton(yes, QMessageBox.AcceptRole)
    box.addButton("Cancel", QMessageBox.RejectRole)
    box.exec()
    return box.clickedButton() is y


def fmt_ts(ts: str | None) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromisoformat(ts).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return str(ts)


# ── report rendering ─────────────────────────────────────────────────────────
def _symptom_rows(cat: dict) -> str:
    rows = []
    cm = cat.get("case_modalities") or {}
    if cm.get("worse") or cm.get("better"):
        rows.append(("Whole case", "All complaints", ", ".join(cm.get("worse", [])) or "—",
                     ", ".join(cm.get("better", [])) or "—"))
    for g in ("psychological", "general", "physical"):
        for s in cat.get(g, []):
            if s.get("unverified"):
                continue
            rows.append((g.title(), s["symptom"], ", ".join(s.get("worse", [])) or "—",
                         ", ".join(s.get("better", [])) or "—"))
    if not rows:
        return ""
    body = "".join(f"<tr><td class='muted'>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td><td>{esc(d)}</td></tr>"
                   for a, b, c, d in rows)
    return ("<table cellspacing='0' width='100%'><tr><th>Category</th><th>Symptom</th><th>Worse</th>"
            f"<th>Better</th></tr>{body}</table>")


def report_html(report: dict, cat: dict, title: str = "", show_evidence: int = 5) -> str:
    """Treatment-plan view. Every value is escaped: symptoms and AI text are untrusted."""
    out = [f"<html><head><style>{REPORT_CSS}</style></head><body>"]
    if title:
        out.append(f"<h1>{esc(title)}</h1>")
    flags = report.get("redFlags") or []
    if flags:
        items = "".join(f"<li><b>{esc(f['title'])}.</b> {esc(f['advice'])}</li>" for f in flags)
        out.append("<div class='flag'><b style='font-size:12pt'>⚠ Seek urgent medical care</b><br>"
                   "This case contains warning signs that need conventional medical assessment first. "
                   f"HoRUS output must not delay it.<ul>{items}</ul></div>")
    if report.get("caseEssence"):
        out.append(f"<h2>Case essence</h2><p>{esc(report['caseEssence'])}</p>")
    out.append(f"<h2>Miasm (rubric-based)</h2><p>{esc(report.get('miasmaticAnalysis', ''))}</p>")
    if cat.get("causation"):
        out.append(f"<h2>Causation</h2><p>{esc(', '.join(cat['causation']))}</p>")
    sym = _symptom_rows(cat)
    if sym:
        out.append(f"<h2>Symptoms used</h2>{sym}")
    if report.get("unverifiedSymptoms"):
        out.append("<h2>Not used — not found in the patient's words</h2>"
                   f"<p class='warn'>{esc(', '.join(report['unverifiedSymptoms']))}</p>")
    top = report.get("top10Remedies") or []
    out.append("<h2>Remedies ranked from the dataset</h2>")
    if not top:
        out.append(f"<p class='warn'>{esc(report.get('primaryRemedy', {}).get('why', ''))}</p>")
    for r in top:
        out.append(f"<p style='margin:10px 0 2px 0'><span class='rem'>{r['rank']}. {esc(r['name'])}</span>"
                   f" <span class='muted'>— {esc(r['role'])} · score {r['datasetScore']:.3f}</span><br>"
                   f"{esc(r['rationale'])}</p>")
        for e in r.get("evidence", [])[:show_evidence]:
            out.append(f"<div class='ev'>• {esc(e['patient'])} ≈ “{esc(e['dataset'])}” "
                       f"<span class='muted'>({e['similarity']:.2f})</span></div>")
    if report.get("remedyRelationships"):
        out.append(f"<h2>How the top remedies differ</h2><p>{esc(report['remedyRelationships'])}</p>")
    if report.get("monitoringPoints"):
        out.append("<h2>Monitoring points</h2><ul>" +
                   "".join(f"<li>{esc(m)}</li>" for m in report["monitoringPoints"]) + "</ul>")
    out.append(f"<p class='muted' style='margin-top:14px'>{esc(report.get('_method', ''))}</p>")
    if report.get("_removed"):
        out.append("<p class='muted'>Filtered from AI text: " + esc("; ".join(report["_removed"])) + "</p>")
    out.append("</body></html>")
    return "".join(out)
