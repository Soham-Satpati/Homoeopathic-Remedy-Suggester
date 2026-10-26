"""Book selection and decision bias: an editor used in Settings (clinic default) and per case."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout, QInputDialog,
                               QScrollArea, QVBoxLayout, QWidget)

from horus.books import (CLASSICAL_CATEGORY_WEIGHTS, CLASSICAL_MODALITY_WEIGHT, Book, DecisionConfig,
                         import_book, remove_book)

from .context import AppContext, run_job
from .widgets import Busy, button, card, confirm, error, info, label, row

HIERARCHY = [
    ("psychological", "Mind & emotions", "Classical: 2"),
    ("general", "Generals (whole body)", "Classical: 1.5"),
    ("physical", "Particulars (local)", "Classical: 1"),
]


def _spin(lo: float, hi: float, step: float, value: float, suffix: str = "") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(2)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    s.setMinimumWidth(90)
    s.setMaximumWidth(110)
    return s


class DecisionEditor(QWidget):
    """Edits a DecisionConfig for the given books."""

    changed = Signal()

    def __init__(self, books: list[Book], config: DecisionConfig, parent=None, editable: bool = True):
        super().__init__(parent)
        self.books = books
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.addWidget(label("Book", "muted"), 0, 0)
        grid.addWidget(label("Weight", "muted"), 0, 2)
        self.book_rows: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        for i, b in enumerate(books, 1):
            cb = QCheckBox(b.title)
            cb.setToolTip(b.description or b.title)
            w = _spin(0.25, 3.0, 0.25, 1.0, "×")
            w.setToolTip("How strongly matches from this book count. 1× = no bias.")
            cb.toggled.connect(w.setEnabled)
            cb.toggled.connect(self.changed)
            w.valueChanged.connect(self.changed)
            desc = label(("Added in this clinic · " if b.source == "user" else "") + (b.description or ""), "muted")
            desc.setMaximumWidth(520)
            grid.addWidget(cb, i, 0)
            grid.addWidget(desc, i, 1)
            grid.addWidget(w, i, 2)
            self.book_rows[b.id] = (cb, w)
        grid.setColumnStretch(1, 1)
        n = len(books) + 1
        self.learned = _spin(0.0, 3.0, 0.25, 1.0, "×")
        self.learned.setToolTip("How much this clinic's saved cases shift remedy scores and clusters. 0 = ignore them.")
        self.learned.valueChanged.connect(self.changed)
        grid.addWidget(label("Saved clinic cases (learning)"), n, 0)
        grid.addWidget(label("Cases saved in HoRUS strengthen the remedies and symptom links they confirm.",
                             "muted"), n, 1)
        grid.addWidget(self.learned, n, 2)
        v.addWidget(card(label("Books", "h2"), grid))

        h = QGridLayout()
        h.setHorizontalSpacing(14)
        self.cat: dict[str, QDoubleSpinBox] = {}
        for i, (key, name, hint) in enumerate(HIERARCHY):
            s = _spin(0.0, 5.0, 0.25, CLASSICAL_CATEGORY_WEIGHTS[key])
            s.valueChanged.connect(self.changed)
            self.cat[key] = s
            h.addWidget(label(name), i, 0)
            h.addWidget(s, i, 1)
            h.addWidget(label(hint, "muted"), i, 2)
        i = len(HIERARCHY)
        self.mod = _spin(0.0, 2.0, 0.1, CLASSICAL_MODALITY_WEIGHT, "×")
        self.mod.valueChanged.connect(self.changed)
        h.addWidget(label("Modalities (worse / better)"), i, 0)
        h.addWidget(self.mod, i, 1)
        h.addWidget(label("Share of the symptom's weight. Classical: 0.6×", "muted"), i, 2)
        self.cause = _spin(0.0, 5.0, 0.25, 0.0)
        self.cause.valueChanged.connect(self.changed)
        h.addWidget(label("Causation (ailments from)"), i + 1, 0)
        h.addWidget(self.cause, i + 1, 1)
        h.addWidget(label("0 = recorded but not used for ranking (as in the online demo)", "muted"), i + 1, 2)
        h.setColumnStretch(2, 1)
        self.reset_btn = button("Reset to classical defaults", slot=self.reset)
        v.addWidget(card(label("Weighting (decision bias)", "h2"),
                         label("How much each part of the case counts when remedies are ranked. Red flags and "
                               "the miasm are not affected.", "muted"), h, row(self.reset_btn, None)))
        self.set_config(config)
        if not editable:
            self.setEnabled(False)

    def set_config(self, cfg: DecisionConfig) -> None:
        chosen = set(cfg.books) if cfg.books else {b.id for b in self.books}
        for bid, (cb, w) in self.book_rows.items():
            cb.setChecked(bid in chosen)
            w.setValue(cfg.weight_of(bid))
            w.setEnabled(bid in chosen)
        self.learned.setValue(cfg.learned_weight)
        cats = cfg.categories
        for k, s in self.cat.items():
            s.setValue(cats[k])
        self.mod.setValue(cfg.modality_weight)
        self.cause.setValue(cats["causation"])

    def reset(self) -> None:
        self.set_config(DecisionConfig())

    def config(self) -> DecisionConfig:
        books = tuple(b.id for b in self.books if self.book_rows[b.id][0].isChecked())
        weights = tuple(sorted((bid, round(w.value(), 3)) for bid, (cb, w) in self.book_rows.items()
                               if cb.isChecked() and round(w.value(), 3) != 1.0))
        cats = {k: round(s.value(), 3) for k, s in self.cat.items()}
        cats["causation"] = round(self.cause.value(), 3)
        return DecisionConfig(books=books, book_weights=weights, learned_weight=round(self.learned.value(), 3),
                              category_weights=tuple(sorted(cats.items())),
                              modality_weight=round(self.mod.value(), 3))


class CaseDecisionDialog(QDialog):
    """Per-case override, starting from the current settings."""

    def __init__(self, ctx: AppContext, current: DecisionConfig, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Books & weighting for this case")
        self.resize(900, 640)
        v = QVBoxLayout(self)
        v.addWidget(label("These settings apply to this case only. The clinic default is set in Settings → "
                          "Books & weighting.", "muted"))
        self.editor = DecisionEditor(ctx.engine.books, current)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        v.addWidget(scroll, 1)
        v.addLayout(row(button("Use clinic default", slot=self._default), None, button("Cancel", slot=self.reject),
                        button("Use for this case", primary=True, slot=self._ok)))

    def _default(self):
        self.editor.set_config(clinic_config(self.ctx))

    def _ok(self):
        if not self.editor.config().books:
            return error(self, "Books", "Select at least one book.")
        self.accept()


def clinic_config(ctx: AppContext) -> DecisionConfig:
    return DecisionConfig.from_dict(ctx.session.decision_config()).resolve(ctx.engine.books) \
        if ctx.engine is not None else DecisionConfig.from_dict(ctx.session.decision_config())


class BooksSettingsTab(QWidget):
    """Settings → Books & weighting: the clinic default, and adding / removing books."""

    def __init__(self, ctx: AppContext, on_books_changed=None):
        super().__init__()
        self.ctx = ctx
        self.on_books_changed = on_books_changed
        self.v = QVBoxLayout(self)
        self.busy = Busy()
        self._build()

    def _build(self):
        while self.v.count():
            item = self.v.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if self.ctx.engine is None:
            self.v.addWidget(label("The books are still loading — open this tab again in a moment.", "muted"))
            return
        can_edit = self.ctx.session.user.can("settings")
        self.v.addWidget(label("Choose which books HoRUS ranks remedies from and how strongly each counts. This "
                               "is the clinic default; doctors can change it for a single case from the New case "
                               "screen. The books and weights used are recorded on every case and its PDF.",
                               "info"))
        self.editor = DecisionEditor(self.ctx.engine.books, clinic_config(self.ctx), editable=can_edit)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        self.v.addWidget(scroll, 1)
        user_books = [b for b in self.ctx.engine.books if b.source == "user"]
        add_btn = button("Add a book…", slot=self._add)
        rem_btn = button("Remove a clinic book…", danger=True, slot=self._remove)
        rem_btn.setVisible(bool(user_books))
        save = button("Save as clinic default", primary=True, slot=self._save)
        for b in (add_btn, rem_btn, save):
            b.setEnabled(can_edit)
        self.v.addLayout(row(add_btn, rem_btn, self.busy, None, save))
        if not can_edit:
            self.v.addWidget(label("Only an administrator can change the clinic default.", "muted"))

    def _save(self):
        cfg = self.editor.config()
        if not cfg.books:
            return error(self, "Books", "Select at least one book.")
        self.ctx.session.set_decision_config(cfg.to_dict())
        self.ctx.engine.set_config(cfg)
        info(self, "Saved", "New cases will use these books and weights.\n\n" + self.ctx.engine.describe(cfg))

    def _add(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a book (HoRUS JSON format)", "", "JSON (*.json)")
        if not path:
            return
        from pathlib import Path
        title, ok = QInputDialog.getText(self, "Book title", "Name shown in HoRUS:", text=Path(path).stem.replace("_", " "))
        if not ok or not title.strip():
            return
        self.busy.start("Adding the book and building its symptom clusters… (a large book takes a few minutes)")
        paths = self.ctx.paths

        def done(book):
            self.busy.stop()
            self.ctx.engine.reload_books()
            cfg = DecisionConfig.from_dict(self.ctx.session.decision_config())
            if cfg.books and book.id not in cfg.books:
                cfg = replace(cfg, books=cfg.books + (book.id,))
                self.ctx.session.set_decision_config(cfg.to_dict())
            self.ctx.engine.set_config(DecisionConfig.from_dict(self.ctx.session.decision_config()))
            self.ctx.session.audit("book.added", book.id)
            info(self, "Book added", f"“{book.title}” was added and is now used for ranking. {book.description}")
            self._build()
            if self.on_books_changed:
                self.on_books_changed()

        def fail(msg, tb):
            self.busy.stop()
            error(self, "Book not added", msg, tb)
        run_job(lambda: import_book(paths, Path(path), title.strip()), done, fail)

    def _remove(self):
        books = [b for b in self.ctx.engine.books if b.source == "user"]
        titles = [b.title for b in books]
        t, ok = QInputDialog.getItem(self, "Remove a book", "Book added in this clinic:", titles, 0, False)
        if not ok:
            return
        book = books[titles.index(t)]
        if not confirm(self, "Remove book", f"Remove “{book.title}” from HoRUS? Saved cases are kept.", "Remove",
                       danger=True):
            return
        try:
            remove_book(self.ctx.paths, book.id)
        except Exception as e:
            return error(self, "Not removed", str(e))
        self.ctx.engine.reload_books()
        self.ctx.session.audit("book.removed", book.id)
        self._build()
        if self.on_books_changed:
            self.on_books_changed()
