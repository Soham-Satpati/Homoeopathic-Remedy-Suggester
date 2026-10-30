"""Book selection and decision bias: an editor used in Settings (clinic default) and per case."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QGridLayout, QInputDialog,
                               QScrollArea, QVBoxLayout, QWidget)

from horus.book_worker import import_books
from horus.books import (CLASSICAL_CATEGORY_WEIGHTS, CLASSICAL_MODALITY_WEIGHT, Book, DecisionConfig,
                         remove_book)

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

    def __init__(self, books: list[Book], config: DecisionConfig, parent=None, editable: bool = True,
                 on_delete=None):
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
            if on_delete is not None:
                d = button("Delete", danger=True, slot=lambda _=False, book=b: on_delete(book))
                d.setToolTip("Delete this book from HoRUS" if b.source == "user" else
                             "Remove this book from use in this clinic (it can be restored later)")
                d.setEnabled(editable)
                d.setFixedWidth(90)
                grid.addWidget(d, i, 3)
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
    """Settings → Books & weighting: the clinic default, adding and deleting books.

    Fills itself in as soon as the books are ready (it listens for the app's engine_ready event), and
    shows live progress while they load, index, or while a new book is being added."""

    def __init__(self, ctx: AppContext, on_books_changed=None):
        super().__init__()
        from PySide6.QtWidgets import QProgressBar
        self.ctx = ctx
        self.on_books_changed = on_books_changed
        self.v = QVBoxLayout(self)
        self.adding = False
        self.prog_title = label("", "h2")
        self.prog_msg = label("", "muted")
        self.prog_bar = QProgressBar()
        self.prog_bar.setRange(0, 100)
        self.prog_bar.setFormat("%p%")
        self.cancel_btn = button("Cancel", slot=self._cancel)
        self.cancel_btn.hide()
        self._cancel_requested = False
        self.prog_card = card(self.prog_title, self.prog_bar, self.prog_msg, row(None, self.cancel_btn))
        ctx.events.prep_progress.connect(self._prep_progress)
        ctx.events.engine_ready.connect(self._ready)
        self._build()

    # ── state ────────────────────────────────────────────────────────────
    @property
    def _loading(self) -> bool:
        return self.ctx.engine is None or not self.ctx.engine_ready

    def _clear(self):
        while self.v.count():
            item = self.v.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.prog_card:
                w.deleteLater()
            elif w is self.prog_card:
                w.setParent(None)
            elif item.layout() is not None:
                _delete_layout(item.layout())

    def _show_progress(self, title: str, fraction: float, message: str):
        self._clear()
        self.prog_title.setText(title)
        self._set_progress(fraction, message)
        self.v.addWidget(self.prog_card)
        self.v.addStretch(1)

    def _set_progress(self, fraction: float, message: str):
        self.prog_msg.setText(message)
        self.prog_bar.setValue(int(round(fraction * 100)))

    def _prep_progress(self, fraction: float, message: str):
        if self.adding:
            return
        if self.prog_card.parent() is None or self.prog_title.text() != "Preparing the books":
            self._show_progress("Preparing the books", fraction, message)
        else:
            self._set_progress(fraction, message)

    def _ready(self):
        if not self.adding:
            self._build()

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self.adding and self._loading and self.prog_card.parent() is None:
            f, m = self.ctx.prep_state
            self._show_progress("Preparing the books", f, m)

    # ── the editor ───────────────────────────────────────────────────────
    def _build(self):
        if self._loading:
            f, m = self.ctx.prep_state
            self._show_progress("Preparing the books", f, m)
            return
        self._clear()
        can_edit = self.ctx.session.user.can("settings")
        self.v.addWidget(label("Choose which books HoRUS ranks remedies from and how strongly each counts. This "
                               "is the clinic default; doctors can change it for a single case from the New case "
                               "screen. The books and weights used are recorded on every case and its PDF.",
                               "info"))
        self.editor = DecisionEditor(self.ctx.engine.books, clinic_config(self.ctx), editable=can_edit,
                                     on_delete=self._delete if can_edit else None)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.editor)
        self.v.addWidget(scroll, 1)
        old = list(getattr(self.ctx.engine, "superseded_books", []))
        if old:
            names = {b.id: b.title for b in self.ctx.engine.all_books}
            text = ("These books you added earlier are now included in HoRUS, cleaned of scanning errors, "
                    "and the built-in copies are used instead: " +
                    "; ".join(f"{b.title} → {names.get(b.superseded_by, b.superseded_by)}" for b in old) +
                    ". Cases and settings that used them now use the built-in copies.")
            note = label(text, "info")
            note.setWordWrap(True)
            self.v.addWidget(note)
            if can_edit:
                self.v.addLayout(row(button(f"Delete the old copies ({len(old)})", slot=self._delete_superseded), None))
        hidden = [b for b in self.ctx.engine.all_books if b.id in self.ctx.engine.hidden_books]
        add_btn = button("Add books…", slot=self._add)
        add_btn.setToolTip("Add one or more books in the HoRUS JSON format")
        restore = button(f"Restore removed books ({len(hidden)})", slot=self._restore)
        restore.setVisible(bool(hidden))
        save = button("Save as clinic default", primary=True, slot=self._save)
        for b in (add_btn, restore, save):
            b.setEnabled(can_edit)
        self.v.addLayout(row(add_btn, restore, None, save))
        if not can_edit:
            self.v.addWidget(label("Only an administrator can change books and the clinic default.", "muted"))

    def _delete_superseded(self):
        from horus.books import remove_book
        old = list(self.ctx.engine.superseded_books)
        if not confirm(self, "Delete old copies", f"Delete the {len(old)} older, uncleaned copies from this "
                       "computer? The built-in cleaned books stay, and settings that named the old copies are "
                       "moved to them.", "Delete", danger=True):
            return
        saved = self.ctx.session.decision_config()
        if saved:      # point the clinic default at the built-in books before the old ids disappear
            self.ctx.session.set_decision_config(
                DecisionConfig.from_dict(saved).resolve(self.ctx.engine.books).to_dict())
        for b in old:
            try:
                remove_book(self.ctx.paths, b.id)
                self.ctx.session.audit("book.deleted", f"{b.id} (replaced by {b.superseded_by})")
            except Exception as e:
                error(self, "Delete", f"{b.title}: {e}")
        self.ctx.engine.reload_books(set(self.ctx.session.hidden_books()))
        self.ctx.engine.set_config(clinic_config(self.ctx))
        self._changed()
        self._build()

    def _save(self):
        cfg = self.editor.config()
        if not cfg.books:
            return error(self, "Books", "Select at least one book.")
        self.ctx.session.set_decision_config(cfg.to_dict())
        self.ctx.engine.set_config(cfg)
        self._changed()
        info(self, "Saved", "New cases will use these books and weights.\n\n" + self.ctx.engine.describe(cfg))

    def _changed(self):
        if self.on_books_changed:
            self.on_books_changed()

    # ── adding books ─────────────────────────────────────────────────────
    def _add(self):
        """Pick one or more book files, name them, and import them in a separate process (the
        window stays responsive even for very large books), with overall progress and Cancel."""
        from pathlib import Path
        files, _ = QFileDialog.getOpenFileNames(self, "Choose one or more books (HoRUS JSON format)", "",
                                                "JSON (*.json)")
        if not files:
            return
        dlg = AddBooksDialog([Path(f) for f in files], self)
        if dlg.exec() != QDialog.Accepted:
            return
        items = dlg.items()
        if not items:
            return
        self.adding = True
        self._cancel_requested = False
        n = len(items)
        self._show_progress(f"Adding {n} book{'s' if n != 1 else ''}", 0.0, "Starting…")
        self.cancel_btn.show()
        self.cancel_btn.setEnabled(True)
        paths = self.ctx.paths
        isolated = getattr(self.ctx, "isolated_book_import", True)

        def done(res):
            self.adding = False
            self.cancel_btn.hide()
            self.ctx.engine.reload_books(self.ctx.engine.hidden_books)
            cfg = DecisionConfig.from_dict(self.ctx.session.decision_config())
            new_ids = tuple(b.id for b in res.added if not cfg.books or b.id not in cfg.books)
            if cfg.books and new_ids:
                self.ctx.session.set_decision_config(replace(cfg, books=cfg.books + new_ids).to_dict())
            self.ctx.engine.set_config(clinic_config(self.ctx))
            for b in res.added:
                self.ctx.session.audit("book.added", b.id)
            lines = []
            if res.added:
                lines.append("Added and now used for ranking:\n" + "\n".join(
                    f"  • {b.title} — {b.description.split(';')[0]}" for b in res.added))
            if res.failed:
                lines.append("Not added:\n" + "\n".join(f"  • {t}: {m}" for t, m, _d in res.failed))
            if res.cancelled:
                lines.append("Cancelled — the remaining books were not added.")
            if res.added:
                lines.append("HoRUS will now index the new symptoms for matching (progress at the bottom).")
            if res.failed and not res.added:
                error(self, "Books not added", "\n\n".join(lines),
                      "\n\n".join(f"{t}\n{d}" for t, _m, d in res.failed))
            else:
                info(self, "Books", "\n\n".join(lines))
            if res.added:
                self._changed()                     # → main window re-indexes (with progress)
            else:
                self._build()

        def fail(msg, tb):
            self.adding = False
            self.cancel_btn.hide()
            self._build()
            error(self, "Books not added", msg, tb)

        run_job(lambda progress: import_books(paths, items, progress=progress,
                                              cancelled=lambda: self._cancel_requested, isolated=isolated),
                done, fail, self._set_progress)

    def _cancel(self):
        self._cancel_requested = True
        self.cancel_btn.setEnabled(False)
        self.prog_msg.setText("Cancelling…")

    # ── deleting / restoring books ───────────────────────────────────────
    def _delete(self, book: Book):
        if book.source == "user":
            text = (f"Delete “{book.title}” permanently? Its file and symptom clusters are removed from this "
                    "computer. Saved cases keep their results.")
        else:
            text = (f"Remove “{book.title}” from use in this clinic? It came with HoRUS, so it stays installed "
                    "and can be restored later with “Restore removed books”. Saved cases keep their results.")
        if not confirm(self, "Delete book", text, "Delete", danger=True):
            return
        if len(self.ctx.engine.books) == 1:
            return error(self, "Delete book", "At least one book is needed.")
        try:
            if book.source == "user":
                remove_book(self.ctx.paths, book.id)
                self.ctx.session.audit("book.deleted", book.id)
            else:
                self.ctx.session.set_hidden_books(sorted(self.ctx.engine.hidden_books | {book.id}))
            cfg = DecisionConfig.from_dict(self.ctx.session.decision_config())
            if cfg.books and book.id in cfg.books:
                self.ctx.session.set_decision_config(
                    replace(cfg, books=tuple(b for b in cfg.books if b != book.id)).to_dict())
        except Exception as e:
            return error(self, "Not deleted", str(e))
        self.ctx.engine.reload_books(set(self.ctx.session.hidden_books()))
        self.ctx.engine.set_config(clinic_config(self.ctx))
        self._changed()

    def _restore(self):
        restored = set(self.ctx.engine.hidden_books)
        cfg = DecisionConfig.from_dict(self.ctx.session.decision_config())
        if cfg.books:
            self.ctx.session.set_decision_config(
                replace(cfg, books=cfg.books + tuple(b for b in sorted(restored) if b not in cfg.books)).to_dict())
        self.ctx.session.set_hidden_books([])
        self.ctx.engine.reload_books(set())
        self.ctx.engine.set_config(clinic_config(self.ctx))
        self._changed()


def _delete_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            item.widget().deleteLater()
        elif item.layout() is not None:
            _delete_layout(item.layout())


class AddBooksDialog(QDialog):
    """Name the chosen books before importing them."""

    def __init__(self, files, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem
        self.setWindowTitle("Add books")
        self.resize(760, 360)
        v = QVBoxLayout(self)
        v.addWidget(label(f"{len(files)} file(s) chosen. Edit the names shown in HoRUS, untick any you don't "
                          "want, then press Add. Large books take a few minutes each; you can keep working.",
                          "muted"))
        self.table = QTableWidget(len(files), 3)
        self.table.setHorizontalHeaderLabels(["Add", "Name shown in HoRUS", "File"])
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.files = list(files)
        for r, f in enumerate(self.files):
            from .widgets import check_cell
            cell, _cb = check_cell(True)
            self.table.setCellWidget(r, 0, cell)
            self.table.setItem(r, 1, QTableWidgetItem(f.stem.replace("_", " ").strip()))
            n = f.stat().st_size if f.exists() else 0
            size = f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{max(1, round(n / 1024))} KB"
            it = QTableWidgetItem(f"{f.name}  ({size})")
            it.setFlags(Qt.ItemIsEnabled)
            it.setToolTip(str(f))
            self.table.setItem(r, 2, it)
        v.addWidget(self.table, 1)
        self.go = button(f"Add {len(files)} book{'s' if len(files) != 1 else ''}", primary=True, slot=self._ok)
        v.addLayout(row(None, button("Cancel", slot=self.reject), self.go))

    def items(self):
        from .widgets import cell_checked
        out = []
        for r, f in enumerate(self.files):
            title = (self.table.item(r, 1).text() if self.table.item(r, 1) else "").strip()
            if cell_checked(self.table, r) and title:
                out.append((f, title))
        return out

    def _ok(self):
        if not self.items():
            return error(self, "Add books", "Tick at least one book and give it a name.")
        self.accept()
