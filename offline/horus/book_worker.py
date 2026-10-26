"""Import books in a separate process, so the app never freezes.

Clustering a large book (HDBSCAN over ~18,000 symptoms) holds Python's interpreter lock for close to a
minute; run in a thread, it freezes the window for that long. A child process has its own interpreter,
so the window stays responsive and the import can be cancelled.

The frozen Windows app must call ``multiprocessing.freeze_support()`` first thing (HoRUS.py does).
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .books import Book, discover_books
from .config import Paths


def _child(resources: str, user: str, source: str, title: str, q) -> None:
    """Runs in the child process."""
    try:
        from .books import import_book
        paths = Paths(resources=Path(resources), user=Path(user))
        book = import_book(paths, Path(source), title, progress=lambda f, m: q.put(("progress", f, m)))
        q.put(("done", book.id))
    except Exception as e:                                   # reported to the parent, never raised there
        q.put(("error", f"{type(e).__name__}: {e}" if not str(e) else str(e), traceback.format_exc()))


class ImportCancelled(Exception):
    pass


def import_book_isolated(paths: Paths, source: Path, title: str, progress=None, cancelled=None,
                         poll: float = 0.2) -> Book:
    """``import_book`` in a child process. ``progress(fraction, message)`` is called in the calling
    thread; ``cancelled()`` returning True stops the child."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_child, args=(str(paths.resources), str(paths.user), str(source), title, q),
                       name="horus-book-import", daemon=True)
    proc.start()
    result = None
    try:
        while True:
            if cancelled is not None and cancelled():
                proc.terminate()
                proc.join(5)
                _cleanup_partial(paths)
                raise ImportCancelled("Cancelled")
            try:
                msg = q.get(timeout=poll)
            except _queue.Empty:
                if not proc.is_alive():
                    # the child may have finished just after our last look
                    try:
                        msg = q.get(timeout=1.0)
                    except _queue.Empty:
                        raise RuntimeError(f"The book import stopped unexpectedly (code {proc.exitcode}). "
                                           "The computer may be low on memory.") from None
                else:
                    continue
            kind = msg[0]
            if kind == "progress":
                if progress:
                    progress(msg[1], msg[2])
            elif kind == "done":
                result = msg[1]
                break
            elif kind == "error":
                raise BookImportError(msg[1], msg[2])
    finally:
        if proc.is_alive():
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
        q.close()
    return next(b for b in discover_books(paths) if b.id == result)


class BookImportError(RuntimeError):
    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail


def _cleanup_partial(paths: Paths) -> None:
    import shutil
    folder = paths.user / "books"
    for p in folder.glob("*.partial") if folder.exists() else []:
        shutil.rmtree(p, ignore_errors=True)


# ── several books at once ────────────────────────────────────────────────────
@dataclass
class BatchResult:
    added: list = field(default_factory=list)              # Book
    failed: list = field(default_factory=list)             # (title, message, detail)
    cancelled: bool = False


def import_books(paths: Paths, items: list[tuple[Path, str]], progress=None, cancelled=None,
                 isolated: bool = True) -> BatchResult:
    """Import several books one after another (one child process at a time keeps memory use low).
    A book that fails is reported and the rest continue. ``progress`` gets the overall fraction."""
    res = BatchResult()
    n = len(items)
    for k, (src, title) in enumerate(items):
        def p(f, m, k=k, title=title):
            if progress:
                progress((k + f) / n, f"Book {k + 1} of {n} — {title}: {m}")
        p(0.0, "starting…")
        try:
            if isolated:
                book = import_book_isolated(paths, Path(src), title, progress=p, cancelled=cancelled)
            else:
                from .books import import_book
                book = import_book(paths, Path(src), title, progress=p)
            res.added.append(book)
        except ImportCancelled:
            res.cancelled = True
            break
        except BookImportError as e:
            res.failed.append((title, str(e), e.detail))
        except Exception as e:
            res.failed.append((title, str(e), traceback.format_exc()))
    if progress:
        progress(1.0, f"Added {len(res.added)} of {n} book(s)")
    return res
