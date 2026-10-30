"""HoRUS desktop app entry point.

    HoRUS.exe                              normal start
    HoRUS.exe --selftest <out> [--require-ai]   headless end-to-end check (used by the installer build)
    HoRUS.exe --gui-smoke <out>             open the window, run a case through it, close
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

APP_VERSION = "1.5.0"
log = logging.getLogger("horus")


def _streams():
    # a windowed .exe has no console: libraries that print would crash on None streams
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _logging(user_dir: Path) -> Path:
    d = user_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "horus.log"
    h = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    return path


# ═════════════════════════════════════════════════════════════════════════════
def run_app() -> int:
    from PySide6.QtCore import QLockFile, Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

    from horus.config import Paths
    from .context import AppContext
    from .login import LoginDialog
    from .main_window import MainWindow
    from .setup_wizard import SetupWizard
    from .theme import QSS

    paths = Paths()
    paths.user.mkdir(parents=True, exist_ok=True)
    log_path = _logging(paths.user)
    app = QApplication(sys.argv)
    from .context import collect_garbage_on_gui_thread
    collect_garbage_on_gui_thread(app)
    app.setApplicationName("HoRUS")
    app.setApplicationVersion(APP_VERSION)
    app.setStyleSheet(QSS)
    icon = paths.resources / "horus.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    def excepthook(et, ev, tb):
        log.error("Unhandled error", exc_info=(et, ev, tb))
        box = QMessageBox(QMessageBox.Critical, "HoRUS", f"Something went wrong: {ev}\n\nDetails are in {log_path}.")
        box.setDetailedText("".join(traceback.format_exception(et, ev, tb)))
        box.exec()
    sys.excepthook = excepthook

    lock = QLockFile(str(paths.user / "horus.lock"))
    lock.setStaleLockTime(0)
    if not lock.tryLock(100):
        QMessageBox.information(None, "HoRUS", "HoRUS is already open on this computer.")
        return 0

    from PySide6.QtCore import QThreadPool
    code = 0
    while True:                                  # loops only when the clinic is deleted → setup again
        ctx = AppContext(paths)
        log.info("HoRUS %s starting; data %s; AI %s", APP_VERSION, ctx.db_path,
                 getattr(ctx.llm, "name", "none"))
        if ctx.open_store() is None:
            wiz = SetupWizard(ctx)
            if wiz.exec() != QDialog.Accepted:
                break
            ctx.session = wiz.session
            if ctx.session is None:              # restored with the recovery key → sign in normally
                dlg = LoginDialog(ctx.store)
                if dlg.exec() != QDialog.Accepted:
                    break
                ctx.session = dlg.session
        else:
            dlg = LoginDialog(ctx.store)
            if dlg.exec() != QDialog.Accepted:
                break
            ctx.session = dlg.session

        win = MainWindow(ctx)
        from .context import freeze_long_lived_objects
        ctx.events.engine_ready.connect(freeze_long_lived_objects)    # books stay loaded: skip them in GC
        win.show()
        code = app.exec()
        ctx.shutdown()
        QThreadPool.globalInstance().waitForDone(15000)
        restart = win.restart_setup
        win.deleteLater()
        if not restart:
            break
        app.setQuitOnLastWindowClosed(True)
    lock.unlock()
    return code


# ═════════════════════════════════════════════════════════════════════════════
SAMPLE_TEXT = ("Since her husband died in March she wakes after midnight with anxiety and restlessness, cannot "
               "stay in bed and walks about. Fears being alone. Burning pains in the stomach, better from warm "
               "drinks. Very chilly, wants to be covered. Thirsty for small sips of water often.")


def selftest(out: Path, require_ai: bool) -> int:
    """Headless end-to-end check of an installed build. Writes ``selftest.json``."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="horus-selftest-"))
    os.environ["HORUS_USER_DIR"] = str(tmp / "user")
    result: dict = {"ok": False, "version": APP_VERSION, "checks": {}}
    checks = result["checks"]

    def check(name, fn):
        t = time.time()
        try:
            val = fn()
            checks[name] = {"ok": True, "seconds": round(time.time() - t, 1), "info": val}
            return val
        except Exception as e:
            checks[name] = {"ok": False, "seconds": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()[-2500:]}
            return None

    try:
        from horus.config import Paths
        from horus.embeddings import load_embedder
        from horus.engine import HorusEngine
        from horus.llama import find_llama
        from horus.store import Store
        from horus.whisper import find_whisper
        paths = Paths()
        _logging(paths.user)

        def manifest():
            import hashlib
            m = json.loads((paths.models_dir / "manifest.json").read_text())
            bad = []
            for rel, digest in m["files"].items():
                h = hashlib.sha256()
                with open(paths.models_dir / rel, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 22), b""):
                        h.update(chunk)
                if h.hexdigest() != digest:
                    bad.append(rel)
            if bad:
                raise RuntimeError(f"corrupted model files: {bad}")
            return {"files": len(m["files"]), "llm": m.get("llm", {}).get("file"),
                    "llama": m.get("llama_server", {}).get("tag")}
        check("models_intact", manifest)

        llm, wh = find_llama(paths.models_dir), find_whisper(paths.models_dir)
        emb = check("embedding_model", lambda: (lambda e: e.model_id if e else None)(
            load_embedder(paths.embedding_model_dir)))
        engine = HorusEngine(paths, embedder=load_embedder(paths.embedding_model_dir), llm=llm, transcriber=wh)
        if engine.matching_mode != "semantic":
            checks["embedding_model"] = {"ok": False, "error": "semantic matching unavailable"}

        def store_flow():
            store, key = Store.create(paths.user / "data" / "horus.db", edition="clinic_lan", clinic_name="Selftest",
                                      admin_username="admin", admin_password="selftest-pass")
            s = store.login("admin", "selftest-pass")
            return store, key, s
        res = check("encrypted_store", store_flow)
        store, rkey, s = res or (None, None, None)
        if res:
            checks["encrypted_store"]["info"] = "created, signed in"   # never write keys to the report

        def seed_index():
            keys = tuple(sorted(engine.dataset.s2r))
            missing = engine._emb_cache.missing(keys) if engine._emb_cache else len(keys)
            assert missing == 0, f"{missing} of {len(keys)} shipped symptoms not in the built-in index"
            return {"indexed_symptoms": len(keys)}
        check("builtin_symptom_index", seed_index)

        def books():
            import json as _json
            from horus.books import import_book
            ids = [b.id for b in engine.books]
            assert {"rheumatic", "case_studies", "guiding_symptoms_1"} <= set(ids), ids
            assert sum(i.startswith("guiding_symptoms_") for i in ids) >= 9, ids     # cleaned vols I–X
            book = {"remedium probandum": {"mind": [{"symptom": f"selftest symptom {i}",
                                                     "labels": {"Modalities": "worse at night"}}
                                                    for i in range(10)]}}
            src = tmp / "selftest_book.json"
            src.write_text(_json.dumps(book), encoding="utf-8")
            from horus.book_worker import import_book_isolated
            added = import_book_isolated(paths, src, "Selftest book")   # child process, scikit-learn
            engine.reload_books()
            return {"books": ids, "added": added.id, "clusters": added.cooccurrence_path.exists(),
                    "symptoms_all_books": len(engine.dataset.s2r)}
        check("books_and_clusters", books)

        def large_book_responsive():
            """The user's scenario: adding an 18k-symptom book must not freeze the app. Import it in the
            background exactly as the app does, while the main thread measures its own stalls."""
            import threading
            from horus.book_worker import import_book_isolated
            # a copy under another name: the shipped volume itself is refused as "already included"
            big = tmp / "large_book.json"
            big.write_bytes((paths.resources / "Guiding_Symptoms_of_our_Materia_Medica-I.json").read_bytes())
            out, worst, last = {}, [0.0], [time.time()]
            t = threading.Thread(target=lambda: out.setdefault(
                "book", import_book_isolated(paths, big, "Large book test")), daemon=True)
            t0 = time.time()
            t.start()
            while t.is_alive():
                time.sleep(0.05)
                now = time.time()
                worst[0] = max(worst[0], now - last[0])
                last[0] = now
            assert "book" in out, "import failed"
            assert worst[0] < 2.0, f"main thread stalled for {worst[0]:.1f}s"
            return {"symptoms": 18635, "seconds": round(time.time() - t0, 1),
                    "longest_stall_s": round(worst[0], 2)}
        check("large_book_import_responsive", large_book_responsive)

        def structured_case():
            cat = engine.empty_case()
            cat["psychological"].append({"symptom": "anxiety after midnight", "worse": ["after midnight"],
                                         "better": []})
            cat["general"].append({"symptom": "chilly, wants to be covered", "worse": [], "better": []})
            cat["physical"].append({"symptom": "burning pain in stomach", "worse": [], "better": ["warm drinks"]})
            cat = engine.finalise_structured(cat)
            rpt = engine.report("PT-TEST", cat)
            pid = s.create_patient({"name": "Self Test"})
            s.add_case(pid, {"raw_symptoms": "", "categorised": cat, "report": rpt,
                             "top_remedies": [r["name"] for r in rpt["top10Remedies"]]})
            engine.learn_from(s.learning_records())
            pdf = engine.pdf(pid, cat, rpt)
            assert pdf[:4] == b"%PDF" and rpt["top10Remedies"]
            return [r["name"] for r in rpt["top10Remedies"][:3]]
        check("structured_case_report_pdf", structured_case)

        if llm:
            def ai_parse():
                cat = engine.categorise(SAMPLE_TEXT)
                n = sum(len(cat[g]) for g in ("physical", "psychological", "general"))
                assert n >= 3, f"only {n} symptoms parsed"
                return {"symptoms": n, "unverified": cat.get("unverified"), "causation": cat.get("causation"),
                        "stats": llm.last_stats.get("seconds")}
            parsed = check("ai_case_parsing", ai_parse)

            def ai_report():
                cat = engine.categorise(SAMPLE_TEXT)
                rpt = engine.report("PT-TEST", cat)
                kept = sum(not r["rationale"].startswith("Matched in dataset") for r in rpt["top10Remedies"])
                return {"rationales_written": kept, "essence": rpt.get("caseEssence", "")[:160]}
            check("ai_explanations", ai_report) if parsed else None
            check("ai_followups", lambda: engine.followups("Main complaint: knee pain\nSleep: poor"))
        if wh:
            def speech():
                import io
                import wave
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(b"\x00\x00" * 16000)
                return {"silence_transcript": wh.transcribe(buf.getvalue())}
            check("speech_model", speech)

        def lan():
            import http.cookiejar
            import ssl
            import urllib.request
            from horus_app.lan import IntakeServer
            got = []
            srv = IntakeServer(lambda d, qa, raw: (got.append(raw),
                                                   s.add_intake(s.create_patient(d), {"qa": qa, "raw_text": raw},
                                                                "lan"))[1] and "PT-LAN",
                               paths.user / "lan", port=0)
            srv.port = _free_port()
            srv.start()
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE          # self-signed; fingerprint shown to the user instead
                jar = http.cookiejar.CookieJar()
                op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx),
                                                 urllib.request.HTTPCookieProcessor(jar),
                                                 urllib.request.ProxyHandler({}))
                base = f"https://127.0.0.1:{srv.port}"

                def post(path, body):
                    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                                 headers={"Content-Type": "application/json"})
                    try:
                        with op.open(req, timeout=10) as r:
                            return r.status
                    except urllib.error.HTTPError as e:
                        return e.code
                assert post("/api/intake", {"details": {"name": "X"}, "answers": {}}) == 401
                assert post("/api/pair", {"pin": "000000" if srv.pin != "000000" else "111111"}) == 403
                assert post("/api/pair", {"pin": srv.pin}) == 200
                assert post("/api/intake", {"details": {"name": "Tablet Patient"},
                                            "answers": {"complaint": "knee pain"}}) == 200
                assert got and "knee pain" in got[0]
                return {"port": srv.port}
            finally:
                srv.stop()
        if s:
            check("lan_tablet_intake", lan)

        def backup():
            bk = s.backup(tmp / "b.horusbak")
            r = Store.restore(bk, tmp / "restored.db", "admin", "selftest-pass")
            r2 = r.login("admin", "selftest-pass")
            assert len(r2.list_patients()) == len(s.list_patients())
            assert store.verify_audit_chain()[0]
            return {"bytes": bk.stat().st_size}
        if s:
            check("backup_restore", backup)

        required = ["models_intact", "embedding_model", "encrypted_store", "structured_case_report_pdf",
                    "lan_tablet_intake", "backup_restore", "books_and_clusters",
                    "builtin_symptom_index", "large_book_import_responsive"]
        if require_ai:
            required += ["ai_case_parsing", "ai_explanations", "ai_followups", "speech_model"]
        missing = [k for k in required if not checks.get(k, {}).get("ok")]
        result["ok"] = not missing
        result["failed"] = missing
        result["ai"] = {"llm": getattr(llm, "name", None), "whisper": getattr(wh, "name", None),
                        "matching": engine.matching_mode}
        if llm:
            llm.stop()
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    (out / "selftest.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return 0 if result["ok"] else 1


def _free_port() -> int:
    import socket
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        return so.getsockname()[1]


def gui_smoke(out: Path) -> int:
    """Open the real main window on a fresh clinic, drive a structured case through it, close."""
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="horus-gui-"))
    os.environ["HORUS_USER_DIR"] = str(tmp / "user")
    os.environ.setdefault("HORUS_NO_AI", "1")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from horus.config import Paths
    from horus.store import Store
    from .context import AppContext
    from .main_window import MainWindow
    from .theme import QSS

    app = QApplication(sys.argv)
    from .context import collect_garbage_on_gui_thread
    collect_garbage_on_gui_thread(app)
    app.setStyleSheet(QSS)
    paths = Paths()
    _logging(paths.user)
    ctx = AppContext(paths)
    ctx.settings.update(ai_enabled=False, backup_on_exit=False)
    ctx.store, _ = Store.create(ctx.db_path, edition="clinic", clinic_name="Smoke Clinic",
                                admin_username="admin", admin_password="smoke-pass")
    ctx.session = ctx.store.login("admin", "smoke-pass")
    win = MainWindow(ctx)
    win.show()
    result = {"ok": False, "steps": []}
    started = time.time()

    def finish(ok, detail=""):
        result["ok"], result["detail"] = ok, detail
        (out / "guismoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        win.close()
        app.quit()

    def step():
        if time.time() - started > 300:
            return finish(False, "timeout")
        if ctx.engine is None:
            return QTimer.singleShot(300, step)
        for key in ("patients", "intakes", "settings", "case"):
            win.go(key)
            result["steps"].append(key)
        cf = win.case
        cf.reset()
        cf.pt_name.setText("Smoke Patient")
        cf.tabs.setCurrentIndex(1)
        cf.st_table.add("psychological", {"symptom": "anxiety after midnight", "worse": ["after midnight"],
                                          "better": []})
        cf.st_table.add("general", {"symptom": "chilly", "worse": [], "better": []})
        cf.st_table.add("physical", {"symptom": "burning pain in stomach", "worse": [], "better": ["warm drinks"]})
        cf._read_case()
        result["steps"].append(f"review rows={cf.rv_table.rowCount()}")
        cf._analyse()
        QTimer.singleShot(300, wait_report)

    def wait_report():
        cf = win.case
        if time.time() - started > 300:
            return finish(False, "report timeout")
        if cf.steps.currentIndex() != 2:
            return QTimer.singleShot(300, wait_report)
        html = cf.rp_view.toPlainText()
        top = (cf.report.get("top10Remedies") or [{}])[0].get("name", "")
        saved = cf.rp_saved.text()
        result["steps"].append(f"report top={top} saved={saved}")
        n_cases = sum(p["case_count"] for p in ctx.session.list_patients())
        finish(bool(top) and top in html and saved.startswith("✓") and n_cases == 1,
               f"top remedy {top}; cases saved {n_cases}")

    QTimer.singleShot(500, step)
    app.exec()
    f = out / "guismoke.json"
    return 0 if f.exists() and json.loads(f.read_text())["ok"] else 1


def main() -> int:
    _streams()
    args = sys.argv[1:]
    if "--selftest" in args:
        i = args.index("--selftest")
        return selftest(Path(args[i + 1]) if len(args) > i + 1 else Path("selftest"), "--require-ai" in args)
    if "--gui-smoke" in args:
        i = args.index("--gui-smoke")
        return gui_smoke(Path(args[i + 1]) if len(args) > i + 1 else Path("guismoke"))
    return run_app()


if __name__ == "__main__":
    sys.exit(main())
