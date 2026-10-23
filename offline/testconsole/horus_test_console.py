"""HoRUS Offline — engine test console.

A small window for trying the step-1 engine on a laptop: enter a case (or load a sample),
press Analyse, and see what the engine decides — red flags, miasm, ranked remedies with
the dataset evidence behind each, and co-occurrence suggestions — then save the PDF.

This is a test harness, not the clinic app (that is step 4). There is no patient database
and no AI parser yet, so symptoms are entered in structured form.

Headless check used by the build pipeline:
    HoRUS-Offline-Test.exe --selftest <out-folder> [--require-model]
"""

from __future__ import annotations

import json
import logging
import queue
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(HERE.parent))           # run from a source checkout

from horus import __version__ as ENGINE_VERSION    # noqa: E402
from horus.config import Paths                     # noqa: E402

APP_TITLE = "HoRUS Offline — Engine Test Console"
GROUPS = (("psychological", "Mind & emotions"), ("general", "Generals (whole body)"),
          ("physical", "Physical / local"))

SAMPLES = {
    "Anxiety after midnight, grief": {
        "id": "SAMPLE-001", "name": "Sample Patient A", "age": "46", "gender": "Female",
        "raw": ("Since her husband died in March she wakes after midnight with anxiety and restlessness, "
                "cannot stay in bed and walks about. Fears being alone. Burning pains in the stomach, "
                "better from warm drinks. Very chilly, wants to be covered. Thirsty for small sips of "
                "water often. Worse after midnight, better from warmth."),
        "psychological": ["anxiety and restlessness after midnight | worse: after midnight",
                          "fear of being alone"],
        "general": ["chilly, wants to be covered", "thirst for small sips of water often"],
        "physical": ["burning pain in stomach | better: warm drinks"],
        "case_worse": "after midnight", "case_better": "warmth", "causation": "grief",
    },
    "Right knee, worse first motion": {
        "id": "SAMPLE-002", "name": "Sample Patient B", "age": "58", "gender": "Male",
        "raw": ("Right knee pain and stiffness for two years after getting soaked in rain. Worse on first "
                "motion and in cold damp weather, better after walking a while and from hot fomentation. "
                "Restless at night, keeps changing position. Irritable."),
        "psychological": ["irritable"],
        "general": ["restless at night, changes position"],
        "physical": ["right knee pain and stiffness | worse: first motion, cold damp weather "
                     "| better: continued walking, hot fomentation"],
        "case_worse": "", "case_better": "", "causation": "getting wet",
    },
    "Chest pain to left arm (red-flag demo)": {
        "id": "SAMPLE-003", "name": "Sample Patient C", "age": "63", "gender": "Male",
        "raw": ("Pressing pain in the chest going to the left arm with sweating since this morning. "
                "Afraid he will die. Worse on exertion."),
        "psychological": ["fear of death"],
        "general": ["sweating"],
        "physical": ["pressing chest pain radiating to left arm | worse: exertion"],
        "case_worse": "", "case_better": "", "causation": "",
    },
}


# ── parsing the structured entry ─────────────────────────────────────────────
def _split_list(text: str) -> list[str]:
    return [p.strip() for p in (text or "").replace(";", ",").split(",") if p.strip()]


def parse_symptom_line(line: str) -> dict | None:
    """`symptom | worse: a, b | better: c` → {symptom, worse, better}."""
    parts = [p.strip() for p in line.split("|")]
    if not parts or not parts[0]:
        return None
    out = {"symptom": parts[0], "worse": [], "better": []}
    for p in parts[1:]:
        key, _, val = p.partition(":")
        key = key.strip().lower()
        if key in ("worse", "agg", "<"):
            out["worse"] += _split_list(val)
        elif key in ("better", "amel", ">"):
            out["better"] += _split_list(val)
    return out


def build_case(fields: dict) -> dict:
    cat = {"physical": [], "psychological": [], "general": [],
           "case_modalities": {"worse": _split_list(fields.get("case_worse", "")),
                               "better": _split_list(fields.get("case_better", ""))},
           "causation": _split_list(fields.get("causation", "")),
           "concomitants": [], "clinical_summary": ""}
    for key, _ in GROUPS:
        for line in fields.get(key, []):
            s = parse_symptom_line(line)
            if s:
                cat[key].append(s)
    return cat


def analyse(engine, fields: dict) -> dict:
    """Run the engine exactly as the clinic app will: verify → rank → report → suggest."""
    cat = build_case(fields)
    raw = (fields.get("raw") or "").strip()
    cat = engine.finalise(cat, raw) if raw else engine.finalise_structured(cat)
    details = {k: fields.get(k, "") for k in ("name", "age", "gender")}
    report = engine.report(fields.get("id") or "TEST", cat, patient_details=details)
    per_symptom, top = engine.suggestions(cat)
    return {"categorised": cat, "report": report, "per_symptom": per_symptom, "top": top,
            "details": details, "id": fields.get("id") or "TEST"}


def load_engine():
    from horus.engine import HorusEngine
    return HorusEngine.from_defaults(Paths())


def _setup_logging() -> Path:
    log_dir = Paths().user / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "test-console.log"
    logging.basicConfig(filename=path, level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return path


# ── headless self-test (used by the build pipeline) ──────────────────────────
def selftest(out_dir: Path, require_model: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging()
    result = {"ok": False}
    try:
        engine = load_engine()
        result.update(matching=engine.matching_mode, dataset=engine.dataset.source,
                      symptoms=len(engine.dataset.s2r), resources=str(engine.paths.resources))
        if require_model and engine.matching_mode != "semantic":
            raise RuntimeError("embedding model not bundled or failed to load")
        for name, sample in SAMPLES.items():
            res = analyse(engine, sample)
            rpt = res["report"]
            result.setdefault("samples", {})[name] = {
                "top": [r["name"] for r in rpt.get("top10Remedies", [])][:5],
                "miasm": rpt.get("miasmLabel"), "red_flags": [f["title"] for f in rpt.get("redFlags", [])],
            }
            pdf = engine.pdf(res["id"], res["categorised"], rpt, patient_details=res["details"])
            assert pdf[:4] == b"%PDF"
            (out_dir / f"{sample['id']}.pdf").write_bytes(pdf)
        flags = result["samples"]["Chest pain to left arm (red-flag demo)"]["red_flags"]
        assert "Possible heart emergency" in flags, flags
        assert all(s["top"] for s in result["samples"].values()), "a sample produced no remedies"
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
    (out_dir / "selftest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


# ── the window ───────────────────────────────────────────────────────────────
def run_gui(smoke_out: Path | None = None) -> None:
    """``smoke_out``: open the window, analyse a sample, record the outcome there, close."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    log_path = _setup_logging()
    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1280x820")
    root.minsize(1000, 640)
    try:
        root.tk.call("tk", "scaling", root.winfo_fpixels("1i") / 72.0)
    except tk.TclError:
        pass

    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    BASE = ("Segoe UI", 10)
    style.configure("Head.TLabel", font=("Segoe UI Semibold", 11))
    style.configure("Muted.TLabel", foreground="#64748b")
    style.configure("Accent.TButton", font=("Segoe UI Semibold", 10))

    state = {"engine": None, "result": None}
    jobs: queue.Queue = queue.Queue()

    def run_bg(fn, done):
        def worker():
            try:
                jobs.put((done, fn(), None))
            except Exception as e:      # shown to the user, full trace in the log
                logging.exception("Background task failed")
                jobs.put((done, None, e))
        threading.Thread(target=worker, daemon=True).start()

    def poll():
        try:
            while True:
                done, value, err = jobs.get_nowait()
                done(value, err)
        except queue.Empty:
            pass
        root.after(80, poll)

    # layout: inputs on the left, results on the right
    paned = ttk.PanedWindow(root, orient="horizontal")
    paned.pack(fill="both", expand=True, padx=10, pady=(10, 0))
    left = ttk.Frame(paned, padding=(4, 4, 10, 4))
    right = ttk.Frame(paned, padding=(10, 4, 4, 4))
    paned.add(left, weight=2)
    paned.add(right, weight=3)

    # ── inputs ──
    top_row = ttk.Frame(left)
    top_row.pack(fill="x")
    ttk.Label(top_row, text="Case", style="Head.TLabel").pack(side="left")
    sample_var = tk.StringVar(value="Load a sample…")
    sample_box = ttk.Combobox(top_row, textvariable=sample_var, values=list(SAMPLES), state="readonly", width=34)
    sample_box.pack(side="right")

    pt = ttk.Frame(left)
    pt.pack(fill="x", pady=(8, 4))
    entries = {}
    for col, (key, label, width) in enumerate((("id", "Patient ID", 14), ("name", "Name", 22),
                                               ("age", "Age", 5))):
        ttk.Label(pt, text=label).grid(row=0, column=col, sticky="w", padx=(0, 6))
        e = ttk.Entry(pt, width=width, font=BASE)
        e.grid(row=1, column=col, sticky="we", padx=(0, 6))
        entries[key] = e
    ttk.Label(pt, text="Gender").grid(row=0, column=3, sticky="w")
    gender = ttk.Combobox(pt, values=["", "Female", "Male", "Other"], state="readonly", width=8)
    gender.grid(row=1, column=3, sticky="we")
    pt.columnconfigure(1, weight=1)

    def text_box(parent, label, hint, height):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(8, 0))
        if hint:
            ttk.Label(parent, text=hint, style="Muted.TLabel", wraplength=470).pack(anchor="w")
        t = tk.Text(parent, height=height, wrap="word", font=BASE, relief="solid", borderwidth=1,
                    padx=6, pady=4, undo=True)
        t.pack(fill="both", expand=True)
        return t

    raw_box = text_box(left, "Patient's words (optional)",
                       "If filled, every symptom below is checked against these words; ones not found "
                       "are marked unverified and left out of the ranking.", 4)
    ttk.Label(left, text="One symptom per line.   Format:  symptom | worse: a, b | better: c",
              style="Muted.TLabel").pack(anchor="w", pady=(10, 0))
    group_boxes = {key: text_box(left, label, "", 3) for key, label in GROUPS}

    mods = ttk.Frame(left)
    mods.pack(fill="x", pady=(8, 0))
    for col, (key, label) in enumerate((("case_worse", "Whole case — worse"),
                                        ("case_better", "Whole case — better"),
                                        ("causation", "Causation (ailments from)"))):
        ttk.Label(mods, text=label).grid(row=0, column=col, sticky="w", padx=(0, 6))
        e = ttk.Entry(mods, font=BASE)
        e.grid(row=1, column=col, sticky="we", padx=(0, 6))
        mods.columnconfigure(col, weight=1)
        entries[key] = e

    buttons = ttk.Frame(left)
    buttons.pack(fill="x", pady=(12, 4))
    analyse_btn = ttk.Button(buttons, text="Analyse", style="Accent.TButton")
    analyse_btn.pack(side="left")
    pdf_btn = ttk.Button(buttons, text="Save PDF…", state="disabled")
    pdf_btn.pack(side="left", padx=8)
    clear_btn = ttk.Button(buttons, text="Clear")
    clear_btn.pack(side="right")

    # ── results ──
    nb = ttk.Notebook(right)
    nb.pack(fill="both", expand=True)

    def result_text(title):
        frame = ttk.Frame(nb)
        t = tk.Text(frame, wrap="word", font=BASE, relief="flat", padx=12, pady=10, state="disabled",
                    background="#ffffff")
        sb = ttk.Scrollbar(frame, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        t.pack(fill="both", expand=True)
        t.tag_configure("h1", font=("Segoe UI Semibold", 15), spacing1=4, spacing3=6)
        t.tag_configure("h2", font=("Segoe UI Semibold", 11), foreground="#1e3a8a", spacing1=12, spacing3=4)
        t.tag_configure("flag", foreground="#7f1d1d", background="#fef2f2", lmargin1=8, lmargin2=8,
                        spacing1=2, spacing3=2)
        t.tag_configure("flaghead", foreground="#b91c1c", background="#fef2f2",
                        font=("Segoe UI Semibold", 12), lmargin1=8)
        t.tag_configure("rem", font=("Segoe UI Semibold", 11), spacing1=8)
        t.tag_configure("muted", foreground="#64748b")
        t.tag_configure("ev", foreground="#334155", lmargin1=22, lmargin2=34)
        t.tag_configure("warn", foreground="#92400e")
        nb.add(frame, text=title)
        return t

    report_txt = result_text("  Report  ")
    sugg_txt = result_text("  Symptom suggestions  ")
    about_txt = result_text("  About this build  ")

    def write(t, chunks):
        t.configure(state="normal")
        t.delete("1.0", "end")
        for text, *tags in chunks:
            t.insert("end", text, tuple(tags))
        t.configure(state="disabled")
        t.yview_moveto(0)

    status = tk.StringVar(value="Loading dataset and embedding model…")
    bar = ttk.Frame(root, padding=(12, 4))
    bar.pack(fill="x")
    ttk.Label(bar, textvariable=status, style="Muted.TLabel").pack(side="left")
    progress = ttk.Progressbar(bar, mode="indeterminate", length=140)
    progress.pack(side="right")
    progress.start(12)

    # ── behaviour ──
    def collect() -> dict:
        f = {k: e.get().strip() for k, e in entries.items()}
        f["gender"] = gender.get()
        f["raw"] = raw_box.get("1.0", "end").strip()
        for key, box in group_boxes.items():
            f[key] = [l for l in box.get("1.0", "end").splitlines() if l.strip()]
        return f

    def fill(sample: dict):
        for k, e in entries.items():
            e.delete(0, "end")
            e.insert(0, sample.get(k, ""))
        gender.set(sample.get("gender", ""))
        raw_box.delete("1.0", "end")
        raw_box.insert("1.0", sample.get("raw", ""))
        for key, box in group_boxes.items():
            box.delete("1.0", "end")
            box.insert("1.0", "\n".join(sample.get(key, [])))

    def on_sample(_event=None):
        if sample_var.get() in SAMPLES:
            fill(SAMPLES[sample_var.get()])

    def on_clear():
        fill({})
        sample_var.set("Load a sample…")
        state["result"] = None
        pdf_btn.configure(state="disabled")
        write(report_txt, [])
        write(sugg_txt, [])

    def render(res: dict):
        rpt, cat = res["report"], res["categorised"]
        c = [(f"{res['details'].get('name') or 'Patient'}  ·  {res['id']}\n", "h1")]
        flags = rpt.get("redFlags") or []
        if flags:
            c.append(("  ⚠ Seek urgent medical care\n", "flaghead"))
            for f in flags:
                c.append((f"  {f['title']}. {f['advice']}\n", "flag"))
        c += [("Miasm (rubric-based)\n", "h2"), (rpt.get("miasmaticAnalysis", "") + "\n",)]
        if cat.get("causation"):
            c += [("Causation\n", "h2"), (", ".join(cat["causation"]) + "\n",)]
        if rpt.get("unverifiedSymptoms"):
            c += [("Unverified — not in the patient's words, left out of the ranking\n", "h2"),
                  (", ".join(rpt["unverifiedSymptoms"]) + "\n", "warn")]
        c.append(("Remedies ranked from the dataset\n", "h2"))
        top10 = rpt.get("top10Remedies") or []
        if not top10:
            c.append((rpt["primaryRemedy"]["why"] + "\n", "warn"))
        for r in top10:
            c.append((f"{r['rank']}.  {r['name']}", "rem"))
            c.append((f"     {r['role']}   ·   score {r['datasetScore']:.3f}\n", "muted"))
            for e in r["evidence"][:6]:
                c.append((f"• {e['patient']}  ≈  “{e['dataset']}”   ({e['similarity']:.2f})\n", "ev"))
        c.append(("\n" + rpt.get("_method", "") + "\n", "muted"))
        write(report_txt, c)

        s = [("Symptoms that often occur with this case\n", "h1"),
             ("From the co-occurrence clusters. Ask the patient about these to strengthen the case.\n", "muted"),
             ("Across the whole case\n", "h2")]
        if res["top"]:
            for x in res["top"]:
                s.append((f"• {x['symptom']}", ))
                s.append((f"   {x['avg_prob']:.0%} · seen with {x['support']} · strongest link: "
                          f"{x['triggered_by']}\n", "muted"))
        else:
            s.append(("No co-occurrence matches — suggestions need symptoms worded like the dataset.\n", "muted"))
        for sym, items in res["per_symptom"].items():
            if items:
                s.append((f"After “{sym}”\n", "h2"))
                for x in items:
                    s.append((f"• {x['symptom']}", ))
                    s.append((f"   {x['prob']:.0%}\n", "muted"))
        write(sugg_txt, s)
        nb.select(0)

    def on_analyse():
        if state["engine"] is None:
            return
        fields = collect()
        if not any(fields[k] for k, _ in GROUPS):
            messagebox.showinfo(APP_TITLE, "Enter at least one symptom, or load a sample.")
            return
        analyse_btn.configure(state="disabled")
        progress.start(12)
        status.set("Analysing…")

        def done(res, err):
            progress.stop()
            analyse_btn.configure(state="normal")
            if err:
                status.set("Analysis failed — see the log.")
                if smoke_out:
                    return smoke_finish(False, f"analysis failed: {err}")
                messagebox.showerror(APP_TITLE, f"Analysis failed:\n{err}\n\nLog: {log_path}")
                return
            state["result"] = res
            pdf_btn.configure(state="normal")
            n = len(res["report"].get("top10Remedies") or [])
            status.set(f"Done — {n} remedies ranked · matching: {state['engine'].matching_mode}")
            render(res)
            if smoke_out:
                shown = report_txt.get("1.0", "end")
                smoke_finish(n > 0 and res["report"]["top10Remedies"][0]["name"] in shown,
                             f"{n} remedies ranked and shown; matching {state['engine'].matching_mode}")

        run_bg(lambda: analyse(state["engine"], fields), done)

    def on_pdf():
        res = state["result"]
        if not res:
            return
        path = filedialog.asksaveasfilename(
            parent=root, defaultextension=".pdf", filetypes=[("PDF", "*.pdf")],
            initialfile=f"HoRUS_{res['id']}_{datetime.now():%Y%m%d}.pdf")
        if not path:
            return
        try:
            pdf = state["engine"].pdf(res["id"], res["categorised"], res["report"],
                                      patient_details=res["details"])
            Path(path).write_bytes(pdf)
            status.set(f"Saved {path}")
        except Exception as e:
            logging.exception("PDF failed")
            messagebox.showerror(APP_TITLE, f"Could not save the PDF:\n{e}")

    def engine_ready(engine, err):
        progress.stop()
        if err:
            status.set("Engine failed to load — see the log.")
            if smoke_out:
                return smoke_finish(False, f"engine failed to load: {err}")
            messagebox.showerror(APP_TITLE, f"The engine failed to load:\n{err}\n\nLog: {log_path}")
            return
        state["engine"] = engine
        analyse_btn.configure(state="normal")
        mode = engine.matching_mode
        status.set(f"Ready · {len(engine.dataset.s2r):,} dataset symptoms · "
                   f"{len(engine.dataset.remedy_universe)} remedies · matching: {mode} · offline")
        write(about_txt, [
            ("About this build\n", "h1"),
            ("This is the engine from step 1 of the offline build, packaged for testing. It runs entirely on "
             "this computer: nothing is sent over the internet.\n",),
            ("What it is not (yet)\n", "h2"),
            ("No patient database, logins or saved history (step 2). No AI: free-text parsing, "
             "explanations, voice and follow-up questions arrive with the local model (step 3). "
             "The real clinic app replaces this window in step 4.\n",),
            ("Engine\n", "h2"),
            (f"Engine version {ENGINE_VERSION}\nDataset: {engine.dataset.source}\n"
             f"Symptom matching: {mode}"
             + (" (all-MiniLM-L6-v2, bundled)" if mode == "semantic"
                else " — embedding model not found, using word overlap") +
             f"\nAI explanations: not installed — reports use dataset evidence only\n"
             f"Data folder: {engine.paths.resources}\nLog file: {log_path}\n",),
        ])

        if smoke_out:
            sample_var.set(next(iter(SAMPLES)))
            on_sample()
            on_analyse()

    def smoke_finish(ok: bool, detail: str):
        smoke_out.mkdir(parents=True, exist_ok=True)
        (smoke_out / "guismoke.json").write_text(json.dumps({"ok": bool(ok), "detail": detail}),
                                                 encoding="utf-8")
        root.after(300, root.destroy)

    sample_box.bind("<<ComboboxSelected>>", on_sample)
    analyse_btn.configure(command=on_analyse, state="disabled")
    pdf_btn.configure(command=on_pdf)
    clear_btn.configure(command=on_clear)
    root.bind("<Control-Return>", lambda e: on_analyse())

    write(report_txt, [("Load a sample or enter a case, then press Analyse.\n", "muted")])
    run_bg(load_engine, engine_ready)
    root.after(80, poll)
    if smoke_out:
        root.after(600_000, lambda: smoke_finish(False, "timed out after 10 minutes"))
    root.mainloop()


def main(argv: list[str]) -> int:
    # A windowed .exe has no console: libraries that print progress would crash on None streams.
    import os
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    if "--selftest" in argv:
        i = argv.index("--selftest")
        out = Path(argv[i + 1]) if len(argv) > i + 1 and not argv[i + 1].startswith("--") else Path("selftest")
        return selftest(out, require_model="--require-model" in argv)
    if "--gui-smoke" in argv:
        i = argv.index("--gui-smoke")
        out = Path(argv[i + 1]) if len(argv) > i + 1 else Path("guismoke")
        run_gui(smoke_out=out)
        f = out / "guismoke.json"
        return 0 if f.exists() and json.loads(f.read_text(encoding="utf-8"))["ok"] else 1
    run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
