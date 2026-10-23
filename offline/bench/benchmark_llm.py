"""Benchmark a local language model on HoRUS's actual jobs.

    python bench/benchmark_llm.py --models ../models --out results.json

Uses whatever model ``models/manifest.json`` names (fetch it with
``scripts/fetch_models.py --llm <key>``) and scores:

* **Case parsing** (the job that matters most) on 14 cases: the 4 saved demo cases against
  Gemini's own parse, and 10 written cases against reference parses. Symptom, modality and
  causation precision/recall (fuzzy word-stem matching), category agreement, how many
  symptoms the engine's verifier had to mark unverified (invented), valid JSON rate, and —
  the clinical bottom line — whether the dataset ranking from the model's parse agrees with
  the ranking from the reference parse (top-1 and top-5 overlap).
* **Explanations** on the 4 saved cases: valid JSON, share of remedy rationales that survive
  the engine's filters, and what the filters removed.
* **Follow-up questions**: valid list of ≤ 3 questions.
* Speed: seconds per job and generated tokens per second on this machine's CPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from horus.config import Paths  # noqa: E402
from horus.embeddings import load_embedder  # noqa: E402
from horus.engine import HorusEngine  # noqa: E402
from horus.llama import find_llama  # noqa: E402
from horus.parsing import SYMPTOM_GROUPS, content_stems  # noqa: E402
from reference_cases import REFERENCE_CASES  # noqa: E402

REPO = HERE.parents[1]
GEMINI_FILES = ("data/horus3_patients.json", "Horus_V2/patients.json")
FOLLOWUP_TEXT = ("Main complaint: pain in my knees\nOnset: two years ago\n"
                 "Mind & emotions: I worry a lot\nSleep: not good")


# ── fuzzy matching ───────────────────────────────────────────────────────────
def similar(a: str, b: str, threshold: float = 0.5) -> bool:
    if a.strip().lower() == b.strip().lower():
        return True
    sa, sb = content_stems(a), content_stems(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / min(len(sa), len(sb)) >= threshold


def greedy_pairs(pred: list, ref: list, key=lambda x: x) -> list[tuple[int, int]]:
    pairs, used = [], set()
    for i, p in enumerate(pred):
        for j, r in enumerate(ref):
            if j not in used and similar(key(p), key(r)):
                pairs.append((i, j))
                used.add(j)
                break
    return pairs


def prf(n_pred: int, n_ref: int, n_match: int) -> tuple[float | None, float | None]:
    return (n_match / n_pred if n_pred else None, n_match / n_ref if n_ref else None)


def flat_symptoms(cat: dict) -> list[tuple[str, str]]:
    return [(g, s["symptom"]) for g in SYMPTOM_GROUPS for s in cat.get(g, []) if not s.get("unverified")]


def flat_modalities(cat: dict) -> list[str]:
    out = []
    for g in SYMPTOM_GROUPS:
        for s in cat.get(g, []):
            out += [f"worse {m}" for m in s.get("worse", [])] + [f"better {m}" for m in s.get("better", [])]
    cm = cat.get("case_modalities") or {}
    out += [f"worse {m}" for m in cm.get("worse", [])] + [f"better {m}" for m in cm.get("better", [])]
    return list(dict.fromkeys(m.lower() for m in out))


def score_parse(pred: dict, ref: dict, engine: HorusEngine) -> dict:
    ps, rs = flat_symptoms(pred), flat_symptoms(ref)
    pairs = greedy_pairs(ps, rs, key=lambda x: x[1])
    sp, sr = prf(len(ps), len(rs), len(pairs))
    cat_ok = sum(ps[i][0] == rs[j][0] for i, j in pairs)

    pm, rm = flat_modalities(pred), flat_modalities(ref)
    mpairs = [(i, j) for i, j in greedy_pairs(pm, rm) if pm[i].split()[0] == rm[j].split()[0]]
    mp, mr = prf(len(pm), len(rm), len(mpairs))

    pc, rc = pred.get("causation", []), ref.get("causation", [])
    cp, cr = prf(len(pc), len(rc), len(greedy_pairs(pc, rc)))

    rank_p = [r["name"] for r in engine.rank(copy.deepcopy(pred), top_n=10)]
    rank_r = [r["name"] for r in engine.rank(copy.deepcopy(ref), top_n=10)]
    return {
        "symptom_precision": sp, "symptom_recall": sr,
        "category_agreement": cat_ok / len(pairs) if pairs else None,
        "modality_precision": mp, "modality_recall": mr,
        "cause_precision": cp, "cause_recall": cr,
        "invented_symptoms": len(pred.get("unverified", [])),
        "top1_same": (rank_p[:1] == rank_r[:1]) if rank_r else None,
        "top5_overlap": len(set(rank_p[:5]) & set(rank_r[:5])) / min(5, len(rank_r)) if rank_r else None,
        "pred_top3": rank_p[:3], "ref_top3": rank_r[:3],
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


# ── run ──────────────────────────────────────────────────────────────────────
def load_cases() -> list[dict]:
    cases = []
    for f in GEMINI_FILES:
        for pid, rec in json.loads((REPO / f).read_text(encoding="utf-8")).items():
            for i, c in enumerate(rec.get("cases", [])):
                ref = {k: v for k, v in c["categorised"].items() if not k.startswith("_")}
                cases.append({"id": f"{pid}#{i}", "raw": c["raw_symptoms"], "reference": ref,
                              "source": "gemini"})
    for c in REFERENCE_CASES:
        cases.append({**copy.deepcopy(c), "source": "written"})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("bench_results.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    llm = find_llama(args.models)
    if llm is None:
        raise SystemExit("No language model in manifest — run scripts/fetch_models.py first")
    paths = Paths(resources=REPO)
    embedder = load_embedder(args.models / "all-MiniLM-L6-v2")
    engine = HorusEngine(paths, embedder=embedder, llm=llm)
    manifest = json.loads((args.models / "manifest.json").read_text())

    t0 = time.time()
    llm.start()
    load_s = round(time.time() - t0, 1)

    cases = load_cases()[: args.limit or None]
    per_case, timings, gen_tps, invalid = [], [], [], 0
    for c in cases:
        ref = engine.finalise(copy.deepcopy(c["reference"]), c["raw"])
        t = time.time()
        try:
            pred = engine.categorise(c["raw"])
            ok = True
        except Exception as e:                    # invalid JSON or engine error
            pred, ok = None, False
            print(f"  {c['id']}: FAILED {type(e).__name__}: {e}")
        dt = time.time() - t
        timings.append(dt)
        tim = (llm.last_stats or {}).get("timings") or {}
        if tim.get("predicted_per_second"):
            gen_tps.append(tim["predicted_per_second"])
        if not ok:
            invalid += 1
            per_case.append({"id": c["id"], "source": c["source"], "valid": False, "seconds": round(dt, 1)})
            continue
        s = score_parse(pred, ref, engine)
        per_case.append({"id": c["id"], "source": c["source"], "valid": True, "seconds": round(dt, 1), **s,
                         "parsed": {k: pred[k] for k in ("physical", "psychological", "general",
                                                         "case_modalities", "causation")}})
        print(f"  {c['id']:<22} {dt:5.1f}s  sym P/R {s['symptom_precision']}/{s['symptom_recall']}  "
              f"top1 {'same' if s['top1_same'] else 'diff'}  {s['pred_top3']} vs {s['ref_top3']}")

    # explanations on the saved (Gemini) cases
    narr = []
    for c in [c for c in cases if c["source"] == "gemini"]:
        cat = engine.finalise(copy.deepcopy(c["reference"]), c["raw"])
        t = time.time()
        rpt = engine.report(c["id"], cat)
        dt = time.time() - t
        top = rpt.get("top10Remedies") or []
        kept = sum(not r["rationale"].startswith("Matched in dataset") for r in top)
        narr.append({"id": c["id"], "seconds": round(dt, 1), "rationales_kept": kept, "remedies": len(top),
                     "essence": rpt.get("caseEssence", "")[:300], "removed": rpt.get("_removed", []),
                     "primary_why": rpt["primaryRemedy"]["why"][:300]})

    t = time.time()
    fu = engine.followups(FOLLOWUP_TEXT)
    fu_s = time.time() - t

    def agg(rows, key):
        return mean([r.get(key) for r in rows if r.get("valid")])

    summary = {
        "model": manifest["llm"]["file"], "llama": manifest.get("llama_server", {}).get("tag"),
        "machine": f"{platform.system()} {platform.machine()} threads={llm.threads}",
        "load_seconds": load_s, "cases": len(cases), "invalid_json": invalid,
        "parse_seconds_median": round(statistics.median(timings), 1) if timings else None,
        "gen_tokens_per_second": round(statistics.median(gen_tps), 1) if gen_tps else None,
        **{k: agg(per_case, k) for k in ("symptom_precision", "symptom_recall", "category_agreement",
                                          "modality_precision", "modality_recall", "cause_precision",
                                          "cause_recall", "top5_overlap")},
        "top1_same": agg(per_case, "top1_same"),
        "invented_symptoms_total": sum(r.get("invented_symptoms", 0) for r in per_case if r.get("valid")),
        "vs_gemini_top5_overlap": mean([r.get("top5_overlap") for r in per_case
                                        if r.get("valid") and r["source"] == "gemini"]),
        "vs_gemini_symptom_recall": mean([r.get("symptom_recall") for r in per_case
                                          if r.get("valid") and r["source"] == "gemini"]),
        "narration_seconds_median": round(statistics.median([n["seconds"] for n in narr]), 1) if narr else None,
        "rationales_kept_share": mean([n["rationales_kept"] / n["remedies"] for n in narr if n["remedies"]]),
        "followups": fu, "followups_seconds": round(fu_s, 1),
    }
    args.out.write_text(json.dumps({"summary": summary, "cases": per_case, "narration": narr}, indent=2))
    llm.stop()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
