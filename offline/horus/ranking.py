"""Dataset-only remedy ranking (repertorisation). Ported from the demo's section 4.

For every patient item (symptom / modality / cause) find matching dataset symptoms, then
credit each remedy listed under them:
    credit = item_weight × similarity × (remedy score ÷ best score in that rubric) × rarity
with rarity = 1 / (1 + ln(number of remedies in that rubric)). A remedy is credited at most
once per patient feature; remedies are ranked by credit × coverage^0.5.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from .embeddings import DatasetEmbeddingCache, Embedder
from .parsing import content_stems

CATEGORY_WEIGHTS = {"causation": 0.0,   # 0 = causes are recorded but do not affect ranking
                    "psychological": 2.0, "general": 1.5, "physical": 1.0}
MODALITY_WEIGHT = 0.6        # modality items count a bit less than the symptom itself
MATCH_TOP_K = 8              # dataset entries considered per patient item
MATCH_MIN_SIM = 0.55         # semantic similarity threshold (0-1)
CAUSE_MIN_SIM = 0.65         # causations must match more closely ("grief" ≠ "fright")
COVERAGE_EXPONENT = 0.5      # score = credit × coverage^0.5 (rewards covering more of the case)
MIN_SCORE_FRACTION = 0.30    # hide remedies scoring under 30% of the top remedy…
MIN_REMEDIES_SHOWN = 3       # …but always show at least this many
LEXICAL_MIN_SIM = 0.34       # fallback threshold when the embedding model is unavailable


def build_query_items(categorised: dict, category_weights: dict | None = None,
                      modality_weight: float | None = None) -> list:
    """Patient items to match: (label, query_text, weight, group_key).

    Items sharing a group_key are alternative phrasings of ONE patient feature — a remedy is
    credited at most once per group. Each modality is its own group, so a modality attached
    to many symptoms still counts only once.
    """
    cw = CATEGORY_WEIGHTS if category_weights is None else {**CATEGORY_WEIGHTS, **category_weights}
    mw = MODALITY_WEIGHT if modality_weight is None else modality_weight
    q = []
    for c in (categorised.get("causation", []) if cw["causation"] > 0 else []):
        core = re.sub(r"^(ailments? from|after|since)\s+", "", c.strip(), flags=re.I)
        g = f"cause:{c.lower()}"
        w = cw["causation"]
        for text in dict.fromkeys([f"ailments from {core}", core, f"{core}, ailments from"]):
            q.append((f"Ailments from {core}", text, w, g))

    mod_weight = {}
    for k in ("psychological", "general", "physical"):
        w = cw[k]
        for s in categorised.get(k, []):
            if s.get("unverified"):
                continue
            sym = s["symptom"].strip()
            q.append((sym, sym, w, f"{k}:{sym.lower()}"))
            for pol in ("worse", "better"):
                for m in s.get(pol, []):
                    g = f"mod:{pol} {m.strip().lower()}"
                    mod_weight[g] = max(mod_weight.get(g, 0), w * mw)
                    q.append((f"{pol.title()} {m}", f"{pol} {m}", None, g))
    # whole-case modalities are generals in the classical hierarchy
    for pol in ("worse", "better"):
        for m in (categorised.get("case_modalities") or {}).get(pol, []):
            g = f"mod:{pol} {m.strip().lower()}"
            mod_weight[g] = max(mod_weight.get(g, 0), cw["general"] * mw)
            q.append((f"{pol.title()} {m} (whole case)", f"{pol} {m}", None, g))
    return [(lab, txt, (wt if wt is not None else mod_weight[g]), g) for lab, txt, wt, g in q]


_LEFT, _RIGHT = re.compile(r"\bleft\b"), re.compile(r"\bright\b")

# A patient modality may only match a dataset entry that is itself a modality.
_MOD_WORDS = re.compile(r"\b(worse|better|agg\w*|amel\w*|reliev\w*|improv\w*|after|before|during|while"
                        r"|on (rising|waking|lying|walking|motion)|from (touch|motion|heat|cold|warm\w*|eating"
                        r"|pressure|rest|lying|walking))\b")
_WORSE_W = re.compile(r"\b(worse|agg\w*|intoleran\w*|cannot bear|can't bear)\b")
_BETTER_W = re.compile(r"\b(better|amel\w*|reliev\w*|improv\w*)\b")


def modality_mismatch(group: str, dataset_text: str) -> bool:
    """Reject modality→symptom matches ('better walking' ≈ 'unsteady walk') and opposite polarity."""
    if not group.startswith("mod:"):
        return False
    d = dataset_text.lower()
    if not _MOD_WORDS.search(d):
        return True
    pol = group[4:].split(" ", 1)[0]
    if pol == "better" and _WORSE_W.search(d) and not _BETTER_W.search(d):
        return True
    if pol == "worse" and _BETTER_W.search(d) and not _WORSE_W.search(d):
        return True
    return False


def laterality_conflict(patient_text: str, dataset_text: str) -> bool:
    p, d = patient_text.lower(), dataset_text.lower()
    return bool((_LEFT.search(p) and _RIGHT.search(d) and not _LEFT.search(d)) or
                (_RIGHT.search(p) and _LEFT.search(d) and not _RIGHT.search(d)))


def lexical_sim(a: str, b: str) -> float:
    sa, sb = content_stems(a), content_stems(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def match_items_to_dataset(queries: list, s2r: dict, embedder: Embedder | None,
                           emb_cache: DatasetEmbeddingCache | None) -> list:
    """For each query → [(dataset_symptom, similarity)], using embeddings (or lexical fallback)."""
    keys = tuple(sorted(s2r.keys()))
    if not keys:
        return [[] for _ in queries]
    emb = emb_cache.get(keys) if (embedder is not None and emb_cache is not None) else None

    out = []
    if emb is not None:
        qv = np.asarray(embedder.encode([q[1] for q in queries]), dtype=np.float32)
        sims = qv @ emb.T
        for q, row in zip(queries, sims):
            th = CAUSE_MIN_SIM if q[3].startswith("cause:") else MATCH_MIN_SIM
            idx = np.argsort(-row)[:MATCH_TOP_K]
            out.append([(keys[i], float(row[i])) for i in idx if row[i] >= th])
    else:
        for q in queries:
            th = LEXICAL_MIN_SIM * (1.4 if q[3].startswith("cause:") else 1.0)
            scored = sorted(((k, lexical_sim(q[1], k)) for k in keys), key=lambda x: -x[1])[:MATCH_TOP_K]
            out.append([(k, s) for k, s in scored if s >= th])
    return out


def rank_remedies(categorised: dict, s2r: dict, embedder: Embedder | None,
                  emb_cache: DatasetEmbeddingCache | None, top_n: int = 10, *,
                  category_weights: dict | None = None, modality_weight: float | None = None,
                  book_weight=None) -> list:
    """``book_weight(dataset_symptom, remedy) -> float`` multiplies each match's credit by the weight
    of the book it came from (None → every book counts 1.0, as in the demo). Items whose category
    weight is 0 are left out entirely, as the demo does for causation."""
    queries = [q for q in build_query_items(categorised, category_weights, modality_weight) if q[2] > 0]
    if not queries:
        return []
    matches = match_items_to_dataset(queries, s2r, embedder, emb_cache)

    best = defaultdict(dict)     # remedy -> group -> (credit, evidence dict)
    for (label, qtext, weight, group), hits in zip(queries, matches):
        for ds_sym, sim in hits:
            rems = s2r.get(ds_sym, {})
            if not rems or laterality_conflict(qtext, ds_sym) or modality_mismatch(group, ds_sym):
                continue
            top = max(rems.values()) or 1.0
            rarity = 1.0 / (1.0 + np.log(len(rems)))   # generic rubrics ("restlessness") count less
            for rem, sc in rems.items():
                credit = weight * sim * (sc / top) * rarity
                if book_weight is not None:
                    bw = book_weight(ds_sym, rem)
                    if bw <= 0:
                        continue
                    credit *= bw
                prev = best[rem].get(group)
                if prev is None or credit > prev[0]:
                    best[rem][group] = (credit, {"patient": label, "dataset": ds_sym,
                                                 "similarity": round(sim, 3)})

    n_groups = len({q[3] for q in queries})
    ranked = []
    for rem, groups in best.items():
        credit = sum(c for c, _ in groups.values())
        # totality: breadth of coverage matters, so one heavy match can't beat many good ones
        score = credit * (len(groups) ** COVERAGE_EXPONENT)
        evidence = [e for _, e in sorted(groups.values(), key=lambda x: -x[0])]
        ranked.append({
            "name": rem,
            "score": round(score, 6),
            "coverage": len(groups),
            "total_items": n_groups,
            "symptoms_covered": [e["patient"] for e in evidence],
            "evidence": evidence,
        })
    ranked.sort(key=lambda r: (-r["score"], -r["coverage"], r["name"]))
    if ranked:
        cutoff = ranked[0]["score"] * MIN_SCORE_FRACTION
        ranked = [r for i, r in enumerate(ranked) if i < MIN_REMEDIES_SHOWN or r["score"] >= cutoff]
    return ranked[:top_n]
