"""``HorusEngine`` — one object holding everything the demo kept in Streamlit globals.

The demo stored the dataset, the embedding model and live-learning state in module
globals and ``st.cache_*``. Here they live on an engine instance, so the desktop app,
the LAN intake server and the tests can each hold their own, and nothing depends on a
UI framework.
"""

from __future__ import annotations

import copy
import logging
import threading
from datetime import datetime
from typing import Mapping

from . import prompts
from .books import Book, DecisionConfig, discover_books
from .config import Paths
from .dataset import (Dataset, build_live_dataset, check_symptoms_against_dataset,
                      learning_payload, load_base_dataset)
from .embeddings import DatasetEmbeddingCache, Embedder, load_embedder
from .llm import LLMBackend, LLMUnavailable, NullLLM, NullTranscriber, Transcriber, parse_json_reply
from .miasm import load_miasm_rubric
from .parsing import SYMPTOM_GROUPS, empty_case, finalise_categorised, refresh_derived
from .pdf import build_pdf
from .probabilities import (SymptomIndex, build_live_conditional_probabilities,
                            get_suggestions_per_symptom, get_top_suggestions, load_book_cooccurrence)
from .ranking import nearest_symptoms, rank_remedies
from .report import assemble_report, narration_inputs, no_match_report
from .safety import detect_red_flags

log = logging.getLogger(__name__)

_UNSET = object()


class HorusEngine:
    def __init__(self, paths: Paths, embedder: Embedder | None = None,
                 llm: LLMBackend | None = None, transcriber: Transcriber | None = None,
                 use_disk_cache: bool = True, config: DecisionConfig | None = None,
                 demo_parity: bool = False):
        """``demo_parity=True`` reproduces the online demo exactly (used by the parity tests):
        remedy names that differ only in capitalisation stay separate, time-of-day modalities may
        match any time, and suggestions need the exact dataset wording."""
        self.paths = paths
        self.demo_parity = demo_parity
        self.embedder = embedder
        self.llm: LLMBackend = llm or NullLLM()
        self.transcriber: Transcriber = transcriber or NullTranscriber()
        self.rubric = load_miasm_rubric(paths.miasm_rubric_path())
        self._emb_cache = (DatasetEmbeddingCache(embedder, paths.cache_dir if use_disk_cache else None,
                                                 seed_dir=paths.models_dir)
                           if embedder is not None else None)
        if self._emb_cache is not None:
            self._emb_cache.low_memory = not demo_parity        # no per-lookup copies of the store
        self.hidden_books: set = set()
        self._learn_lock = threading.RLock()
        self._payload: tuple = ()
        self._payload_version = 0
        self._bases: dict = {}          # books → Dataset
        self._lives: dict = {}          # (books, payload version, learned weight) → Dataset
        self._cp_bases: dict = {}       # books → combined co-occurrence table
        self._cp_indexes: dict = {}     # (books, payload version, learned weight) → SymptomIndex
        self.reload_books()
        self.config = (config or DecisionConfig()).resolve(self.books)

    # ── books & decision settings ────────────────────────────────────────
    def reload_books(self, hidden: set | None = None) -> None:
        """Re-scan shipped and clinic-added books (after adding, deleting or hiding one).
        ``hidden``: shipped books this clinic has removed from use."""
        with self._learn_lock:
            if hidden is not None:
                self.hidden_books = set(hidden)
            found = discover_books(self.paths, include_superseded=True)
            # clinic-added copies of a book HoRUS now ships cleaned; kept on disk until deleted
            self.superseded_books: list[Book] = [b for b in found if b.superseded_by]
            self.all_books: list[Book] = [b for b in found if not b.superseded_by]
            self.books: list[Book] = [b for b in self.all_books if b.id not in self.hidden_books]
            self._bases.clear()
            self._lives.clear()
            self._cp_bases.clear()
            self._cp_indexes.clear()
            if hasattr(self, "config"):
                self.config = self.config.resolve(self.books)

    def set_config(self, config: DecisionConfig) -> None:
        self.config = config.resolve(self.books)

    def _cfg(self, config: DecisionConfig | None) -> DecisionConfig:
        return self.config if config is None else config.resolve(self.books)

    def _book_objs(self, ids: tuple) -> list[Book]:
        by_id = {b.id: b for b in self.books}
        return [by_id[i] for i in ids if i in by_id]

    def warm_up(self, config: DecisionConfig | None = None, progress=None) -> None:
        """Load the selected books, index their symptoms and build the suggestion clusters now, so the
        first analysis is fast. ``progress(fraction, message)`` reports how far it has got."""
        def stage(lo, hi):
            return (lambda f, m: progress(lo + (hi - lo) * f, m)) if progress else None
        need = 0
        ds = self.dataset_for(config, progress=stage(0.0, 0.15))
        if self._emb_cache is not None:
            keys = tuple(sorted(ds.s2r))
            need = self._emb_cache.missing(keys)
            prog = stage(0.15, 0.9) if need else None
            if self._emb_cache.low_memory:
                self._emb_cache.subset(keys, progress=prog)
            else:
                self._emb_cache.get(keys, progress=prog)
        if progress:
            progress(0.9, "Building symptom clusters for suggestions…")
        self._suggestion_index(config)
        if progress:
            progress(1.0, "Ready")

    def describe(self, config: DecisionConfig | None = None) -> str:
        return self._cfg(config).describe(self.books)

    def _base_for(self, books: tuple, progress=None) -> Dataset:
        with self._learn_lock:
            if books not in self._bases:
                if len(self._bases) >= 4:
                    self._bases.pop(next(iter(self._bases)))
                self._bases[books] = load_base_dataset(self.paths, self._book_objs(books), progress=progress,
                                                       merge_remedy_names=not self.demo_parity)
            return self._bases[books]

    def dataset_for(self, config: DecisionConfig | None = None, progress=None) -> Dataset:
        cfg = self._cfg(config)
        with self._learn_lock:
            key = (cfg.books, self._payload_version, cfg.learned_weight)
            if key not in self._lives:
                if len(self._lives) >= 3:
                    self._lives.pop(next(iter(self._lives)))
                self._lives[key] = build_live_dataset(self._base_for(cfg.books, progress), self._payload,
                                                      cfg.learned_weight)
            return self._lives[key]

    @staticmethod
    def _book_weight_fn(ds: Dataset, cfg: DecisionConfig):
        """None when every book counts 1.0 (the demo's behaviour)."""
        weights = {b: cfg.weight_of(b) for b in cfg.books}
        if all(w == 1.0 for w in weights.values()) or not ds.origin:
            return None
        origin = ds.origin

        def bw(sym: str, rem: str) -> float:
            src = origin.get(sym, {}).get(rem)
            return max(weights.get(b, 1.0) for b in src) if src else 1.0   # learned-only pairs: 1.0
        return bw

    @classmethod
    def from_defaults(cls, paths: Paths | None = None, llm: LLMBackend | None = None,
                      transcriber: Transcriber | None = None, load_embedding: bool = True) -> "HorusEngine":
        paths = paths or Paths()
        embedder = load_embedder(paths.embedding_model_dir) if load_embedding else None
        return cls(paths, embedder=embedder, llm=llm, transcriber=transcriber)

    # ── dataset & live learning ──────────────────────────────────────────
    @property
    def dataset(self) -> Dataset:
        return self.dataset_for(None)

    @property
    def matching_mode(self) -> str:
        return "semantic" if self.embedder is not None else "lexical"

    @property
    def ai_available(self) -> bool:
        return not isinstance(self.llm, NullLLM)

    def learn_from(self, patients: Mapping) -> None:
        """Merge every saved case into the remedy index and co-occurrence table.

        Call after loading patient records and after each saved case — the demo did this
        implicitly by re-reading GitHub on every render.
        """
        payload = learning_payload(patients)
        with self._learn_lock:
            if payload == self._payload:
                return
            self._payload = payload
            self._payload_version += 1
            self._lives.clear()
            self._cp_indexes.clear()

    def check_symptoms(self, symptoms: list, config: DecisionConfig | None = None) -> tuple:
        return check_symptoms_against_dataset(self.dataset_for(config), symptoms)

    # ── case parsing ─────────────────────────────────────────────────────
    def categorise(self, raw_text: str, mode: str = "case_studies") -> dict:
        """Free text → verified structured case. Needs a language model."""
        t, n = prompts.GENERATION["categorise"]
        reply = self.llm.generate(prompts.categorise_prompt(raw_text),
                                  system=prompts.categorise_system(mode),
                                  temperature=t, max_tokens=n, json_output=True,
                                  schema=prompts.SCHEMAS["categorise"])
        cat = parse_json_reply(reply)
        if not isinstance(cat, dict):
            raise ValueError("Parser did not return a JSON object.")
        return self.finalise(cat, raw_text)

    def finalise(self, cat: dict, raw_text: str) -> dict:
        """Verify a structured case (from a model or structured entry) against the raw text."""
        return finalise_categorised(cat, raw_text, self.rubric)

    def finalise_structured(self, cat: dict) -> dict:
        """Structured entry with no free text: the doctor's own entries are the evidence,
        so verification is run against the text of those entries."""
        own_words = _structured_text(cat)
        return finalise_categorised(cat, own_words, self.rubric)

    @staticmethod
    def empty_case() -> dict:
        return empty_case()

    def refresh(self, cat: dict) -> dict:
        return refresh_derived(cat, self.rubric)

    def red_flags(self, raw_text: str, cat: dict | None = None) -> list:
        return detect_red_flags(raw_text, cat)

    # ── ranking & report ─────────────────────────────────────────────────
    def rank(self, cat: dict, top_n: int = 10, config: DecisionConfig | None = None) -> list:
        cfg = self._cfg(config)
        ds = self.dataset_for(cfg)
        return rank_remedies(cat, ds.s2r, self.embedder, self._emb_cache, top_n=top_n,
                             category_weights=cfg.categories, modality_weight=cfg.modality_weight,
                             book_weight=self._book_weight_fn(ds, cfg),
                             strict_times=not self.demo_parity)

    def report(self, patient_id: str, categorised: dict, patient_details: dict | None = None,
               mode: str = "case_studies", extra_label: str = "", narrate: bool = True,
               config: DecisionConfig | None = None) -> dict:
        """Rank from the dataset, then (if a model is available and ``narrate``) add filtered
        explanations. ``narrate=False`` returns the instant, dataset-only report.
        With an explicit ``config`` the report records which books and weights were used."""
        out = self._report(patient_id, categorised, mode, extra_label, narrate, config)
        if config is not None:
            cfg = self._cfg(config)
            out["_sources"] = cfg.describe(self.books)
            out["_decision"] = cfg.to_dict()
        return out

    def _report(self, patient_id, categorised, mode, extra_label, narrate, config) -> dict:
        categorised = self.refresh(categorised)
        ranked = self.rank(categorised, top_n=10, config=config)
        universe = self.dataset_for(config).remedy_universe
        if not ranked:
            return no_match_report(categorised, self.rubric)
        g: dict = {}
        if not narrate:
            return assemble_report(categorised, ranked, g, universe, self.rubric)
        try:
            lines, remedy_block = narration_inputs(categorised, ranked)
            t, n = prompts.GENERATION["narrate"]
            reply = self.llm.generate(prompts.narrate_prompt(patient_id, extra_label, lines, remedy_block),
                                      system=prompts.NARRATE_SYSTEM, temperature=t, max_tokens=n,
                                      json_output=True,
                                  schema=prompts.SCHEMAS["narrate"])
            parsed = parse_json_reply(reply)
            g = parsed if isinstance(parsed, dict) else {}
        except LLMUnavailable:
            pass
        except Exception:
            # Explanation is optional — the dataset decision stands without it.
            log.exception("Narration failed; report built from dataset evidence only")
        return assemble_report(categorised, ranked, g, universe, self.rubric)

    def comparative_summary(self, report_base: dict, report_enhanced: dict) -> str:
        """Short narrative comparing two plans; empty string when no model is installed."""
        t, n = prompts.GENERATION["comparative"]
        try:
            return self.llm.generate(prompts.comparative_prompt(report_base, report_enhanced),
                                     system=prompts.COMPARATIVE_SYSTEM, temperature=t,
                                     max_tokens=n).strip()
        except LLMUnavailable:
            return ""

    # ── symptom suggestions (co-occurrence) ──────────────────────────────
    @property
    def learned_cases(self) -> int:
        return len(self._payload)

    def cooccurrence_base(self, config: DecisionConfig | None = None):
        """Combined co-occurrence table of the selected books (before learning)."""
        books = self._cfg(config).books
        with self._learn_lock:
            if books not in self._cp_bases:
                self._cp_bases[books] = load_book_cooccurrence(self._book_objs(books))
            return self._cp_bases[books]

    def _suggestion_index(self, config: DecisionConfig | None = None) -> SymptomIndex | None:
        cfg = self._cfg(config)
        base = self.cooccurrence_base(cfg)
        with self._learn_lock:
            key = (cfg.books, self._payload_version, cfg.learned_weight)
            if key not in self._cp_indexes:
                if len(self._cp_indexes) >= 2:
                    self._cp_indexes.pop(next(iter(self._cp_indexes)))
                cp = build_live_conditional_probabilities(base, self._payload, cfg.learned_weight)
                self._cp_indexes[key] = SymptomIndex(cp) if cp is not None else None
            return self._cp_indexes[key]

    def suggestions(self, categorised: dict, top_n: int = 10,
                    config: DecisionConfig | None = None) -> tuple[dict, list]:
        """(per-symptom suggestions, global top suggestions) for the case's symptoms."""
        syms = [s["symptom"] for k in SYMPTOM_GROUPS for s in categorised.get(k, [])]
        index = self._suggestion_index(config)
        if self.demo_parity or index is None:
            return (get_suggestions_per_symptom(syms, index, top_n),
                    get_top_suggestions(syms, index, top_n))
        resolve = self._suggestion_resolver(index, syms, config)
        leaders = self._leaders(categorised, config)
        s2r = self.dataset_for(config).s2r
        per_pool = get_suggestions_per_symptom(syms, index, max(40, top_n * 4), resolve)
        per = {ps: [x for _, x in self._leader_order(items, "prob", leaders, s2r)][:top_n]
               for ps, items in per_pool.items()}
        pool = get_top_suggestions(syms, index, max(60, top_n * 6), resolve)
        return per, self._rank_suggestions(pool, top_n, leaders, s2r)

    def _leaders(self, categorised: dict, config) -> list:
        try:
            return [r["name"] for r in self.rank(categorised, top_n=5, config=config)]
        except Exception:
            log.exception("Ranking for suggestions failed")
            return []

    @staticmethod
    def _leader_order(items: list, prob_key: str, leaders: list, s2r: dict) -> list:
        """[(score, item + "remedies")], best first: a symptom listed under the case's leading
        remedies helps confirm or tell them apart, so it is worth asking about first."""
        out = []
        for x in items:
            rems = s2r.get(x["symptom"].strip().lower(), {})
            hits = [r for r in leaders if r in rems]
            bonus = sum(1.0 / (1 + leaders.index(r)) for r in hits)
            score = x[prob_key] * x.get("support", 1) ** 0.5 * (1.0 + bonus)
            out.append((score, {**x, "remedies": hits}))
        out.sort(key=lambda t: (-t[0], -t[1].get("support", 1), len(t[1]["symptom"]), t[1]["symptom"]))
        return out

    def _rank_suggestions(self, pool: list, top_n: int, leaders: list, s2r: dict) -> list:
        """The case-wide list: leader order, taking turns between the patient's symptoms."""
        if not pool:
            return []
        out = self._leader_order(pool, "avg_prob", leaders, s2r)
        # take turns between the patient's symptoms, so one symptom with many links can't fill the list
        groups: dict = {}
        for _, x in out:
            groups.setdefault(x["triggered_by"], []).append(x)
        picked = []
        while len(picked) < top_n and any(groups.values()):
            for g in list(groups):
                if groups[g] and len(picked) < top_n:
                    picked.append(groups[g].pop(0))
        return picked

    def _suggestion_resolver(self, index: SymptomIndex, syms: list, config: DecisionConfig | None = None):
        """Production suggestions:

        * each patient symptom is mapped to the closest symptoms the co-occurrence tables contain
          ("headache in the forehead for ten days" → "pain in forehead"), so suggestions no longer
          depend on the doctor using the book's exact wording;
        * links are weighted by how many remedies share them (``probabilities.SHARED_PRIOR``).
        """
        s2r = self.dataset_for(config).s2r
        lows = [s.strip().lower() for s in syms if s.strip()]
        # every symptom also borrows the links of its close neighbours ("heaviness of head" ←
        # "heaviness in head", "head heavy"): a single book entry rarely has links worth showing
        found = nearest_symptoms(list(dict.fromkeys(lows)), index.keys, self.embedder, self._emb_cache)
        table = {s: ([(s, 1.0)] if index.has(s) else []) + [(k, sim) for k, sim in found.get(s, []) if k != s]
                 for s in lows}
        table = {s: [(k, sim, len(s2r.get(k, ())) or 1) for k, sim in v] for s, v in table.items()}
        return lambda ps: table.get(ps, [])

    @property
    def has_cooccurrence_data(self) -> bool:
        return self._suggestion_index() is not None

    @staticmethod
    def enhanced_case(categorised: dict, selected: Mapping) -> tuple[dict, list]:
        """Add doctor-selected suggestions to a copy of the case (as the demo's Tab 3 did).

        ``selected`` maps symptom → {worse, better, avg_prob, triggered_by}.
        Returns (enhanced case, list of added entries for the PDF).
        """
        enhanced = copy.deepcopy(categorised)
        added = []
        for sym, mods in selected.items():
            entry = {
                "symptom": sym,
                "worse": mods.get("worse", []),
                "better": mods.get("better", []),
                "avg_prob": mods.get("avg_prob", 0.0),
                "triggered_by": mods.get("triggered_by", ""),
            }
            enhanced["physical"].append({"symptom": sym, "worse": entry["worse"], "better": entry["better"]})
            added.append(entry)
        return enhanced, added

    # ── patient intake helpers ───────────────────────────────────────────
    def followups(self, answers_text: str) -> list:
        """Up to 3 follow-up questions; [] when no model is installed or the reply is unusable."""
        t, n = prompts.GENERATION["followups"]
        try:
            reply = self.llm.generate(answers_text, system=prompts.FOLLOWUPS_SYSTEM,
                                      temperature=t, max_tokens=n, json_output=True,
                                  schema=prompts.SCHEMAS["followups"])
            items = parse_json_reply(reply)
        except LLMUnavailable:
            return []
        except Exception:
            log.exception("Follow-up generation failed")
            return []
        return [str(q).strip() for q in items if str(q).strip()][:3] if isinstance(items, list) else []

    def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        return self.transcriber.transcribe(audio, mime).strip()

    # ── output ───────────────────────────────────────────────────────────
    @staticmethod
    def pdf(patient_id: str, categorised: dict, report: dict, patient_details: dict | None = None,
            report_enhanced: dict | None = None, added_symptoms: list | None = None,
            comparative_summary: str | None = None, generated_on: datetime | None = None,
            sources_note: str | None = None) -> bytes:
        return build_pdf(patient_id, categorised, report, patient_details=patient_details,
                         report_enhanced=report_enhanced, added_symptoms=added_symptoms,
                         comparative_summary=comparative_summary, generated_on=generated_on,
                         sources_note=sources_note or report.get("_sources"))


def _structured_text(cat: dict) -> str:
    parts = []
    for k in SYMPTOM_GROUPS:
        for s in cat.get(k, []) or []:
            if isinstance(s, dict):
                parts += [str(s.get("symptom", ""))] + list(s.get("worse") or []) + list(s.get("better") or [])
    cm = cat.get("case_modalities") or {}
    parts += list(cm.get("worse") or []) + list(cm.get("better") or [])
    parts += list(cat.get("causation") or []) + list(cat.get("concomitants") or [])
    return ". ".join(p for p in parts if p)
