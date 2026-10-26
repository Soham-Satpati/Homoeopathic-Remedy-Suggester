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
from .ranking import rank_remedies
from .report import assemble_report, narration_inputs, no_match_report
from .safety import detect_red_flags

log = logging.getLogger(__name__)

_UNSET = object()


class HorusEngine:
    def __init__(self, paths: Paths, embedder: Embedder | None = None,
                 llm: LLMBackend | None = None, transcriber: Transcriber | None = None,
                 use_disk_cache: bool = True, config: DecisionConfig | None = None):
        self.paths = paths
        self.embedder = embedder
        self.llm: LLMBackend = llm or NullLLM()
        self.transcriber: Transcriber = transcriber or NullTranscriber()
        self.rubric = load_miasm_rubric(paths.miasm_rubric_path())
        self._emb_cache = (DatasetEmbeddingCache(embedder, paths.cache_dir if use_disk_cache else None,
                                                 seed_dir=paths.models_dir)
                           if embedder is not None else None)
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
            self.all_books: list[Book] = discover_books(self.paths)
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
            self._emb_cache.get(keys, progress=stage(0.15, 0.9) if need else None)
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
                self._bases[books] = load_base_dataset(self.paths, self._book_objs(books), progress=progress)
            return self._bases[books]

    def dataset_for(self, config: DecisionConfig | None = None, progress=None) -> Dataset:
        cfg = self._cfg(config)
        with self._learn_lock:
            key = (cfg.books, self._payload_version, cfg.learned_weight)
            if key not in self._lives:
                if len(self._lives) >= 6:
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
                             book_weight=self._book_weight_fn(ds, cfg))

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
                if len(self._cp_indexes) >= 3:
                    self._cp_indexes.pop(next(iter(self._cp_indexes)))
                cp = build_live_conditional_probabilities(base, self._payload, cfg.learned_weight)
                self._cp_indexes[key] = SymptomIndex(cp) if cp is not None else None
            return self._cp_indexes[key]

    def suggestions(self, categorised: dict, top_n: int = 10,
                    config: DecisionConfig | None = None) -> tuple[dict, list]:
        """(per-symptom suggestions, global top suggestions) for the case's symptoms."""
        syms = [s["symptom"] for k in SYMPTOM_GROUPS for s in categorised.get(k, [])]
        index = self._suggestion_index(config)
        return (get_suggestions_per_symptom(syms, index, top_n),
                get_top_suggestions(syms, index, top_n))

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
