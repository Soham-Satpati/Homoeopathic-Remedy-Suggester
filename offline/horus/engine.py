"""``HorusEngine`` — one object holding everything the demo kept in Streamlit globals.

The demo stored the dataset, the embedding model and live-learning state in module
globals and ``st.cache_*``. Here they live on an engine instance, so the desktop app,
the LAN intake server and the tests can each hold their own, and nothing depends on a
UI framework.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Mapping

from . import prompts
from .config import Paths
from .dataset import (Dataset, build_live_dataset, check_symptoms_against_dataset,
                      learning_payload, load_base_dataset)
from .embeddings import DatasetEmbeddingCache, Embedder, load_embedder
from .llm import LLMBackend, LLMUnavailable, NullLLM, NullTranscriber, Transcriber, parse_json_reply
from .miasm import load_miasm_rubric
from .parsing import SYMPTOM_GROUPS, empty_case, finalise_categorised, refresh_derived
from .pdf import build_pdf
from .probabilities import (SymptomIndex, build_live_conditional_probabilities,
                            get_suggestions_per_symptom, get_top_suggestions,
                            load_conditional_probabilities)
from .ranking import rank_remedies
from .report import assemble_report, narration_inputs, no_match_report
from .safety import detect_red_flags

log = logging.getLogger(__name__)

_UNSET = object()


class HorusEngine:
    def __init__(self, paths: Paths, embedder: Embedder | None = None,
                 llm: LLMBackend | None = None, transcriber: Transcriber | None = None,
                 use_disk_cache: bool = True):
        self.paths = paths
        self.embedder = embedder
        self.llm: LLMBackend = llm or NullLLM()
        self.transcriber: Transcriber = transcriber or NullTranscriber()
        self.rubric = load_miasm_rubric(paths.miasm_rubric_path())
        self._emb_cache = (DatasetEmbeddingCache(embedder, paths.cache_dir if use_disk_cache else None)
                           if embedder is not None else None)
        self._base: Dataset = load_base_dataset(paths)
        self._live: Dataset = self._base
        self._payload: tuple = ()
        self._cp_base = _UNSET
        self._cp_key = _UNSET
        self._cp_index: SymptomIndex | None = None

    @classmethod
    def from_defaults(cls, paths: Paths | None = None, llm: LLMBackend | None = None,
                      transcriber: Transcriber | None = None, load_embedding: bool = True) -> "HorusEngine":
        paths = paths or Paths()
        embedder = load_embedder(paths.embedding_model_dir) if load_embedding else None
        return cls(paths, embedder=embedder, llm=llm, transcriber=transcriber)

    # ── dataset & live learning ──────────────────────────────────────────
    @property
    def dataset(self) -> Dataset:
        return self._live

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
        if payload == self._payload:
            return
        self._payload = payload
        self._live = build_live_dataset(self._base, payload)

    def check_symptoms(self, symptoms: list) -> tuple:
        return check_symptoms_against_dataset(self._live, symptoms)

    # ── case parsing ─────────────────────────────────────────────────────
    def categorise(self, raw_text: str, mode: str = "case_studies") -> dict:
        """Free text → verified structured case. Needs a language model."""
        t, n = prompts.GENERATION["categorise"]
        reply = self.llm.generate(prompts.categorise_prompt(raw_text),
                                  system=prompts.categorise_system(mode),
                                  temperature=t, max_tokens=n, json_output=True)
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
    def rank(self, cat: dict, top_n: int = 10) -> list:
        return rank_remedies(cat, self._live.s2r, self.embedder, self._emb_cache, top_n=top_n)

    def report(self, patient_id: str, categorised: dict, patient_details: dict | None = None,
               mode: str = "case_studies", extra_label: str = "") -> dict:
        """Rank from the dataset, then (if a model is available) add filtered explanations."""
        categorised = self.refresh(categorised)
        ranked = self.rank(categorised, top_n=10)
        if not ranked:
            return no_match_report(categorised, self.rubric)
        g: dict = {}
        try:
            lines, remedy_block = narration_inputs(categorised, ranked)
            t, n = prompts.GENERATION["narrate"]
            reply = self.llm.generate(prompts.narrate_prompt(patient_id, extra_label, lines, remedy_block),
                                      system=prompts.NARRATE_SYSTEM, temperature=t, max_tokens=n,
                                      json_output=True)
            parsed = parse_json_reply(reply)
            g = parsed if isinstance(parsed, dict) else {}
        except LLMUnavailable:
            pass
        except Exception:
            # Explanation is optional — the dataset decision stands without it.
            log.exception("Narration failed; report built from dataset evidence only")
        return assemble_report(categorised, ranked, g, self._live.remedy_universe, self.rubric)

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
    def _suggestion_index(self) -> SymptomIndex | None:
        if self._cp_base is _UNSET:
            self._cp_base = load_conditional_probabilities(self.paths)
        if self._cp_key is _UNSET or self._cp_key != self._payload:
            cp = build_live_conditional_probabilities(self._cp_base, self._payload)
            self._cp_index = SymptomIndex(cp) if cp is not None else None
            self._cp_key = self._payload
        return self._cp_index

    def suggestions(self, categorised: dict, top_n: int = 10) -> tuple[dict, list]:
        """(per-symptom suggestions, global top suggestions) for the case's symptoms."""
        syms = [s["symptom"] for k in SYMPTOM_GROUPS for s in categorised.get(k, [])]
        index = self._suggestion_index()
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
                                      temperature=t, max_tokens=n, json_output=True)
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
            comparative_summary: str | None = None, generated_on: datetime | None = None) -> bytes:
        return build_pdf(patient_id, categorised, report, patient_details=patient_details,
                         report_enhanced=report_enhanced, added_symptoms=added_symptoms,
                         comparative_summary=comparative_summary, generated_on=generated_on)


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
