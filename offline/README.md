# HoRUS Offline

The offline edition of HoRUS: a desktop app (Windows and macOS) that runs with no internet connection. This folder is being built in five steps:

| Step | What | Status |
|---|---|---|
| 1 | **Core engine**: the demo's decision logic as a plain Python package, proven identical | ✅ done |
| 2 | Encrypted local database, user accounts and roles, audit log, backup/restore, import of existing `patients.json` | next |
| 3 | Offline AI: local language model + Whisper, benchmarked against Gemini's saved output | |
| 4 | PySide6 desktop app (doctor and patient screens) and the clinic-LAN tablet intake server | |
| 5 | Windows `.exe` and macOS `.dmg` installers with models bundled | |

The demo (`horus3_1_streamlit.py`) and its data are left untouched; this package reads the same dataset files from the repository root.

## The engine (`horus/`)

```python
from horus import HorusEngine

engine = HorusEngine.from_defaults()       # loads dataset + local embedding model
engine.learn_from(patients)                # saved cases, same shape as patients.json

cat = engine.categorise(raw_text)          # free text → verified case (needs a local model, step 3)
cat = engine.finalise(parsed, raw_text)    # or verify a case parsed any other way
report = engine.report("PT-2026-001", cat) # dataset ranking + filtered explanations
per_symptom, top = engine.suggestions(cat) # co-occurrence suggestions
pdf_bytes = engine.pdf("PT-2026-001", cat, report)
```

| Module | Ported from the demo |
|---|---|
| `dataset.py` | `load_all_dataset_symptoms`, `_learning_payload`, `build_live_dataset` |
| `probabilities.py` | conditional probabilities, live update, symptom suggestions |
| `parsing.py` | `finalise_categorised`, `refresh_derived`, support checks |
| `safety.py` | red-flag rules |
| `miasm.py` | miasm rubric |
| `ranking.py` | query items, matching, `rank_remedies_from_dataset` |
| `report.py` | remedy-name / potency / evidence filters, `assemble_report` |
| `prompts.py` | every prompt sent to Gemini, word for word |
| `pdf.py` | `build_pdf` |
| `intake.py` | patient questionnaire |
| `llm.py`, `embeddings.py`, `engine.py`, `config.py` | new: model interfaces, local-only loading, the engine object, file locations |

### What changed, and why

Decisions are unchanged: ranking, scores, miasm, red flags, filters and the PDF are byte-for-byte what the demo produces (see Tests). The changes are all outside the decision logic:

- **No network.** The embedding model loads only from `models/all-MiniLM-L6-v2`, with Hugging Face's offline switches forced on. If it's missing, matching falls back to the demo's lexical similarity.
- **No Streamlit globals.** The dataset, model and learned cases live on a `HorusEngine` object instead of module globals and `st.cache_*`.
- **AI is pluggable.** The engine talks to an `LLMBackend` interface. With no model installed, it degrades the way the demo did when Gemini failed: reports are built from dataset evidence alone, and the doctor uses structured entry (`finalise_structured`).
- **More robust JSON parsing.** The demo stripped code fences with `str.lstrip("```json")`, which strips a *set of characters* and can eat the start of a reply. Local models also wrap JSON in prose more often than Gemini. `parse_json_reply` handles both.
- **Faster.** Dataset embeddings are cached on disk instead of recomputed on every start. Suggestions index the co-occurrence table once instead of scanning it per symptom, with the same row order and therefore the same results.
- **Deterministic.** The demo added case-wide modalities in set order, which varies between runs. The engine sorts them.
- **No stray patient files.** The demo wrote each PDF to a temp file it never deleted. The engine builds PDFs in memory.

## Tests

```bash
cd offline
pip install -e ".[test]"
python -m pytest
```

`tests/legacy.py` loads the demo's own functions straight from `horus3_1_streamlit.py` (with stand-ins for Streamlit, Gemini, GitHub and the embedding model) and runs them side by side with the engine. The inputs are the 4 real saved cases plus 5 synthetic cases aimed at the tricky paths, run with and without learned cases and in both matching modes. Results must be identical, including the prompts and sampling settings sent to the model, and the PDF bytes.

`tests/test_offline.py` checks the new guarantees: no online libraries are imported, a full workflow runs with sockets disabled, and the engine degrades cleanly with no AI.

## Models

```bash
pip install -e ".[build]"
python scripts/fetch_models.py --dest ../models
```

This downloads the embedding model, writes a SHA-256 manifest, and reloads the model with networking disabled to prove the copy is complete. It runs at installer-build time (and in CI); end users never need it. When the model is present, `test_ranking_with_real_model_matches_demo` also checks parity with the real weights.

## Notes on the existing data

- The root-level `horus3_patients.json` contains only a blank line, so it holds no records. The sample records the tests use are in `data/horus3_patients.json` and `Horus_V2/patients.json`.
