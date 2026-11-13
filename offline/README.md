# HoRUS Offline macOS Edition

HoRUS Offline macOS Edition: the offline edition of HoRUS as a Mac app (Apple silicon) that runs with no internet connection. Patient records are encrypted on the Mac, and the AI runs locally, on the Mac's GPU. The demo (`horus3_1_streamlit.py`) and its data are untouched; this folder reads the same dataset files from the repository root.

This branch (`horus-offline-macos`) is the macOS Edition. The Windows Edition lives on `horus-offline`; both share the engine, the encrypted data format and backups.

| Step | What | Status |
|---|---|---|
| 1 | **Core engine** — the demo's decision logic as a plain Python package, proven identical | ✅ |
| 2 | **Encrypted store** — users and roles, audit log, sealed backups, demo import | ✅ |
| 3 | **Offline AI** — local language model (llama.cpp) + Whisper speech, chosen by benchmark | ✅ |
| 4 | **Desktop app** (PySide6) + Clinic LAN tablet intake server | ✅ |
| 5 | **macOS app and disk image** — built, signed, installed and tested on an Apple silicon Mac by CI | ✅ |

## Layout

```
offline/
  horus/            engine (no UI): dataset, parsing checks, red flags, miasm, ranking, report filters,
                    suggestions, PDF, prompts, encrypted store, crypto, local AI backends
  horus_app/        desktop app: setup, sign-in/lock, case flow, patients, kiosk, settings, LAN server
  HoRUS.py          app launcher (also: --selftest, --gui-smoke)
  scripts/          fetch_models.py — downloads every bundled model (build time only)
  bench/            AI model benchmark on HoRUS's real jobs
  packaging/        PyInstaller .app build, signing, disk image, linked-library check, user README, release notes
  tests/            parity with the demo, store security, LAN server, UI flows
```

## Editions

One codebase; the edition is chosen at first start and can be changed later in Settings.

| Edition | Who |
|---|---|
| **Solo** | One doctor, one account |
| **Clinic** | Several accounts on one Mac: `admin` (everything), `doctor` (patients, cases, reports, backups), `staff` (register patients, take intakes; no access to cases or reports) |
| **Clinic LAN** | As Clinic, plus tablets on the clinic Wi-Fi can submit the intake questionnaire |

## How decisions are made

Unchanged from the demo, and proven identical by `tests/test_parity.py`, which loads the demo's own functions from `horus3_1_streamlit.py`:

- Remedy ranking, scores, miasm, red flags and the PDF come from the dataset and fixed rules.
- The AI only:
  - reads free text into symptoms (every symptom is then checked against the patient's words)
  - writes explanations, which are filtered for remedies not in the ranking, potency or dose wording, and claims the dataset doesn't support
  - transcribes speech
  - suggests follow-up questions
- Without the AI, structured entry works, and reports use dataset evidence alone.

The dataset ranking appears instantly. The AI's explanations fill in afterwards: about 1–2 minutes on a 4-core CPU, and considerably faster on Apple silicon, where the model runs on the GPU (Metal). `HORUS_LLAMA_GPU_LAYERS=0` forces the CPU.

## Security

- **Encryption at rest.** Patient details, cases and intakes are sealed with AES-256-GCM under a random clinic data key, and each record is bound to its row. The key is stored only in wrapped form: once per user (scrypt-hardened password) and once under a recovery key shown at setup.
- **Sign-in protection.** Accounts lock after 5 failed sign-ins. The screen auto-locks after inactivity. New users must change their temporary password.
- **Audit log.** It records ids and actions only, never patient details. It is hash-chained, with a keyed checkpoint, so edits, deletions and truncation are all detectable.
- **Backups** are fully sealed files. Restoring needs a user password or the recovery key.
- **Local AI.** The llama.cpp server binds to 127.0.0.1 on a random port, uses a per-launch API key, and is stopped along with the app (a watchdog process ends it even if HoRUS is force-quit).
- **LAN server.**
  - HTTPS with a certificate generated on the Mac
  - 6-digit pairing, with a per-device lockout after failed attempts
  - Write-only API, with size limits and strict headers

## Building the Mac app

CI runs this on every push to `horus-offline-macos` (`.github/workflows/macos-app.yml`, on GitHub's Apple silicon `macos-15` runner). Pushing a tag `offline-macos-vX.Y.Z` also publishes the release.

```bash
pip install -e "offline[test]" PySide6-Essentials PySide6-Addons onnxruntime tokenizers faster-whisper \
            huggingface_hub qrcode pyinstaller
python offline/scripts/fetch_models.py --dest models --llm gemma4-e4b --embedding onnx   # macOS llama.cpp build (Metal)
python -m pytest offline
python offline/packaging/build_app.py --models models          # → dist/HoRUS.app (signed ad hoc)
python offline/packaging/check_deps.py dist/HoRUS.app            # every linked library is inside the app or part of macOS
dist/HoRUS.app/Contents/MacOS/HoRUS --selftest out --require-ai  # end-to-end check of the build
python offline/packaging/build_dmg.py --version 1.5.0            # → installer/HoRUS-1.5.0-macOS-arm64.dmg(.partNN)
```

- **Info.plist.** `build_app.py` adds the bundle id (`com.sohamsatpati.horus`), version, minimum macOS (13.0), and the microphone and local-network usage texts. Without the microphone text macOS ends the app when dictation starts.
- **Signing.** `sign_app.sh` signs every binary inside out (including the llama.cpp server, which ships as data), then the bundle. By default it signs ad hoc; Apple silicon refuses unsigned code. With `--sign "Developer ID Application: …"` it uses the hardened runtime and `entitlements.plist` (microphone, Python's executable memory, loading bundled libraries). CI does this and notarises when the `MACOS_CERT_*` and `APPLE_*` secrets are set, and staples the ticket to the app so Gatekeeper accepts it on Macs with no internet.
- **Disk image.** HFS+, LZFSE-compressed, with an Applications shortcut and the user README. The bundled models exceed GitHub's 2 GB per-file limit, so the image is split into `.dmg.part01`, `.part02`, … that users join with `cat` (the user README and release notes give the one-line command). `SHA256SUMS.txt` covers the joined image and each part.

The pipeline then joins the parts, mounts the image, copies HoRUS to an Applications folder, runs the self-test and the window test on that copy, removes it and checks that patient data survived.

Intel Macs: `fetch_models.py --platform mac-x64` fetches llama.cpp's Intel build, and `build_app.py` targets the architecture of the Mac it runs on. CI builds Apple silicon only.

## Tests

```bash
cd offline && python -m pytest
```

| Test file | What it covers |
|---|---|
| `test_parity.py` | The demo's functions against the engine, across 9 cases, both matching modes, and with or without learned cases. Covers prompts and byte-identical PDFs, plus the real model when it has been fetched. |
| `test_offline.py` | No online libraries, a workflow with sockets disabled, degrading cleanly without AI, and the JSON parser. |
| `test_store.py` | Encryption (no plaintext on disk), lockout, roles, the recovery key, audit tampering and truncation, backup and restore, demo import, and concurrent writers. |
| `test_lan.py` | Pairing, lockout, the write-only API, limits and headers. |
| `test_app.py` | Headless UI flows: setup, the kiosk, the inbox, a case from an intake, settings per role, backup, lock, and instant reports with later AI text. |

## AI model benchmark

`bench/benchmark_llm.py` runs in CI (`.github/workflows/ai-benchmark.yml`, now on Apple silicon `macos-15` runners). It scores each candidate model on HoRUS's real jobs, using 14 cases:

- the 4 saved cases, compared against Gemini's own parse
- 10 written reference cases, with polarity, negation and causation traps

It measures:

- **parsing quality:** symptom, modality and cause precision and recall, plus symptoms invented
- **ranking agreement:** how often the remedy ranking from the model's parse matches the ranking from the reference parse
- **explanations:** what share survive the filters
- **speed**

### Results and choice

The model was chosen from these runs on a 4-vCPU Windows machine (CPU only), comparable to a mid-range laptop. The same model ships on the Mac, where the GPU makes it faster:

| Model (all Apache-2.0) | File | RAM in use | Symptoms found | Invented | Category correct | Explanations kept | Parse | Explain |
|---|---|---|---|---|---|---|---|---|
| **Gemma 4 E4B (shipped)** | 4.8 GB | 3.1 GB | 99–100% | 0–1 | 68–86% | 100% | 38–67 s | ~110 s |
| Gemma 4 E2B | 3.1 GB | 1.6 GB | 98–100% | 1 | 60–74% | 73% | 35–60 s | ~100 s |
| Qwen3.5-4B | 2.6 GB | — | 92% | 4 | 82% | 65% | 66 s | 160 s |
| Qwen3-4B-2507 | 2.5 GB | — | 96% | 0 | 84% | 18% | 46 s | 150 s |
| Qwen3.5-2B | 1.2 GB | — | 87% | 27 | 64% | 50% | 23 s | 60 s |

Where a figure is a range, it spans two runs: all 14 cases, and a separate 4-case run that also measured memory.

Gemma 4 E4B was the most faithful model:
- it found almost every stated symptom and invented virtually none
- every explanation it wrote survived the engine's filters
- it uses about 3 GB of RAM while running

Low agreement on the top-1 remedy across all models reflects granularity: the models split symptoms more finely than the reference parses. It does not reflect invented content. The doctor reviews the symptoms before analysis in any case.

Switching models later only needs `--llm <key>` in `fetch_models.py`; no code changes.

## Books and decision bias (1.1)

Books are listed in `books.json` at the repository root (shipped with the app). Each entry names:
- the book JSON
- an optional pickled index
- the co-occurrence table used for suggestions
- optionally, chapters to leave out

Books a clinic adds in the app live in `~/Library/Application Support/HoRUS/books`.

| Book | Remedies | Symptoms | Co-occurrence table |
|---|---|---|---|
| Rheumatic remedies | 125 | 4,604 | shipped (original) |
| Case studies (combined) | 33 | — | shipped (original) |
| Hering, *Guiding Symptoms* vol. I | 45 | 18,635 (the "relationship" chapter is left out) | generated with `horus/clusters.py` |

`horus/clusters.py` builds a co-occurrence table with the method of the original tables:
1. Cluster the book's entries with HDBSCAN on remedy and modality features (as in `symptom_clustering_pro.py`).
2. Within a cluster, P(B|A) = shared remedies ÷ remedies listing A, rounded to 4 places.

On the rheumatic book this reproduces 96% of the original table's pairs, with identical probabilities on 98% of them. For large books, each symptom keeps its 15 strongest partners.

`DecisionConfig` (`horus/books.py`) holds which books are used and every weight:
- per book (multiplies the credit of matches from that book)
- saved clinic cases (scales live learning)
- mind / generals / particulars / modalities / causation

With the demo's two books and every weight at its classical value, ranking is identical to the online demo; `tests/test_parity.py` checks this. The clinic default is stored in the encrypted database: only administrators can change it, and each change is logged in the audit log.

Data note: the Guiding Symptoms JSON contains some non-symptom text from the printed book, such as publication history in "general characteristics". It is kept as supplied. Such lines rarely match a patient's words, but it is worth cleaning at source.
