# HoRUS Offline

The offline edition of HoRUS: a Windows desktop app that runs with no internet connection. Patient records are encrypted on the PC, and the AI runs locally. The demo (`horus3_1_streamlit.py`) and its data are untouched; this folder reads the same dataset files from the repository root.

| Step | What | Status |
|---|---|---|
| 1 | **Core engine** — the demo's decision logic as a plain Python package, proven identical | ✅ |
| 2 | **Encrypted store** — users and roles, audit log, sealed backups, demo import | ✅ |
| 3 | **Offline AI** — local language model (llama.cpp) + Whisper speech, chosen by benchmark | ✅ |
| 4 | **Desktop app** (PySide6) + Clinic LAN tablet intake server | ✅ |
| 5 | **Windows installer** — built, installed and tested on a clean Windows machine by CI | ✅ |

## Layout

```
offline/
  horus/            engine (no UI): dataset, parsing checks, red flags, miasm, ranking, report filters,
                    suggestions, PDF, prompts, encrypted store, crypto, local AI backends
  horus_app/        desktop app: setup, sign-in/lock, case flow, patients, kiosk, settings, LAN server
  HoRUS.py          app launcher (also: --selftest, --gui-smoke)
  scripts/          fetch_models.py — downloads every bundled model (build time only)
  bench/            AI model benchmark on HoRUS's real jobs
  packaging/        PyInstaller build, Inno Setup script, DLL check, user README, release notes
  tests/            parity with the demo, store security, LAN server, UI flows
```

## Editions

One codebase; the edition is chosen at first start and can be changed later in Settings.

| Edition | Who |
|---|---|
| **Solo** | One doctor, one account |
| **Clinic** | Several accounts on one PC: `admin` (everything), `doctor` (patients, cases, reports, backups), `staff` (register patients, take intakes; no access to cases or reports) |
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

The dataset ranking appears instantly. The AI's explanations fill in afterwards, about 1–2 minutes on a 4-core CPU.

## Security

- **Encryption at rest.** Patient details, cases and intakes are sealed with AES-256-GCM under a random clinic data key, and each record is bound to its row. The key is stored only in wrapped form: once per user (scrypt-hardened password) and once under a recovery key shown at setup.
- **Sign-in protection.** Accounts lock after 5 failed sign-ins. The screen auto-locks after inactivity. New users must change their temporary password.
- **Audit log.** It records ids and actions only, never patient details. It is hash-chained, with a keyed checkpoint, so edits, deletions and truncation are all detectable.
- **Backups** are fully sealed files. Restoring needs a user password or the recovery key.
- **Local AI.** The llama.cpp server binds to 127.0.0.1 on a random port, uses a per-launch API key, and is killed along with the app.
- **LAN server.**
  - HTTPS with a certificate generated on the PC
  - 6-digit pairing, with a per-device lockout after failed attempts
  - Write-only API, with size limits and strict headers

## Building the Windows installer

CI runs this on every push (`.github/workflows/windows-installer.yml`). Pushing a tag `offline-vX.Y.Z` also publishes the release.

```bash
pip install -e "offline[test]" PySide6-Essentials PySide6-Addons onnxruntime tokenizers faster-whisper \
            huggingface_hub qrcode pyinstaller pefile
python offline/scripts/fetch_models.py --dest models --llm qwen3.5-4b --embedding onnx
python -m pytest offline
python offline/packaging/build_app.py --models models          # → dist/HoRUS/HoRUS.exe
python offline/packaging/check_deps.py dist/HoRUS                # every DLL shipped or part of Windows
dist/HoRUS/HoRUS.exe --selftest out --require-ai                 # end-to-end check of the build
iscc offline/packaging/horus.iss /DSourceDir=…\dist\HoRUS /DOutputDir=…\installer
```

The pipeline then installs the result silently on the runner and runs the self-test and the window test on the installed copy. It uninstalls and checks that patient data survived. The installer is split into parts (`HoRUS-Setup.exe` plus `.bin` files) because the bundled models exceed GitHub's 2 GB per-file limit.

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

`bench/benchmark_llm.py` runs on Windows CPU machines in CI (`.github/workflows/ai-benchmark.yml`). It scores each candidate model on HoRUS's real jobs, using 14 cases:

- the 4 saved cases, compared against Gemini's own parse
- 10 written reference cases, with polarity, negation and causation traps

It measures:

- **parsing quality:** symptom, modality and cause precision and recall, plus symptoms invented
- **ranking agreement:** how often the remedy ranking from the model's parse matches the ranking from the reference parse
- **explanations:** what share survive the filters
- **speed**
