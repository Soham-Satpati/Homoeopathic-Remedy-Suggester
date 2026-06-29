"""
HoRUS 3 — Clinical Homeopathic Intelligence
Streamlit Cloud deployment with GitHub-backed patient storage.

Secrets required (set in Streamlit Cloud → App settings → Secrets):
    GEMINI_API_KEY   = "AIza..."
    GITHUB_TOKEN     = "github_pat_..."   # fine-grained PAT, contents: read+write
    GITHUB_REPO      = "owner/repo-name"
    GITHUB_FILE_PATH = "data/horus3_patients.json"   # path inside the repo

Requirements (requirements.txt):
    streamlit
    google-generativeai
    sentence-transformers
    reportlab
    PyGithub
    pandas
    numpy
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import json
import base64
import tempfile
from collections import defaultdict
from datetime import datetime

import google.generativeai as genai
from github import Github, GithubException
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HoRUS 3",
    page_icon="⚕",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2.5rem 2rem 4rem; max-width: 860px; }
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    h1 { font-size: 1.5rem !important; font-weight: 500 !important; letter-spacing: -0.02em; }
    h2 { font-size: 1.1rem !important; font-weight: 500 !important; margin-top: 2rem !important; }
    h3 { font-size: 0.95rem !important; font-weight: 500 !important; color: #555 !important; }
    .stButton > button {
        border: 1px solid #d0d0d0 !important; background: white !important;
        color: #111 !important; border-radius: 6px !important;
        font-size: 0.85rem !important; padding: 0.45rem 1.1rem !important;
        transition: background 0.15s, border-color 0.15s;
    }
    .stButton > button:hover { background: #f5f5f5 !important; border-color: #aaa !important; }
    .stButton > button[kind="primary"] {
        background: #111 !important; color: white !important; border-color: #111 !important;
    }
    .stButton > button[kind="primary"]:hover { background: #333 !important; }
    .stTextInput > div > div > input, .stTextArea textarea {
        border: 1px solid #d0d0d0 !important; border-radius: 6px !important;
        font-size: 0.9rem !important;
    }
    .stTextArea textarea { min-height: 180px !important; line-height: 1.7 !important; }
    .step-pill {
        display: inline-block; font-size: 0.7rem; font-weight: 500;
        letter-spacing: 0.08em; text-transform: uppercase; padding: 3px 10px;
        border-radius: 20px; border: 1px solid #d0d0d0; color: #555; margin-bottom: 0.75rem;
    }
    .step-pill.active { background: #111; color: white; border-color: #111; }
    .remedy-card {
        border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 1.1rem 1.3rem; margin-bottom: 0.75rem; background: white;
    }
    .remedy-primary { border-left: 3px solid #111; }
    .remedy-name { font-size: 1rem; font-weight: 500; margin-bottom: 0.3rem; }
    .remedy-role {
        display: inline-block; font-size: 0.7rem; padding: 2px 8px;
        border-radius: 20px; background: #f0f0f0; color: #444;
        margin-left: 8px; text-transform: capitalize;
    }
    .remedy-rationale { font-size: 0.88rem; color: #444; line-height: 1.65; margin-top: 0.5rem; }
    .remedy-meta { font-size: 0.8rem; color: #777; margin-top: 0.5rem; }
    .sym-tag {
        display: inline-block; font-size: 0.8rem; padding: 3px 10px;
        border-radius: 4px; margin: 3px; background: #f5f5f5;
        color: #333; border: 1px solid #e0e0e0;
    }
    .sym-worse { background: #fff5f5; border-color: #fcc; color: #900; }
    .sym-better { background: #f0fff4; border-color: #9e9; color: #2a5; }
    .section-label {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #888;
        margin-bottom: 0.6rem; margin-top: 1.5rem;
    }
    hr { border: none; border-top: 1px solid #eee; margin: 1.5rem 0; }
    .patient-badge {
        font-size: 0.75rem; color: #888; border: 1px solid #e8e8e8;
        border-radius: 4px; padding: 3px 10px;
        display: inline-block; margin-bottom: 1rem;
    }
    .info-box {
        background: #f9f9f9; border: 1px solid #eee; border-radius: 6px;
        padding: 0.9rem 1.1rem; font-size: 0.85rem; color: #555;
        line-height: 1.65; margin: 0.75rem 0;
    }
    .monitor-item {
        font-size: 0.88rem; color: #333; padding: 0.4rem 0;
        border-bottom: 1px solid #f0f0f0; line-height: 1.55;
    }
    .monitor-item:last-child { border-bottom: none; }
    .streamlit-expanderHeader { font-size: 0.88rem !important; }
    div[data-baseweb="select"] { font-size: 0.88rem !important; }
    .patient-card {
        border: 1px solid #e8e8e8; border-radius: 8px;
        padding: 1rem 1.2rem; margin-bottom: 0.6rem; background: #fafafa;
    }
    .patient-card:hover { border-color: #bbb; background: white; }
    .patient-name { font-size: 1rem; font-weight: 500; color: #111; }
    .patient-meta { font-size: 0.78rem; color: #888; margin-top: 2px; }
    .case-timeline {
        border-left: 2px solid #e0e0e0; padding-left: 1rem; margin-top: 0.5rem;
    }
    .case-entry {
        position: relative; padding: 0.5rem 0;
        border-bottom: 1px solid #f5f5f5; font-size: 0.85rem;
    }
    .case-entry:last-child { border-bottom: none; }
    .case-date { font-size: 0.75rem; color: #999; margin-bottom: 2px; }
    .mode-badge {
        display: inline-block; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 10px;
        border-radius: 4px; margin-bottom: 0.75rem;
    }
    .mode-case { background: #e8f4fd; color: #1565c0; border: 1px solid #bbdefb; }
    .mode-materia { background: #f3e5f5; color: #6a1b9a; border: 1px solid #e1bee7; }
    .detail-row {
        display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.75rem;
    }
    .detail-chip {
        font-size: 0.8rem; color: #444; background: #f5f5f5;
        border: 1px solid #e5e5e5; border-radius: 4px; padding: 3px 10px;
    }
    .save-success {
        font-size: 0.8rem; color: #2a7a2a; background: #f0fff4;
        border: 1px solid #9e9; border-radius: 4px; padding: 4px 10px;
        display: inline-block;
    }
    /* Future symptoms tab */
    .sug-card {
        border: 1px solid #e0e0e0; border-radius: 8px;
        padding: 0.85rem 1.1rem; margin-bottom: 0.6rem; background: white;
        display: flex; align-items: flex-start; gap: 0.75rem;
    }
    .sug-card.selected { border-color: #111; border-left: 3px solid #111; }
    .prob-bar-bg {
        background: #f0f0f0; border-radius: 4px; height: 6px;
        width: 100%; margin-top: 5px;
    }
    .prob-bar-fill {
        background: #111; border-radius: 4px; height: 6px;
    }
    .prob-label {
        font-size: 0.72rem; color: #888; margin-top: 2px;
    }
    .diff-added { color: #2a7a2a; font-weight: 500; }
    .diff-neutral { color: #555; }
    .compare-col {
        background: #fafafa; border: 1px solid #eee; border-radius: 8px;
        padding: 1rem 1.1rem; height: 100%;
    }
    .compare-col.enhanced { background: #f0fff4; border-color: #c3e6cb; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECRETS
# ─────────────────────────────────────────────
def _secret(key: str, fallback: str = "") -> str:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return fallback

GEMINI_API_KEY   = _secret("GEMINI_API_KEY")
GITHUB_TOKEN     = _secret("GITHUB_TOKEN")
GITHUB_REPO      = _secret("GITHUB_REPO")
GITHUB_FILE_PATH = _secret("GITHUB_FILE_PATH", "data/horus3_patients.json")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_READY = bool(GEMINI_API_KEY)

# ─────────────────────────────────────────────
# GITHUB-BACKED PATIENT STORAGE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _gh_repo():
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None
    try:
        g = Github(GITHUB_TOKEN)
        return g.get_repo(GITHUB_REPO)
    except Exception:
        return None


def load_patients() -> dict:
    if "patients_cache" in st.session_state and st.session_state.get("patients_cache_valid"):
        return st.session_state.patients_cache
    repo = _gh_repo()
    if repo is None:
        st.session_state.patients_cache = {}
        st.session_state.patients_cache_valid = True
        return {}
    try:
        contents = repo.get_contents(GITHUB_FILE_PATH)
        raw = base64.b64decode(contents.content).decode("utf-8")
        data = json.loads(raw)
        st.session_state.patients_cache = data
        st.session_state.patients_cache_valid = True
        return data
    except GithubException as e:
        if e.status == 404:
            st.session_state.patients_cache = {}
            st.session_state.patients_cache_valid = True
            return {}
        st.warning(f"GitHub read error: {e.data.get('message', e)}")
        return {}
    except Exception as e:
        st.warning(f"Could not load patient history: {e}")
        return {}


def invalidate_patient_cache():
    st.session_state.patients_cache_valid = False


def save_patients(patients: dict) -> bool:
    repo = _gh_repo()
    if repo is None:
        st.error("GitHub not configured — case not saved.")
        return False
    payload = json.dumps(patients, indent=2, ensure_ascii=False)
    commit_msg = f"HoRUS3 update — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    try:
        try:
            existing = repo.get_contents(GITHUB_FILE_PATH)
            repo.update_file(path=GITHUB_FILE_PATH, message=commit_msg,
                             content=payload, sha=existing.sha)
        except GithubException as e:
            if e.status == 404:
                repo.create_file(path=GITHUB_FILE_PATH, message=commit_msg, content=payload)
            else:
                raise
        st.session_state.patients_cache = patients
        st.session_state.patients_cache_valid = True
        return True
    except Exception as e:
        st.error(f"GitHub write error: {e}")
        return False


def save_case(pid: str, case_data: dict) -> bool:
    patients = load_patients()
    patients.setdefault(pid, {"details": {}, "cases": []})
    if isinstance(patients[pid], list):
        old_cases = patients[pid]
        patients[pid] = {"details": {}, "cases": old_cases}
    patients[pid]["cases"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), **case_data})
    return save_patients(patients)


def save_patient_details(pid: str, details: dict) -> bool:
    patients = load_patients()
    patients.setdefault(pid, {"details": {}, "cases": []})
    if isinstance(patients[pid], list):
        old_cases = patients[pid]
        patients[pid] = {"details": {}, "cases": old_cases}
    patients[pid]["details"] = details
    return save_patients(patients)


def delete_patient(pid: str) -> bool:
    patients = load_patients()
    if pid not in patients:
        return False
    del patients[pid]
    return save_patients(patients)


def get_patient_details(pid: str) -> dict:
    patients = load_patients()
    record = patients.get(pid, {})
    if isinstance(record, list):
        return {}
    return record.get("details", {})


def get_patient_cases(pid: str) -> list:
    patients = load_patients()
    record = patients.get(pid, {})
    if isinstance(record, list):
        return record
    return record.get("cases", [])


def next_patient_id() -> str:
    patients = load_patients()
    year = datetime.now().year
    prefix = f"PT-{year}-"
    nums = [
        int(k.split("-")[-1])
        for k in patients
        if k.startswith(prefix) and k.split("-")[-1].isdigit()
    ]
    return f"{prefix}{(max(nums, default=0) + 1):03d}"


# ─────────────────────────────────────────────
# UNIFIED DATASET LOADER
# pkl first → JSON fallback (same source as Tab 3)
# Both rheumatic.json + Case_studies_combined.json
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_all_dataset_symptoms() -> tuple:
    """
    Single loader used by Tab 1 (validation) AND Tab 3 (suggestions).
    Returns (known_set, s2r_dict, source_label).

    Priority:
      1. pkl files  → symptom_to_remedies already computed
      2. raw JSONs  → parse same way as SymptomRemedyMatcherTrainer
    Both sources merged: pkl gives fast lookup; JSON fills gaps.
    """
    from collections import Counter as _C

    known: set = set()
    s2r: dict = defaultdict(dict)
    sources = []

    # ── Layer 1: pkl files ─────────────────────
    for fname, label in [
        ("case_studies_model.pkl", "case_studies"),
        ("rheumatic_model.pkl",    "rheumatic"),
    ]:
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "rb") as f:
                d = pickle.load(f)
            for sym, rem_dict in d.get("symptom_to_remedies", {}).items():
                sl = sym.strip().lower()
                known.add(sl)
                for rem, score in rem_dict.items():
                    s2r[sl][rem] = max(s2r[sl].get(rem, 0), score)
            sources.append(fname)
        except Exception:
            pass

    # ── Layer 2: raw JSON files ────────────────
    # Always parse JSONs — fills gaps pkl may miss
    all_entries = []
    for fname in ["rheumatic.json", "Case_studies_combined.json"]:
        if not os.path.exists(fname):
            continue
        try:
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            for remedy, sections in data.items():
                for section_list in sections.values():
                    for item in section_list:
                        if isinstance(item, dict):
                            sym = item.get("symptom", "").strip().lower()
                            if sym:
                                known.add(sym)
                                all_entries.append((sym, remedy))
            sources.append(fname)
        except Exception:
            pass

    # Compute JSON-derived scores and merge (max wins)
    counts = _C(sym for sym, _ in all_entries)
    for sym, rem in all_entries:
        score = round(1.0 / max(1, counts[sym]), 6)
        s2r[sym][rem] = max(s2r[sym].get(rem, 0), score)

    source_label = ", ".join(sources) if sources else "none"
    return known, dict(s2r), source_label


# Load once at startup — shared by Tab 1 + Tab 3
_known_symptoms: set = set()
_s2r_unified: dict = {}
_dataset_source: str = "none"
try:
    _known_symptoms, _s2r_unified, _dataset_source = load_all_dataset_symptoms()
except Exception:
    pass


@st.cache_resource(show_spinner=False)
def load_system():
    """SentenceTransformer + cluster CSVs. s2r now from load_all_dataset_symptoms."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    clusters = {}
    for name in ["remedy_modalities", "remedy_area_modalities", "remedy_area"]:
        fpath = f"clusters_{name}.csv"
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df["Cluster_ID"] = df["Cluster_ID"].astype(str)
            clusters[name] = df
    return {"model": model, "clusters": clusters}


system_data = None
try:
    system_data = load_system()
except Exception:
    pass


def check_symptoms_against_dataset(symptom_list: list) -> tuple:
    """
    Split into (found, missing) using unified known-symptom set.
    Covers both pkl + JSON sources.
    """
    found, missing = [], []
    for s in symptom_list:
        (found if s.strip().lower() in _known_symptoms else missing).append(s)
    return found, missing


# ─────────────────────────────────────────────
# CONDITIONAL PROBABILITY LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_conditional_probabilities() -> pd.DataFrame | None:
    """
    Load symptom_conditional_probabilities.csv produced by the new clustering pipeline.
    Falls back to per-file versions (rheumatic_conditional_probabilities.csv, etc.)
    Returns None if no file found.
    """
    candidates = [
        "symptom_conditional_probabilities.csv",
        "rheumatic_conditional_probabilities.csv",
        "Case_studies_combined_conditional_probabilities.csv",
    ]
    for fname in candidates:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            # Normalise column names to lowercase for safety
            df.columns = [c.strip() for c in df.columns]
            return df
    return None


def get_suggestions_per_symptom(
    patient_symptoms: list,
    cp_df: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """
    Returns dict: { patient_symptom → [top_n suggested {symptom, prob}] }
    Each patient symptom gets its own top-10 list.
    """
    if cp_df is None or not patient_symptoms:
        return {}

    cp_df["_s1"] = cp_df["Symptom1"].str.strip().str.lower()
    cp_df["_s2"] = cp_df["Symptom2"].str.strip().str.lower()
    patient_syms_lower = {s.strip().lower() for s in patient_symptoms if s.strip()}

    result = {}
    for ps_orig in patient_symptoms:
        ps = ps_orig.strip().lower()
        if not ps:
            continue
        scores = {}         # candidate_lower → (prob, display_str)

        # Forward: ps is Symptom1
        for _, row in cp_df[cp_df["_s1"] == ps].iterrows():
            s2l = row["_s2"]
            if s2l not in patient_syms_lower:
                prev = scores.get(s2l, (0, row["Symptom2"]))
                scores[s2l] = (max(prev[0], float(row["P_B_given_A"])), row["Symptom2"])

        # Reverse: ps is Symptom2
        for _, row in cp_df[cp_df["_s2"] == ps].iterrows():
            s1l = row["_s1"]
            if s1l not in patient_syms_lower:
                prev = scores.get(s1l, (0, row["Symptom1"]))
                scores[s1l] = (max(prev[0], float(row["P_A_given_B"])), row["Symptom1"])

        top = sorted(scores.values(), key=lambda x: -x[0])[:top_n]
        result[ps_orig] = [{"symptom": disp, "prob": round(prob, 4)} for prob, disp in top]

    return result


def get_top_suggestions(
    patient_symptoms: list,
    cp_df: pd.DataFrame,
    top_n: int = 10,
) -> list:
    """
    Aggregate across all patient symptoms → top_n globally (for comparative report use).
    Each entry also carries triggered_by = patient symptom with highest contribution.
    """
    if cp_df is None or not patient_symptoms:
        return []

    cp_df["_s1"] = cp_df["Symptom1"].str.strip().str.lower()
    cp_df["_s2"] = cp_df["Symptom2"].str.strip().str.lower()
    patient_syms_lower = {s.strip().lower() for s in patient_symptoms if s.strip()}

    # candidate_lower → {probs:[], display:str, best_trigger:str, best_prob:float}
    agg = {}
    for ps_orig in patient_symptoms:
        ps = ps_orig.strip().lower()
        if not ps:
            continue
        for _, row in cp_df[cp_df["_s1"] == ps].iterrows():
            s2l = row["_s2"]
            if s2l in patient_syms_lower:
                continue
            p = float(row["P_B_given_A"])
            if s2l not in agg:
                agg[s2l] = {"probs": [], "display": row["Symptom2"], "best_trigger": ps_orig, "best_prob": p}
            agg[s2l]["probs"].append(p)
            if p > agg[s2l]["best_prob"]:
                agg[s2l]["best_prob"] = p
                agg[s2l]["best_trigger"] = ps_orig

        for _, row in cp_df[cp_df["_s2"] == ps].iterrows():
            s1l = row["_s1"]
            if s1l in patient_syms_lower:
                continue
            p = float(row["P_A_given_B"])
            if s1l not in agg:
                agg[s1l] = {"probs": [], "display": row["Symptom1"], "best_trigger": ps_orig, "best_prob": p}
            agg[s1l]["probs"].append(p)
            if p > agg[s1l]["best_prob"]:
                agg[s1l]["best_prob"] = p
                agg[s1l]["best_trigger"] = ps_orig

    averaged = [
        {
            "symptom": v["display"],
            "avg_prob": round(sum(v["probs"]) / len(v["probs"]), 4),
            "support": len(v["probs"]),
            "triggered_by": v["best_trigger"],
        }
        for v in agg.values()
    ]
    averaged.sort(key=lambda x: (-x["avg_prob"], -x["support"]))
    return averaged[:top_n]


# ─────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────
def gemini_categorise(raw_text: str, mode: str = "case_studies") -> dict:
    mode_instruction = (
        "Focus on symptom totality, constitutional picture, and classical case-taking "
        "as used in clinical case studies. Emphasise mental generals, physical generals, "
        "and particulars in that hierarchy."
        if mode == "case_studies"
        else
        "Focus on materia medica keynotes, drug pictures, and characteristic symptoms. "
        "Emphasise strange-rare-peculiar symptoms, keynote symptoms, and pathognomonic "
        "features that directly point to specific remedies in the materia medica."
    )
    system = f"""You are a classical homeopathic repertorisation assistant.
{mode_instruction}

Read the patient's description and return ONLY valid JSON — no markdown, no preamble.

Rules:
- Split symptoms into physical, psychological, and general categories.
- For each symptom extract worse[] and better[] modalities.
- Keep symptom text concise but clinically precise.
- Identify the dominant miasmatic tendency (psoric / sycotic / syphilitic / tubercular / mixed).
- List any concomitants (symptoms that appear together).
- Write a one-sentence clinical_summary capturing the case essence.

JSON shape (return exactly this, nothing else):
{{
  "physical": [{{"symptom": "...", "worse": ["..."], "better": ["..."]}}],
  "psychological": [{{"symptom": "...", "worse": ["..."], "better": ["..."]}}],
  "general": [{{"symptom": "...", "worse": ["..."], "better": ["..."]}}],
  "miasm": "...",
  "concomitants": ["..."],
  "clinical_summary": "..."
}}"""
    model = genai.GenerativeModel(model_name="gemini-3.1-flash-lite", system_instruction=system)
    response = model.generate_content(
        f"Patient describes: {raw_text}",
        generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=3000),
    )
    raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(raw)


def rank_remedies_from_dataset(categorised: dict, s2r: dict, top_n: int = 10) -> list:
    """
    Score every remedy in the dataset by how many patient symptoms it covers.
    Returns ranked list of dicts:
      {name, score, symptoms_covered: [symptom_str, ...]}
    100% dataset-driven — no Gemini involvement.
    """
    from collections import defaultdict as _dd
    remedy_scores = _dd(lambda: {"score": 0.0, "symptoms_covered": []})

    all_syms = [
        s["symptom"]
        for k in ("physical", "psychological", "general")
        for s in categorised.get(k, [])
    ]

    for sym in all_syms:
        sym_lower = sym.strip().lower()
        # exact match first
        remedies = s2r.get(sym_lower, {})
        # fuzzy: if no exact, try partial key match
        if not remedies:
            for key, val in s2r.items():
                if sym_lower in key or key in sym_lower:
                    remedies = val
                    break
        for rem, weight in remedies.items():
            remedy_scores[rem]["score"] += weight
            remedy_scores[rem]["symptoms_covered"].append(sym)

    ranked = sorted(
        [{"name": rem, **info} for rem, info in remedy_scores.items()],
        key=lambda x: -x["score"]
    )
    return ranked[:top_n]


def gemini_narrate_remedies(
    patient_id: str,
    categorised: dict,
    ranked_remedies: list,
    patient_details: dict = None,
    mode: str = "case_studies",
    extra_label: str = "",
) -> dict:
    """
    Gemini receives the dataset-ranked remedy list.
    Job: ONLY narrate/explain WHY each remedy fits.
    Cannot add or remove remedies from the list.
    """
    lines = []
    for cat in ("physical", "psychological", "general"):
        for s in categorised.get(cat, []):
            worse  = ", ".join(s.get("worse",  [])) or "—"
            better = ", ".join(s.get("better", [])) or "—"
            lines.append(f"[{cat.title()}] {s['symptom']} | ↓ {worse} | ↑ {better}")

    remedy_list_str = "\n".join(
        f"{i+1}. {r['name']} (dataset score={r['score']:.4f}, "
        f"covers: {', '.join(r['symptoms_covered'][:5])})"
        for i, r in enumerate(ranked_remedies)
    )

    mode_instruction = (
        "Use classical case-analysis methodology referencing symptom totality and miasmatic background."
        if mode == "case_studies"
        else
        "Use materia medica drug-picture methodology referencing keynotes and characteristic modalities."
    )

    system = f"""You are a classical homeopathic analyst. {mode_instruction}

CRITICAL RULES:
- The remedies below were ranked by a dataset scoring algorithm. You CANNOT change the order or add new remedies.
- Your ONLY job is to explain WHY each remedy fits this specific patient's symptoms.
- Reference actual symptoms from the case for every remedy.
- Never suggest dosages or potencies.
- Return ONLY valid JSON, no markdown, no preamble.

JSON shape:
{{
  "primaryRemedy": {{
    "name": "<rank 1 remedy name — copy exactly>",
    "why": "Clinical paragraph explaining why this remedy fits this case.",
    "keyIndications": ["patient symptom → this remedy keynote", "..."]
  }},
  "top10Remedies": [
    {{
      "rank": 1,
      "name": "<copy from list>",
      "role": "simillimum|complementary|intercurrent|acute|anti-miasmatic|constitutional",
      "rationale": "Why this remedy fits — cite specific patient symptoms.",
      "keySymptoms": ["patient symptom matched", "..."],
      "datasetScore": 0.0
    }}
  ],
  "miasmaticAnalysis": "Miasmatic background paragraph.",
  "caseEssence": "Strange, rare, peculiar features pointing to the simillimum.",
  "remedyRelationships": "Sequence plan — what follows what and why.",
  "monitoringPoints": ["Observable sign to watch", "..."]
}}"""

    patient_context = ""
    if patient_details:
        parts = []
        if patient_details.get("name"):       parts.append(f"Name: {patient_details['name']}")
        if patient_details.get("age"):        parts.append(f"Age: {patient_details['age']}")
        if patient_details.get("gender"):     parts.append(f"Gender: {patient_details['gender']}")
        if patient_details.get("occupation"): parts.append(f"Occupation: {patient_details['occupation']}")
        if parts:
            patient_context = "Patient details: " + " | ".join(parts) + "\n"

    prompt = (
        f"Patient: {patient_id} {extra_label}\n"
        f"{patient_context}"
        f"Miasm: {categorised.get('miasm', 'unknown')}\n"
        f"Summary: {categorised.get('clinical_summary', '')}\n"
        f"Concomitants: {', '.join(categorised.get('concomitants', [])) or 'none'}\n\n"
        f"Symptoms:\n" + "\n".join(lines) +
        f"\n\nDATASET-RANKED REMEDIES (DO NOT CHANGE THIS LIST):\n{remedy_list_str}"
    )

    model = genai.GenerativeModel(model_name="gemini-3.1-flash-lite", system_instruction=system)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=4000),
    )
    raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    result = json.loads(raw)

    # Enforce dataset scores into top10Remedies regardless of what Gemini returned
    name_to_score = {r["name"].lower(): r["score"] for r in ranked_remedies}
    for entry in result.get("top10Remedies", []):
        entry["datasetScore"] = round(
            name_to_score.get(entry.get("name", "").lower(), 0.0), 6
        )
    return result


def gemini_report(patient_id: str, categorised: dict, patient_details: dict = None,
                  mode: str = "case_studies", extra_label: str = "") -> dict:
    """
    Entry point for all tabs.
    Step 1: rank remedies purely from dataset (_s2r_unified).
    Step 2: Gemini narrates WHY — no remedy name invention.
    """
    ranked = rank_remedies_from_dataset(categorised, _s2r_unified, top_n=10)

    if not ranked:
        return {
            "top10Remedies": [],
            "primaryRemedy": {
                "name": "No dataset match",
                "why": "No remedies found in dataset for these symptoms.",
                "keyIndications": [],
            },
            "miasmaticAnalysis": "",
            "caseEssence": "No matching symptoms found in loaded datasets.",
            "remedyRelationships": "",
            "monitoringPoints": [],
            "_warning": "No remedies matched in dataset.",
        }

    return gemini_narrate_remedies(
        patient_id, categorised, ranked,
        patient_details=patient_details,
        mode=mode,
        extra_label=extra_label,
    )


def gemini_comparative_summary(report_base: dict, report_enhanced: dict) -> str:
    """Generate a short AI narrative comparing the two prescriptions."""
    system = """You are a classical homeopath. 
Compare two treatment plans — one based on original symptoms, one with AI-suggested additional symptoms.
Write a concise clinical paragraph (150-200 words) summarising:
- Where the prescriptions agree (same simillimum or complementary remedies)
- Key differences in remedy selection or rationale
- Whether the added symptoms strengthened or changed the prescription
- Clinical recommendation: does adding the suggested symptoms materially improve the case analysis?

Return plain text only. No JSON. No headers."""
    base_top10    = [r["name"] for r in report_base.get("top10Remedies", [])]
    enhanced_top10 = [r["name"] for r in report_enhanced.get("top10Remedies", [])]
    prompt = (
        f"BASE PLAN — Primary: {report_base['primaryRemedy']['name']} | "
        f"Essence: {report_base.get('caseEssence','')}\n"
        f"Base top-10 (dataset-ranked): {base_top10}\n\n"
        f"ENHANCED PLAN — Primary: {report_enhanced['primaryRemedy']['name']} | "
        f"Essence: {report_enhanced.get('caseEssence','')}\n"
        f"Enhanced top-10 (dataset-ranked): {enhanced_top10}"
    )
    model = genai.GenerativeModel(model_name="gemini-3.1-flash-lite", system_instruction=system)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=1000),
    )
    return response.text.strip()


# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def build_pdf(patient_id: str, categorised: dict, report: dict,
              patient_details: dict = None,
              report_enhanced: dict = None,
              added_symptoms: list = None,
              comparative_summary: str = None) -> bytes:
    buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(
        buf.name, pagesize=A4,
        rightMargin=55, leftMargin=55, topMargin=70, bottomMargin=60,
    )
    styles = getSampleStyleSheet()
    S = lambda name, **kw: ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_s  = S("T",  fontSize=18, fontName="Helvetica-Bold", textColor=colors.HexColor("#111"), spaceAfter=4)
    sub_s    = S("Su", fontSize=9,  textColor=colors.HexColor("#888"), spaceAfter=20)
    label_s  = S("Lb", fontSize=7,  fontName="Helvetica-Bold", textColor=colors.HexColor("#999"),
                 spaceBefore=14, spaceAfter=5, leading=10)
    body_s   = S("Bd", fontSize=10, textColor=colors.HexColor("#222"), leading=16, spaceAfter=8)
    rem_s    = S("RN", fontSize=12, fontName="Helvetica-Bold", textColor=colors.HexColor("#111"), spaceAfter=3)
    bullet_s = S("Bl", fontSize=10, textColor=colors.HexColor("#333"), leading=14, leftIndent=12, spaceAfter=3)
    small_s  = S("Sm", fontSize=9,  textColor=colors.HexColor("#666"), leading=13)
    foot_s   = S("Ft", fontSize=7,  textColor=colors.HexColor("#aaa"))
    h2_s     = S("H2", fontSize=13, fontName="Helvetica-Bold", textColor=colors.HexColor("#111"),
                 spaceBefore=18, spaceAfter=6)

    patient_line = f"Patient {patient_id}"
    if patient_details:
        if patient_details.get("name"):
            patient_line = f"{patient_details['name']} ({patient_id})"
        extras = []
        if patient_details.get("age"):    extras.append(f"Age {patient_details['age']}")
        if patient_details.get("gender"): extras.append(patient_details["gender"])
        if extras:
            patient_line += " · " + " · ".join(extras)

    # FIX: build elems list, pass explicitly to inner fn to avoid closure rebind bug
    elems = [
        Paragraph("HoRUS 3 — Treatment Plan", title_s),
        Paragraph(f"{patient_line} · {datetime.now().strftime('%d %B %Y')}", sub_s),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd"), spaceAfter=14),
    ]

    def _add_report_section(elems_ref: list, rpt: dict, cat: dict, section_label: str):
        """Inner fn receives elems_ref explicitly — no closure rebind issue."""
        elems_ref.append(Paragraph(section_label, h2_s))
        elems_ref.append(Paragraph("CASE ESSENCE", label_s))
        elems_ref.append(Paragraph(rpt.get("caseEssence", ""), body_s))
        elems_ref.append(Paragraph("MIASMATIC PICTURE", label_s))
        elems_ref.append(Paragraph(rpt.get("miasmaticAnalysis", ""), body_s))

        # Symptom table
        elems_ref.append(Paragraph("SYMPTOM SUMMARY", label_s))
        rows = [["Category", "Symptom", "↓ Worse", "↑ Better"]]
        for c in ("physical", "psychological", "general"):
            for s in cat.get(c, []):
                rows.append([
                    c.title(), s["symptom"][:60],
                    ", ".join(s.get("worse", []))[:40] or "—",
                    ", ".join(s.get("better", []))[:40] or "—",
                ])
        if len(rows) > 1:
            t = Table(rows, colWidths=[1*inch, 2.4*inch, 1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#111")),
                ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
                ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
                ("TOPPADDING",     (0,0), (-1,-1), 5),
                ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
                ("LEFTPADDING",    (0,0), (-1,-1), 6),
                ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ]))
            elems_ref.extend([t, Spacer(1, 14)])

        # Top 10 remedies table
        top10 = rpt.get("top10Remedies", [])
        pr = rpt.get("primaryRemedy", {})
        elems_ref.append(Paragraph("TOP 10 REMEDIES", label_s))
        if top10:
            rem_rows = [["#", "Remedy", "Role", "Key Rationale"]]
            for i, r in enumerate(top10, 1):
                rem_rows.append([
                    str(i),
                    r.get("name", ""),
                    r.get("role", ""),
                    r.get("rationale", "")[:80],
                ])
            rt = Table(rem_rows, colWidths=[0.3*inch, 1.4*inch, 1.1*inch, 3.6*inch])
            rt.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#111")),
                ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
                ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
                ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#ddd")),
                ("TOPPADDING",     (0,0), (-1,-1), 4),
                ("BOTTOMPADDING",  (0,0), (-1,-1), 4),
                ("LEFTPADDING",    (0,0), (-1,-1), 5),
                ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ]))
            elems_ref.extend([rt, Spacer(1, 10)])
        else:
            # Fallback: primary + secondary
            elems_ref.extend([
                Paragraph("SIMILLIMUM", label_s),
                Paragraph(pr.get("name", ""), rem_s),
                Paragraph(pr.get("why", ""), body_s),
            ])

        if pr.get("keyIndications"):
            elems_ref.append(Paragraph("Key indications in this case:", small_s))
            for ind in pr["keyIndications"]:
                elems_ref.append(Paragraph(f"• {ind}", bullet_s))
        if pr.get("followedBy"):
            elems_ref.append(Paragraph(f"Followed well by: {', '.join(pr['followedBy'])}", small_s))
        elems_ref.append(Spacer(1, 10))

        if rpt.get("remedyRelationships"):
            elems_ref.extend([Paragraph("REMEDY SEQUENCE", label_s), Paragraph(rpt["remedyRelationships"], body_s)])
        mon = rpt.get("monitoringPoints", [])
        if mon:
            elems_ref.append(Paragraph("MONITORING POINTS", label_s))
            for pt in mon:
                elems_ref.append(Paragraph(f"• {pt}", bullet_s))

    # Base report
    _add_report_section(elems, report, categorised, "A. Original Symptom Analysis")

    # Enhanced report (if present)
    if report_enhanced and added_symptoms:
        import copy
        elems.extend([Spacer(1, 20), HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#ddd"), spaceAfter=14)])

        elems.append(Paragraph("B. AI-Suggested Additional Symptoms", h2_s))
        elems.append(Paragraph("Symptoms suggested by cluster model, added by clinician:", small_s))
        for s in added_symptoms:
            worse  = ", ".join(s.get("worse",  [])) or "—"
            better = ", ".join(s.get("better", [])) or "—"
            prob   = s.get("avg_prob", 0)
            trigger = s.get("triggered_by", "")
            trigger_str = f" ← {trigger}" if trigger else ""
            elems.append(Paragraph(
                f"• {s['symptom']}  [P={prob:.0%}]{trigger_str} | ↓ {worse} | ↑ {better}",
                bullet_s
            ))
        elems.append(Spacer(1, 10))

        enhanced_cat = copy.deepcopy(categorised)
        for s in added_symptoms:
            enhanced_cat["physical"].append({
                "symptom": s["symptom"],
                "worse": s.get("worse", []),
                "better": s.get("better", []),
            })

        _add_report_section(elems, report_enhanced, enhanced_cat, "C. Enhanced Analysis (with suggested symptoms)")

        if comparative_summary:
            elems.extend([
                Spacer(1, 14),
                HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd"), spaceAfter=10),
                Paragraph("D. Comparative Clinical Summary", h2_s),
                Paragraph(comparative_summary, body_s),
            ])

    elems.extend([
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#ddd"), spaceAfter=6),
        Paragraph("For clinical reference only. Prescribing decisions rest with the practitioner.", foot_s),
    ])
    doc.build(elems)
    with open(buf.name, "rb") as f:
        return f.read()


# ─────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────
def sym_tags(items, cls="sym-tag"):
    return " ".join(f'<span class="{cls}">{i}</span>' for i in items if i)


def render_symptom_category(label, symptoms):
    if not symptoms:
        return
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    for s in symptoms:
        worse  = sym_tags(s.get("worse",  []), "sym-tag sym-worse")
        better = sym_tags(s.get("better", []), "sym-tag sym-better")
        mod = ""
        if worse:  mod += f'<div style="margin-top:4px">↓ {worse}</div>'
        if better: mod += f'<div style="margin-top:2px">↑ {better}</div>'
        st.markdown(
            f'<div class="remedy-card" style="padding:0.75rem 1rem;margin-bottom:6px">'
            f'<div style="font-size:0.9rem;font-weight:500;color:#111">{s["symptom"]}</div>'
            f'<div style="font-size:0.8rem;color:#888">{mod}</div></div>',
            unsafe_allow_html=True,
        )


def render_report_column(rpt: dict):
    """Render compact report card — top 10 remedies ranked."""
    pr = rpt.get("primaryRemedy", {})
    top10 = rpt.get("top10Remedies", [])

    if top10:
        for r in top10:
            rank = r.get("rank", "")
            role_badge = f'<span class="remedy-role">{r.get("role","")}</span>'
            border = "border-left:3px solid #111;" if rank == 1 else ""
            st.markdown(
                f'<div class="remedy-card" style="{border}margin-bottom:5px;padding:0.6rem 0.9rem">'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<span style="font-size:0.72rem;color:#aaa;min-width:16px">#{rank}</span>'
                f'<span style="font-size:0.9rem;font-weight:{"600" if rank==1 else "400"};color:#111">'
                f'{r.get("name","")}</span>{role_badge}</div>'
                f'<div style="font-size:0.8rem;color:#555;margin-top:3px;line-height:1.5">'
                f'{r.get("rationale","")[:160]}…</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        # Fallback if old schema
        st.markdown(f"**Simillimum: {pr.get('name','—')}**")
        st.markdown(f'<div style="font-size:0.83rem;color:#444;line-height:1.6">{pr.get("why","")[:400]}…</div>',
                    unsafe_allow_html=True)
        # fallback: show secondary from top10 if old schema
        secs = [r for r in rpt.get("top10Remedies", [])[1:] if r.get("name")]
        if secs:
            st.markdown('<div class="section-label" style="margin-top:0.75rem">Supporting</div>', unsafe_allow_html=True)
            for r in secs[:4]:
                st.markdown(f'<span class="sym-tag">{r["name"]} · {r.get("role","")}</span>', unsafe_allow_html=True)

    if rpt.get("caseEssence"):
        st.markdown(
            f'<div style="font-size:0.8rem;color:#666;margin-top:0.75rem;font-style:italic">'
            f'{rpt["caseEssence"][:300]}…</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "step": "intake",
    "patient_id": "",
    "patient_details": {},
    "raw_symptoms": "",
    "categorised": None,
    "report": None,
    "case_saved": False,
    "analysis_mode": "case_studies",
    "patients_cache_valid": False,
    # Future symptoms tab state
    "future_selected": {},        # {symptom_str: {worse:[], better:[]}}
    "future_report_base": None,
    "future_report_enhanced": None,
    "future_comparative": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.patient_id:
    st.session_state.patient_id = next_patient_id()

# ─────────────────────────────────────────────
# STARTUP CHECKS
# ─────────────────────────────────────────────
missing = []
if not GEMINI_API_KEY:   missing.append("`GEMINI_API_KEY`")
if not GITHUB_TOKEN:     missing.append("`GITHUB_TOKEN`")
if not GITHUB_REPO:      missing.append("`GITHUB_REPO`")

if missing:
    st.error(
        f"**Missing secrets:** {', '.join(missing)}  \n"
        "Go to **Streamlit Cloud → your app → Settings → Secrets** and add them."
    )
    st.stop()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# ⚕ HoRUS 3")
st.markdown(
    '<p style="color:#888;font-size:0.9rem;margin-top:-0.75rem;margin-bottom:1.5rem">'
    "Clinical homeopathic intelligence</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_intake, tab_patients, tab_future = st.tabs(["📋  New Case", "👥  All Patients", "🔬  Symptom Suggestions"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — INTAKE / CASE WORKFLOW
# ═══════════════════════════════════════════════════════════════
with tab_intake:
    STEPS       = ["intake", "analysis", "report"]
    STEP_LABELS = ["Intake", "Categorisation", "Report"]
    step_idx = STEPS.index(st.session_state.step)
    cols_prog = st.columns(len(STEPS))
    for i, (col, lbl) in enumerate(zip(cols_prog, STEP_LABELS)):
        with col:
            cls   = "step-pill active" if i == step_idx else "step-pill"
            check = "✓ " if i < step_idx else ""
            st.markdown(f'<span class="{cls}">{check}{lbl}</span>', unsafe_allow_html=True)

    st.markdown('<hr style="margin-bottom:1.5rem">', unsafe_allow_html=True)

    # ── STEP 1 — INTAKE ──────────────────────────────────────
    if st.session_state.step == "intake":
        st.markdown("## Patient Details")
        patients_all = load_patients()
        patient_options = sorted(patients_all.keys()) if patients_all else []

        col_mode1, col_mode2 = st.columns([1, 1])
        with col_mode1:
            patient_entry_mode = st.radio(
                "Patient entry", ["New patient", "Existing patient"],
                horizontal=True, label_visibility="collapsed",
            )

        if patient_entry_mode == "Existing patient" and patient_options:
            with col_mode2:
                selected_pid = st.selectbox(
                    "Select patient", patient_options,
                    key="existing_patient_select", label_visibility="collapsed",
                )
            if selected_pid:
                st.session_state.patient_id = selected_pid
                existing_details = get_patient_details(selected_pid)
                if existing_details:
                    st.session_state.patient_details = existing_details
                    d = existing_details
                    chips = []
                    if d.get("name"):       chips.append(d["name"])
                    if d.get("age"):        chips.append(f"Age {d['age']}")
                    if d.get("gender"):     chips.append(d["gender"])
                    if d.get("occupation"): chips.append(d["occupation"])
                    chips_html = "".join(f'<span class="detail-chip">{c}</span>' for c in chips)
                    st.markdown(f'<div style="margin:0.5rem 0 1rem">{chips_html}</div>', unsafe_allow_html=True)
                    prior_cases = get_patient_cases(selected_pid)
                    if prior_cases:
                        st.markdown(
                            f'<span class="patient-meta">📁 {len(prior_cases)} prior case(s) on record</span>',
                            unsafe_allow_html=True,
                        )
        elif patient_entry_mode == "Existing patient" and not patient_options:
            st.info("No patients on record yet.")

        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

        with st.expander(
            "✎  Patient information" + (" (filled)" if st.session_state.patient_details.get("name") else ""),
            expanded=(patient_entry_mode == "New patient")
        ):
            d_col1, d_col2, d_col3 = st.columns([2, 1, 1])
            with d_col1:
                p_name = st.text_input("Full name", value=st.session_state.patient_details.get("name", ""), placeholder="Optional")
            with d_col2:
                p_age = st.text_input("Age", value=st.session_state.patient_details.get("age", ""), placeholder="e.g. 34")
            with d_col3:
                p_gender = st.selectbox("Gender", ["", "Male", "Female", "Other"],
                    index=["", "Male", "Female", "Other"].index(st.session_state.patient_details.get("gender", ""))
                    if st.session_state.patient_details.get("gender", "") in ["", "Male", "Female", "Other"] else 0)
            d_col4, d_col5 = st.columns([2, 2])
            with d_col4:
                p_occupation = st.text_input("Occupation", value=st.session_state.patient_details.get("occupation", ""), placeholder="Optional")
            with d_col5:
                p_contact = st.text_input("Contact / notes", value=st.session_state.patient_details.get("contact", ""), placeholder="Phone, email, or notes")

            pid_col1, pid_col2 = st.columns([3, 1])
            with pid_col1:
                new_pid = st.text_input("Patient ID", value=st.session_state.patient_id)
                if new_pid.strip().upper() != st.session_state.patient_id:
                    st.session_state.patient_id = new_pid.strip().upper()
            with pid_col2:
                st.markdown('<div style="margin-top:1.65rem"></div>', unsafe_allow_html=True)
                if st.button("↺ Generate"):
                    st.session_state.patient_id = next_patient_id()
                    st.rerun()

            st.session_state.patient_details = {
                "name": p_name.strip(), "age": p_age.strip(), "gender": p_gender,
                "occupation": p_occupation.strip(), "contact": p_contact.strip(),
            }

        st.markdown('<hr style="margin:1.5rem 0">', unsafe_allow_html=True)
        st.markdown("## Analysis Mode")
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            cs_selected = st.session_state.analysis_mode == "case_studies"
            if st.button("📚  Case Studies" + (" ✓" if cs_selected else ""),
                         use_container_width=True, type="primary" if cs_selected else "secondary"):
                st.session_state.analysis_mode = "case_studies"
                st.rerun()
        with mode_col2:
            mm_selected = st.session_state.analysis_mode == "materia_medica"
            if st.button("🌿  Materia Medica" + (" ✓" if mm_selected else ""),
                         use_container_width=True, type="primary" if mm_selected else "secondary"):
                st.session_state.analysis_mode = "materia_medica"
                st.rerun()

        mode_label = "case_studies" if st.session_state.analysis_mode == "case_studies" else "materia_medica"
        mode_cls   = "mode-case" if mode_label == "case_studies" else "mode-materia"
        mode_desc  = (
            "Analyses using symptom totality and clinical case precedent — constitutional approach, miasmatic background, mental/physical/general hierarchy."
            if mode_label == "case_studies" else
            "Analyses using materia medica drug pictures — keynote symptoms, characteristic modalities, strange-rare-peculiar features, and remedy essence."
        )
        st.markdown(
            f'<div class="info-box" style="margin-top:0.75rem">'
            f'<span class="mode-badge {mode_cls}">{mode_label.replace("_", " ").title()}</span>'
            f'<br>{mode_desc}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr style="margin:1.5rem 0">', unsafe_allow_html=True)
        st.markdown("## Symptoms")

        pid = st.session_state.patient_id
        pd_name = st.session_state.patient_details.get("name", "")
        badge_text = f"{pd_name} · {pid}" if pd_name else f"Patient {pid}"
        st.markdown(f'<div class="patient-badge">{badge_text}</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.85rem;color:#777;margin-top:-0.25rem;margin-bottom:1rem">'
            "Write freely — physical complaints, mental state, what makes things worse or better, "
            "time of day, thermal preferences, sleep, appetite, any relevant details.</p>",
            unsafe_allow_html=True,
        )

        raw = st.text_area(
            label="Symptoms", value=st.session_state.raw_symptoms, height=220,
            placeholder="Example: Sharp throbbing headache above the left eye for 3 days, worse in cold air and any touch, better lying still in a dark room. Very anxious, restless at night...",
            label_visibility="collapsed",
        )
        st.session_state.raw_symptoms = raw

        st.markdown(
            '<div class="info-box">The AI will categorise these symptoms into physical, '
            'psychological, and general groups with their modalities. '
            'You can review and edit the result before generating the treatment plan.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Analyse symptoms →", type="primary", disabled=not raw.strip(), use_container_width=True):
            with st.spinner("Parsing symptoms…"):
                try:
                    result = gemini_categorise(raw, mode=st.session_state.analysis_mode)
                    st.session_state.categorised = result
                    st.session_state.step = "analysis"
                    # Reset future tab state when new case starts
                    st.session_state.future_selected = {}
                    st.session_state.future_report_base = None
                    st.session_state.future_report_enhanced = None
                    st.session_state.future_comparative = None
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Could not parse AI response — try rephrasing. ({e})")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── STEP 2 — CATEGORISATION REVIEW ───────────────────────
    elif st.session_state.step == "analysis":
        cat = st.session_state.categorised

        pid = st.session_state.patient_id
        pd_name = st.session_state.patient_details.get("name", "")
        badge_text = f"{pd_name} · {pid}" if pd_name else f"Patient {pid}"
        st.markdown(f'<div class="patient-badge">{badge_text}</div>', unsafe_allow_html=True)

        mode_label = st.session_state.analysis_mode.replace("_", " ").title()
        mode_cls = "mode-case" if st.session_state.analysis_mode == "case_studies" else "mode-materia"
        st.markdown(f'<span class="mode-badge {mode_cls}">{mode_label}</span>', unsafe_allow_html=True)

        # ── Dataset validation ──────────────────────────────────
        if cat.get("clinical_summary"):
            st.markdown(
                f'<div class="info-box" style="border-left:3px solid #111;padding-left:1rem">'
                f'<strong>Case essence</strong><br>{cat["clinical_summary"]}</div>',
                unsafe_allow_html=True,
            )

        meta_cols = st.columns(2)
        with meta_cols[0]:
            if cat.get("miasm"):
                st.markdown(f'<span class="sym-tag" style="background:#f0f0f0">Miasm: {cat["miasm"]}</span>', unsafe_allow_html=True)
        with meta_cols[1]:
            if cat.get("concomitants"):
                st.markdown(f'<span style="font-size:0.8rem;color:#777">Concomitants: {", ".join(cat["concomitants"])}</span>', unsafe_allow_html=True)

        st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
        render_symptom_category("Physical",      cat.get("physical",      []))
        render_symptom_category("Psychological", cat.get("psychological", []))
        render_symptom_category("General",       cat.get("general",       []))

        with st.expander("✎  Edit categorised symptoms (optional)"):
            st.caption("Edit symptom text or modalities directly.")
            for cat_key, cat_label in [("physical", "Physical"), ("psychological", "Psychological"), ("general", "General")]:
                st.markdown(f"**{cat_label}**")
                edited = []
                for idx, s in enumerate(cat.get(cat_key, [])):
                    c1, c2, c3 = st.columns([3, 2, 2])
                    sym = c1.text_input("Symptom", value=s["symptom"], key=f"edit_{cat_key}_{idx}_sym", label_visibility="collapsed")
                    w   = c2.text_input("↓ Worse",  value=", ".join(s.get("worse",  [])), key=f"edit_{cat_key}_{idx}_w",   label_visibility="collapsed")
                    b   = c3.text_input("↑ Better", value=", ".join(s.get("better", [])), key=f"edit_{cat_key}_{idx}_b",   label_visibility="collapsed")
                    edited.append({
                        "symptom": sym,
                        "worse":  [x.strip() for x in w.split(",") if x.strip()],
                        "better": [x.strip() for x in b.split(",") if x.strip()],
                    })
                cat[cat_key] = edited
            st.session_state.categorised = cat

        st.markdown('<hr style="margin:1.5rem 0">', unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1, 3])
        with btn_col1:
            if st.button("← Back"):
                st.session_state.step = "intake"
                st.rerun()
        with btn_col2:
            if st.button("Generate treatment plan →", type="primary", use_container_width=True):
                with st.spinner("Composing treatment plan…"):
                    try:
                        rpt = gemini_report(
                            st.session_state.patient_id,
                            st.session_state.categorised,
                            patient_details=st.session_state.patient_details,
                            mode=st.session_state.analysis_mode,
                        )
                        st.session_state.report = rpt
                        st.session_state.step = "report"
                        st.session_state.case_saved = False
                        st.rerun()
                    except json.JSONDecodeError as e:
                        st.error(f"Could not parse report response. ({e})")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── STEP 3 — TREATMENT PLAN ───────────────────────────────
    elif st.session_state.step == "report":
        rpt = st.session_state.report
        cat = st.session_state.categorised
        pid = st.session_state.patient_id
        pd  = st.session_state.patient_details

        if not rpt:
            st.error("No report found — please go back.")
            st.stop()

        if not st.session_state.case_saved:
            top_remedies = [r["name"] for r in rpt.get("top10Remedies", [])] or [rpt.get("primaryRemedy", {}).get("name", "")]
            with st.spinner("Saving case to GitHub…"):
                ok = save_case(pid, {
                    "raw_symptoms":    st.session_state.raw_symptoms,
                    "categorised":     cat,
                    "top_remedies":    top_remedies,
                    "clinical_notes":  rpt.get("caseEssence", ""),
                    "patient_details": pd,
                    "analysis_mode":   st.session_state.analysis_mode,
                })
                if pd.get("name") or pd.get("age"):
                    save_patient_details(pid, pd)
            if ok:
                st.session_state.case_saved = True
                st.markdown('<span class="save-success">✓ Case saved to GitHub</span>', unsafe_allow_html=True)
            else:
                st.warning("Case could not be saved — check GitHub configuration.")

        pd_name = pd.get("name", "")
        badge_text = f"{pd_name} · {pid}" if pd_name else f"Patient {pid}"
        extras = []
        if pd.get("age"):    extras.append(f"Age {pd['age']}")
        if pd.get("gender"): extras.append(pd["gender"])
        if extras:
            badge_text += " · " + " · ".join(extras)
        st.markdown(
            f'<div class="patient-badge">{badge_text} &nbsp;·&nbsp; {datetime.now().strftime("%d %B %Y")}</div>',
            unsafe_allow_html=True,
        )

        mode_label = st.session_state.analysis_mode.replace("_", " ").title()
        mode_cls = "mode-case" if st.session_state.analysis_mode == "case_studies" else "mode-materia"
        st.markdown(f'<span class="mode-badge {mode_cls}">{mode_label}</span>', unsafe_allow_html=True)

        st.markdown("## Treatment Plan")

        if rpt.get("caseEssence"):
            st.markdown(
                f'<div class="info-box" style="border-left:3px solid #111;padding-left:1rem">'
                f'<strong>Case essence</strong><br>{rpt["caseEssence"]}</div>',
                unsafe_allow_html=True,
            )

        if rpt.get("miasmaticAnalysis"):
            st.markdown('<div class="section-label">Miasmatic picture</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:0.9rem;color:#333;line-height:1.7">{rpt["miasmaticAnalysis"]}</p>', unsafe_allow_html=True)

        pr = rpt.get("primaryRemedy", {})
        top10 = rpt.get("top10Remedies", [])

        if top10:
            st.markdown('<div class="section-label">Top 10 Remedies</div>', unsafe_allow_html=True)
            for r in top10:
                rank = r.get("rank", "")
                role_badge = f'<span class="remedy-role">{r.get("role","")}</span>'
                border = "border-left:3px solid #111;" if rank == 1 else ""
                key_syms = r.get("keySymptoms", [])
                key_html = ""
                if key_syms:
                    key_html = (
                        '<ul style="margin:0.3rem 0 0 1rem;padding:0;font-size:0.82rem;color:#555">'
                        + "".join(f"<li>{k}</li>" for k in key_syms[:3])
                        + "</ul>"
                    )
                st.markdown(
                    f'<div class="remedy-card" style="{border}margin-bottom:6px">'
                    f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<span style="font-size:0.75rem;color:#aaa;min-width:20px;font-weight:600">#{rank}</span>'
                    f'<span class="remedy-name" style="margin:0">{r.get("name","")}</span>{role_badge}</div>'
                    f'<div class="remedy-rationale">{r.get("rationale","")}</div>'
                    f'{key_html}</div>',
                    unsafe_allow_html=True,
                )
        else:
            # Fallback: old schema
            st.markdown('<div class="section-label">Simillimum</div>', unsafe_allow_html=True)
            followed_html = ""
            if pr.get("followedBy"):
                followed_html = f'<div class="remedy-meta">Followed well by: {", ".join(pr["followedBy"])}</div>'
            indications_html = ""
            if pr.get("keyIndications"):
                items = "".join(f"<li>{i}</li>" for i in pr["keyIndications"])
                indications_html = (
                    f'<div class="remedy-meta" style="margin-top:0.6rem"><strong>Key indications</strong>'
                    f'<ul style="margin:0.3rem 0 0 1rem;padding:0;font-size:0.85rem;color:#444">{items}</ul></div>'
                )
            st.markdown(
                f'<div class="remedy-card remedy-primary">'
                f'<div class="remedy-name">{pr.get("name","")}</div>'
                f'<div class="remedy-rationale">{pr.get("why","")}</div>'
                f'{indications_html}{followed_html}</div>',
                unsafe_allow_html=True,
            )
            secs = rpt.get("top10Remedies", [])[1:]
            if secs:
                st.markdown('<div class="section-label">Supporting remedies</div>', unsafe_allow_html=True)
                for rem in secs[:4]:
                    role_badge = f'<span class="remedy-role">{rem.get("role","")}</span>'
                    st.markdown(
                        f'<div class="remedy-card">'
                        f'<div class="remedy-name">{rem["name"]}{role_badge}</div>'
                        f'<div class="remedy-rationale">{rem.get("rationale","")}</div></div>',
                        unsafe_allow_html=True,
                    )

        if rpt.get("remedyRelationships"):
            st.markdown('<div class="section-label">Remedy sequence & relationships</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size:0.9rem;color:#333;line-height:1.7">{rpt["remedyRelationships"]}</p>', unsafe_allow_html=True)

        mon = rpt.get("monitoringPoints", [])
        if mon:
            st.markdown('<div class="section-label">What to monitor</div>', unsafe_allow_html=True)
            items_html = "".join(f'<div class="monitor-item">• {pt}</div>' for pt in mon)
            st.markdown(f'<div class="info-box" style="padding:0.6rem 1rem">{items_html}</div>', unsafe_allow_html=True)

        with st.expander("Full symptom categorisation"):
            render_symptom_category("Physical",      cat.get("physical",      []))
            render_symptom_category("Psychological", cat.get("psychological", []))
            render_symptom_category("General",       cat.get("general",       []))

        st.markdown('<hr style="margin:1.5rem 0">', unsafe_allow_html=True)
        act1, act2, act3 = st.columns(3)

        with act1:
            if st.button("← Back to categorisation"):
                st.session_state.step = "analysis"
                st.rerun()
        with act2:
            try:
                pdf_bytes = build_pdf(pid, cat, rpt, patient_details=pd)
                st.download_button(
                    label="↓ Download PDF", data=pdf_bytes,
                    file_name=f"HoRUS3_{pid}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF error: {e}")
        with act3:
            if st.button("↺  New case", use_container_width=True):
                for k in ("step", "raw_symptoms", "categorised", "report", "case_saved", "patient_details"):
                    st.session_state[k] = defaults[k]
                st.session_state.step = "intake"
                st.session_state.patient_id = next_patient_id()
                st.rerun()

        st.markdown(
            '<p style="font-size:0.75rem;color:#bbb;margin-top:1rem;text-align:center">'
            "For clinical reference only. Prescribing decisions rest with the practitioner.</p>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — ALL PATIENTS
# ═══════════════════════════════════════════════════════════════
with tab_patients:
    st.markdown("## Patient Registry")
    col_h1, col_h2 = st.columns([5, 1])
    with col_h2:
        if st.button("↺ Refresh", use_container_width=True):
            invalidate_patient_cache()
            st.rerun()

    patients_all = load_patients()

    if not patients_all:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:2rem">'
            '📭 No patients on record yet.<br>'
            '<span style="font-size:0.8rem;color:#aaa">Cases will appear here after the first consultation.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        search_q = st.text_input("Search patients", placeholder="Search by ID or name…", label_visibility="collapsed")

        patient_rows = []
        for pid, record in patients_all.items():
            if isinstance(record, list):
                details, cases = {}, record
            else:
                details = record.get("details", {})
                cases   = record.get("cases", [])
            patient_rows.append({
                "pid": pid, "name": details.get("name", ""), "age": details.get("age", ""),
                "gender": details.get("gender", ""),
                "last_visit": cases[-1]["timestamp"] if cases else "—",
                "total_cases": len(cases),
                "last_remedy": (cases[-1].get("top_remedies", ["—"])[0] if cases else "—"),
                "last_mode": cases[-1].get("analysis_mode", "") if cases else "",
                "cases": cases, "details": details,
            })
        patient_rows.sort(key=lambda r: r["last_visit"], reverse=True)
        if search_q.strip():
            q = search_q.strip().lower()
            patient_rows = [r for r in patient_rows if q in r["pid"].lower() or q in r["name"].lower()]

        st.markdown(f'<p style="font-size:0.8rem;color:#888;margin-bottom:1rem">{len(patient_rows)} patient(s) found</p>', unsafe_allow_html=True)

        for row in patient_rows:
            pid  = row["pid"]
            name = row["name"]
            disp = f"{name} <span style='color:#aaa'>·</span> {pid}" if name else pid
            meta_parts = []
            if row["age"]:    meta_parts.append(f"Age {row['age']}")
            if row["gender"]: meta_parts.append(row["gender"])
            meta_parts.append(f"{row['total_cases']} case(s)")
            meta_parts.append(f"Last visit: {row['last_visit']}")
            meta_str = " &nbsp;·&nbsp; ".join(meta_parts)

            with st.expander(f"{'👤 ' + name if name else '🆔'} {pid}"):
                st.markdown(
                    f'<div><div style="font-size:1.05rem;font-weight:500;color:#111">{disp}</div>'
                    f'<div class="patient-meta">{meta_str}</div></div>',
                    unsafe_allow_html=True,
                )
                chips = []
                if row["details"].get("occupation"): chips.append(f"🧑‍💼 {row['details']['occupation']}")
                if row["details"].get("contact"):    chips.append(f"📞 {row['details']['contact']}")
                if chips:
                    chips_html = " ".join(f'<span class="detail-chip">{c}</span>' for c in chips)
                    st.markdown(f'<div style="margin:0.5rem 0">{chips_html}</div>', unsafe_allow_html=True)

                st.markdown('<hr style="margin:0.75rem 0">', unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:0.75rem;font-weight:600;letter-spacing:0.08em;'
                    'text-transform:uppercase;color:#888;margin-bottom:0.5rem">Case History</div>',
                    unsafe_allow_html=True,
                )

                if not row["cases"]:
                    st.caption("No cases recorded.")
                else:
                    for i, case in enumerate(reversed(row["cases"]), 1):
                        remedies   = case.get("top_remedies", [])
                        primary    = remedies[0] if remedies else "—"
                        secondary  = remedies[1:4] if len(remedies) > 1 else []
                        mode_str   = case.get("analysis_mode", "")
                        mode_label = mode_str.replace("_", " ").title() if mode_str else ""
                        mode_cls_  = "mode-case" if mode_str == "case_studies" else "mode-materia"
                        mode_badge = (
                            f'<span class="mode-badge {mode_cls_}" style="font-size:0.65rem;padding:2px 7px">'
                            f'{mode_label}</span>' if mode_label else ""
                        )
                        secondary_html = (
                            f'<span style="font-size:0.78rem;color:#888"> + {", ".join(secondary)}</span>'
                            if secondary else ""
                        )
                        notes = case.get("clinical_notes", "")
                        notes_html = (
                            f'<div style="font-size:0.8rem;color:#666;margin-top:3px;font-style:italic;line-height:1.5">'
                            f'{notes[:200]}{"…" if len(notes)>200 else ""}</div>' if notes else ""
                        )
                        st.markdown(
                            f'<div class="case-entry">'
                            f'<div class="case-date">{case.get("timestamp","—")} &nbsp;{mode_badge}</div>'
                            f'<div style="font-size:0.9rem;font-weight:500;color:#111">{primary}{secondary_html}</div>'
                            f'{notes_html}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown('<div style="margin-top:0.75rem"></div>', unsafe_allow_html=True)
                btn_c1, btn_c2 = st.columns([3, 1])
                with btn_c1:
                    if st.button(f"📋  New case for {pid}", key=f"new_case_{pid}"):
                        for k in ("step", "raw_symptoms", "categorised", "report", "case_saved"):
                            st.session_state[k] = defaults[k]
                        st.session_state.step = "intake"
                        st.session_state.patient_id = pid
                        st.session_state.patient_details = row["details"]
                        st.rerun()
                with btn_c2:
                    confirm_key = f"confirm_delete_{pid}"
                    if st.session_state.get(confirm_key):
                        if st.button(f"⚠️ Confirm delete", key=f"do_delete_{pid}", type="primary"):
                            with st.spinner("Deleting…"):
                                ok = delete_patient(pid)
                            if ok:
                                st.session_state[confirm_key] = False
                                st.success(f"Patient {pid} deleted.")
                                st.rerun()
                    else:
                        if st.button(f"🗑 Delete", key=f"del_{pid}"):
                            st.session_state[confirm_key] = True
                            st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 3 — SYMPTOM SUGGESTIONS (powered by cluster conditional probabilities)
# ═══════════════════════════════════════════════════════════════
with tab_future:
    st.markdown("## 🔬 Symptom Suggestions")

    # ── Guard: need a categorised case ──────────────────────────
    cat = st.session_state.get("categorised")
    if not cat:
        st.markdown(
            '<div class="info-box" style="text-align:center;padding:2rem">'
            '<div style="font-size:1.5rem;margin-bottom:0.5rem">📋</div>'
            '<div style="font-weight:500;color:#111;margin-bottom:0.4rem">No active case</div>'
            '<div style="font-size:0.85rem;color:#888">Complete symptom intake in the <strong>New Case</strong> tab first.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Load conditional probability table ──────────────────────
    cp_df = load_conditional_probabilities()

    if cp_df is None:
        st.markdown(
            '<div class="info-box" style="border-left:3px solid #f0a500;padding-left:1rem">'
            '⚠️ <strong>No cluster data found.</strong><br>'
            'Run <code>symptom_clustering_pro.py</code> to generate '
            '<code>*_conditional_probabilities.csv</code> and place it alongside this app.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Extract patient symptoms (all categories, flat list) ────
    all_patient_syms = []
    for cat_key in ("physical", "psychological", "general"):
        for s in cat.get(cat_key, []):
            all_patient_syms.append(s["symptom"])

    if not all_patient_syms:
        st.info("No symptoms found in the current case.")
        st.stop()

    # ── Compute suggestions: both per-symptom AND global aggregate ──
    per_sym_sugs = get_suggestions_per_symptom(all_patient_syms, cp_df, top_n=10)
    global_sugs  = get_top_suggestions(all_patient_syms, cp_df, top_n=10)

    if not per_sym_sugs:
        st.info("No co-occurring symptoms found in cluster data.")
        st.stop()

    # ── Show current case symptoms as reference ─────────────────
    with st.expander("Current case symptoms (reference)", expanded=False):
        for s in all_patient_syms:
            st.markdown(f'<span class="sym-tag">{s}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.85rem;color:#777;margin-bottom:1.25rem">'
        'Top 10 co-occurring symptoms shown per entered symptom. '
        'Select clinically relevant ones, add modalities, then generate comparative report.</p>',
        unsafe_allow_html=True,
    )

    # ── Per-symptom suggestion groups ────────────────────────────
    selected: dict = st.session_state.future_selected  # {symptom_str: {worse:[], better:[]}}

    # Build flat global index for checkbox keys (unique across all groups)
    global_idx = 0

    for ps_orig in all_patient_syms:
        sug_list = per_sym_sugs.get(ps_orig, [])
        if not sug_list:
            continue

        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;letter-spacing:0.08em;'
            f'text-transform:uppercase;color:#555;margin:1.25rem 0 0.5rem 0;'
            f'border-bottom:1px solid #eee;padding-bottom:4px">'
            f'Triggered by: <span style="color:#111">{ps_orig}</span></div>',
            unsafe_allow_html=True,
        )

        for sug in sug_list:
            sym      = sug["symptom"]
            prob     = sug["prob"]
            is_sel   = sym in selected
            prob_pct = int(prob * 100)
            bar_w    = max(4, prob_pct)
            card_cls = "sug-card selected" if is_sel else "sug-card"
            ck_key   = f"sug_check_{global_idx}"
            global_idx += 1

            col_check, col_body = st.columns([0.06, 0.94])

            with col_check:
                checked = st.checkbox("", value=is_sel, key=ck_key, label_visibility="collapsed")
                if checked and sym not in selected:
                    selected[sym] = {"worse": [], "better": [], "triggered_by": ps_orig, "avg_prob": prob}
                    st.session_state.future_selected = selected
                    st.rerun()
                elif not checked and sym in selected:
                    del selected[sym]
                    st.session_state.future_selected = selected
                    st.rerun()

            with col_body:
                st.markdown(
                    f'<div class="{card_cls}">'
                    f'<div style="flex:1">'
                    f'<div style="font-size:0.9rem;font-weight:{"600" if is_sel else "400"};color:#111">{sym}</div>'
                    f'<div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{bar_w}%"></div></div>'
                    f'<div class="prob-label">P(this | <em>{ps_orig[:40]}</em>) = <strong>{prob_pct}%</strong></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                if is_sel:
                    m_col1, m_col2 = st.columns(2)
                    w_key = f"worse_g{global_idx}"
                    b_key = f"better_g{global_idx}"
                    with m_col1:
                        worse_val = st.text_input(
                            "↓ Worse", value=", ".join(selected[sym].get("worse", [])),
                            key=w_key, placeholder="e.g. cold, motion, night",
                        )
                        selected[sym]["worse"] = [x.strip() for x in worse_val.split(",") if x.strip()]
                    with m_col2:
                        better_val = st.text_input(
                            "↑ Better", value=", ".join(selected[sym].get("better", [])),
                            key=b_key, placeholder="e.g. warmth, rest, pressure",
                        )
                        selected[sym]["better"] = [x.strip() for x in better_val.split(",") if x.strip()]
                    st.session_state.future_selected = selected

    st.markdown("---")

    # ── Summary of selected ──────────────────────────────────────
    n_selected = len(selected)
    if n_selected == 0:
        st.markdown(
            '<div class="info-box">Select one or more suggested symptoms above to generate a comparative report.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="info-box" style="border-left:3px solid #111;padding-left:1rem">'
            f'<strong>{n_selected} symptom(s) selected</strong><br>'
            + " ".join(f'<span class="sym-tag">{s}</span>' for s in selected.keys())
            + '</div>',
            unsafe_allow_html=True,
        )

        if st.button("🔬  Generate comparative report", type="primary", use_container_width=True):
            pid    = st.session_state.patient_id
            pd_det = st.session_state.patient_details
            mode   = st.session_state.analysis_mode

            import copy
            enhanced_cat = copy.deepcopy(cat)
            added_list = []
            for sym, mods in selected.items():
                entry = {
                    "symptom":      sym,
                    "worse":        mods.get("worse", []),
                    "better":       mods.get("better", []),
                    "avg_prob":     mods.get("avg_prob", 0.0),
                    "triggered_by": mods.get("triggered_by", ""),
                }
                enhanced_cat["physical"].append({"symptom": sym, "worse": entry["worse"], "better": entry["better"]})
                added_list.append(entry)

            col_prog1, col_prog2, col_prog3 = st.columns(3)

            with col_prog1:
                with st.spinner("Base report…"):
                    try:
                        base_rpt = gemini_report(pid, cat, patient_details=pd_det, mode=mode,
                                                  extra_label="[BASE — original symptoms]")
                        st.session_state.future_report_base = base_rpt
                    except Exception as e:
                        st.error(f"Base report error: {e}")
                        st.stop()

            with col_prog2:
                with st.spinner("Enhanced report…"):
                    try:
                        enh_rpt = gemini_report(pid, enhanced_cat, patient_details=pd_det, mode=mode,
                                                 extra_label="[ENHANCED — with suggested symptoms]")
                        st.session_state.future_report_enhanced = enh_rpt
                    except Exception as e:
                        st.error(f"Enhanced report error: {e}")
                        st.stop()

            with col_prog3:
                with st.spinner("Comparative summary…"):
                    try:
                        comp = gemini_comparative_summary(
                            st.session_state.future_report_base,
                            st.session_state.future_report_enhanced,
                        )
                        st.session_state.future_comparative = comp
                        st.session_state["_future_added_list"] = added_list
                    except Exception as e:
                        st.warning(f"Comparative summary skipped: {e}")
                        st.session_state.future_comparative = None

            st.rerun()

    # ── Display comparative report ───────────────────────────────
    base_rpt = st.session_state.get("future_report_base")
    enh_rpt  = st.session_state.get("future_report_enhanced")
    comp_txt = st.session_state.get("future_comparative")

    if base_rpt and enh_rpt:
        st.markdown("## Comparative Analysis")

        if comp_txt:
            st.markdown(
                f'<div class="info-box" style="border-left:3px solid #111;padding-left:1rem">'
                f'<div class="section-label" style="margin-top:0">AI Clinical Commentary</div>'
                f'{comp_txt}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(
                '<div class="compare-col">'
                '<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;'
                'text-transform:uppercase;color:#888;margin-bottom:0.75rem">'
                'A · Original symptoms only</div>',
                unsafe_allow_html=True,
            )
            render_report_column(base_rpt)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown(
                '<div class="compare-col enhanced">'
                '<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;'
                'text-transform:uppercase;color:#2a7a2a;margin-bottom:0.75rem">'
                '✦ B · With suggested symptoms</div>',
                unsafe_allow_html=True,
            )
            render_report_column(enh_rpt)

            # Highlight added symptoms
            added_list = st.session_state.get("_future_added_list", [])
            if added_list:
                st.markdown(
                    '<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.08em;'
                    'text-transform:uppercase;color:#2a7a2a;margin-top:0.75rem;margin-bottom:0.3rem">'
                    'Added symptoms</div>',
                    unsafe_allow_html=True,
                )
                for s in added_list:
                    prob_pct = int(s.get("avg_prob", 0) * 100)
                    worse_str  = ", ".join(s.get("worse",  [])) or "—"
                    better_str = ", ".join(s.get("better", [])) or "—"
                    st.markdown(
                        f'<div style="font-size:0.82rem;color:#2a5;margin-bottom:4px">'
                        f'<strong>+ {s["symptom"]}</strong> '
                        f'<span style="color:#aaa">[{prob_pct}%]</span><br>'
                        f'<span style="color:#888;font-size:0.77rem">↓ {worse_str} &nbsp;|&nbsp; ↑ {better_str}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown('</div>', unsafe_allow_html=True)

        # Detailed expanders
        with st.expander("Full base report details"):
            st.markdown(f'**Miasmatic analysis:** {base_rpt.get("miasmaticAnalysis","")}')
            if base_rpt.get("remedyRelationships"):
                st.markdown(f'**Remedy sequence:** {base_rpt["remedyRelationships"]}')
            mon = base_rpt.get("monitoringPoints", [])
            if mon:
                st.markdown("**Monitor:**")
                for pt in mon:
                    st.markdown(f"- {pt}")

        with st.expander("Full enhanced report details"):
            st.markdown(f'**Miasmatic analysis:** {enh_rpt.get("miasmaticAnalysis","")}')
            if enh_rpt.get("remedyRelationships"):
                st.markdown(f'**Remedy sequence:** {enh_rpt["remedyRelationships"]}')
            mon = enh_rpt.get("monitoringPoints", [])
            if mon:
                st.markdown("**Monitor:**")
                for pt in mon:
                    st.markdown(f"- {pt}")

        st.markdown("---")

        # PDF download (comparative)
        pid      = st.session_state.patient_id
        pd_det   = st.session_state.patient_details
        added_list = st.session_state.get("_future_added_list", [])

        try:
            pdf_bytes = build_pdf(
                pid, cat, base_rpt,
                patient_details=pd_det,
                report_enhanced=enh_rpt,
                added_symptoms=added_list,
                comparative_summary=comp_txt,
            )
            st.download_button(
                label="↓ Download Comparative PDF",
                data=pdf_bytes,
                file_name=f"HoRUS3_{pid}_comparative_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF error: {e}")

        st.markdown(
            '<p style="font-size:0.75rem;color:#bbb;margin-top:1rem;text-align:center">'
            "For clinical reference only. Prescribing decisions rest with the practitioner.</p>",
            unsafe_allow_html=True,
        )
