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
    .block-container { padding: 2.5rem 2rem 4rem; max-width: 780px; }
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECRETS — read once at startup
# ─────────────────────────────────────────────
def _secret(key: str, fallback: str = "") -> str:
    """Read from st.secrets; return fallback if missing (e.g. local dev)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return fallback

GEMINI_API_KEY   = _secret("GEMINI_API_KEY")
GITHUB_TOKEN     = _secret("GITHUB_TOKEN")
GITHUB_REPO      = _secret("GITHUB_REPO")       # "owner/repo"
GITHUB_FILE_PATH = _secret("GITHUB_FILE_PATH", "data/horus3_patients.json")

# Configure Gemini once at module load; no user input needed.
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_READY = bool(GEMINI_API_KEY)

# ─────────────────────────────────────────────
# GITHUB-BACKED PATIENT STORAGE
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _gh_repo():
    """Return a PyGithub Repository object, cached for the session."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None
    try:
        g = Github(GITHUB_TOKEN)
        return g.get_repo(GITHUB_REPO)
    except Exception:
        return None


def load_patients() -> dict:
    """
    Fetch horus3_patients.json from GitHub.
    Falls back to an empty dict if the file doesn't exist yet or on any error.
    """
    repo = _gh_repo()
    if repo is None:
        return {}
    try:
        contents = repo.get_contents(GITHUB_FILE_PATH)
        raw = base64.b64decode(contents.content).decode("utf-8")
        return json.loads(raw)
    except GithubException as e:
        if e.status == 404:
            return {}           # file doesn't exist yet — first run
        st.warning(f"GitHub read error: {e.data.get('message', e)}")
        return {}
    except Exception as e:
        st.warning(f"Could not load patient history: {e}")
        return {}


def save_patients(patients: dict) -> bool:
    """
    Write the full patients dict back to GitHub as a single JSON file.
    Creates the file on first save, updates (with the correct SHA) on subsequent saves.
    Returns True on success.
    """
    repo = _gh_repo()
    if repo is None:
        st.error("GitHub not configured — case not saved. Check GITHUB_TOKEN and GITHUB_REPO in secrets.")
        return False

    payload = json.dumps(patients, indent=2, ensure_ascii=False)
    commit_msg = f"HoRUS3 update — {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    try:
        try:
            existing = repo.get_contents(GITHUB_FILE_PATH)
            repo.update_file(
                path=GITHUB_FILE_PATH,
                message=commit_msg,
                content=payload,
                sha=existing.sha,
            )
        except GithubException as e:
            if e.status == 404:
                # First save — create the file (and any parent dirs implicitly)
                repo.create_file(
                    path=GITHUB_FILE_PATH,
                    message=commit_msg,
                    content=payload,
                )
            else:
                raise
        return True
    except Exception as e:
        st.error(f"GitHub write error: {e}")
        return False


def save_case(pid: str, case_data: dict):
    """Append one case to the patient record and push to GitHub."""
    patients = load_patients()
    patients.setdefault(pid, [])
    patients[pid].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **case_data,
    })
    save_patients(patients)


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
# ML SYSTEM (pkl files — optional)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_system():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    s2r = {}
    chapters = defaultdict(list)
    for fname in ["case_studies_model.pkl", "rheumatic_model.pkl"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                d = pickle.load(f)
            for sym, chap in d.get("categories", {}).items():
                chapters[chap].append(sym)
            s2r.update(d.get("symptom_to_remedies", {}))
    clusters = {}
    for name in ["remedy_modalities", "remedy_area_modalities", "remedy_area"]:
        fpath = f"clusters_{name}.csv"
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            df["Cluster_ID"] = df["Cluster_ID"].astype(str)
            clusters[name] = df
    return {"model": model, "s2r": s2r, "chapters": dict(chapters), "clusters": clusters}


system_data = None
try:
    system_data = load_system()
except Exception:
    pass  # Gemini covers the core clinical logic even without pkl files

# ─────────────────────────────────────────────
# GEMINI HELPERS
# ─────────────────────────────────────────────
def gemini_categorise(raw_text: str) -> dict:
    system = """You are a classical homeopathic repertorisation assistant.
Read the patient's description and return ONLY valid JSON — no markdown, no preamble.

Rules:
- Split symptoms into physical, psychological, and general categories.
- For each symptom extract worse[] and better[] modalities.
- Keep symptom text concise but clinically precise.
- Identify the dominant miasmatic tendency (psoric / sycotic / syphilitic / tubercular / mixed).
- List any concomitants (symptoms that appear together).
- Write a one-sentence clinical_summary capturing the case essence.

JSON shape (return exactly this, nothing else):
{
  "physical": [{"symptom": "...", "worse": ["..."], "better": ["..."]}],
  "psychological": [{"symptom": "...", "worse": ["..."], "better": ["..."]}],
  "general": [{"symptom": "...", "worse": ["..."], "better": ["..."]}],
  "miasm": "...",
  "concomitants": ["..."],
  "clinical_summary": "..."
}"""
    model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite", system_instruction=system)
    response = model.generate_content(
        f"Patient describes: {raw_text}",
        generation_config=genai.GenerationConfig(temperature=0.2, max_output_tokens=3000),
    )
    raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(raw)


def gemini_report(patient_id: str, categorised: dict) -> dict:
    lines = []
    for cat in ("physical", "psychological", "general"):
        for s in categorised.get(cat, []):
            worse  = ", ".join(s.get("worse",  [])) or "—"
            better = ", ".join(s.get("better", [])) or "—"
            lines.append(f"[{cat.title()}] {s['symptom']} | ↓ {worse} | ↑ {better}")

    system = """You are an experienced classical homeopath writing a comprehensive treatment plan.

Rules:
- Never suggest dosages, potencies, or repetition schedules — only remedy names.
- Justify every remedy with specific symptoms from this case.
- Be clinically precise. Reference the totality, miasm, and modalities.
- Return ONLY valid JSON, no markdown, no preamble.

JSON shape (return exactly this):
{
  "primaryRemedy": {
    "name": "...",
    "why": "Detailed paragraph — mind, body, generals, modalities from this patient's symptoms.",
    "keyIndications": ["specific symptom → remedy keynote", "..."],
    "followedBy": ["Remedy A", "Remedy B"]
  },
  "secondaryRemedies": [
    {"name": "...", "role": "complementary|intercurrent|acute|anti-miasmatic",
     "rationale": "Why this remedy fits this case."}
  ],
  "miasmaticAnalysis": "Paragraph on miasmatic background and how it shapes the prescription.",
  "caseEssence": "The fundamental disturbance — strange, rare, peculiar features pointing to the simillimum.",
  "remedyRelationships": "Planned remedy sequence — what follows what, what antidotes what, and why.",
  "monitoringPoints": ["Observable sign to watch", "..."]
}"""

    prompt = (
        f"Patient: {patient_id}\n"
        f"Miasm: {categorised.get('miasm','unknown')}\n"
        f"Summary: {categorised.get('clinical_summary','')}\n"
        f"Concomitants: {', '.join(categorised.get('concomitants',[])) or 'none'}\n\n"
        f"Symptoms:\n" + "\n".join(lines)
    )
    model = genai.GenerativeModel(model_name="gemini-2.5-flash-lite", system_instruction=system)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=4000),
    )
    raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(raw)


# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def build_pdf(patient_id: str, categorised: dict, report: dict) -> bytes:
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

    elems = [
        Paragraph("HoRUS 3 — Treatment Plan", title_s),
        Paragraph(f"Patient {patient_id} &nbsp;·&nbsp; {datetime.now().strftime('%d %B %Y')}", sub_s),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddd"), spaceAfter=14),
        Paragraph("CASE ESSENCE", label_s),
        Paragraph(report.get("caseEssence", ""), body_s),
        Paragraph("MIASMATIC PICTURE", label_s),
        Paragraph(report.get("miasmaticAnalysis", ""), body_s),
    ]

    # Symptom table
    elems.append(Paragraph("SYMPTOM SUMMARY", label_s))
    rows = [["Category", "Symptom", "↓ Worse", "↑ Better"]]
    for cat in ("physical", "psychological", "general"):
        for s in categorised.get(cat, []):
            rows.append([
                cat.title(), s["symptom"][:60],
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
        elems += [t, Spacer(1, 14)]

    # Primary remedy
    pr = report.get("primaryRemedy", {})
    elems += [Paragraph("SIMILLIMUM", label_s), Paragraph(pr.get("name",""), rem_s), Paragraph(pr.get("why",""), body_s)]
    if pr.get("keyIndications"):
        elems.append(Paragraph("Key indications in this case:", small_s))
        for ind in pr["keyIndications"]:
            elems.append(Paragraph(f"• {ind}", bullet_s))
    if pr.get("followedBy"):
        elems.append(Paragraph(f"Followed well by: {', '.join(pr['followedBy'])}", small_s))
    elems.append(Spacer(1, 10))

    # Secondary remedies
    for rem in report.get("secondaryRemedies", []):
        elems += [
            Paragraph("SUPPORTING REMEDIES", label_s),
            Paragraph(f"{rem['name']}  <font size='8' color='#888'>[{rem.get('role','').upper()}]</font>", rem_s),
            Paragraph(rem.get("rationale",""), body_s),
        ]

    if report.get("remedyRelationships"):
        elems += [Paragraph("REMEDY SEQUENCE", label_s), Paragraph(report["remedyRelationships"], body_s)]

    for pt in report.get("monitoringPoints", []):
        elems.append(Paragraph(f"• {pt}", bullet_s))

    elems += [
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=0.3, color=colors.HexColor("#ddd"), spaceAfter=6),
        Paragraph("For clinical reference only. Prescribing decisions rest with the practitioner.", foot_s),
    ]
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


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
defaults = {
    "step": "intake",
    "patient_id": "",
    "raw_symptoms": "",
    "categorised": None,
    "report": None,
    "case_saved": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.patient_id:
    st.session_state.patient_id = next_patient_id()

# ─────────────────────────────────────────────
# STARTUP CHECKS — shown only when secrets missing
# ─────────────────────────────────────────────
missing = []
if not GEMINI_API_KEY:   missing.append("`GEMINI_API_KEY`")
if not GITHUB_TOKEN:     missing.append("`GITHUB_TOKEN`")
if not GITHUB_REPO:      missing.append("`GITHUB_REPO`")

if missing:
    st.error(
        f"**Missing secrets:** {', '.join(missing)}  \n"
        "Go to **Streamlit Cloud → your app → Settings → Secrets** and add them. "
        "See the `secrets.toml.example` in the repo for the exact format."
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

# ─────────────────────────────────────────────
# SIDEBAR — patient management + history
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Patient")
    new_pid = st.text_input("Patient ID", value=st.session_state.patient_id)
    if new_pid.strip().upper() != st.session_state.patient_id:
        st.session_state.patient_id = new_pid.strip().upper()
    if st.button("↺  Generate new ID"):
        st.session_state.patient_id = next_patient_id()
        st.rerun()

    st.divider()

    st.markdown("### History")
    patients = load_patients()
    if patients:
        pid_sel = st.selectbox("Load patient", ["—"] + sorted(patients.keys()), key="hist_sel")
        if pid_sel and pid_sel != "—":
            cases = patients[pid_sel]
            st.markdown(f"**{len(cases)} case(s)**")
            for c in reversed(cases):
                with st.expander(c["timestamp"]):
                    for i, r in enumerate(c.get("top_remedies", [])[:5], 1):
                        st.markdown(f"{i}. **{r}**")
                    if c.get("clinical_notes"):
                        st.caption(c["clinical_notes"])
    else:
        st.caption("No history yet.")

    st.divider()
    if st.button("↺  Start new case", use_container_width=True):
        for k in ("step", "raw_symptoms", "categorised", "report", "case_saved"):
            st.session_state[k] = defaults[k]
        st.session_state.step = "intake"
        st.session_state.patient_id = next_patient_id()
        st.rerun()


# ─────────────────────────────────────────────
# STEP 1 — INTAKE
# ─────────────────────────────────────────────
if st.session_state.step == "intake":
    st.markdown(
        f'<div class="patient-badge">Patient {st.session_state.patient_id}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Describe the patient's symptoms")
    st.markdown(
        '<p style="font-size:0.85rem;color:#777;margin-top:-0.5rem;margin-bottom:1rem">'
        "Write freely — physical complaints, mental state, what makes things worse or better, "
        "time of day, thermal preferences, sleep, appetite, any relevant details.</p>",
        unsafe_allow_html=True,
    )

    raw = st.text_area(
        label="Symptoms",
        value=st.session_state.raw_symptoms,
        height=220,
        placeholder=(
            "Example: Sharp throbbing headache above the left eye for 3 days, "
            "worse in cold air and any touch, better lying still in a dark room. "
            "Very anxious, restless at night, fear of being alone. "
            "Profuse sweating during sleep, thirsty for cold water in large sips. "
            "Burning sensation in stomach after meals, better from warm drinks. "
            "Generally chilly but cannot tolerate a stuffy room..."
        ),
        label_visibility="collapsed",
    )
    st.session_state.raw_symptoms = raw

    st.markdown(
        '<div class="info-box">The AI will categorise these symptoms into physical, '
        "psychological, and general groups with their modalities. "
        "You can review and edit the result before generating the treatment plan.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Analyse symptoms →", type="primary", disabled=not raw.strip(), use_container_width=True):
        with st.spinner("Parsing symptoms…"):
            try:
                result = gemini_categorise(raw)
                st.session_state.categorised = result
                st.session_state.step = "analysis"
                st.rerun()
            except json.JSONDecodeError as e:
                st.error(f"Could not parse AI response — try rephrasing. ({e})")
            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# STEP 2 — CATEGORISATION REVIEW
# ─────────────────────────────────────────────
elif st.session_state.step == "analysis":
    cat = st.session_state.categorised

    st.markdown(
        f'<div class="patient-badge">Patient {st.session_state.patient_id}</div>',
        unsafe_allow_html=True,
    )

    if cat.get("clinical_summary"):
        st.markdown(
            f'<div class="info-box" style="border-left:3px solid #111;padding-left:1rem">'
            f'<strong>Case essence</strong><br>{cat["clinical_summary"]}</div>',
            unsafe_allow_html=True,
        )

    meta_cols = st.columns(2)
    with meta_cols[0]:
        if cat.get("miasm"):
            st.markdown(
                f'<span class="sym-tag" style="background:#f0f0f0">Miasm: {cat["miasm"]}</span>',
                unsafe_allow_html=True,
            )
    with meta_cols[1]:
        if cat.get("concomitants"):
            st.markdown(
                f'<span style="font-size:0.8rem;color:#777">'
                f'Concomitants: {", ".join(cat["concomitants"])}</span>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)
    render_symptom_category("Physical",      cat.get("physical",      []))
    render_symptom_category("Psychological", cat.get("psychological", []))
    render_symptom_category("General",       cat.get("general",       []))

    with st.expander("✎  Edit categorised symptoms (optional)"):
        st.caption("Edit symptom text or modalities directly.")
        for cat_key, cat_label in [
            ("physical", "Physical"), ("psychological", "Psychological"), ("general", "General")
        ]:
            st.markdown(f"**{cat_label}**")
            edited = []
            for idx, s in enumerate(cat.get(cat_key, [])):
                c1, c2, c3 = st.columns([3, 2, 2])
                sym = c1.text_input("Symptom", value=s["symptom"],
                                    key=f"edit_{cat_key}_{idx}_sym", label_visibility="collapsed")
                w   = c2.text_input("↓ Worse",  value=", ".join(s.get("worse",  [])),
                                    key=f"edit_{cat_key}_{idx}_w",   label_visibility="collapsed")
                b   = c3.text_input("↑ Better", value=", ".join(s.get("better", [])),
                                    key=f"edit_{cat_key}_{idx}_b",   label_visibility="collapsed")
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
                    rpt = gemini_report(st.session_state.patient_id, st.session_state.categorised)
                    st.session_state.report = rpt
                    st.session_state.step = "report"
                    st.session_state.case_saved = False
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Could not parse report response. ({e})")
                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────
# STEP 3 — TREATMENT PLAN
# ─────────────────────────────────────────────
elif st.session_state.step == "report":
    rpt = st.session_state.report
    cat = st.session_state.categorised
    pid = st.session_state.patient_id

    if not rpt:
        st.error("No report found — please go back.")
        st.stop()

    # Save once to GitHub
    if not st.session_state.case_saved:
        top_remedies = [rpt["primaryRemedy"]["name"]] + [
            r["name"] for r in rpt.get("secondaryRemedies", [])
        ]
        with st.spinner("Saving case to GitHub…"):
            save_case(pid, {
                "raw_symptoms":  st.session_state.raw_symptoms,
                "categorised":   cat,
                "top_remedies":  top_remedies,
                "clinical_notes": rpt.get("caseEssence", ""),
            })
        st.session_state.case_saved = True

    st.markdown(
        f'<div class="patient-badge">Patient {pid} &nbsp;·&nbsp; '
        f'{datetime.now().strftime("%d %B %Y")}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Treatment Plan")

    if rpt.get("caseEssence"):
        st.markdown(
            f'<div class="info-box" style="border-left:3px solid #111;padding-left:1rem">'
            f'<strong>Case essence</strong><br>{rpt["caseEssence"]}</div>',
            unsafe_allow_html=True,
        )

    if rpt.get("miasmaticAnalysis"):
        st.markdown('<div class="section-label">Miasmatic picture</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.9rem;color:#333;line-height:1.7">{rpt["miasmaticAnalysis"]}</p>',
            unsafe_allow_html=True,
        )

    pr = rpt.get("primaryRemedy", {})
    st.markdown('<div class="section-label">Simillimum</div>', unsafe_allow_html=True)

    followed_html = ""
    if pr.get("followedBy"):
        followed_html = f'<div class="remedy-meta">Followed well by: {", ".join(pr["followedBy"])}</div>'

    indications_html = ""
    if pr.get("keyIndications"):
        items = "".join(f"<li>{i}</li>" for i in pr["keyIndications"])
        indications_html = (
            f'<div class="remedy-meta" style="margin-top:0.6rem">'
            f'<strong>Key indications in this case</strong>'
            f'<ul style="margin:0.3rem 0 0 1rem;padding:0;font-size:0.85rem;color:#444">{items}</ul>'
            f'</div>'
        )

    st.markdown(
        f'<div class="remedy-card remedy-primary">'
        f'<div class="remedy-name">{pr.get("name","")}</div>'
        f'<div class="remedy-rationale">{pr.get("why","")}</div>'
        f'{indications_html}{followed_html}</div>',
        unsafe_allow_html=True,
    )

    secs = rpt.get("secondaryRemedies", [])
    if secs:
        st.markdown('<div class="section-label">Supporting remedies</div>', unsafe_allow_html=True)
        for rem in secs:
            role_badge = f'<span class="remedy-role">{rem.get("role","")}</span>'
            st.markdown(
                f'<div class="remedy-card">'
                f'<div class="remedy-name">{rem["name"]}{role_badge}</div>'
                f'<div class="remedy-rationale">{rem.get("rationale","")}</div></div>',
                unsafe_allow_html=True,
            )

    if rpt.get("remedyRelationships"):
        st.markdown('<div class="section-label">Remedy sequence & relationships</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.9rem;color:#333;line-height:1.7">{rpt["remedyRelationships"]}</p>',
            unsafe_allow_html=True,
        )

    mon = rpt.get("monitoringPoints", [])
    if mon:
        st.markdown('<div class="section-label">What to monitor</div>', unsafe_allow_html=True)
        items_html = "".join(f'<div class="monitor-item">• {pt}</div>' for pt in mon)
        st.markdown(f'<div class="info-box" style="padding:0.6rem 1rem">{items_html}</div>',
                    unsafe_allow_html=True)

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
            pdf_bytes = build_pdf(pid, cat, rpt)
            st.download_button(
                label="↓ Download PDF",
                data=pdf_bytes,
                file_name=f"HoRUS3_{pid}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF error: {e}")

    with act3:
        if st.button("↺  New case", use_container_width=True):
            for k in ("step", "raw_symptoms", "categorised", "report", "case_saved"):
                st.session_state[k] = defaults[k]
            st.session_state.step = "intake"
            st.session_state.patient_id = next_patient_id()
            st.rerun()

    st.markdown(
        '<p style="font-size:0.75rem;color:#bbb;margin-top:1rem;text-align:center">'
        "For clinical reference only. Prescribing decisions rest with the practitioner.</p>",
        unsafe_allow_html=True,
    )
