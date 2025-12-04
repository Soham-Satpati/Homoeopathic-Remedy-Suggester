# HORUS3_PERFECT_FINAL_WORKFLOW.py
# THE ULTIMATE CLINICAL VERSION — EXACT WORKFLOW YOU WANTED
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
from collections import defaultdict
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import json
from datetime import datetime

st.set_page_config(page_title="HoRUS 3 — Final Clinical Edition", layout="wide")

# =============================================
# DATA & CACHING
# =============================================
MODALITIES = {
    "worse": ["cold", "damp", "motion", "night", "touch", "pressure", "rest", "heat", "lying", "standing", "sitting", "warm"],
    "better": ["motion", "warmth", "rest", "pressure", "open air", "rubbing", "lying", "cold", "warm applications", "walking", "sitting"]
}

@st.cache_resource
def load_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open("case_studies_model.pkl", "rb") as f:
        case_dict = pickle.load(f)
    with open("rheumatic_model.pkl", "rb") as f:
        rhe_dict = pickle.load(f)
    
    clusters = {}
    for name in ["remedy_modalities", "remedy_area_modalities", "remedy_area"]:
        file = f"clusters_{name}.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['Cluster_ID'] = df['Cluster_ID'].astype(str)
            clusters[name] = df

    chapters = defaultdict(list)
    for data in [case_dict, rhe_dict]:
        for sym, chap in data.get('categories', {}).items():
            chapters[chap].append(sym)
    
    s2r = {**case_dict.get('symptom_to_remedies', {}), **rhe_dict.get('symptom_to_remedies', {})}
    
    return {
        'model': model,
        'chapters': dict(chapters),
        'clusters': clusters,
        's2r': s2r,
        'rhe_dict': rhe_dict
    }

data = load_system()
model = data['model']
chapters = data['chapters']
clusters = data['clusters']
s2r_global = data['s2r']
rhe_dict = data['rhe_dict']

# Pre-encode all chapters for speed
@st.cache_resource
def precompute_embeddings():
    emb = {}
    for chap, syms in chapters.items():
        if len(syms) >= 5:
            emb[chap] = model.encode(syms, convert_to_tensor=True, show_progress_bar=False)
    return emb

chapter_emb = precompute_embeddings()

# =============================================
# SESSION & HISTORY
# =============================================
HISTORY_FILE = "horus3_patients.json"
def load_patients():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump({}, f, indent=2)
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_patient_once(pid, data):
    patients = load_patients()
    patients.setdefault(pid, [])
    patients[pid].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), **data})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump(patients, f, indent=2)

if 'initialized' not in st.session_state:
    st.session_state.update({
        "step": 1,
        "physical": [], "psychological": [], "generals": [],
        "refined": {},  # original → rubric
        "modalities": defaultdict(lambda: {"worse": [], "better": []}),
        "selected_pattern_symptoms": [],
        "rhe_weight": 0.5, "case_weight": 0.5,
        "patient_id": "", "patient_mode": "new",
        "report_generated": False,
        "initialized": True
    })

tab1, tab2 = st.tabs(["Patient Case", "Patient History"])

with tab1:
    st.title("HoRUS 3")
    st.markdown("### **True Clinical Intelligence — Final Edition**")

    # =============================================
    # PATIENT
    # =============================================
    st.header("Patient")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("New Patient", type="primary", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k != "initialized": del st.session_state[k]
            st.rerun()
    with c2:
        if st.button("Former Patient", type="secondary", use_container_width=True):
            st.session_state.patient_mode = "former"
            st.rerun()

    if st.session_state.patient_mode == "new":
        pid = st.text_input("Patient ID", placeholder="PT-2025-001")
        if pid: st.session_state.patient_id = pid.strip().upper()
    else:
        sel = st.selectbox("Select", [""] + sorted(load_patients().keys()))
        if sel: st.session_state.patient_id = sel

    if not st.session_state.patient_id: st.stop()
    st.success(f"Active: **{st.session_state.patient_id}**")

    # Dual sliders
    st.sidebar.header("Dataset Balance")
    cA, cB = st.sidebar.columns(2)
    with cA:
        rhe = st.slider("Rheumatic", 0.0, 1.0, st.session_state.rhe_weight, 0.05)
    with cB:
        case = st.slider("Case Studies", 0.0, 1.0, 1.0 - rhe, 0.05)
    st.session_state.rhe_weight = rhe
    st.session_state.case_weight = case

    # =============================================
    # STEP 1: ENTER SYMPTOMS
    # =============================================
    st.header("1. Enter Patient Symptoms")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Physical")
        phys = st.text_area("e.g. knee pain, headache", height=130, key="phys_in")
    with c2:
        st.subheader("Psychological")
        psych = st.text_area("e.g. anxiety, fear", height=130, key="psych_in")
    with c3:
        st.subheader("Generals")
        gen = st.text_area("e.g. worse cold", height=130, key="gen_in")

    if st.button("Analyze Symptoms", type="primary", use_container_width=True):
        p = [s.strip() for s in phys.replace('\n', ',').split(',') if s.strip()]
        ps = [s.strip() for s in psych.replace('\n', ',').split(',') if s.strip()]
        g = [s.strip() for s in gen.replace('\n', ',').split(',') if s.strip()]
        if not (p or ps or g):
            st.error("Enter at least one symptom")
            st.stop()
        st.session_state.physical = p
        st.session_state.psychological = ps
        st.session_state.generals = g
        st.session_state.step = 1.5  # After Step 1
        st.session_state.refined = {}
        st.session_state.selected_pattern_symptoms = []
        st.rerun()

    if st.session_state.step < 1.5: st.stop()

    # =============================================
    # AFTER STEP 1 — THREE DIRECT PATHS
    # =============================================
    st.success("Symptoms recorded. Choose your path:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Refine Rubrics\n(Step 2)", type="secondary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Discover Hidden Patterns\n(Step 3)", type="secondary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("Generate Report Now\n(Step 4)", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.session_state.selected_keywords = (
                st.session_state.physical + 
                st.session_state.psychological + 
                st.session_state.generals
            )
            st.rerun()

    # =============================================
    # STEP 2: REFINEMENT
    # =============================================
    if st.session_state.step == 2:
        st.header("2. Refine Symptoms (Optional)")
        all_syms = (
            [("Physical", s) for s in st.session_state.physical] +
            [("Psychological", s) for s in st.session_state.psychological] +
            [("Generals", s) for s in st.session_state.generals]
        )

        refined_count = 0
        for cat, sym in all_syms:
            st.subheader(cat)
            with st.expander(f"**{sym}**", expanded=True):
                sym_emb = model.encode(sym, convert_to_tensor=True)
                chapter_options = ["Keep original"]
                match_counts = {}
                for chap, emb_tensor in chapter_emb.items():
                    if emb_tensor.shape[0] == 0: continue
                    scores = util.cos_sim(sym_emb, emb_tensor)[0]
                    good = sum(1 for s in scores if s > 0.58)
                    if good > 0:
                        match_counts[chap] = good
                        chapter_options.append(f"{chap.title()} ({good} matches)")

                chosen_chap = st.selectbox("Chapter", chapter_options, key=f"chap_{sym}")
                if "Keep original" not in chosen_chap:
                    chap_name = chosen_chap.split(" (")[0].lower()
                    scores = util.cos_sim(sym_emb, chapter_emb[chap_name])[0]
                    topk = scores.topk(min(12, len(scores)))
                    rubrics = [chapters[chap_name][i] for i in topk.indices]
                    scores_list = topk.values.tolist()
                    chosen_rubric = st.selectbox("Rubric", 
                        [f"{r} ({s:.2f})" for r, s in zip(rubrics, scores_list)], 
                        key=f"rub_{sym}")
                    rubric = chosen_rubric.split(" (")[0]
                    st.session_state.refined[sym] = rubric
                    refined_count += 1

                    with st.expander("Modalities", expanded=True):
                        c1, c2 = st.columns(2)
                        worse = c1.multiselect("Worse", MODALITIES["worse"], key=f"w_{sym}")
                        better = c2.multiselect("Better", MODALITIES["better"], key=f"b_{sym}")
                        if worse or better:
                            st.session_state.modalities[rubric] = {"worse": worse, "better": better}

        core = [st.session_state.refined.get(s, s) for _, s in all_syms]
        st.session_state.selected_keywords = core
        st.success(f"{len(core)} symptoms ready ({refined_count} refined)")

        st.markdown("---")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Discover Hidden Patterns (Step 3)", type="secondary", use_container_width=True):
                st.session_state.step = 3
                st.rerun()
        with colB:
            if st.button("Generate Final Report", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()

    # =============================================
    # STEP 3: HIDDEN PATTERNS
    # =============================================
    if st.session_state.step == 3:
        st.header("3. Discover Hidden Clinical Patterns")
        core = st.session_state.selected_keywords or (
            st.session_state.physical + st.session_state.psychological + st.session_state.generals
        )
        rarity = st.radio("Pattern Type", ["Common", "Uncommon", "Rare"], horizontal=True)
        min_size = {"Common": 25, "Uncommon": 12, "Rare": 0}[rarity]

        found = []
        seen = set()
        for _, df in clusters.items():
            for kw in core:
                rows = df[df['Symptom'].str.contains(kw, case=False, na=False)]
                for _, row in rows.iterrows():
                    cid = str(row['Cluster_ID'])
                    if cid in {'NOISE', '-1'} or cid in seen: continue
                    cluster = df[df['Cluster_ID'] == cid]
                    if len(cluster) >= min_size or rarity == "Rare":
                        seen.add(cid)
                        symptoms = [s for s in cluster['Symptom'].tolist() if s not in core]
                        if symptoms:
                            found.append(symptoms)

        added = []
        if found:
            st.write(f"**{len(found)} clinical pattern(s) found**")
            for i, cluster in enumerate(found[:7]):
                with st.expander(f"Pattern {i+1} — {len(cluster)} symptoms"):
                    chosen = st.multiselect("Add", cluster, key=f"p{i}")
                    added.extend(chosen)

        if st.button("Include Patterns & Generate Report", type="primary", use_container_width=True):
            st.session_state.selected_pattern_symptoms = list(set(added))
            st.session_state.step = 4
            st.rerun()

    # =============================================
    # STEP 4: FINAL REPORT
    # =============================================
    if st.session_state.step >= 4:
        st.header("Final Clinical Report")
        core = st.session_state.selected_keywords or (
            st.session_state.physical + st.session_state.psychological + st.session_state.generals
        )
        pattern = st.session_state.selected_pattern_symptoms
        only_core = len(pattern) == 0

        # Expansion
        expanded = set()
        for c in core:
            emb = model.encode(c, convert_to_tensor=True)
            for syms in chapters.values():
                if len(syms) < 10: continue
                batch = model.encode(syms, convert_to_tensor=True)
                scores = util.cos_sim(emb, batch)[0]
                expanded.update(sym for sc, sym in zip(scores.tolist(), syms) if sc > 0.65)

        # Coverage
        coverage = defaultdict(int)
        for remedy in {r for v in s2r_global.values() for r in v}:
            for c in core:
                if c in s2r_global and remedy in s2r_global[c]:
                    coverage[remedy] += 1
                else:
                    for exp in expanded:
                        if exp in s2r_global and remedy in s2r_global[exp]:
                            if any(w in exp.lower() for w in c.lower().split() if len(w)>2):
                                coverage[remedy] += 1
                                break
        top10 = sorted(coverage.items(), key=lambda x: x[1], reverse=True)[:10]

        st.subheader("Top 10 Remedies — True Clinical Coverage")
        for i, (rem, cov) in enumerate(top10, 1):
            with st.expander(f"**{i}. {rem.upper()}** — Covers {cov}/{len(core)} symptoms", expanded=(i<=3)):
                st.metric("Coverage", f"{cov}/{len(core)}")
                for c in core:
                    covered = any(rem in s2r_global.get(e, {}) for e in [c] + list(expanded) 
                                if any(w in e.lower() for w in c.lower().split() if len(w)>2))
                    st.write(f"{'Checkmark' if covered else 'Crossmark'} {c}")

        st.divider()
        st.subheader("Per-Symptom Analysis")
        for sym in core:
            with st.expander(f"**{sym}**"):
                hits = []
                for rem, _ in top10:
                    if sym in s2r_global and rem in s2r_global[sym]:
                        hits.append((rem.upper(), s2r_global[sym][rem], "Direct"))
                    else:
                        for exp in expanded:
                            if exp in s2r_global and rem in s2r_global[exp]:
                                if any(w in exp.lower() for w in sym.lower().split() if len(w)>2):
                                    hits.append((rem.upper(), s2r_global[exp].get(rem, 0), "AI Match"))
                                    break
                for r, sc, typ in sorted(hits, key=lambda x: x[1], reverse=True)[:7]:
                    st.write(f"**{r}** → {sc:.2f} — *{typ}*")

        if not st.session_state.report_generated:
            save_patient_once(st.session_state.patient_id, {
                "core": core, "pattern": pattern, "top10": top10, "mode": "coverage"
            })
            st.session_state.report_generated = True
            st.success("Case saved")

        def pdf():
            buf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            doc = SimpleDocTemplate(buf.name, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = [Paragraph("HoRUS 3 Report", styles['Title']),
                        Paragraph(f"Patient: {st.session_state.patient_id}", styles['Normal']),
                        Paragraph(f"Date: {datetime.now().strftime('%d %B %Y')}", styles['Normal']),
                        Spacer(1, 20)]
            for i, (r, v) in enumerate(top10, 1):
                elements.append(Paragraph(f"{i}. {r.upper()} — {v}/{len(core)}", styles['Normal']))
            doc.build(elements)
            return buf.name

        with open(pdf(), "rb") as f:
            st.download_button("Download PDF", f.read(), f"HoRUS3_{st.session_state.patient_id}.pdf", "application/pdf")

# History Tab
with tab2:
    st.header("Patient History")
    patients = load_patients()
    if patients:
        pid = st.selectbox("Select", [""] + sorted(patients.keys()))
        if pid:
            for case in reversed(patients[pid][-15:]):
                with st.expander(f"{case['timestamp']}"):
                    for i, (r, v) in enumerate(case.get("top10", [])[:5]):
                        st.write(f"**{i+1}. {r.upper()}** — {v}/{len(case.get('core',[]))}")
