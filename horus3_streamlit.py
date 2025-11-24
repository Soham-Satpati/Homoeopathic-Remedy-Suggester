# HORUS3_PURE.py
# FINAL — NO "RUBRICS" — ONLY SYMPTOMS
# HoRUS 3: The Remedy Suggester

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile
import json
from datetime import datetime

st.set_page_config(page_title="HoRUS 3", layout="wide")

# === LOAD DATA ===
@st.cache_resource
def load_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    with open("case_studies_model.pkl", "rb") as f:
        case = pickle.load(f)
    with open("rheumatic_model.pkl", "rb") as f:
        rhe = pickle.load(f)
    
    clusters = {}
    for name in ["remedy_modalities", "remedy_area_modalities"]:
        file = f"clusters_{name}.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            df['Cluster_ID'] = df['Cluster_ID'].astype(str)
            clusters[name] = df
    
    chapters = defaultdict(list)
    for data in [case, rhe]:
        cats = data.get('categories', {})
        for sym, chap in cats.items():
            chapters[chap].append(sym)
    
    s2r = {**case.get('symptom_to_remedies', {}), **rhe.get('symptom_to_remedies', {})}
    
    return {
        'model': model,
        'case': case,
        'rhe': rhe,
        'clusters': clusters,
        'chapters': dict(chapters),
        's2r': s2r
    }

data = load_system()
model = data['model']
chapters = data['chapters']
clusters = data['clusters']
s2r_global = data['s2r']

# === MODALITIES ===
MODALITIES = {
    "worse": ["cold", "damp", "motion", "night", "touch", "pressure", "rest", "heat", "lying"],
    "better": ["motion", "warmth", "rest", "pressure", "open air", "rubbing", "lying", "cold"]
}

# === HISTORY FILE ===
HISTORY_FILE = "horus3_history.json"
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)

def save_case_to_history(case_data):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **case_data
    })
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# === SESSION STATE ===
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.symptoms = []
    st.session_state.selected_keywords = []
    st.session_state.selected_pattern_symptoms = []
    st.session_state.selected_modalities = defaultdict(lambda: {"worse": [], "better": []})
    st.session_state.rhe_weight = 0.5
    st.session_state.case_weight = 0.5
    st.session_state._last_slider = None

# === TABS ===
tab1, tab2 = st.tabs(["New Case", "History"])

# === TAB 1: NEW CASE ===
with tab1:
    # === SIDEBAR: BOTH SLIDERS INTERACTIVE & AUTO-LINKED ===
    st.sidebar.header("Dataset Weights")
    rhe_slider = st.sidebar.slider(
        "Rheumatic", 0.0, 1.0, st.session_state.rhe_weight, 0.05, key="rhe_interactive"
    )
    case_slider = st.sidebar.slider(
        "Case Studies", 0.0, 1.0, st.session_state.case_weight, 0.05, key="case_interactive"
    )

    # Auto-sync
    if st.session_state._last_slider != "case" and rhe_slider != st.session_state.rhe_weight:
        new_case = max(0.0, min(1.0, 1.0 - rhe_slider))
        st.session_state.rhe_weight = rhe_slider
        st.session_state.case_weight = new_case
        st.session_state._last_slider = "rhe"
        st.rerun()

    elif st.session_state._last_slider != "rhe" and case_slider != st.session_state.case_weight:
        new_rhe = max(0.0, min(1.0, 1.0 - case_slider))
        st.session_state.case_weight = case_slider
        st.session_state.rhe_weight = new_rhe
        st.session_state._last_slider = "case"
        st.rerun()

    rhe_weight = st.session_state.rhe_weight
    case_weight = st.session_state.case_weight

    st.title("HoRUS 3")
    st.markdown("**AI-Powered Remedy Suggester**")
    st.header("1. Patient Case")
    case_text = st.text_area("Enter symptoms", height=140, placeholder="joint pain worse cold, swelling knees, better motion")

    if st.button("Analyze"):
        syms = [s.strip() for s in case_text.replace('\n', ',').split(',') if s.strip()]
        st.session_state.symptoms = syms
        st.session_state.step = 2
        st.session_state.selected_keywords = []
        st.session_state.selected_pattern_symptoms = []
        st.session_state.selected_modalities.clear()
        st.rerun()

    if st.session_state.step < 2:
        st.stop()

    # === 2. KEYWORD-BASED SUGGESTIONS (SYMPTOMS ONLY) ===
    st.header("2. Keyword-Based Symptom Suggestions")
    st.markdown("**AI finds the most relevant symptoms from your input**")

    chapter_matches = {}
    with st.spinner("Matching symptoms..."):
        for chapter, symptoms in chapters.items():
            if len(symptoms) < 5: continue
            emb_r = model.encode(symptoms, convert_to_tensor=True)
            hits = []
            seen = set()
            for sym in st.session_state.symptoms:
                emb_s = model.encode(sym, convert_to_tensor=True)
                scores = util.cos_sim(emb_s, emb_r)[0]
                top = scores.topk(min(10, len(scores)))
                for sc, idx in zip(top.values.tolist(), top.indices.tolist()):
                    if sc > 0.57:
                        r = symptoms[idx]
                        if r not in seen:
                            hits.append((r, sc))
                            seen.add(r)
            if hits:
                chapter_matches[chapter] = sorted(hits, key=lambda x: x[1], reverse=True)[:15]

    selected_keywords = []
    cols = st.columns(3)
    for idx, (chap, items) in enumerate(chapter_matches.items()):
        with cols[idx % 3]:
            st.subheader(chap)
            options = ["—"] + [f"{r} ({s:.2f})" for r, s in items]
            choice = st.selectbox("Select", options, key=f"sel_{chap}_{idx}")
            if choice != "—":
                symptom = choice.split(" (")[0]
                selected_keywords.append(symptom)
                with st.expander(f"Modalities for **{symptom}**"):
                    c1, c2 = st.columns(2)
                    with c1:
                        worse = st.multiselect("Worse", MODALITIES["worse"], key=f"w_{symptom}_{idx}")
                    with c2:
                        better = st.multiselect("Better", MODALITIES["better"], key=f"b_{symptom}_{idx}")
                    st.session_state.selected_modalities[symptom] = {"worse": worse, "better": better}

    if selected_keywords:
        st.session_state.selected_keywords = list(dict.fromkeys(selected_keywords))
        st.success(f"Selected {len(st.session_state.selected_keywords)} symptoms")

        if st.button("Next: Pattern Selection"):
            st.session_state.step = 3
            st.rerun()

    # === 3. PATTERN SELECTION + MODALITIES ===
    if st.session_state.step >= 3:
        st.header("3. Select Pattern Type → Add Symptoms + Modalities")
        pattern_type = st.radio("Pattern:", ["Common", "Uncommon", "Rare"], horizontal=True)

        size_map = {"Common": 25, "Uncommon": 12, "Rare": 0}
        min_size = size_map[pattern_type]

        pattern_clusters = []
        seen = set()

        for _, df in clusters.items():
            for kw in st.session_state.selected_keywords:
                rows = df[df['Symptom'].str.contains(kw, case=False, na=False)]
                for _, row in rows.iterrows():
                    cid = str(row['Cluster_ID'])
                    if cid in {'NOISE', '-1'} or cid in seen: continue
                    cluster = df[df['Cluster_ID'] == cid]
                    if len(cluster) > min_size or pattern_type == "Rare":
                        seen.add(cid)
                        top_rems = cluster['Remedy'].value_counts().head(2).index.tolist()
                        all_syms = cluster['Symptom'].tolist()
                        pattern_clusters.append({
                            'id': cid,
                            'size': len(cluster),
                            'remedies': top_rems,
                            'symptoms': all_syms
                        })

        selected_pattern_symptoms = []
        if pattern_clusters:
            st.write(f"Found **{len(pattern_clusters)}** {pattern_type.lower()} clusters")
            for pc in pattern_clusters[:10]:
                with st.expander(f"Cluster {pc['id']} — {', '.join(pc['remedies']).upper()} ({pc['size']} symptoms)"):
                    sym_options = [s for s in pc['symptoms'] if s not in st.session_state.selected_keywords]
                    chosen = st.multiselect("Add symptoms", sym_options, key=f"pat_{pc['id']}")
                    selected_pattern_symptoms.extend(chosen)
                    for sym in chosen:
                        with st.expander(f"Modalities for **{sym}**"):
                            c1, c2 = st.columns(2)
                            with c1:
                                worse = st.multiselect("Worse", MODALITIES["worse"], key=f"pw_{sym}_{pc['id']}")
                            with c2:
                                better = st.multiselect("Better", MODALITIES["better"], key=f"pb_{sym}_{pc['id']}")
                            st.session_state.selected_modalities[sym] = {"worse": worse, "better": better}
        else:
            st.info("No clusters found.")

        st.session_state.selected_pattern_symptoms = selected_pattern_symptoms

        if st.button("Final Results + PDF"):
            st.session_state.step = 4
            st.rerun()

    # === 4. FINAL + SAVE + PDF ===
    if st.session_state.step >= 4:
        st.header("4. Final Remedy Suggestion")

        all_selected = st.session_state.selected_keywords + st.session_state.selected_pattern_symptoms

        # Coverage
        coverage = defaultdict(float)
        count = defaultdict(int)
        for sym in all_selected:
            if sym in s2r_global:
                for rem, sc in s2r_global[sym].items():
                    w = rhe_weight if sym in data['rhe'].get('symptom_to_remedies', {}) else case_weight
                    coverage[rem] += sc * w
                    count[rem] += 1

        top5 = sorted(coverage.items(), key=lambda x: x[1], reverse=True)[:5]

        st.subheader("Top 5 Suggested Remedies")
        cols = st.columns(5)
        for i, (rem, sc) in enumerate(top5):
            with cols[i]:
                st.metric(rem.upper(), f"{sc:.1f}")
                st.caption(f"{count[rem]} symptoms")

        st.subheader("Top 5 Per Selected Symptom")
        per_symptom_data = {}
        for sym in all_selected[:15]:
            with st.expander(sym):
                if sym in s2r_global:
                    ranked = sorted(s2r_global[sym].items(), key=lambda x: x[1], reverse=True)[:5]
                    for rem, sc in ranked:
                        st.write(f"**{rem}** — {sc:.2f}")
                    per_symptom_data[sym] = [(rem, sc) for rem, sc in ranked]

        # Save to history
        case_data = {
            "patient_symptoms": st.session_state.symptoms,
            "selected_keywords": st.session_state.selected_keywords,
            "selected_pattern_symptoms": st.session_state.selected_pattern_symptoms,
            "rhe_weight": rhe_weight,
            "case_weight": case_weight,
            "top5_coverage": top5,
            "per_symptom": per_symptom_data
        }
        save_case_to_history(case_data)

        # PDF
        def create_pdf():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                doc = SimpleDocTemplate(tmp.name, pagesize=letter)
                styles = getSampleStyleSheet()
                elements = [
                    Paragraph("HoRUS 3 — Remedy Suggestion Report", styles['Title']),
                    Spacer(1, 0.2*inch),
                    Paragraph(f"Patient: {', '.join(st.session_state.symptoms)}", styles['Normal']),
                    Paragraph(f"Rheumatic: {rhe_weight:.2f} | Case Studies: {case_weight:.2f}", styles['Normal']),
                    Spacer(1, 0.2*inch),
                    Paragraph("Top 5 Suggested Remedies", styles['Heading2']),
                ]
                table_data = [["Rank", "Remedy", "Score", "Count"]]
                for i, (rem, sc) in enumerate(top5, 1):
                    table_data.append([i, rem.upper(), f"{sc:.1f}", count[rem]])
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e40af")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                ]))
                elements.append(table)

                elements.append(Spacer(1, 0.3*inch))
                elements.append(Paragraph("Top 5 Remedies Per Symptom", styles['Heading2']))
                for sym, ranked in per_symptom_data.items():
                    elements.append(Paragraph(f"<b>{sym}</b>", styles['Normal']))
                    tdata = [["Remedy", "Score"]]
                    for rem, sc in ranked:
                        tdata.append([rem, f"{sc:.2f}"])
                    t = Table(tdata)
                    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.gray)]))
                    elements.append(t)
                    elements.append(Spacer(1, 0.1*inch))

                doc.build(elements)
                return tmp.name

        pdf_path = create_pdf()
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f.read(), "HORUS3_Remedy_Report.pdf", "application/pdf")

        if st.button("New Case"):
            st.session_state.clear()
            st.rerun()

# === TAB 2: HISTORY ===
with tab2:
    st.header("Case History")
    if not os.path.exists(HISTORY_FILE):
        st.info("No history file found.")
    else:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
        
        if not history:
            st.info("No past cases yet.")
        else:
            for case in reversed(history[-10:]):
                with st.expander(f"{case['timestamp']} — {', '.join(case['patient_symptoms'][:3])}{'...' if len(case['patient_symptoms']) > 3 else ''}"):
                    st.write(f"**Rheumatic:** {case['rhe_weight']:.2f} | **Case Studies:** {case['case_weight']:.2f}")
                    st.write("**Top 5 Suggested Remedies:**")
                    cols = st.columns(5)
                    for i, (rem, sc) in enumerate(case['top5_coverage']):
                        with cols[i]:
                            st.metric(rem.upper(), f"{sc:.1f}")
                    if case['per_symptom']:
                        st.write("**Per Symptom:**")
                        for sym, ranked in list(case['per_symptom'].items())[:5]:
                            with st.expander(sym):
                                for rem, sc in ranked:
                                    st.write(f"**{rem}** — {sc:.2f}")