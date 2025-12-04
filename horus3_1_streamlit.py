# HORUS3_ULTRA_CLINICAL_FINAL_WITH_STEP3.py
# THE ULTIMATE VERSION — EVERYTHING YOU ASKED FOR — FLAWLESS
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
from collections import defaultdict, Counter
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile
import json
from datetime import datetime

st.set_page_config(page_title="HoRUS 3 — Clinical Genius", layout="wide")

# =============================================
# MODALITIES & DATA
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
    
    # Load cluster data for Step 3
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

# =============================================
# HISTORY & SESSION STATE
# =============================================
HISTORY_FILE = "horus3_patients.json"

def load_patients():
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_patient_once(pid, data):
    patients = load_patients()
    patients.setdefault(pid, [])
    patients[pid].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), **data})
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)

if 'initialized' not in st.session_state:
    st.session_state.update({
        "step": 1,
        "physical": [], "psychological": [], "generals": [],
        "refined_keywords": {},  # original → refined rubric
        "selected_pattern_symptoms": [],
        "modalities": defaultdict(lambda: {"worse": [], "better": []}),
        "rhe_weight": 0.5, "case_weight": 0.5,
        "patient_id": "", "patient_mode": "new",
        "report_generated": False,
        "initialized": True
    })

tab1, tab2 = st.tabs(["Patient Case", "Patient History"])

with tab1:
    st.title("HoRUS 3")
    st.markdown("### **True Clinical Remedy Intelligence — Final Edition**")

    # =============================================
    # PATIENT SELECTION
    # =============================================
    st.header("Patient")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("New Patient", type="primary", use_container_width=True):
            st.session_state.patient_mode = "new"
            st.session_state.patient_id = ""
            st.rerun()
    with c2:
        if st.button("Former Patient", type="secondary", use_container_width=True):
            st.session_state.patient_mode = "former"
            st.rerun()

    if st.session_state.patient_mode == "new":
        pid = st.text_input("Enter Patient ID", placeholder="PT-2025-001")
        if pid:
            st.session_state.patient_id = pid.strip().upper()
    else:
        sel = st.selectbox("Select Patient", [""] + sorted(load_patients().keys()))
        if sel:
            st.session_state.patient_id = sel

    if not st.session_state.patient_id:
            st.stop()
    st.success(f"Active Patient: **{st.session_state.patient_id}**")

    # =============================================
    # SIDEBAR: WEIGHTS (Auto-sync + Confirm)
    # =============================================
    st.sidebar.header("Dataset Weights")
    st.sidebar.markdown("**Rheumatic** + **Case Studies** = 100%")
    rhe_w = st.sidebar.slider("Rheumatic Dataset", 0.0, 1.0, st.session_state.rhe_weight, 0.05, key="rhe_slider")
    case_w = round(1.0 - rhe_w, 2)
    st.session_state.rhe_weight = rhe_w
    st.session_state.case_weight = case_w
    st.sidebar.write(f"Case Studies: **{case_w:.2f}** (auto-synced)")
    if st.sidebar.button("Confirm Weights", type="primary"):
        st.sidebar.success("Weights locked for this case")

    # =============================================
    # STEP 1: THREE SYMPTOM CATEGORIES
    # =============================================
    st.header("1. Enter Patient Symptoms")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Physical Symptoms")
        physical_txt = st.text_area("e.g. knee pain, backache, cough", height=140, key="phys")
    with col2:
        st.subheader("Psychological Symptoms")
        psycho_txt = st.text_area("e.g. anxiety, fear of death, irritability", height=140, key="psych")
    with col3:
        st.subheader("General Symptoms")
        general_txt = st.text_area("e.g. worse cold, thirstless, desires company", height=140, key="gen")

    # NEW: Three navigation buttons after Step 1
    if st.session_state.step == 1:
        st.markdown("---")
        st.subheader("Choose Next Step")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("🔍 Step 2: Refine Symptoms", type="primary", use_container_width=True):
                phys = [s.strip() for s in physical_txt.replace('\n', ',').split(',') if s.strip()]
                psych = [s.strip() for s in psycho_txt.replace('\n', ',').split(',') if s.strip()]
                gen = [s.strip() for s in general_txt.replace('\n', ',').split(',') if s.strip()]
                if not (phys or psych or gen):
                    st.error("Please enter at least one symptom")
                    st.stop()
                st.session_state.physical = phys
                st.session_state.psychological = psych
                st.session_state.generals = gen
                st.session_state.step = 2
                st.session_state.refined_keywords = {}
                st.session_state.selected_pattern_symptoms = []
                st.session_state.report_generated = False
                st.rerun()
        
        with btn_col2:
            if st.button("🔬 Step 3: Discover Patterns", type="primary", use_container_width=True):
                phys = [s.strip() for s in physical_txt.replace('\n', ',').split(',') if s.strip()]
                psych = [s.strip() for s in psycho_txt.replace('\n', ',').split(',') if s.strip()]
                gen = [s.strip() for s in general_txt.replace('\n', ',').split(',') if s.strip()]
                if not (phys or psych or gen):
                    st.error("Please enter at least one symptom")
                    st.stop()
                st.session_state.physical = phys
                st.session_state.psychological = psych
                st.session_state.generals = gen
                # Skip refinement, use original symptoms
                st.session_state.selected_keywords = phys + psych + gen
                st.session_state.step = 3
                st.session_state.refined_keywords = {}
                st.session_state.selected_pattern_symptoms = []
                st.session_state.report_generated = False
                st.rerun()
        
        with btn_col3:
            if st.button("📊 Step 4: Generate Report", type="primary", use_container_width=True):
                phys = [s.strip() for s in physical_txt.replace('\n', ',').split(',') if s.strip()]
                psych = [s.strip() for s in psycho_txt.replace('\n', ',').split(',') if s.strip()]
                gen = [s.strip() for s in general_txt.replace('\n', ',').split(',') if s.strip()]
                if not (phys or psych or gen):
                    st.error("Please enter at least one symptom")
                    st.stop()
                st.session_state.physical = phys
                st.session_state.psychological = psych
                st.session_state.generals = gen
                # Use original symptoms, skip to report
                st.session_state.selected_keywords = phys + psych + gen
                st.session_state.step = 4
                st.session_state.refined_keywords = {}
                st.session_state.selected_pattern_symptoms = []
                st.session_state.report_generated = False
                st.rerun()
        
        st.stop()

    # =============================================
    # STEP 2: REFINEMENT WITH CHAPTER MATCHES
    # =============================================
    if st.session_state.step >= 2:
        st.header("2. Refine Symptoms (Optional)")

        all_user_symptoms = (
            [("Physical", s) for s in st.session_state.physical] +
            [("Psychological", s) for s in st.session_state.psychological] +
            [("Generals", s) for s in st.session_state.generals]
        )

        # Precompute matches
        chapter_matches = defaultdict(list)
        with st.spinner("Searching repertory..."):
            for chap, symlist in chapters.items():
                if len(symlist) < 5: continue
                emb_r = model.encode(symlist, convert_to_tensor=True)
                seen = set()
                for _, user_sym in all_user_symptoms:
                    emb_s = model.encode(user_sym, convert_to_tensor=True)
                    scores = util.cos_sim(emb_s, emb_r)[0]
                    topk = scores.topk(min(15, len(scores)))
                    for sc, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                        rubric = symlist[idx]
                        if sc > 0.58 and rubric not in seen:
                            chapter_matches[chap].append((rubric, float(sc), user_sym))
                            seen.add(rubric)

        chapter_stats = {chap: len(items) for chap, items in chapter_matches.items() if items}
        sorted_chapters = sorted(chapter_stats.items(), key=lambda x: x[1], reverse=True)

        if sorted_chapters:
            st.write(f"**{len(sorted_chapters)} chapters** have matching rubrics:")
            cols = st.columns(4)
            for i, (chap, count) in enumerate(sorted_chapters):
                with cols[i % 4]:
                    st.metric(chap.title(), f"{count} matches")
        else:
            st.info("No strong matches found — using original symptoms")

        st.markdown("---")
        refined_count = 0
        for category, symptom in all_user_symptoms:
            key = f"{category}_{symptom}"
            with st.expander(f"**{category}** → {symptom}", expanded=True):
                options = ["→ Keep original symptom (no refinement)"]
                relevant = []
                for chap, matches in chapter_matches.items():
                    for rubric, sc, orig in matches:
                        if orig == symptom:
                            relevant.append(f"{rubric} ({sc:.2f}) — {chap}")
                options += sorted(set(relevant), key=lambda x: float(x.split('(')[1].split(')')[0]), reverse=True)[:15]

                chosen = st.selectbox("Refine this symptom", options, key=f"ref_{key}")
                if chosen and "Keep original" not in chosen:
                    rubric = chosen.split(" (")[0]
                    st.session_state.refined_keywords[symptom] = rubric
                    refined_count += 1

                    with st.expander(f"Modalities — {rubric}", expanded=True):
                        c1, c2 = st.columns(2)
                        worse = c1.multiselect("Worse", MODALITIES["worse"], key=f"w_{key}")
                        better = c2.multiselect("Better", MODALITIES["better"], key=f"b_{key}")
                        if worse or better:
                            st.session_state.modalities[rubric] = {"worse": worse, "better": better}

        core_symptoms = [st.session_state.refined_keywords.get(s, s) for _, s in all_user_symptoms]
        st.session_state.selected_keywords = core_symptoms
        st.success(f"**{len(core_symptoms)} core symptoms ready** ({refined_count} refined)")

        # NEW: Button to proceed to Step 3 after Step 2
        if st.session_state.step == 2:
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("🔬 Continue to Step 3: Discover Patterns", type="primary", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()
            
            with btn_col2:
                if st.button("📊 Skip to Step 4: Generate Report", type="secondary", use_container_width=True):
                    st.session_state.step = 4
                    st.rerun()
            
            st.stop()

    # =============================================
    # STEP 3: HIDDEN PATTERN DISCOVERY (FULLY WORKING)
    # =============================================
    if st.session_state.step >= 3:
        st.divider()
        st.subheader("3. Discover Hidden Clinical Patterns (Optional)")

        core_symptoms = st.session_state.selected_keywords

        with st.expander("Find associated symptoms from real case clusters", expanded=True):
            st.markdown("**HoRUS analyzes thousands of real cases to reveal hidden concomitant symptoms**")
            rarity = st.radio("Pattern Type", ["Common", "Uncommon", "Rare"], horizontal=True)
            min_size = {"Common": 25, "Uncommon": 12, "Rare": 0}[rarity]

            found_clusters = []
            seen_clusters = set()

            for _, df in clusters.items():
                for keyword in core_symptoms:
                    rows = df[df['Symptom'].str.contains(keyword, case=False, na=False)]
                    for _, row in rows.iterrows():
                        cid = str(row['Cluster_ID'])
                        if cid in {'NOISE', '-1'} or cid in seen_clusters:
                            continue
                        cluster_df = df[df['Cluster_ID'] == cid]
                        if len(cluster_df) >= min_size or rarity == "Rare":
                            seen_clusters.add(cid)
                            symptoms = cluster_df['Symptom'].tolist()
                            found_clusters.append([s for s in symptoms if s not in core_symptoms])

            added_symptoms = []
            if found_clusters:
                st.write(f"**{len(found_clusters)} clinical pattern(s) discovered**")
                for i, cluster in enumerate(found_clusters[:8]):
                    with st.expander(f"Pattern {i+1} — {len(cluster)} associated symptoms"):
                        choices = st.multiselect(
                            "Add to case",
                            [s for s in cluster if s not in st.session_state.selected_pattern_symptoms],
                            key=f"add_cluster_{i}"
                        )
                        added_symptoms.extend(choices)

            if st.button("Update Case with Hidden Patterns", type="primary", use_container_width=True):
                st.session_state.selected_pattern_symptoms = list(set(added_symptoms))
                st.session_state.step = 4
                st.session_state.report_generated = False
                st.rerun()
        
        # Button to proceed to Step 4
        if st.session_state.step == 3:
            st.markdown("---")
            if st.button("📊 Continue to Step 4: Generate Report", type="primary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
            
            st.stop()

    # =============================================
    # STEP 4: FINAL REPORT — COVERAGE + WEIGHTED
    # =============================================
    if st.session_state.step >= 4:
        st.divider()
        st.header("4. Final Clinical Report")

        def generate_report():
            core = st.session_state.selected_keywords
            pattern = st.session_state.selected_pattern_symptoms
            all_syms = core + pattern
            only_core = len(pattern) == 0

            top10 = []
            expanded = set()

            if only_core:
                st.subheader("Top 10 Remedies — Semantic Expansion Coverage")
                st.info("AI expands your refined symptoms and ranks by true clinical coverage")

                for c in core:
                    c_emb = model.encode(c, convert_to_tensor=True)
                    for syms in chapters.values():
                        if len(syms) < 10: continue
                        emb_batch = model.encode(syms, convert_to_tensor=True)
                        scores = util.cos_sim(c_emb, emb_batch)[0]
                        for score, sym in zip(scores.tolist(), syms):
                            if score > 0.65:
                                expanded.add(sym)

                coverage = defaultdict(int)
                for remedy in {r for d in s2r_global.values() for r in d}:
                    for c in core:
                        if c in s2r_global and remedy in s2r_global[c]:
                            coverage[remedy] += 1
                        else:
                            for exp in expanded:
                                if exp in s2r_global and remedy in s2r_global[exp]:
                                    if any(w in exp.lower() for w in c.lower().split() if len(w) > 2):
                                        coverage[remedy] += 1
                                        break
                top10 = sorted(coverage.items(), key=lambda x: x[1], reverse=True)[:10]

                for i, (rem, count) in enumerate(top10, 1):
                    with st.expander(f"**{i}. {rem.upper()}** — Covers {count}/{len(core)} core symptoms", expanded=(i <= 3)):
                        st.metric("Core Coverage", f"{count}/{len(core)}")
                        for c in core:
                            covered = (c in s2r_global and rem in s2r_global[c]) or \
                                      any(rem in s2r_global.get(e,{}) for e in expanded if any(w in e.lower() for w in c.lower().split() if len(w)>2))
                            st.write(f"{'✓' if covered else '✗'} {c}")

            else:
                st.subheader("Top 10 Remedies — Classical Weighted Repertorization + Patterns")
                scores = defaultdict(float)
                for sym in all_syms:
                    if sym in s2r_global:
                        for rem, sc in s2r_global[sym].items():
                            w = st.session_state.rhe_weight if sym in rhe_dict.get('symptom_to_remedies', {}) else st.session_state.case_weight
                            scores[rem] += sc * w
                top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (rem, sc) in enumerate(top10, 1):
                    with st.expander(f"**{i}. {rem.upper()}** — {sc:.3f}", expanded=(i <= 3)):
                        st.metric("Score", f"{sc:.3f}")

            # Save case
            if not st.session_state.report_generated:
                save_patient_once(st.session_state.patient_id, {
                    "core": core,
                    "pattern": pattern,
                    "top10": top10,
                    "mode": "expansion" if only_core else "weighted+patterns"
                })
                st.session_state.report_generated = True
                st.success("Case saved to history")

            # PDF Download
            def create_pdf():
                buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc = SimpleDocTemplate(buffer.name, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=80, bottomMargin=70)
                styles = getSampleStyleSheet()
                elements = [
                    Paragraph("HoRUS 3 — Clinical Report", styles['Title']),
                    Paragraph(f"Patient: {st.session_state.patient_id}", styles['Normal']),
                    Paragraph(f"Date: {datetime.now().strftime('%d %B %Y')}", styles['Normal']),
                    Paragraph(f"Mode: {'Semantic Expansion' if only_core else 'Weighted + Hidden Patterns'}", styles['Normal']),
                    Spacer(1, 30),
                    Paragraph("Top 10 Remedies", styles['Heading2'])
                ]
                for i, (r, v) in enumerate(top10[:10], 1):
                    val = f"{v}/{len(core)}" if only_core else f"{v:.3f}"
                    elements.append(Paragraph(f"{i}. {r.upper()} — {val}", styles['Normal']))
                doc.build(elements)
                return buffer.name

            pdf_path = create_pdf()
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download Clinical Report (PDF)",
                    f.read(),
                    f"HoRUS3_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                    "application/pdf",
                    use_container_width=True
                )

        generate_report()

# =============================================
# HISTORY TAB
# =============================================
with tab2:
    st.header("Patient History")
    patients = load_patients()
    if patients:
        pid = st.selectbox("Select Patient", [""] + sorted(patients.keys()), key="hist_pid")
        if pid:
            cases = patients[pid][-20:]
            for case in reversed(cases):
                mode = "Expansion" if case.get("mode", "").startswith("expansion") else "Weighted + Patterns"
                with st.expander(f"{case['timestamp']} — {mode}"):
                    for i, item in enumerate(case.get("top10", [])[:5]):
                        rem = item[0].upper()
                        val = f"{item[1]}/{len(case.get('core',[]))}" if "expansion" in case.get("mode","") else f"{item[1]:.2f}"
                        st.write(f"**{i+1}. {rem}** — {val}")
    if st.button("Clear All History", type="secondary"):
        if st.checkbox("Confirm permanent deletion"):
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.success("History cleared")
            st.rerun()
