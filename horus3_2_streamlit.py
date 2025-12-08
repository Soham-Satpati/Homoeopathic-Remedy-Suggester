# HORUS3_ULTRA_CLINICAL_FINAL_WITH_QOL_ENHANCEMENTS.py
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
from collections import defaultdict, Counter
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
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

# Remedy context data (simplified materia medica)
REMEDY_CONTEXT = {
    "arnica": {"follows": ["Rhus-t", "Calc"], "antidoted": ["Camph"], "thumb": "Trauma, bruising, soreness"},
    "rhus-t": {"follows": ["Arn", "Bry"], "antidoted": ["Bell"], "thumb": "Restlessness, worse rest, better motion"},
    "bryonia": {"follows": ["Rhus-t", "Arn"], "antidoted": ["Acon"], "thumb": "Worse motion, better rest, irritable"},
    "pulsatilla": {"follows": ["Kali-s", "Sil"], "antidoted": ["Cham"], "thumb": "Changeable, mild, desires company"},
    "sulphur": {"follows": ["Acon", "Ars"], "antidoted": ["Lyc"], "thumb": "Burning, untidy, worse heat"},
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

def auto_generate_patient_id():
    """Generate next available patient ID"""
    patients = load_patients()
    today = datetime.now().strftime("%Y")
    prefix = f"PT-{today}-"
    existing_nums = []
    for pid in patients.keys():
        if pid.startswith(prefix):
            try:
                num = int(pid.split('-')[-1])
                existing_nums.append(num)
            except:
                pass
    next_num = max(existing_nums, default=0) + 1
    return f"{prefix}{next_num:03d}"

if 'initialized' not in st.session_state:
    st.session_state.update({
        "step": 1,
        "physical": [], "psychological": [], "generals": [],
        "refined_keywords": {},
        "selected_pattern_symptoms": [],
        "pattern_frequencies": {},
        "modalities": defaultdict(lambda: {"worse": [], "better": []}),
        "rhe_weight": 0.5, "case_weight": 0.5,
        "weights_locked": False,
        "patient_id": "", "patient_mode": "new",
        "report_generated": False,
        "case_notes": "",
        "initialized": True
    })

# =============================================
# SIDEBAR: NAVIGATION & WEIGHTS
# =============================================
st.sidebar.title("🧭 Navigation")
nav_steps = ["1. Enter Symptoms", "2. Refine Symptoms", "3. Discover Patterns", "4. Generate Report"]
current_step_idx = st.session_state.step - 1

for i, step_name in enumerate(nav_steps):
    if i < current_step_idx:
        st.sidebar.success(f"✓ {step_name}")
    elif i == current_step_idx:
        st.sidebar.info(f"→ {step_name}")
    else:
        st.sidebar.text(f"  {step_name}")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Dataset Weights")
st.sidebar.caption("📘 Higher Rheumatic = Traditional texts emphasis\n📊 Higher Case Studies = Real clinical outcomes")

if not st.session_state.weights_locked:
    rhe_w = st.sidebar.slider("Rheumatic Dataset", 0.0, 1.0, st.session_state.rhe_weight, 0.05, key="rhe_slider")
    case_w = round(1.0 - rhe_w, 2)
    st.session_state.rhe_weight = rhe_w
    st.session_state.case_weight = case_w
    st.sidebar.write(f"Case Studies: **{case_w:.2f}** (auto-synced)")
    if st.sidebar.button("🔒 Lock Weights", type="primary"):
        st.session_state.weights_locked = True
        st.rerun()
else:
    st.sidebar.success(f"🔒 Locked: Rheumatic {st.session_state.rhe_weight:.2f} | Cases {st.session_state.case_weight:.2f}")
    if st.sidebar.button("🔓 Unlock Weights"):
        st.session_state.weights_locked = False
        st.rerun()

# =============================================
# TABS
# =============================================
tab1, tab2 = st.tabs(["📋 Patient Case", "📚 Patient History"])

with tab1:
    st.title("HoRUS 3")
    st.markdown("### **True Clinical Remedy Intelligence — Enhanced Edition**")

    # =============================================
    # PATIENT SELECTION
    # =============================================
    st.header("👤 Patient Management")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New Patient", type="primary", use_container_width=True):
            st.session_state.patient_mode = "new"
            st.session_state.patient_id = ""
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("📂 Former Patient", type="secondary", use_container_width=True):
            st.session_state.patient_mode = "former"
            st.rerun()

    if st.session_state.patient_mode == "new":
        col_a, col_b = st.columns([3, 1])
        with col_a:
            pid = st.text_input("Enter Patient ID (or leave blank for auto-generate)", placeholder="PT-2025-001")
        with col_b:
            st.write("")
            st.write("")
            if st.button("🔄 Auto-Generate", use_container_width=True):
                st.session_state.patient_id = auto_generate_patient_id()
                st.rerun()
        if pid:
            st.session_state.patient_id = pid.strip().upper()
        elif not st.session_state.patient_id:
            st.session_state.patient_id = auto_generate_patient_id()
    else:
        patients_list = sorted(load_patients().keys())
        search = st.text_input("🔍 Search Patient", placeholder="Type to filter...")
        if search:
            patients_list = [p for p in patients_list if search.upper() in p.upper()]
        sel = st.selectbox("Select Patient", [""] + patients_list)
        if sel:
            st.session_state.patient_id = sel

    if not st.session_state.patient_id:
        st.stop()
    st.success(f"✅ Active Patient: **{st.session_state.patient_id}**")

    # =============================================
    # SYMPTOM ENTRY WITH DYNAMIC LISTS
    # =============================================
    st.header("1. 📝 Enter Patient Symptoms")
    
    def add_symptom_interface(category, key_prefix):
        """Dynamic symptom entry with tags"""
        current_symptoms = getattr(st.session_state, category, [])
        
        col_input, col_add = st.columns([4, 1])
        with col_input:
            new_symptom = st.text_input(
                f"Add {category.title()} Symptom",
                placeholder=f"Type and press Add...",
                key=f"{key_prefix}_input"
            )
        with col_add:
            st.write("")
            st.write("")
            if st.button("➕ Add", key=f"{key_prefix}_btn", use_container_width=True):
                if new_symptom.strip():
                    current_symptoms.append(new_symptom.strip())
                    setattr(st.session_state, category, current_symptoms)
                    st.rerun()
        
        if current_symptoms:
            st.caption(f"**{len(current_symptoms)} symptom(s) entered:**")
            for i, sym in enumerate(current_symptoms):
                col_sym, col_del = st.columns([5, 1])
                with col_sym:
                    st.text(f"• {sym}")
                with col_del:
                    if st.button("🗑️", key=f"{key_prefix}_del_{i}"):
                        current_symptoms.pop(i)
                        setattr(st.session_state, category, current_symptoms)
                        st.rerun()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🏥 Physical Symptoms")
        add_symptom_interface("physical", "phys")
    with col2:
        st.subheader("🧠 Psychological Symptoms")
        add_symptom_interface("psychological", "psych")
    with col3:
        st.subheader("🌡️ General Symptoms")
        add_symptom_interface("generals", "gen")

    # Alternative: Bulk text entry
    with st.expander("💡 Or paste symptoms in bulk (comma/line separated)"):
        bulk_txt = st.text_area("Paste symptoms here", height=100, key="bulk_input")
        cat_choice = st.selectbox("Add to category", ["Physical", "Psychological", "General"])
        if st.button("Import Bulk Symptoms"):
            bulk_syms = [s.strip() for s in bulk_txt.replace('\n', ',').split(',') if s.strip()]
            target = cat_choice.lower()
            current = getattr(st.session_state, target, [])
            current.extend(bulk_syms)
            setattr(st.session_state, target, current)
            st.success(f"Added {len(bulk_syms)} symptoms to {cat_choice}")
            st.rerun()

    # Navigation buttons
    if st.session_state.step == 1:
        total_syms = len(st.session_state.physical) + len(st.session_state.psychological) + len(st.session_state.generals)
        if total_syms == 0:
            st.warning("⚠️ Please enter at least one symptom to continue")
            st.stop()
        
        st.markdown("---")
        st.subheader("Choose Next Action")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button("🔍 Refine Symptoms", type="primary", use_container_width=True):
                st.session_state.step = 2
                st.session_state.refined_keywords = {}
                st.session_state.selected_pattern_symptoms = []
                st.session_state.report_generated = False
                st.rerun()
        
        with btn_col2:
            if st.button("🔬 Discover Patterns", type="primary", use_container_width=True):
                st.session_state.selected_keywords = st.session_state.physical + st.session_state.psychological + st.session_state.generals
                st.session_state.step = 3
                st.session_state.refined_keywords = {}
                st.session_state.selected_pattern_symptoms = []
                st.session_state.report_generated = False
                st.rerun()
        
        with btn_col3:
            if st.button("📊 Generate Report", type="primary", use_container_width=True):
                st.session_state.selected_keywords = st.session_state.physical + st.session_state.psychological + st.session_state.generals
                st.session_state.step = 4
                st.session_state.refined_keywords = {}
                st.session_state.selected_pattern_symptoms = []
                st.session_state.report_generated = False
                st.rerun()
        
        st.stop()

    # =============================================
    # REFINEMENT WITH CHAPTER MATCHES
    # =============================================
    if st.session_state.step >= 2:
        st.header("2. 🔍 Refine Symptoms")

        all_user_symptoms = (
            [("Physical", s) for s in st.session_state.physical] +
            [("Psychological", s) for s in st.session_state.psychological] +
            [("Generals", s) for s in st.session_state.generals]
        )

        # Quick action buttons
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            if st.button("✅ Use All As-Is (Skip Refinement)", use_container_width=True):
                st.session_state.selected_keywords = [s for _, s in all_user_symptoms]
                st.session_state.step = 3
                st.rerun()
        with action_col2:
            refine_count = sum(1 for s in all_user_symptoms if all_user_symptoms[1] in st.session_state.refined_keywords)
            st.metric("Refined So Far", f"{len(st.session_state.refined_keywords)}/{len(all_user_symptoms)}")

        # Precompute matches
        chapter_matches = defaultdict(list)
        with st.spinner("🔍 Searching repertory..."):
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

        st.markdown("---")
        for category, symptom in all_user_symptoms:
            key = f"{category}_{symptom}"
            has_matches = any(orig == symptom for chap, matches in chapter_matches.items() for rubric, sc, orig in matches)
            
            expand_default = not has_matches or symptom not in st.session_state.refined_keywords
            
            with st.expander(f"**{category}** → {symptom} {'✓' if symptom in st.session_state.refined_keywords else '⚪'}", expanded=expand_default):
                if not has_matches:
                    st.info("ℹ️ No close matches found in repertory — keeping original symptom")
                    continue
                
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

        # Modalities Summary Section
        if st.session_state.refined_keywords:
            st.markdown("---")
            st.subheader("🌡️ Modalities Summary (Optional)")
            st.caption("Add worse/better modifiers for refined symptoms")
            
            for orig_sym, refined_sym in st.session_state.refined_keywords.items():
                with st.expander(f"Modalities for: {refined_sym}"):
                    c1, c2 = st.columns(2)
                    worse = c1.multiselect("Worse", MODALITIES["worse"], 
                                          default=st.session_state.modalities[refined_sym]["worse"],
                                          key=f"w_sum_{refined_sym}")
                    better = c2.multiselect("Better", MODALITIES["better"],
                                           default=st.session_state.modalities[refined_sym]["better"],
                                           key=f"b_sum_{refined_sym}")
                    
                    # Conflict detection
                    conflicts = set(worse) & set(better)
                    if conflicts:
                        st.warning(f"⚠️ Conflicting modalities: {', '.join(conflicts)}")
                    
                    st.session_state.modalities[refined_sym] = {"worse": worse, "better": better}

        core_symptoms = [st.session_state.refined_keywords.get(s, s) for _, s in all_user_symptoms]
        st.session_state.selected_keywords = core_symptoms
        
        refined_count = len(st.session_state.refined_keywords)
        st.success(f"✅ **{len(core_symptoms)} core symptoms ready** ({refined_count} refined)")

        if st.session_state.step == 2:
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("🔬 Continue to Discover Patterns", type="primary", use_container_width=True):
                    st.session_state.step = 3
                    st.rerun()
            
            with btn_col2:
                if st.button("📊 Skip to Generate Report", type="secondary", use_container_width=True):
                    st.session_state.step = 4
                    st.rerun()
            
            st.stop()

    # =============================================
    # PATTERN DISCOVERY WITH CONFIDENCE FILTER
    # =============================================
    if st.session_state.step >= 3:
        st.divider()
        st.subheader("3. 🔬 Discover Hidden Clinical Patterns")

        core_symptoms = st.session_state.selected_keywords

        with st.expander("Find associated symptoms from real case clusters", expanded=True):
            st.markdown("**HoRUS analyzes thousands of real cases to reveal hidden concomitant symptoms**")
            
            col_rarity, col_confidence = st.columns(2)
            with col_rarity:
                rarity = st.radio("Pattern Type", ["Common", "Uncommon", "Rare"], horizontal=True)
            with col_confidence:
                confidence_threshold = st.slider("Confidence Filter (%)", 0, 100, 50, 5,
                                                help="Higher = only show patterns from more cases")
            
            size_ranges = {
                "Common": (25, float('inf')),
                "Uncommon": (12, 24),
                "Rare": (0, 11)
            }
            min_size, max_size = size_ranges[rarity]

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
                        cluster_size = len(cluster_df)
                        
                        if min_size <= cluster_size <= max_size:
                            frequency = min(100, cluster_size * 2)
                            
                            if frequency >= confidence_threshold:
                                seen_clusters.add(cid)
                                symptoms = cluster_df['Symptom'].tolist()
                                pattern_syms = [s for s in symptoms if s not in core_symptoms]
                                found_clusters.append((pattern_syms, frequency))
                                st.session_state.pattern_frequencies[cid] = frequency

            # Track added symptoms
            if 'pattern_accumulator' not in st.session_state:
                st.session_state.pattern_accumulator = []

            if found_clusters:
                st.write(f"**{len(found_clusters)} clinical pattern(s) discovered**")
                
                for i, (cluster, freq) in enumerate(found_clusters[:8]):
                    with st.expander(f"Pattern {i+1} — {len(cluster)} symptoms | 📊 {freq}% frequency"):
                        choices = st.multiselect(
                            "Add to case",
                            [s for s in cluster if s not in st.session_state.pattern_accumulator],
                            key=f"add_cluster_{i}"
                        )
                        if choices:
                            st.session_state.pattern_accumulator.extend(choices)
            else:
                st.info("No patterns match current filters. Try adjusting rarity or confidence threshold.")

            # Show accumulator
            if st.session_state.pattern_accumulator:
                st.markdown("---")
                st.subheader("📋 Added Patterns Summary")
                st.caption(f"**{len(set(st.session_state.pattern_accumulator))} unique symptoms** selected from patterns:")
                for sym in sorted(set(st.session_state.pattern_accumulator)):
                    st.text(f"• {sym}")

            if st.button("✅ Update Case with Selected Patterns", type="primary", use_container_width=True):
                st.session_state.selected_pattern_symptoms = list(set(st.session_state.pattern_accumulator))
                st.session_state.step = 4
                st.session_state.report_generated = False
                st.rerun()
        
        if st.session_state.step == 3:
            st.markdown("---")
            if st.button("📊 Continue to Generate Report (without patterns)", type="secondary", use_container_width=True):
                st.session_state.step = 4
                st.rerun()
            
            st.stop()

    # =============================================
    # FINAL REPORT WITH ENHANCED CONTEXT
    # =============================================
    if st.session_state.step >= 4:
        st.divider()
        st.header("4. 📊 Final Clinical Report")

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
                        col_metric, col_context = st.columns([1, 2])
                        
                        with col_metric:
                            st.metric("Core Coverage", f"{count}/{len(core)}")
                        
                        with col_context:
                            if rem.lower() in REMEDY_CONTEXT:
                                ctx = REMEDY_CONTEXT[rem.lower()]
                                st.caption(f"💡 **{ctx['thumb']}**")
                                st.caption(f"▶️ Follows: {', '.join(ctx['follows'])}")
                                st.caption(f"⊗ Antidoted: {', '.join(ctx['antidoted'])}")
                        
                        st.markdown("**Symptom Coverage:**")
                        for c in core:
                            covered = (c in s2r_global and rem in s2r_global[c]) or \
                                      any(rem in s2r_global.get(e,{}) for e in expanded if any(w in e.lower() for w in c.lower().split() if len(w)>2))
                            st.write(f"{'✅' if covered else '❌'} {c}")

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
                        col_metric, col_context = st.columns([1, 2])
                        
                        with col_metric:
                            st.metric("Score", f"{sc:.3f}")
                        
                        with col_context:
                            if rem.lower() in REMEDY_CONTEXT:
                                ctx = REMEDY_CONTEXT[rem.lower()]
                                st.caption(f"💡 **{ctx['thumb']}**")
                                st.caption(f"▶️ Follows: {', '.join(ctx['follows'])}")
                                st.caption(f"⊗ Antidoted: {', '.join(ctx['antidoted'])}")

            # Case notes
            st.markdown("---")
            st.subheader("📝 Clinical Notes (Optional)")
            notes = st.text_area(
                "Add follow-up notes, remedy response, or observations",
                value=st.session_state.case_notes,
                height=100,
                placeholder="e.g., Patient responded well to Nat-m, followed with Sulphur after 3 weeks..."
            )
            st.session_state.case_notes = notes

            # Save case
            if not st.session_state.report_generated:
                save_patient_once(st.session_state.patient_id, {
                    "core": core,
                    "pattern": pattern,
                    "top10": top10,
                    "notes": notes,
                    "weights": {"rheumatic": st.session_state.rhe_weight, "cases": st.session_state.case_weight},
                    "mode": "expansion" if only_core else "weighted"
                })
                st.session_state.report_generated = True

            # PDF Export
            st.markdown("---")
            st.subheader("📄 Export Report")
            
            def create_pdf():
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc = SimpleDocTemplate(tmp.name, pagesize=A4)
                styles = getSampleStyleSheet()
                story = []
                
                # Header
                title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                            fontSize=18, textColor=colors.HexColor('#1f77b4'), spaceAfter=12)
                story.append(Paragraph("HoRUS 3 — Clinical Report", title_style))
                story.append(Spacer(1, 0.2*inch))
                
                # Patient Info
                info_data = [
                    ["Patient ID:", st.session_state.patient_id],
                    ["Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
                    ["Analysis Mode:", "Semantic Expansion" if only_core else "Weighted Repertorization"],
                    ["Dataset Weights:", f"Rheumatic {st.session_state.rhe_weight:.2f} | Cases {st.session_state.case_weight:.2f}"]
                ]
                info_table = Table(info_data, colWidths=[2*inch, 4*inch])
                info_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                story.append(info_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Core Symptoms
                story.append(Paragraph("<b>Core Symptoms:</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                for sym in core:
                    story.append(Paragraph(f"• {sym}", styles['BodyText']))
                story.append(Spacer(1, 0.2*inch))
                
                # Pattern Symptoms
                if pattern:
                    story.append(Paragraph("<b>Pattern-Derived Symptoms:</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
                    for sym in pattern:
                        story.append(Paragraph(f"• {sym}", styles['BodyText']))
                    story.append(Spacer(1, 0.2*inch))
                
                # Modalities
                if st.session_state.modalities:
                    story.append(Paragraph("<b>Modalities:</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
                    for sym, mods in st.session_state.modalities.items():
                        if mods["worse"] or mods["better"]:
                            story.append(Paragraph(f"<b>{sym}</b>", styles['BodyText']))
                            if mods["worse"]:
                                story.append(Paragraph(f"  Worse: {', '.join(mods['worse'])}", styles['BodyText']))
                            if mods["better"]:
                                story.append(Paragraph(f"  Better: {', '.join(mods['better'])}", styles['BodyText']))
                    story.append(Spacer(1, 0.2*inch))
                
                # Top Remedies
                story.append(Paragraph("<b>Top 10 Remedies:</b>", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                remedy_data = [["Rank", "Remedy", "Score/Coverage"]]
                for i, (rem, val) in enumerate(top10, 1):
                    score_str = f"{val}/{len(core)}" if only_core else f"{val:.3f}"
                    remedy_data.append([str(i), rem.upper(), score_str])
                
                remedy_table = Table(remedy_data, colWidths=[0.7*inch, 2*inch, 1.5*inch])
                remedy_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
                ]))
                story.append(remedy_table)
                story.append(Spacer(1, 0.3*inch))
                
                # Clinical Notes
                if notes:
                    story.append(Paragraph("<b>Clinical Notes:</b>", styles['Heading2']))
                    story.append(Spacer(1, 0.1*inch))
                    story.append(Paragraph(notes, styles['BodyText']))
                
                doc.build(story)
                return tmp.name
            
            if st.button("📥 Download PDF Report", type="primary", use_container_width=True):
                pdf_path = create_pdf()
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "💾 Save PDF",
                        f.read(),
                        file_name=f"HoRUS3_Report_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                os.unlink(pdf_path)
        
        generate_report()
        
        st.markdown("---")
        if st.button("🔄 Start New Case", type="secondary", use_container_width=True):
            for key in ["physical", "psychological", "generals", "refined_keywords", 
                       "selected_pattern_symptoms", "pattern_accumulator", "case_notes"]:
                if key in st.session_state:
                    if isinstance(st.session_state[key], list):
                        st.session_state[key] = []
                    elif isinstance(st.session_state[key], dict):
                        st.session_state[key] = {}
                    else:
                        st.session_state[key] = ""
            st.session_state.step = 1
            st.session_state.patient_mode = "new"
            st.session_state.patient_id = ""
            st.session_state.report_generated = False
            st.rerun()

# =============================================
# PATIENT HISTORY TAB
# =============================================
with tab2:
    st.title("📚 Patient History & Records")
    
    patients = load_patients()
    
    if not patients:
        st.info("No patient records found. Create your first case in the Patient Case tab!")
        st.stop()
    
    # Search and filter
    col_search, col_sort = st.columns([3, 1])
    with col_search:
        search_term = st.text_input("🔍 Search patients", placeholder="Enter patient ID or date...")
    with col_sort:
        sort_order = st.selectbox("Sort by", ["Recent First", "Oldest First", "Patient ID"])
    
    # Filter patients
    filtered = {k: v for k, v in patients.items() 
                if not search_term or search_term.upper() in k.upper()}
    
    # Sort
    if sort_order == "Recent First":
        sorted_patients = sorted(filtered.items(), 
                                key=lambda x: x[1][-1]['timestamp'] if x[1] else "", 
                                reverse=True)
    elif sort_order == "Oldest First":
        sorted_patients = sorted(filtered.items(), 
                                key=lambda x: x[1][0]['timestamp'] if x[1] else "")
    else:
        sorted_patients = sorted(filtered.items())
    
    st.write(f"**{len(sorted_patients)} patient(s) found**")
    st.markdown("---")
    
    # Display patient cards
    for pid, records in sorted_patients:
        with st.expander(f"👤 **{pid}** — {len(records)} visit(s)", expanded=False):
            for i, rec in enumerate(records, 1):
                st.markdown(f"### Visit {i} — {rec['timestamp']}")
                
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Core Symptoms", len(rec.get('core', [])))
                    st.metric("Pattern Symptoms", len(rec.get('pattern', [])))
                with col_info2:
                    st.metric("Analysis Mode", rec.get('mode', 'N/A').title())
                    weights = rec.get('weights', {})
                    st.caption(f"Weights: R{weights.get('rheumatic', 0):.2f} / C{weights.get('cases', 0):.2f}")
                
                # Symptoms
                with st.expander("View Symptoms", expanded=False):
                    st.markdown("**Core Symptoms:**")
                    for sym in rec.get('core', []):
                        st.text(f"• {sym}")
                    
                    if rec.get('pattern'):
                        st.markdown("**Pattern Symptoms:**")
                        for sym in rec['pattern']:
                            st.text(f"• {sym}")
                
                # Top remedies
                with st.expander("Top Remedies", expanded=False):
                    top10 = rec.get('top10', [])
                    for j, (rem, score) in enumerate(top10[:5], 1):
                        if isinstance(score, float):
                            st.text(f"{j}. {rem.upper()} — {score:.3f}")
                        else:
                            st.text(f"{j}. {rem.upper()} — {score}")
                
                # Notes
                if rec.get('notes'):
                    with st.expander("Clinical Notes", expanded=False):
                        st.text(rec['notes'])
                
                st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.caption("HoRUS 3 Ultra Clinical Edition")
st.sidebar.caption("v3.0 — Enhanced with QoL features")
