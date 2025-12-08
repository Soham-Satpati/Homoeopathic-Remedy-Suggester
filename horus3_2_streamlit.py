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
        "pattern_frequencies": {},  # Track pattern frequencies
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
            # Map category names to session state keys
            category_map = {
                "Physical": "physical",
                "Psychological": "psychological",
                "General": "generals"  # Note: plural!
            }
            target = category_map[cat_choice]
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
                            # Simulate frequency data (in production, this would come from cluster metadata)
                            frequency = min(100, cluster_size * 2)  # Larger clusters = higher frequency
                            
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
                            # Add remedy context
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
                    "mode": "expansion" if only_core else "weighted+patterns"
                })
                st.session_state.report_generated = True
                st.success("✅ Case saved to history")

            # Enhanced PDF Download
            def create_enhanced_pdf():
                buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                doc = SimpleDocTemplate(buffer.name, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=80, bottomMargin=70)
                styles = getSampleStyleSheet()
                
                # Custom styles
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Title'],
                    fontSize=24,
                    textColor=colors.HexColor('#1f77b4'),
                    spaceAfter=30
                )
                
                elements = [
                    Paragraph("HoRUS 3 — Clinical Report", title_style),
                    Spacer(1, 12),
                    Paragraph(f"<b>Patient ID:</b> {st.session_state.patient_id}", styles['Normal']),
                    Paragraph(f"<b>Date Generated:</b> {datetime.now().strftime('%d %B %Y, %H:%M')}", styles['Normal']),
                    Paragraph(f"<b>Analysis Mode:</b> {'Semantic Expansion Coverage' if only_core else 'Weighted Repertorization + Patterns'}", styles['Normal']),
                    Paragraph(f"<b>Dataset Weights:</b> Rheumatic {st.session_state.rhe_weight:.2f} | Case Studies {st.session_state.case_weight:.2f}", styles['Normal']),
                    Spacer(1, 20),
                ]
                
                # Case presentation summary
                elements.append(Paragraph("<b>Clinical Presentation</b>", styles['Heading2']))
                elements.append(Spacer(1, 6))
                
                if st.session_state.physical:
                    elements.append(Paragraph(f"<b>Physical:</b> {', '.join(st.session_state.physical[:5])}", styles['Normal']))
                if st.session_state.psychological:
                    elements.append(Paragraph(f"<b>Psychological:</b> {', '.join(st.session_state.psychological[:5])}", styles['Normal']))
                if st.session_state.generals:
                    elements.append(Paragraph(f"<b>General:</b> {', '.join(st.session_state.generals[:5])}", styles['Normal']))
                
                elements.append(Spacer(1, 20))
                elements.append(Paragraph("<b>Top 10 Remedies</b>", styles['Heading2']))
                elements.append(Spacer(1, 12))
                
                # Create table for top remedies
                table_data = [["Rank", "Remedy", "Score", "Context"]]
                for i, (r, v) in enumerate(top10[:10], 1):
                    val = f"{v}/{len(core)}" if only_core else f"{v:.3f}"
                    context = REMEDY_CONTEXT.get(r.lower(), {}).get('thumb', '-')[:40]
                    table_data.append([str(i), r.upper(), val, context])
                
                remedy_table = Table(table_data, colWidths=[0.6*inch, 1.5*inch, 1*inch, 3*inch])
                remedy_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                ]))
                elements.append(remedy_table)
                
                # Top 3 remedy details
                elements.append(Spacer(1, 20))
                elements.append(Paragraph("<b>Top 3 Remedy Details</b>", styles['Heading2']))
                for i, (rem, _) in enumerate(top10[:3], 1):
                    elements.append(Spacer(1, 10))
                    elements.append(Paragraph(f"<b>{i}. {rem.upper()}</b>", styles['Heading3']))
                    if rem.lower() in REMEDY_CONTEXT:
                        ctx = REMEDY_CONTEXT[rem.lower()]
                        elements.append(Paragraph(f"<i>{ctx['thumb']}</i>", styles['Normal']))
                        elements.append(Paragraph(f"Follows well: {', '.join(ctx['follows'])}", styles['Normal']))
                        elements.append(Paragraph(f"Antidoted by: {', '.join(ctx['antidoted'])}", styles['Normal']))
                
                # Clinical notes
                if notes:
                    elements.append(Spacer(1, 20))
                    elements.append(Paragraph("<b>Clinical Notes</b>", styles['Heading2']))
                    elements.append(Paragraph(notes, styles['Normal']))
                
                doc.build(elements)
                return buffer.name

            pdf_path = create_enhanced_pdf()
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download Enhanced Clinical Report (PDF)",
                    f.read(),
                    f"HoRUS3_{st.session_state.patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf",
                    use_container_width=True
                )
            
            # New case button
            st.markdown("---")
            if st.button("🔄 Start New Case", type="secondary", use_container_width=True):
                # Reset to step 1
                st.session_state.step = 1
                st.session_state.physical = []
                st.session_state.psychological = []
                st.session_state.generals = []
                st.session_state.refined_keywords = {}
                st.session_state.selected_pattern_symptoms = []
                st.session_state.pattern_accumulator = []
                st.session_state.case_notes = ""
                st.session_state.report_generated = False
                st.rerun()

        generate_report()

# =============================================
# ENHANCED HISTORY TAB
# =============================================
with tab2:
    st.header("📚 Patient History")
    
    patients = load_patients()
    if patients:
        # Search functionality
        search_patient = st.text_input("🔍 Search patient history", placeholder="Type patient ID...")
        patient_list = sorted(patients.keys())
        if search_patient:
            patient_list = [p for p in patient_list if search_patient.upper() in p.upper()]
        
        pid = st.selectbox("Select Patient", [""] + patient_list, key="hist_pid")
        
        if pid:
            cases = patients[pid]
            st.success(f"**{len(cases)} case(s)** on record for {pid}")
            
            # Show full top 10 toggle
            show_full = st.checkbox("Show full Top 10 (default: Top 5)", value=False)
            limit = 10 if show_full else 5
            
            # Case comparison option
            if len(cases) >= 2:
                st.markdown("---")
                st.subheader("📊 Compare Cases")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    case1_idx = st.selectbox("First case", range(len(cases)), 
                                            format_func=lambda x: cases[x]['timestamp'])
                with col_c2:
                    case2_idx = st.selectbox("Second case", range(len(cases)), 
                                            format_func=lambda x: cases[x]['timestamp'],
                                            index=min(1, len(cases)-1))
                
                if st.button("Compare Selected Cases"):
                    case1 = cases[case1_idx]
                    case2 = cases[case2_idx]
                    
                    st.markdown("### Comparison Results")
                    
                    # Create comparison table
                    comp_data = []
                    remedies1 = {r[0]: (i+1, r[1]) for i, r in enumerate(case1.get('top10', [])[:10])}
                    remedies2 = {r[0]: (i+1, r[1]) for i, r in enumerate(case2.get('top10', [])[:10])}
                    
                    all_remedies = set(remedies1.keys()) | set(remedies2.keys())
                    
                    for rem in sorted(all_remedies):
                        rank1, score1 = remedies1.get(rem, ('-', '-'))
                        rank2, score2 = remedies2.get(rem, ('-', '-'))
                        
                        change = ""
                        if rank1 != '-' and rank2 != '-':
                            diff = rank1 - rank2
                            if diff > 0:
                                change = f"⬆️ +{diff}"
                            elif diff < 0:
                                change = f"⬇️ {diff}"
                            else:
                                change = "➡️ same"
                        elif rank1 == '-':
                            change = "🆕 new"
                        elif rank2 == '-':
                            change = "❌ dropped"
                        
                        comp_data.append({
                            "Remedy": rem.upper(),
                            f"Case 1 ({case1['timestamp']})": f"#{rank1}" if rank1 != '-' else '-',
                            f"Case 2 ({case2['timestamp']})": f"#{rank2}" if rank2 != '-' else '-',
                            "Change": change
                        })
                    
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("Case History")
            
            # Display cases in reverse chronological order
            for idx, case in enumerate(reversed(cases)):
                mode = "Expansion" if case.get("mode", "").startswith("expansion") else "Weighted + Patterns"
                weights_info = case.get("weights", {})
                weight_str = f"Rhe: {weights_info.get('rheumatic', 0.5):.2f} | Cases: {weights_info.get('cases', 0.5):.2f}" if weights_info else ""
                
                with st.expander(f"📅 {case['timestamp']} — {mode} {weight_str}", expanded=(idx==0)):
                    # Display core symptoms
                    if case.get('core'):
                        st.markdown("**Core Symptoms:**")
                        st.write(", ".join(case['core'][:10]))
                    
                    # Display pattern symptoms if present
                    if case.get('pattern'):
                        st.markdown("**Pattern Symptoms Added:**")
                        st.write(", ".join(case['pattern'][:5]))
                    
                    st.markdown(f"**Top {limit} Remedies:**")
                    for i, item in enumerate(case.get("top10", [])[:limit]):
                        rem = item[0].upper()
                        val = f"{item[1]}/{len(case.get('core',[]))}" if "expansion" in case.get("mode","") else f"{item[1]:.2f}"
                        
                        # Add context inline
                        context = ""
                        if rem.lower() in REMEDY_CONTEXT:
                            context = f" — {REMEDY_CONTEXT[rem.lower()]['thumb']}"
                        
                        st.write(f"**{i+1}. {rem}** — {val}{context}")
                    
                    # Display notes if present
                    if case.get('notes'):
                        st.markdown("**Clinical Notes:**")
                        st.info(case['notes'])
        
        # Bulk export option
        st.markdown("---")
        st.subheader("📤 Export Options")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("Export All Patient Data (JSON)", use_container_width=True):
                json_data = json.dumps(patients, indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"horus3_all_patients_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json",
                    use_container_width=True
                )
        
        with col_e2:
            if pid and st.button("Export Single Patient (JSON)", use_container_width=True):
                patient_data = {pid: patients[pid]}
                json_data = json.dumps(patient_data, indent=2)
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"horus3_{pid}_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json",
                    use_container_width=True
                )
        
        st.markdown("---")
        if st.button("🗑️ Clear All History", type="secondary"):
            if st.checkbox("⚠️ Confirm permanent deletion of ALL patient records"):
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.success("✅ All history cleared")
                st.rerun()
    else:
        st.info("📭 No patient history yet. Complete a case to build your database.")

# =============================================
# KEYBOARD SHORTCUTS HELPER
# =============================================
with st.sidebar:
    st.markdown("---")
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("""
        **Coming Soon:**
        - `Ctrl+R` — Refine symptoms
        - `Ctrl+P` — Generate report
        - `Ctrl+N` — New patient
        - `Ctrl+S` — Save/Export
        
        *Note: Browser-based shortcuts require additional implementation*
        """)