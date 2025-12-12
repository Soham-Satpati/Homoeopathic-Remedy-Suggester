import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
from collections import defaultdict, Counter
import json
from datetime import datetime
import google.generativeai as genai
import re

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="HoRUS 3 AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CONFIGURATION
# =============================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")  # Store in Streamlit secrets
HISTORY_FILE = "horus3_patients.json"

# Remedy context data
REMEDY_CONTEXT = {
    "arnica": {"follows": ["Rhus-t", "Calc"], "antidoted": ["Camph"], "thumb": "Trauma, bruising, soreness"},
    "rhus-t": {"follows": ["Arn", "Bry"], "antidoted": ["Bell"], "thumb": "Restlessness, worse rest, better motion"},
    "bryonia": {"follows": ["Rhus-t", "Arn"], "antidoted": ["Acon"], "thumb": "Worse motion, better rest, irritable"},
    "pulsatilla": {"follows": ["Kali-s", "Sil"], "antidoted": ["Cham"], "thumb": "Changeable, mild, desires company"},
    "sulphur": {"follows": ["Acon", "Ars"], "antidoted": ["Lyc"], "thumb": "Burning, untidy, worse heat"},
    "aconite": {"follows": ["Arn", "Sulph"], "antidoted": ["Coff"], "thumb": "Sudden onset, fear, anxiety"},
    "belladonna": {"follows": ["Calc", "Hep"], "antidoted": ["Camph"], "thumb": "Throbbing, redness, heat"},
    "natrum-mur": {"follows": ["Ign", "Sep"], "antidoted": ["Ars"], "thumb": "Grief, salt craving, worse consolation"},
    "phosphorus": {"follows": ["Ars", "All-c"], "antidoted": ["Nux-v"], "thumb": "Sympathetic, fearful, desires company"},
    "sepia": {"follows": ["Nat-m", "Puls"], "antidoted": ["Acon"], "thumb": "Indifferent, bearing down, worse pregnancy"},
}

# =============================================
# LOAD SYSTEM
# =============================================
@st.cache_resource
def load_system():
    """Load ML models and datasets"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load case studies
        if os.path.exists("case_studies_model.pkl"):
            with open("case_studies_model.pkl", "rb") as f:
                case_dict = pickle.load(f)
        else:
            case_dict = {'symptom_to_remedies': {}, 'categories': {}}
        
        # Load rheumatic data
        if os.path.exists("rheumatic_model.pkl"):
            with open("rheumatic_model.pkl", "rb") as f:
                rhe_dict = pickle.load(f)
        else:
            rhe_dict = {'symptom_to_remedies': {}, 'categories': {}}
        
        # Load clusters
        clusters = {}
        for name in ["remedy_modalities", "remedy_area_modalities", "remedy_area"]:
            file = f"clusters_{name}.csv"
            if os.path.exists(file):
                df = pd.read_csv(file)
                df['Cluster_ID'] = df['Cluster_ID'].astype(str)
                clusters[name] = df
        
        # Combine chapters
        chapters = defaultdict(list)
        for data in [case_dict, rhe_dict]:
            for sym, chap in data.get('categories', {}).items():
                chapters[chap].append(sym)
        
        # Combine symptom to remedies
        s2r = {**case_dict.get('symptom_to_remedies', {}), **rhe_dict.get('symptom_to_remedies', {})}
        
        return {
            'model': model,
            'chapters': dict(chapters),
            'clusters': clusters,
            's2r': s2r,
            'case_dict': case_dict,
            'rhe_dict': rhe_dict
        }
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None

# =============================================
# GEMINI AGENT
# =============================================
def initialize_gemini():
    """Initialize Gemini API"""
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-pro')
    return None

def call_gemini_agent(prompt, context, model):
    """Call Gemini API with context"""
    if not model:
        return "⚠️ Gemini API key not configured. Please add it to Streamlit secrets."
    
    try:
        full_prompt = f"""You are HoRUS 3, an expert homeopathic case analysis assistant with deep knowledge of repertorization and materia medica.

Current Conversation Context:
- Stage: {context['stage']}
- Patient ID: {context.get('patient_id', 'Not set')}
- Symptoms collected: Physical={len(context.get('symptoms', {}).get('physical', []))}, Psychological={len(context.get('symptoms', {}).get('psychological', []))}, General={len(context.get('symptoms', {}).get('general', []))}

Your capabilities:
1. Extract and categorize symptoms from natural language
2. Suggest repertory-style refinements
3. Identify concomitant symptoms and patterns
4. Explain remedy relationships and materia medica

Guidelines:
- Be conversational but professional
- When extracting symptoms, format them clearly
- For physical symptoms, look for: pain, sensations, locations, modalities
- For psychological symptoms, look for: emotions, mental states, fears
- For general symptoms, look for: temperature reactions, time aggravations, food desires
- Always maintain clinical accuracy

User's message: {prompt}

Respond appropriately based on the current stage. If extracting symptoms, format them as:
PHYSICAL: [list]
PSYCHOLOGICAL: [list]
GENERAL: [list]
"""
        
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# =============================================
# SYMPTOM PARSING AGENT
# =============================================
def parse_symptoms_with_ai(text, gemini_model):
    """Use AI to intelligently parse and categorize symptoms"""
    prompt = f"""Parse these symptoms into categories. Be thorough and specific:

Text: {text}

Return in this exact format:
PHYSICAL: symptom1 | symptom2 | symptom3
PSYCHOLOGICAL: symptom1 | symptom2
GENERAL: symptom1 | symptom2

Physical symptoms include: pains, sensations, locations, physical complaints
Psychological symptoms include: emotions, mental states, fears, desires
General symptoms include: temperature modalities, time modalities, food cravings/aversions, general states

Be specific and preserve the exact wording where possible."""

    try:
        response = gemini_model.generate_content(prompt)
        text = response.text
        
        symptoms = {'physical': [], 'psychological': [], 'general': []}
        
        # Parse AI response
        if 'PHYSICAL:' in text:
            phys_text = text.split('PHYSICAL:')[1].split('PSYCHOLOGICAL:')[0].strip()
            symptoms['physical'] = [s.strip() for s in phys_text.split('|') if s.strip()]
        
        if 'PSYCHOLOGICAL:' in text:
            psych_text = text.split('PSYCHOLOGICAL:')[1].split('GENERAL:')[0].strip()
            symptoms['psychological'] = [s.strip() for s in psych_text.split('|') if s.strip()]
        
        if 'GENERAL:' in text:
            gen_text = text.split('GENERAL:')[1].strip()
            symptoms['general'] = [s.strip() for s in gen_text.split('|') if s.strip()]
        
        return symptoms
    except:
        # Fallback to simple parsing
        return simple_symptom_parse(text)

def simple_symptom_parse(text):
    """Simple fallback parsing without AI"""
    symptoms = {'physical': [], 'psychological': [], 'general': []}
    
    physical_keywords = ['pain', 'ache', 'sore', 'swelling', 'rash', 'fever', 'cough', 
                         'throat', 'stomach', 'head', 'joint', 'muscle', 'chest']
    psych_keywords = ['anxiety', 'fear', 'anger', 'sad', 'irritable', 'restless', 
                      'depressed', 'worried', 'nervous', 'stressed']
    general_keywords = ['worse', 'better', 'morning', 'evening', 'cold', 'heat', 
                        'motion', 'rest', 'weather', 'night']
    
    sentences = re.split(r'[.,;]', text.lower())
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if any(kw in sentence for kw in physical_keywords):
            symptoms['physical'].append(sentence)
        elif any(kw in sentence for kw in psych_keywords):
            symptoms['psychological'].append(sentence)
        elif any(kw in sentence for kw in general_keywords):
            symptoms['general'].append(sentence)
        else:
            symptoms['physical'].append(sentence)
    
    return symptoms

# =============================================
# REPERTORIZATION AGENT
# =============================================
def repertorize_symptoms(symptoms, system_data, rhe_weight=0.5, case_weight=0.5):
    """Perform weighted repertorization"""
    s2r = system_data['s2r']
    rhe_dict = system_data['rhe_dict']
    
    remedy_scores = defaultdict(float)
    remedy_coverage = defaultdict(list)
    
    all_symptoms = symptoms['physical'] + symptoms['psychological'] + symptoms['general']
    
    for symptom in all_symptoms:
        if symptom in s2r:
            for remedy, score in s2r[symptom].items():
                # Apply dataset weighting
                weight = rhe_weight if symptom in rhe_dict.get('symptom_to_remedies', {}) else case_weight
                remedy_scores[remedy] += score * weight
                remedy_coverage[remedy].append(symptom)
    
    # Sort by score
    sorted_remedies = sorted(remedy_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Add coverage information
    results = []
    for remedy, score in sorted_remedies[:10]:
        context = REMEDY_CONTEXT.get(remedy.lower(), {})
        results.append({
            'remedy': remedy,
            'score': score,
            'coverage': remedy_coverage[remedy],
            'context': context
        })
    
    return results

# =============================================
# PATTERN DISCOVERY AGENT
# =============================================
def discover_patterns(symptoms, system_data, rarity='Common', confidence=50):
    """Discover concomitant symptom patterns from clusters"""
    clusters = system_data['clusters']
    
    size_ranges = {
        "Common": (25, float('inf')),
        "Uncommon": (12, 24),
        "Rare": (0, 11)
    }
    min_size, max_size = size_ranges[rarity]
    
    found_patterns = []
    seen_clusters = set()
    all_symptoms = symptoms['physical'] + symptoms['psychological'] + symptoms['general']
    
    for _, df in clusters.items():
        for keyword in all_symptoms:
            rows = df[df['Symptom'].str.contains(keyword, case=False, na=False)]
            for _, row in rows.iterrows():
                cid = str(row['Cluster_ID'])
                if cid in {'NOISE', '-1'} or cid in seen_clusters:
                    continue
                
                cluster_df = df[df['Cluster_ID'] == cid]
                cluster_size = len(cluster_df)
                
                if min_size <= cluster_size <= max_size:
                    frequency = min(100, cluster_size * 2)
                    
                    if frequency >= confidence:
                        seen_clusters.add(cid)
                        cluster_symptoms = cluster_df['Symptom'].tolist()
                        new_symptoms = [s for s in cluster_symptoms if s not in all_symptoms]
                        
                        if new_symptoms:
                            found_patterns.append({
                                'symptoms': new_symptoms,
                                'frequency': frequency,
                                'size': cluster_size,
                                'trigger': keyword
                            })
    
    return found_patterns[:8]

# =============================================
# PATIENT MANAGEMENT
# =============================================
def load_patients():
    """Load patient history"""
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_patient_case(patient_id, case_data):
    """Save a patient case"""
    patients = load_patients()
    patients.setdefault(patient_id, [])
    
    case_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patients[patient_id].append(case_data)
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(patients, f, indent=2)

def generate_patient_id():
    """Generate next available patient ID"""
    patients = load_patients()
    year = datetime.now().year
    prefix = f"PT-{year}-"
    
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

# =============================================
# SESSION STATE INITIALIZATION
# =============================================
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = {
        'stage': 'greeting',  # greeting, symptoms_collection, refinement, pattern_discovery, report
        'patient_id': '',
        'symptoms': {'physical': [], 'psychological': [], 'general': []},
        'refined_symptoms': {},
        'pattern_symptoms': [],
        'top_remedies': [],
        'notes': '',
        'rhe_weight': 0.5,
        'case_weight': 0.5
    }

if 'system_loaded' not in st.session_state:
    with st.spinner("🔄 Loading HoRUS 3 AI System..."):
        st.session_state.system_data = load_system()
        st.session_state.gemini_model = initialize_gemini()
        st.session_state.system_loaded = True

# =============================================
# SIDEBAR
# =============================================
with st.sidebar:
    st.title("🤖 HoRUS 3 AI")
    st.caption("Clinical Homeopathy Chatbot")
    
    st.divider()
    
    # Patient Info
    if st.session_state.conversation_state['patient_id']:
        st.subheader("📋 Current Case")
        st.info(f"**Patient:** {st.session_state.conversation_state['patient_id']}")
        
        symptoms = st.session_state.conversation_state['symptoms']
        st.metric("Physical Symptoms", len(symptoms['physical']))
        st.metric("Psychological Symptoms", len(symptoms['psychological']))
        st.metric("General Symptoms", len(symptoms['general']))
        
        # Stage indicator
        stage = st.session_state.conversation_state['stage']
        stage_names = {
            'greeting': '1️⃣ Setup',
            'symptoms_collection': '2️⃣ Collecting',
            'refinement': '3️⃣ Refining',
            'pattern_discovery': '4️⃣ Patterns',
            'report': '5️⃣ Report'
        }
        st.success(f"**Stage:** {stage_names.get(stage, stage)}")
    
    st.divider()
    
    # Dataset Weights
    st.subheader("⚖️ Dataset Weights")
    rhe_w = st.slider(
        "Rheumatic Dataset",
        0.0, 1.0,
        st.session_state.conversation_state['rhe_weight'],
        0.05,
        help="Higher = More emphasis on traditional texts"
    )
    case_w = round(1.0 - rhe_w, 2)
    
    st.session_state.conversation_state['rhe_weight'] = rhe_w
    st.session_state.conversation_state['case_weight'] = case_w
    st.caption(f"Case Studies: {case_w:.2f}")
    
    st.divider()
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 New Case", use_container_width=True):
        st.session_state.conversation_state = {
            'stage': 'greeting',
            'patient_id': '',
            'symptoms': {'physical': [], 'psychological': [], 'general': []},
            'refined_symptoms': {},
            'pattern_symptoms': [],
            'top_remedies': [],
            'notes': '',
            'rhe_weight': 0.5,
            'case_weight': 0.5
        }
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📚 View History", use_container_width=True):
        st.session_state.show_history = not st.session_state.get('show_history', False)
        st.rerun()
    
    if st.session_state.conversation_state['top_remedies']:
        if st.button("📥 Export Report", use_container_width=True):
            st.session_state.show_export = True
            st.rerun()
    
    st.divider()
    
    # API Status
    if st.session_state.gemini_model:
        st.success("✅ Gemini API Connected")
    else:
        st.error("❌ Gemini API Not Configured")
        with st.expander("How to configure"):
            st.code("""
# Add to .streamlit/secrets.toml
GEMINI_API_KEY = "your-api-key-here"
            """)

# =============================================
# MAIN CHAT INTERFACE
# =============================================
st.title("💬 HoRUS 3 AI Assistant")
st.caption("Conversational Homeopathic Case Analysis")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initial greeting
if len(st.session_state.messages) == 0:
    greeting = """👋 **Welcome to HoRUS 3 AI Assistant!**

I'm here to help you analyze homeopathic cases intelligently. I can:
- 🎯 Extract and categorize symptoms from natural descriptions
- 🔍 Refine symptoms to repertory language
- 🔬 Discover hidden concomitant patterns
- 📊 Generate comprehensive remedy reports

**Let's begin!** Please provide a Patient ID, or I can generate one for you.

You can say:
- "Generate a new patient ID"
- "Use patient PT-2025-001"
- "Show patient history"
"""
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    with st.chat_message("assistant"):
        st.markdown(greeting)

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process message
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            context = st.session_state.conversation_state
            stage = context['stage']
            response = ""
            
            # ===== STAGE: GREETING (Patient ID Setup) =====
            if stage == 'greeting':
                lower_prompt = prompt.lower()
                
                if 'generate' in lower_prompt or 'new' in lower_prompt or 'create' in lower_prompt:
                    new_id = generate_patient_id()
                    context['patient_id'] = new_id
                    context['stage'] = 'symptoms_collection'
                    
                    response = f"""✅ **Patient ID Generated:** `{new_id}`

Great! Now let's collect the case symptoms. You can describe them naturally - I'll organize them for you.

**Tell me about:**
🏥 Physical symptoms (pains, sensations, locations)
🧠 Psychological symptoms (emotions, mental states)
🌡️ General symptoms (modalities, time aggravations)

For example: *"The patient has severe headache worse from motion, anxiety about health, and feels worse in cold weather."*
"""
                
                elif re.search(r'pt-\d{4}-\d{3}', lower_prompt, re.IGNORECASE):
                    match = re.search(r'pt-\d{4}-\d{3}', lower_prompt, re.IGNORECASE)
                    pid = match.group(0).upper()
                    context['patient_id'] = pid
                    context['stage'] = 'symptoms_collection'
                    
                    response = f"""✅ **Using Patient ID:** `{pid}`

Perfect! Now let's gather the symptoms. Describe the case naturally, and I'll categorize everything.
"""
                
                elif 'history' in lower_prompt or 'previous' in lower_prompt:
                    patients = load_patients()
                    if patients:
                        patient_list = "\n".join([f"- {pid} ({len(cases)} cases)" for pid, cases in patients.items()])
                        response = f"📚 **Patient History:**\n\n{patient_list}\n\nWhich patient would you like to work with?"
                    else:
                        response = "📭 No patient history found yet. Let's create a new case!"
                
                else:
                    response = call_gemini_agent(prompt, context, st.session_state.gemini_model)
            
            # ===== STAGE: SYMPTOMS COLLECTION =====
            elif stage == 'symptoms_collection':
                lower_prompt = prompt.lower()
                
                if 'done' in lower_prompt or 'finished' in lower_prompt or 'next' in lower_prompt or 'proceed' in lower_prompt:
                    total = sum(len(v) for v in context['symptoms'].values())
                    
                    if total == 0:
                        response = "⚠️ No symptoms collected yet. Please describe at least one symptom before proceeding."
                    else:
                        context['stage'] = 'refinement'
                        response = f"""📊 **Symptoms Summary:**

✅ Physical: {len(context['symptoms']['physical'])} symptoms
✅ Psychological: {len(context['symptoms']['psychological'])} symptoms
✅ General: {len(context['symptoms']['general'])} symptoms

**Total: {total} symptoms collected**

**What would you like to do next?**

1️⃣ **Refine symptoms** - Convert to repertory language (recommended)
2️⃣ **Discover patterns** - Find concomitant symptoms from case clusters
3️⃣ **Generate report** - Skip to remedy analysis

Just tell me your choice (e.g., "refine symptoms" or "option 1")
"""
                
                else:
                    # Parse symptoms using AI
                    parsed = parse_symptoms_with_ai(prompt, st.session_state.gemini_model)
                    
                    # Add to context
                    for category in ['physical', 'psychological', 'general']:
                        context['symptoms'][category].extend(parsed[category])
                    
                    count = sum(len(v) for v in parsed.values())
                    
                    if count > 0:
                        response = f"""✅ **Added {count} symptom(s)!**

**Breakdown:**
- 🏥 Physical: {len(parsed['physical'])} → Total: {len(context['symptoms']['physical'])}
- 🧠 Psychological: {len(parsed['psychological'])} → Total: {len(context['symptoms']['psychological'])}
- 🌡️ General: {len(parsed['general'])} → Total: {len(context['symptoms']['general'])}

"""
                        # Show what was added
                        if parsed['physical']:
                            response += f"\n**Physical:** {', '.join(parsed['physical'][:3])}"
                        if parsed['psychological']:
                            response += f"\n**Psychological:** {', '.join(parsed['psychological'][:3])}"
                        if parsed['general']:
                            response += f"\n**General:** {', '.join(parsed['general'][:3])}"
                        
                        response += "\n\n💡 Add more symptoms, or say **'done'** to proceed to the next stage."
                    else:
                        response = "🤔 I couldn't identify specific symptoms in that message. Could you rephrase? For example: 'The patient has throbbing headache worse from noise and light.'"
            
            # ===== STAGE: REFINEMENT =====
            elif stage == 'refinement':
                lower_prompt = prompt.lower()
                
                if 'skip' in lower_prompt or 'report' in lower_prompt or '3' in lower_prompt:
                    # Generate report directly
                    with st.spinner("🔮 Performing repertorization..."):
                        results = repertorize_symptoms(
                            context['symptoms'],
                            st.session_state.system_data,
                            context['rhe_weight'],
                            context['case_weight']
                        )
                        context['top_remedies'] = results
                        context['stage'] = 'report'
                        
                        response = "📊 **Clinical Report Generated!**\n\n"
                        response += f"**Top 10 Remedies for {context['patient_id']}:**\n\n"
                        
                        for i, remedy_data in enumerate(results, 1):
                            remedy = remedy_data['remedy'].upper()
                            score = remedy_data['score']
                            ctx = remedy_data['context']
                            
                            response += f"**{i}. {remedy}** — Score: {score:.3f}\n"
                            if ctx:
                                response += f"   💡 *{ctx.get('thumb', '')}*\n"
                                response += f"   ▶️ Follows: {', '.join(ctx.get('follows', []))}\n"
                            response += "\n"
                        
                        response += "\n📝 You can now add clinical notes or start a new case."
                
                elif 'pattern' in lower_prompt or 'discover' in lower_prompt or '2' in lower_prompt:
                    context['stage'] = 'pattern_discovery'
                    response = """🔬 **Pattern Discovery Mode Activated**

I'll analyze thousands of real cases to find concomitant symptoms that frequently appear with your current symptom picture.

**Choose pattern type:**
- **Common** patterns (appear in 25+ cases)
- **Uncommon** patterns (12-24 cases)
- **Rare** patterns (less than 12 cases)

Say something like: "Show me common patterns" or "Find rare patterns"
"""
                
                else:
                    # Use Gemini for refinement suggestions
                    response = call_gemini_agent(
                        f"Suggest repertory-style refinements for these symptoms: {context['symptoms']}",
                        context,
                        st.session_state.gemini_model
                    )
                    response += "\n\n💡 Say **'generate report'** when ready, or **'discover patterns'** to find concomitants."
            
            # ===== STAGE: PATTERN DISCOVERY =====
            elif stage == 'pattern_discovery':
                lower_prompt = prompt.lower()
                
                # Determine rarity
                rarity = 'Common'
                if 'uncommon' in lower_prompt:
                    rarity = 'Uncommon'
                elif 'rare' in lower_prompt:
                    rarity = 'Rare'
                
                if 'skip' in lower_prompt or 'report' in lower_prompt or 'done' in lower_prompt:
                    # Generate report
                    with st.spinner("🔮 Performing repertorization..."):
                        results = repertorize_symptoms(
                            context['symptoms'],
                            st.session_state.system_data,
                            context['rhe_weight'],
                            context['case_weight']
                        )
                        context['top_remedies'] = results
                        context['stage'] = 'report'
                        
                        response = "📊 **Report Generated!** (See above for top remedies)"
                
                else:
                    # Discover patterns
                    with st.spinner(f"🔍 Finding {rarity.lower()} patterns..."):
                        patterns = discover_patterns(
                            context['symptoms'],
                            st.session_state.system_data,
                            rarity,
                            confidence=50
                        )
                        
                        if patterns:
                            response = f"🎯 **Found {len(patterns)} {rarity} Pattern(s):**\n\n"
                            
                            for i, pattern in enumerate(patterns[:5], 1):
                                response += f"**Pattern {i}** (Frequency: {pattern['frequency']}% | Triggered by: *{pattern['trigger']}*)\n"
                                response += f"Symptoms: {', '.join(pattern['symptoms'][:5])}\n\n"
                            
                            response += "\n💡 Would you like to add any patterns, or say **'generate report'** to proceed?"
                        else:
                            response = f"🔍 No {rarity.lower()} patterns found matching your criteria. Try a different rarity level or proceed to generate the report."
            
            # ===== STAGE: REPORT =====
            elif stage == 'report':
                lower_prompt = prompt.lower()
                
                if 'new case' in lower_prompt or 'start over' in lower_prompt:
                    # Save current case
                    save_patient_case(context['patient_id'], {
                        'symptoms': context['symptoms'],
                        'top_remedies': [(r['remedy'], r['score']) for r in context['top_remedies']],
                        'notes': context.get('notes', ''),
                        'weights': {'rheumatic': context['rhe_weight'], 'cases': context['case_weight']}
                    })
                    
                    # Reset
                    st.session_state.conversation_state = {
                        'stage': 'greeting',
                        'patient_id': '',
                        'symptoms': {'physical': [], 'psychological': [], 'general': []},
                        'refined_symptoms': {},
                        'pattern_symptoms': [],
                        'top_remedies': [],
                        'notes': '',
                        'rhe_weight': 0.5,
                        'case_weight': 0.5
                    }
                    
                    response = "✅ Case saved! 🔄 Starting new case...\n\nPlease provide a Patient ID or say 'generate new patient ID'."
                
                elif 'note' in lower_prompt:
                    context['notes'] = prompt
                    response = "📝 Clinical notes saved! Anything else you'd like to document?"
                
                else:
                    # General query about remedies
                    response = call_gemini_agent(prompt, context, st.session_state.gemini_model)
            
            # Display response
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# =============================================
# PATIENT HISTORY PANEL
# =============================================
if st.session_state.get('show_history', False):
    st.divider()
    st.subheader("📚 Patient History")
    
    patients = load_patients()
    
    if patients:
        cols = st.columns([3, 1])
        with cols[0]:
            search_query = st.text_input("🔍 Search patients", placeholder="Type patient ID...")
        with cols[1]:
            st.write("")
            st.write("")
            if st.button("❌ Close", use_container_width=True):
                st.session_state.show_history = False
                st.rerun()
        
        patient_list = sorted(patients.keys())
        if search_query:
            patient_list = [p for p in patient_list if search_query.upper() in p.upper()]
        
        for pid in patient_list:
            cases = patients[pid]
            with st.expander(f"👤 {pid} — {len(cases)} case(s)", expanded=False):
                for i, case in enumerate(reversed(cases), 1):
                    st.caption(f"**Case {i}** — {case.get('timestamp', 'Unknown date')}")
                    
                    symptoms = case.get('symptoms', {})
                    st.write(f"Symptoms: P={len(symptoms.get('physical', []))}, Ps={len(symptoms.get('psychological', []))}, G={len(symptoms.get('general', []))}")
                    
                    top_remedies = case.get('top_remedies', [])[:5]
                    if top_remedies:
                        remedy_text = ", ".join([f"{r[0]}" for r in top_remedies])
                        st.write(f"**Top 5:** {remedy_text}")
                    
                    if case.get('notes'):
                        st.info(f"📝 {case['notes']}")
                    
                    st.divider()
    else:
        st.info("📭 No patient history yet.")

# =============================================
# EXPORT PANEL
# =============================================
if st.session_state.get('show_export', False):
    st.divider()
    st.subheader("📥 Export Report")
    
    context = st.session_state.conversation_state
    
    report_text = f"""HoRUS 3 Clinical Report
{'='*50}

Patient ID: {context['patient_id']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset Weights: Rheumatic={context['rhe_weight']:.2f}, Cases={context['case_weight']:.2f}

SYMPTOM SUMMARY
{'-'*50}
Physical: {len(context['symptoms']['physical'])} symptoms
Psychological: {len(context['symptoms']['psychological'])} symptoms
General: {len(context['symptoms']['general'])} symptoms

TOP 10 REMEDIES
{'-'*50}
"""
    
    for i, remedy_data in enumerate(context['top_remedies'], 1):
        report_text += f"\n{i}. {remedy_data['remedy'].upper()} — Score: {remedy_data['score']:.3f}\n"
        if remedy_data['context']:
            report_text += f"   {remedy_data['context'].get('thumb', '')}\n"
    
    if context.get('notes'):
        report_text += f"\n\nCLINICAL NOTES\n{'-'*50}\n{context['notes']}\n"
    
    st.download_button(
        "📄 Download as TXT",
        report_text,
        file_name=f"HoRUS3_{context['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain"
    )
    
    if st.button("❌ Close Export Panel"):
        st.session_state.show_export = False
        st.rerun()

# =============================================
# FOOTER
# =============================================
st.divider()
st.caption("🤖 HoRUS 3 AI Assistant • Powered by Google Gemini • Streamlit Cloud Ready")
