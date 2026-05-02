#!/usr/bin/env python3
"""
NeuroVision — AI-Powered Brain Disease Detection System
────────────────────────────────────────────────────────
Developed by:
  · Dhruv Sharma  — 11023210198
  · Akshat Jain   — 11023210142

Guided by : Ms. Anju Goel Ma'am
Institution: SRM University Delhi-NCR, Sonepat
Department : B.Tech Computer Science

PASSWORD : nanital

ASSETS (same folder as app.py):
  · neurovision_model.keras
  · 220232_medium.mp4
  · 1777098010701_image.png   ← brain hero
  · 1777098001570_image.png   ← SRM logo
"""

import os, base64, hashlib
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import tensorflow as tf

# ════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroVision | AI Brain Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════════
#  PASSWORD  — sha256("nanital")
# ════════════════════════════════════════════════════════════════════════════
_CORRECT_HASH = hashlib.sha256("nanital".encode()).hexdigest()

def _check_pw(raw: str) -> bool:
    return hashlib.sha256(raw.strip().encode()).hexdigest() == _CORRECT_HASH

# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
for _k, _v in [("authenticated", False), ("page", "intro"),
               ("chat_open", False), ("chat_history", []), ("pw_error", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"]{background:#040810!important;color:#cfd8dc!important;font-family:'Rajdhani',sans-serif!important;}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stSidebar"]{display:none!important;}
#MainMenu,footer{visibility:hidden!important;}
.main .block-container{padding:0!important;max-width:100%!important;}

/* Buttons */
.stButton>button{
  background:linear-gradient(135deg,#00bcd4,#0288d1)!important;color:#fff!important;
  border:none!important;border-radius:30px!important;padding:.65rem 1.8rem!important;
  font-family:'Rajdhani',sans-serif!important;font-size:1rem!important;font-weight:700!important;
  letter-spacing:1.5px!important;text-transform:uppercase!important;transition:all .3s!important;
  box-shadow:0 4px 15px rgba(0,188,212,.25)!important;}
.stButton>button:hover{
  background:linear-gradient(135deg,#00acc1,#0277bd)!important;
  transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(0,188,212,.5)!important;}

/* File uploader */
[data-testid="stFileUploader"]{
  background:rgba(0,188,212,.05)!important;border:2px dashed rgba(0,188,212,.35)!important;
  border-radius:14px!important;padding:10px!important;}
[data-testid="stFileUploader"] section{background:transparent!important;}

/* Password input */
[data-testid="stTextInput"] input{
  background:rgba(0,188,212,0.08)!important;border:1px solid rgba(0,188,212,0.4)!important;
  border-radius:10px!important;color:#e0f7fa!important;
  font-family:'Rajdhani',sans-serif!important;font-size:16px!important;letter-spacing:2px!important;}

/* Plotly */
.js-plotly-plot{border-radius:12px!important;}

/* Scrollbar */
::-webkit-scrollbar{width:6px;}
::-webkit-scrollbar-track{background:#040810;}
::-webkit-scrollbar-thumb{background:#00bcd430;border-radius:3px;}

/* Floating chatbot FAB — purely CSS, positioned fixed */
@keyframes fabPulse{
  0%,100%{box-shadow:0 4px 24px rgba(0,188,212,0.55);}
  50%{box-shadow:0 4px 44px rgba(0,188,212,0.95);}
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  MODEL  — ✅ BUG FIXED: alphabetical class order + ResNet50 preprocessing
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("neurovision_model.keras", compile=False)
    except Exception:
        return None

CLASS_NAMES = ['alzheimer', 'stroke', 'tumor']   # ✅ alphabetical
IMG_SIZE    = (224, 224)

def predict(img: Image.Image):
    model = load_model()
    if model is None:
        return None, 0.0, None
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)   # ✅ NOT /255
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASS_NAMES[idx], float(preds[idx]) * 100, preds

# ════════════════════════════════════════════════════════════════════════════
#  ASSET HELPER
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

# ════════════════════════════════════════════════════════════════════════════
#  DISEASE DATA
# ════════════════════════════════════════════════════════════════════════════
DISEASE_DATA = {
    "alzheimer": {
        "color":"#4fc3f7","icon":"🧩",
        "gradient":"linear-gradient(145deg,rgba(13,27,110,0.5),rgba(26,35,126,0.6))",
        "border":"#4fc3f7","glow":"rgba(79,195,247,0.25)",
        "title":"Alzheimer's Disease",
        "short":"Progressive neurological disorder destroying memory and cognitive function.",
        "description":("Alzheimer's is the most prevalent cause of dementia — accounting for 60–80% of all cases. "
                       "It is a progressive brain disorder that systematically destroys memory, thinking skills, and "
                       "eventually the ability to perform even simple tasks. Brain changes begin a decade or more "
                       "before symptoms appear, making early screening critical."),
        "symptoms":["Progressive memory loss (especially recent events)",
                    "Confusion with time, place and familiar people",
                    "Rapid mood and personality changes",
                    "Difficulty with language and writing",
                    "Trouble completing familiar tasks",
                    "Social withdrawal and loss of initiative"],
        "danger":85,"danger_text":"Very High","danger_color":"#ff7043",
        "medicines":["Donepezil (Aricept) — Improves memory and cognitive function",
                     "Rivastigmine (Exelon) — Slows cognitive decline significantly",
                     "Memantine (Namenda) — Reduces worsening of moderate–severe symptoms",
                     "Galantamine — Cholinesterase inhibitor for mild–moderate stages"],
        "habits":["Daily mental exercises — puzzles, crosswords, reading aloud",
                  "30 minutes of aerobic exercise every single day",
                  "Mediterranean diet — omega-3, antioxidants, leafy greens",
                  "7–9 hours of quality sleep every night",
                  "Stay socially active and maintain close relationships",
                  "Keep blood pressure, diabetes and cholesterol under control"],
        "remedies":("Early diagnosis allows pharmacological intervention before severe damage occurs. "
                    "Cognitive behavioral therapy, memory care programs, speech therapy, and caregiver "
                    "education form the core of management. Regular cognitive assessments help adjust treatment "
                    "over time and track progression."),
        "overcome":("No cure exists, but medications meaningfully slow progression. "
                    "A combination of mental stimulation, physical activity, social engagement, "
                    "and strict medication adherence improves quality of life considerably. "
                    "Support groups and caregiver training are equally essential for families."),
    },
    "stroke": {
        "color":"#ef5350","icon":"⚡",
        "gradient":"linear-gradient(145deg,rgba(92,0,0,0.5),rgba(183,28,28,0.55))",
        "border":"#ef5350","glow":"rgba(239,83,80,0.25)",
        "title":"Stroke",
        "short":"Life-threatening emergency: blood supply to brain blocked or vessel ruptures.",
        "description":("A stroke occurs when blood flow to part of the brain is cut off — either by a clot "
                       "(ischemic stroke, 87% of cases) or a burst blood vessel (hemorrhagic stroke). "
                       "Brain cells begin dying within minutes. Stroke is the #2 killer worldwide and the #1 "
                       "cause of long-term adult disability. Recognize using FAST: "
                       "Face drooping · Arm weakness · Speech difficulty · Time to call 112."),
        "symptoms":["Sudden face drooping on one side (FAST — F)",
                    "Arm or leg weakness on one side (FAST — A)",
                    "Slurred or incomprehensible speech (FAST — S)",
                    "Sudden severe 'thunderclap' headache",
                    "Vision loss in one or both eyes",
                    "Sudden loss of balance and coordination"],
        "danger":95,"danger_text":"CRITICAL — Emergency","danger_color":"#d32f2f",
        "medicines":["tPA (Alteplase) — Clot-dissolving drug (ONLY within 4.5 hrs of onset)",
                     "Aspirin / Clopidogrel — Antiplatelet agents to prevent future clots",
                     "Atorvastatin (Lipitor) — Reduces cholesterol and stroke recurrence risk",
                     "Amlodipine / Lisinopril — Antihypertensives for strict BP control"],
        "habits":["Control blood pressure — target below 120/80 mmHg at all times",
                  "Quit smoking — the single biggest modifiable stroke risk factor",
                  "150+ minutes of aerobic exercise every week",
                  "DASH diet — low sodium, high potassium, fruits and vegetables",
                  "Strictly limit or eliminate alcohol consumption",
                  "Keep diabetes and cholesterol under rigorous medical control"],
        "remedies":("Call 112 immediately — every minute without blood flow costs 1.9 million neurons. "
                    "tPA (Alteplase) or mechanical thrombectomy can restore circulation if administered early. "
                    "Post-stroke rehabilitation — physical therapy, speech therapy, occupational therapy — "
                    "is the cornerstone of recovery and independent living."),
        "overcome":("Learn and act on FAST. Up to 80% of strokes are entirely preventable through lifestyle changes. "
                    "Aggressive rehabilitation post-stroke significantly restores independence. "
                    "Medication adherence, BP control, and quitting smoking prevent most recurrences. "
                    "Time is brain — never delay seeking emergency care."),
    },
    "tumor": {
        "color":"#66bb6a","icon":"🔬",
        "gradient":"linear-gradient(145deg,rgba(10,46,10,0.5),rgba(27,94,32,0.55))",
        "border":"#66bb6a","glow":"rgba(102,187,106,0.25)",
        "title":"Brain Tumor",
        "short":"Abnormal cell mass in the brain — benign or malignant — needing immediate investigation.",
        "description":("Brain tumors are abnormal growths of cells within or surrounding the brain. They can be "
                       "primary (originating in the brain: gliomas, meningiomas) or secondary (spread from lung, "
                       "breast, etc.). Glioblastoma (GBM) is the most aggressive primary brain tumor. "
                       "MRI with contrast is the gold standard for detection. Treatment requires a full "
                       "multidisciplinary neuro-oncology team."),
        "symptoms":["New persistent headaches (worse in the mornings)",
                    "New-onset seizures without prior history",
                    "Nausea and vomiting without clear cause",
                    "Progressive cognitive decline and memory gaps",
                    "Changes in vision, hearing or speech quality",
                    "Personality shifts and unexplained mood changes"],
        "danger":80,"danger_text":"High","danger_color":"#f57c00",
        "medicines":["Temozolomide (Temodar) — Standard chemo for GBM via Stupp Protocol",
                     "Bevacizumab (Avastin) — Anti-angiogenic; cuts tumor blood supply",
                     "Dexamethasone — Corticosteroid to reduce dangerous brain swelling",
                     "Lomustine (CCNU) — Alkylating agent for recurrent high-grade tumors"],
        "habits":["Regular MRI scans and neuro-oncology follow-up appointments",
                  "Nutrient-dense, anti-inflammatory and antioxidant-rich diet",
                  "Minimize unnecessary ionizing radiation exposure",
                  "Daily meditation and yoga for stress and inflammation control",
                  "Structured physical therapy for motor function rehabilitation",
                  "Cognitive therapy and brain exercises to aid recovery"],
        "remedies":("Surgical resection (when anatomically safe) is first-line treatment. "
                    "This is followed by the Stupp Protocol: concurrent Temozolomide + radiation, then adjuvant TMZ. "
                    "Targeted therapy (EGFR inhibitors), tumor-treating fields (TTFields), and immunotherapy "
                    "are emerging as powerful additional options."),
        "overcome":("Early MRI detection dramatically improves surgical outcomes and prognosis. "
                    "Participation in clinical trials may provide access to cutting-edge therapies. "
                    "Immunotherapy and CAR-T cell therapy are showing strong promise for GBM. "
                    "Robust palliative care, support networks, and patient advocacy groups are vital to quality of life."),
    },
}

# ════════════════════════════════════════════════════════════════════════════
#  HOSPITALS DATA
# ════════════════════════════════════════════════════════════════════════════
HOSPITALS = [
    {"name":"AIIMS New Delhi","full":"All India Institute of Medical Sciences",
     "location":"Ansari Nagar East, New Delhi — 110029",
     "specialty":"Neurology · Neurosurgery · Neuro-Oncology · Stroke Unit",
     "phone":"+91-11-2658-8500","website":"aiims.edu",
     "color":"#1565c0","glow":"rgba(21,101,192,0.3)","icon":"🏛️","tag":"Premier Govt. Institute"},
    {"name":"Medanta – The Medicity","full":"Medanta The Medicity, Gurugram",
     "location":"Sector 38, CH Baktawar Singh Rd, Gurugram — 122001",
     "specialty":"Institute of Neurosciences · Brain & Spine · Stroke Centre",
     "phone":"+91-124-414-1414","website":"medanta.org",
     "color":"#00695c","glow":"rgba(0,105,92,0.3)","icon":"🏥","tag":"Top Rated Neuro Centre"},
    {"name":"Fortis Memorial Research Institute","full":"Fortis Memorial Research Institute, Gurugram",
     "location":"Sector 44, Gurugram — 122002",
     "specialty":"Neurology · Neurosurgery · Gamma Knife · Neuro-Rehab",
     "phone":"+91-124-496-2200","website":"fortishealthcare.com",
     "color":"#b71c1c","glow":"rgba(183,28,28,0.3)","icon":"🏨","tag":"Gamma Knife Technology"},
    {"name":"Indraprastha Apollo Hospitals","full":"Indraprastha Apollo Hospitals, New Delhi",
     "location":"Sarita Vihar, Mathura Road, New Delhi — 110076",
     "specialty":"Comprehensive Stroke Centre · Neuro-Oncology · CyberKnife",
     "phone":"+91-11-7179-1090","website":"apollohospitals.com",
     "color":"#6a1b9a","glow":"rgba(106,27,154,0.3)","icon":"🏗️","tag":"JCI Accredited"},
    {"name":"Max Super Speciality Hospital","full":"Max Super Speciality Hospital, Saket",
     "location":"1 Press Enclave Road, Saket, New Delhi — 110017",
     "specialty":"Neurosciences Centre · Brain Tumor · Deep Brain Stimulation",
     "phone":"+91-11-2651-0050","website":"maxhealthcare.in",
     "color":"#e65100","glow":"rgba(230,81,0,0.3)","icon":"🏢","tag":"Best Brain Tumor Care"},
    {"name":"NIMHANS Bengaluru","full":"Nat. Institute of Mental Health & Neuro Sciences",
     "location":"Hosur Road, Bengaluru — 560029",
     "specialty":"Neurology · Psychiatry · Neuro-Oncology · Alzheimer's Research",
     "phone":"+91-80-2699-5000","website":"nimhans.ac.in",
     "color":"#00838f","glow":"rgba(0,131,143,0.3)","icon":"🎓","tag":"National Centre of Excellence"},
]

# ════════════════════════════════════════════════════════════════════════════
#  DEATHS DATA
# ════════════════════════════════════════════════════════════════════════════
YEARS = [2018,2019,2020,2021,2022,2023]
INDIA_DEATHS = {
    "Stroke":                 [1520,1582,1615,1654,1718,1782],
    "Alzheimer's / Dementia": [285, 308, 326, 342, 361, 383],
    "Brain Tumor":            [26,  28,  29,  31,  33,  36],
}
DEATH_COLORS = {"Stroke":"#ef5350","Alzheimer's / Dementia":"#4fc3f7","Brain Tumor":"#66bb6a"}

# ════════════════════════════════════════════════════════════════════════════
#  CHATBOT  — 50 Q&A pairs
# ════════════════════════════════════════════════════════════════════════════
CHATBOT_QA = {
    "what is neurovision":"NeuroVision is an AI-powered brain MRI analysis system built on ResNet50 deep learning that detects Alzheimer's Disease, Stroke, and Brain Tumor from uploaded MRI scans.",
    "who made neurovision":"NeuroVision was developed by Dhruv Sharma (11023210198) and Akshat Jain (11023210142), B.Tech CSE students at SRM University Delhi-NCR, Sonepat, under guidance of Ms. Anju Goel Ma'am.",
    "what diseases can you detect":"NeuroVision detects three critical brain conditions: 🧩 Alzheimer's Disease, ⚡ Stroke, and 🔬 Brain Tumor from brain MRI scans.",
    "how accurate is the model":"NeuroVision uses a fine-tuned ResNet50 model trained on brain MRI datasets. Always confirm AI results with a certified neurologist for clinical decisions.",
    "what is resnet50":"ResNet50 is a 50-layer deep Convolutional Neural Network by Microsoft Research. It uses residual (skip) connections to avoid vanishing gradients and excels at medical image classification.",
    "how do i use neurovision":"Go to Home page → scroll to 'Upload MRI Scan' → upload a JPG/PNG brain MRI → the AI instantly predicts the condition with confidence scores.",
    "what image format should i upload":"Upload brain MRI scans in JPG, JPEG, or PNG format. Proper axial or coronal brain MRI slices give the best results.",
    "is this a replacement for a doctor":"No. NeuroVision is a decision-support tool only. Always consult a certified neurologist or neurosurgeon for clinical diagnosis and treatment.",
    "what is alzheimer":"Alzheimer's Disease is a progressive neurodegenerative disorder that destroys memory, thinking skills, and eventually the ability to perform basic tasks. It accounts for 60–80% of all dementia cases.",
    "what causes alzheimer":"Alzheimer's is caused by abnormal protein build-up — amyloid plaques and tau tangles — destroying brain neurons. Age, genetics, lifestyle, and cardiovascular factors all contribute.",
    "what are symptoms of alzheimer":"Key symptoms: progressive memory loss, confusion about time/place, personality changes, difficulty with language, trouble with familiar tasks, and social withdrawal.",
    "is alzheimer curable":"There is currently no cure for Alzheimer's. However, Donepezil, Rivastigmine, and Memantine can significantly slow progression and improve quality of life.",
    "what medicines treat alzheimer":"Main medicines: Donepezil (Aricept), Rivastigmine (Exelon), Memantine (Namenda), and Galantamine — all slow cognitive decline.",
    "how to prevent alzheimer":"Stay mentally active (puzzles, reading), exercise daily, follow Mediterranean diet, sleep 7–9 hours, stay socially engaged, control BP and diabetes.",
    "how many people have alzheimer in india":"Approximately 5.3 million Indians live with dementia/Alzheimer's. About 383,000 deaths are attributed to it annually in India.",
    "what is the danger level of alzheimer":"Alzheimer's has a Very High danger level (85/100 severity index) due to its progressive, irreversible nature affecting all aspects of daily life.",
    "how fast does alzheimer progress":"Alzheimer's progresses over 8–20 years from mild cognitive impairment to moderate dementia to severe dementia requiring full-time care.",
    "what is a stroke":"A stroke occurs when blood supply to part of the brain is cut off — by a clot (ischemic, 87%) or a burst blood vessel (hemorrhagic). Brain cells die within minutes.",
    "what is fast for stroke":"FAST = Face drooping, Arm weakness, Speech difficulty, Time to call 112. These are the four key warning signs of stroke.",
    "what causes stroke":"Stroke is caused by high blood pressure, atrial fibrillation, diabetes, smoking, obesity, high cholesterol, or physical inactivity — all damage blood vessels.",
    "what are symptoms of stroke":"Sudden face drooping, arm or leg weakness, slurred speech, thunderclap headache, sudden vision loss, and loss of balance.",
    "how dangerous is stroke":"Stroke has a CRITICAL danger level (95/100). It is the #2 killer globally and #1 cause of adult disability. Every minute costs 1.9 million neurons.",
    "what medicines treat stroke":"tPA (Alteplase) within 4.5 hours, Aspirin/Clopidogrel, Atorvastatin (Lipitor), Amlodipine/Lisinopril for BP control.",
    "can stroke be prevented":"Up to 80% of strokes are preventable. Control BP, quit smoking, exercise regularly, eat DASH diet, limit alcohol, manage diabetes rigorously.",
    "what is tpa for stroke":"tPA (Alteplase) is a clot-dissolving drug that restores blood flow in ischemic stroke — MUST be given within 4.5 hours of symptom onset.",
    "how many people die from stroke in india":"Approximately 1.78 million Indians die from stroke every year — the leading cause of adult disability in the country.",
    "what is ischemic vs hemorrhagic stroke":"Ischemic stroke (87%): blood clot blocks an artery. Hemorrhagic stroke (13%): blood vessel bursts and bleeds into the brain.",
    "what is stroke rehabilitation":"Post-stroke rehab includes physical therapy (movement), speech therapy (language), and occupational therapy (daily tasks) — crucial for regaining independence.",
    "what number to call for stroke":"Call 112 (India emergency number) immediately. Every second counts — do not wait to see if symptoms improve.",
    "what is a brain tumor":"A brain tumor is an abnormal mass of cells in or around the brain. It can be benign (non-cancerous) or malignant (cancerous). Primary tumors originate in brain; secondary spread from other organs.",
    "what causes brain tumor":"Exact cause often unknown. Risk factors: ionizing radiation exposure, family history/genetics, certain hereditary syndromes, and immune system disorders.",
    "what are symptoms of brain tumor":"Persistent morning headaches, new-onset seizures, nausea/vomiting, progressive cognitive decline, vision/hearing/speech changes, and unexplained personality shifts.",
    "what is glioblastoma":"Glioblastoma (GBM) is the most aggressive primary brain tumor. Average survival is 14–16 months with treatment. It represents about 15% of all brain tumors.",
    "how is brain tumor treated":"Surgery (resection) + Stupp Protocol: Temozolomide chemo + radiation, followed by adjuvant TMZ. May also include Bevacizumab, TTFields therapy, or immunotherapy.",
    "what medicines treat brain tumor":"Temozolomide (Temodar), Bevacizumab (Avastin), Dexamethasone (reduces swelling), and Lomustine (CCNU) are key medicines.",
    "how dangerous is brain tumor":"Brain tumor has a High danger level (80/100). Malignant tumors like GBM are life-threatening. Even benign tumors cause serious neurological damage from pressure.",
    "what is the stupp protocol":"The Stupp Protocol: 6 weeks of concurrent Temozolomide + radiation, followed by 6 cycles of adjuvant Temozolomide. Standard GBM treatment.",
    "can brain tumor be cured":"Benign tumors are often curable with surgery. Malignant tumors like GBM have no cure, but treatment significantly extends survival. Clinical trials offer hope.",
    "how many people get brain tumors in india":"India reports ~40,000+ new brain tumor cases annually, with ~36,000 deaths per year attributed to brain tumors.",
    "what is mri for brain tumor":"MRI with gadolinium contrast is the gold standard for detecting, locating, and monitoring brain tumors — far superior to CT scans.",
    "what is bevacizumab":"Bevacizumab (Avastin) is an anti-angiogenic drug that cuts tumor blood supply by blocking VEGF, preventing tumor growth.",
    "how can i keep my brain healthy":"Exercise regularly, sleep 7–9 hours, eat brain foods (omega-3, berries, leafy greens), stay mentally and socially active, avoid smoking and excess alcohol.",
    "what foods are good for brain":"Fatty fish (omega-3), blueberries, turmeric (curcumin), broccoli, pumpkin seeds, dark chocolate, nuts, eggs, and green tea are excellent for brain health.",
    "what is dementia":"Dementia is a decline in cognitive function severe enough to interfere with daily life. Alzheimer's is the most common cause (60–80% of cases).",
    "what is a neurologist":"A neurologist is a medical doctor specialising in diagnosing and treating disorders of the brain, spinal cord, and nervous system.",
    "which are the best hospitals for brain disease in india":"Top Indian hospitals: AIIMS New Delhi, Medanta Gurugram, Fortis Gurugram, Apollo New Delhi, Max Saket, and NIMHANS Bengaluru.",
    "what is the blood brain barrier":"The blood-brain barrier (BBB) is a highly selective membrane preventing harmful substances from entering the brain — it also makes drug delivery to the brain very difficult.",
    "what is deep brain stimulation":"DBS involves implanting electrodes in specific brain areas to send electrical impulses — used for Parkinson's, tremors, and some movement disorders.",
    "what does confidence score mean":"The confidence score is the model's probability estimate for the detected class — e.g., 92.5% means the model is 92.5% confident in its prediction.",
    "how do i contact a neurologist":"In India: AIIMS (+91-11-2658-8500), Medanta (+91-124-414-1414), Apollo (+91-11-7179-1090), Max Hospital (+91-11-2651-0050).",
}

def chatbot_response(user_input: str) -> str:
    text = user_input.lower().strip()
    for ch in ["?","!",".",","]:
        text = text.replace(ch, "")
    best_key, best_score = None, 0
    for key in CHATBOT_QA:
        kw = set(key.split()); iw = set(text.split())
        score = len(kw & iw) / max(len(kw), 1)
        if score > best_score:
            best_score, best_key = score, key
    if best_score >= 0.35:
        return CHATBOT_QA[best_key]
    if any(w in text for w in ["alzheimer","memory","dementia","forget"]):
        return CHATBOT_QA["what is alzheimer"]
    if any(w in text for w in ["stroke","fast","clot","paralysis"]):
        return CHATBOT_QA["what is a stroke"]
    if any(w in text for w in ["tumor","tumour","cancer","glioma","gbm"]):
        return CHATBOT_QA["what is a brain tumor"]
    if any(w in text for w in ["hospital","doctor","where","treat"]):
        return CHATBOT_QA["which are the best hospitals for brain disease in india"]
    if any(w in text for w in ["hello","hi","hey","namaste"]):
        return "Hello! 👋 I'm NeuroBot. Ask me anything about Alzheimer's, Stroke, Brain Tumor, or how NeuroVision works!"
    return ("I'm not sure about that. Try asking me about:\n"
            "• Alzheimer's symptoms or treatment\n"
            "• Stroke warning signs or medicines\n"
            "• Brain Tumor detection or hospitals\n"
            "• How NeuroVision works")

# ════════════════════════════════════════════════════════════════════════════
#  SHARED UI HELPERS
# ════════════════════════════════════════════════════════════════════════════
def section_title(title: str, subtitle: str = ""):
    sub = (f"<div style='font-family:Rajdhani,sans-serif;font-size:15px;color:#78909c;"
           f"margin-top:10px;'>{subtitle}</div>") if subtitle else ""
    st.markdown(
        f"""<div style='text-align:center;margin:52px 0 30px;'>
          <div style='font-family:Orbitron,monospace;font-size:clamp(18px,2.6vw,26px);
            font-weight:700;color:#00e5ff;letter-spacing:3px;margin-bottom:10px;'>{title.upper()}</div>
          <div style='width:70px;height:2px;background:linear-gradient(90deg,transparent,#00bcd4,transparent);
            margin:0 auto;'></div>{sub}
        </div>""", unsafe_allow_html=True)

def water_wave(height=100):
    """Animated three-layer fluid water wave banner."""
    h = height
    mid = h // 2
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#040e1a;overflow:hidden;width:100%;height:{h}px;}}
.ww{{position:relative;width:100%;height:{h}px;overflow:hidden;
  background:linear-gradient(180deg,#040e1a 0%,#061525 100%);}}
svg.w{{position:absolute;bottom:0;width:200%;}}
svg.w1{{animation:mw 7s linear infinite;opacity:.7;}}
svg.w2{{animation:mw 11s linear infinite reverse;opacity:.45;bottom:6px;}}
svg.w3{{animation:mw 15s linear infinite;opacity:.28;bottom:12px;}}
@keyframes mw{{0%{{transform:translateX(0);}}100%{{transform:translateX(-50%);}}}}
.lbl{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-family:'Orbitron',monospace;font-size:clamp(9px,1.4vw,13px);
  color:rgba(0,229,255,0.6);letter-spacing:5px;text-transform:uppercase;white-space:nowrap;}}
</style></head><body>
<div class="ww">
  <svg class="w w1" viewBox="0 0 1440 {h}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
    <path fill="rgba(0,188,212,0.4)" d="M0,{mid} C180,{mid-28} 360,{mid+28} 540,{mid} C720,{mid-28} 900,{mid+28} 1080,{mid} C1260,{mid-28} 1440,{mid+28} 1440,{mid} L1440,{h} L0,{h} Z"/>
  </svg>
  <svg class="w w2" viewBox="0 0 1440 {h}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
    <path fill="rgba(2,136,209,0.28)" d="M0,{mid+10} C200,{mid-18} 400,{mid+30} 600,{mid+10} C800,{mid-18} 1000,{mid+30} 1200,{mid+10} C1300,{mid-18} 1440,{mid+20} 1440,{mid+10} L1440,{h} L0,{h} Z"/>
  </svg>
  <svg class="w w3" viewBox="0 0 1440 {h}" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
    <path fill="rgba(79,195,247,0.18)" d="M0,{mid+20} C240,{mid-8} 480,{mid+34} 720,{mid+20} C960,{mid-8} 1200,{mid+34} 1440,{mid+20} L1440,{h} L0,{h} Z"/>
  </svg>
  <div class="lbl">⬡ &nbsp; NEUROVISION &nbsp;·&nbsp; AI NEURAL INTERFACE &nbsp; ⬡</div>
</div>
</body></html>"""
    st.components.v1.html(html, height=h, scrolling=False)

def nav_bar():
    srm = b64("1777098001570_image.png")
    itag = (f'<img src="data:image/png;base64,{srm}" style="height:34px;'
            f'filter:drop-shadow(0 0 6px rgba(0,188,212,0.4));">') if srm else "🧠"
    st.markdown(
        f"""<div style='background:linear-gradient(90deg,#040810,#0a1628,#040810);
          border-bottom:1px solid rgba(0,188,212,0.2);padding:12px 40px;
          display:flex;align-items:center;justify-content:space-between;'>
          <div style='display:flex;align-items:center;gap:14px;'>
            {itag}
            <span style='font-family:Orbitron,monospace;font-size:20px;font-weight:900;
              background:linear-gradient(135deg,#00e5ff,#0288d1);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              background-clip:text;letter-spacing:3px;'>NEUROVISION</span>
          </div>
          <div style='font-family:Rajdhani,sans-serif;font-size:12px;color:#78909c;
            letter-spacing:2px;text-transform:uppercase;'>AI Brain Disease Detection</div>
        </div>""", unsafe_allow_html=True)
    water_wave(height=95)
    gap, c1, c2, end = st.columns([4.2, 1, 1.1, 0.4])
    with c1:
        if st.button("🏠  Home", use_container_width=True, key="nav_home"):
            st.session_state.page = "main"; st.rerun()
    with c2:
        if st.button("🏥  Medical Panel", use_container_width=True, key="nav_med"):
            st.session_state.page = "medical"; st.rerun()
    st.markdown("<hr style='border:none;border-top:1px solid rgba(0,188,212,0.1);margin:0 0 8px;'>",
                unsafe_allow_html=True)

def footer():
    srm = b64("1777098001570_image.png")
    itag = (f'<img src="data:image/png;base64,{srm}" style="height:58px;margin-bottom:18px;'
            f'opacity:.9;filter:drop-shadow(0 0 10px rgba(0,188,212,0.3));">') if srm else "🧠"
    st.markdown(
        f"""<div style='background:linear-gradient(135deg,#02050d,#060d1c);
          border-top:1px solid rgba(0,188,212,0.15);margin-top:60px;padding:50px 40px;text-align:center;'>
          {itag}
          <div style='font-family:Orbitron,monospace;font-size:12px;color:#4fc3f7;
            letter-spacing:4px;text-transform:uppercase;margin-bottom:16px;'>
            NeuroVision — AI-Powered Brain Disease Detection
          </div>
          <div style='font-family:Rajdhani,sans-serif;font-size:18px;color:#80deea;margin-bottom:10px;'>
            Developed by &nbsp;
            <strong style='color:#00e5ff;font-size:19px;'>Dhruv Sharma</strong>
            &nbsp;<span style='color:#4fc3f7;'>— 11023210198</span>
            &emsp;|&emsp;
            <strong style='color:#00e5ff;font-size:19px;'>Akshat Jain</strong>
            &nbsp;<span style='color:#4fc3f7;'>— 11023210142</span>
          </div>
          <div style='font-family:Rajdhani,sans-serif;font-size:16px;color:#90a4ae;margin-bottom:8px;'>
            Special thanks to &nbsp;
            <strong style='color:#80cbc4;font-size:17px;'>Ms. Anju Goel Ma'am</strong>
            &nbsp;for her invaluable guidance and support
          </div>
          <div style='font-family:Rajdhani,sans-serif;font-size:14px;color:#78909c;margin-top:10px;'>
            <span style='color:#4dd0e1;'>SRM University Delhi-NCR, Sonepat</span>
            &nbsp;•&nbsp;
            <span style='color:#4dd0e1;'>B.Tech Computer Science</span>
            &nbsp;•&nbsp;
            <span style='color:#4dd0e1;'>2024–25</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  FLOATING CHATBOT
# ════════════════════════════════════════════════════════════════════════════
def render_chatbot():
    # CSS for chat panel + FAB
    st.markdown("""
    <style>
    .chat-panel{
      position:fixed;bottom:105px;right:24px;width:360px;max-height:500px;
      background:linear-gradient(145deg,#060f1e,#0a1a2e);
      border:1px solid rgba(0,188,212,0.4);border-radius:20px;
      box-shadow:0 8px 50px rgba(0,188,212,0.3);z-index:9998;
      overflow:hidden;display:flex;flex-direction:column;}
    .ch{background:linear-gradient(90deg,#00bcd4,#0288d1);padding:13px 18px;
      display:flex;align-items:center;gap:10px;
      font-family:Orbitron,monospace;font-size:13px;font-weight:700;
      color:#fff;letter-spacing:1.5px;}
    .cb{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;
      gap:9px;max-height:310px;}
    .mb{background:rgba(0,188,212,0.1);border:1px solid rgba(0,188,212,0.2);
      border-radius:14px 14px 14px 4px;padding:10px 13px;
      font-family:Rajdhani,sans-serif;font-size:13.5px;color:#b0bec5;
      max-width:92%;line-height:1.55;}
    .mu{background:rgba(2,136,209,0.18);border:1px solid rgba(2,136,209,0.32);
      border-radius:14px 14px 4px 14px;padding:10px 13px;
      font-family:Rajdhani,sans-serif;font-size:13.5px;color:#e0f7fa;
      max-width:92%;align-self:flex-end;line-height:1.55;}
    .cb::-webkit-scrollbar{width:4px;}
    .cb::-webkit-scrollbar-thumb{background:#00bcd430;border-radius:2px;}
    /* FAB */
    .fab-wrap{position:fixed;bottom:28px;right:28px;z-index:9999;
      display:flex;flex-direction:column;align-items:center;gap:6px;}
    .fab{width:62px;height:62px;border-radius:50%;
      background:linear-gradient(135deg,#00bcd4,#0288d1);
      display:flex;align-items:center;justify-content:center;font-size:28px;
      box-shadow:0 4px 24px rgba(0,188,212,0.6);
      animation:fabPulse 2.5s ease-in-out infinite;}
    .fab-lbl{font-family:Rajdhani,sans-serif;font-size:10px;color:#80deea;
      letter-spacing:1px;text-transform:uppercase;}
    @keyframes fabPulse{
      0%,100%{box-shadow:0 4px 24px rgba(0,188,212,0.55);}
      50%{box-shadow:0 4px 44px rgba(0,188,212,0.95);}}
    </style>
    <div class="fab-wrap">
      <div class="fab" title="Open NeuroBot AI Chatbot">🤖</div>
      <div class="fab-lbl">NeuroBot</div>
    </div>
    """, unsafe_allow_html=True)

    # Toggle button (real Streamlit button positioned at bottom-right area)
    _, rb = st.columns([9, 1])
    with rb:
        label = "✕" if st.session_state.chat_open else "💬"
        if st.button(label, key="chat_toggle_fab"):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()

    if st.session_state.chat_open:
        history_html = ""
        for role, msg in st.session_state.chat_history[-16:]:
            css = "mu" if role == "user" else "mb"
            pfx = "You: " if role == "user" else "🧠 NeuroBot: "
            safe = msg.replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            history_html += f"<div class='{css}'><b>{pfx}</b>{safe}</div>"
        if not history_html:
            history_html = "<div class='mb'><b>🧠 NeuroBot: </b>Hello! 👋 Ask me anything about Alzheimer's, Stroke, Brain Tumor, or how NeuroVision works!</div>"

        st.markdown(
            f"""<div class='chat-panel'>
              <div class='ch'>🤖 &nbsp; NEUROBOT — AI Assistant</div>
              <div class='cb'>{history_html}</div>
            </div>""", unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            uq = st.text_input("Message", placeholder="e.g. What are stroke symptoms?",
                               label_visibility="collapsed")
            sc1, sc2 = st.columns([3,1])
            with sc1: sub = st.form_submit_button("Send ➤", use_container_width=True)
            with sc2: cls = st.form_submit_button("✕ Close", use_container_width=True)
        if sub and uq.strip():
            st.session_state.chat_history.append(("user", uq))
            st.session_state.chat_history.append(("bot", chatbot_response(uq)))
            st.rerun()
        if cls:
            st.session_state.chat_open = False; st.rerun()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE — PASSWORD LOCK
# ════════════════════════════════════════════════════════════════════════════
def page_password():
    st.components.v1.html("""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#040810;width:100%;height:340px;display:flex;align-items:center;justify-content:center;overflow:hidden;}
.wrap{text-align:center;position:relative;z-index:2;}
.lock{font-size:64px;animation:lp 2s ease-in-out infinite;display:block;margin-bottom:18px;}
.title{font-family:'Orbitron',monospace;font-size:clamp(26px,6vw,60px);font-weight:900;
  background:linear-gradient(135deg,#00e5ff,#4fc3f7,#0288d1);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  letter-spacing:8px;margin-bottom:10px;}
.sub{font-family:'Rajdhani',sans-serif;font-size:16px;color:#78909c;letter-spacing:3px;text-transform:uppercase;}
.pc{position:fixed;inset:0;pointer-events:none;z-index:0;}
.ww{position:absolute;bottom:0;left:0;width:100%;overflow:hidden;height:55px;}
svg.wv{width:200%;animation:mw 8s linear infinite;}
@keyframes lp{0%,100%{transform:scale(1);filter:drop-shadow(0 0 10px rgba(0,229,255,0.4));}
  50%{transform:scale(1.1);filter:drop-shadow(0 0 28px rgba(0,229,255,0.9));}}
@keyframes mw{0%{transform:translateX(0);}100%{transform:translateX(-50%);}}
@keyframes pf{0%{transform:translateY(130vh) scale(0);opacity:0;}8%{opacity:.6;}92%{opacity:.25;}100%{transform:translateY(-60px) scale(1);opacity:0;}}
</style></head><body>
<div class="pc" id="pc"></div>
<div class="wrap">
  <span class="lock">🔐</span>
  <div class="title">NEUROVISION</div>
  <div class="sub">Secure Access Required</div>
</div>
<div class="ww">
  <svg class="wv" viewBox="0 0 1440 55" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
    <path fill="rgba(0,188,212,0.25)" d="M0,27 C180,8 360,46 540,27 C720,8 900,46 1080,27 C1260,8 1440,46 1440,27 L1440,55 L0,55 Z"/>
  </svg>
</div>
<script>
const pc=document.getElementById('pc');
const c=['#00e5ff','#4fc3f7','#00bcd4','#80deea'];
for(let i=0;i<50;i++){
  const p=document.createElement('div'),s=Math.random()*3+1;
  p.style.cssText='position:absolute;width:'+s+'px;height:'+s+'px;background:'
    +c[Math.floor(Math.random()*c.length)]+';border-radius:50%;left:'
    +Math.random()*100+'%;animation:pf '+(Math.random()*16+10)+'s linear '
    +(Math.random()*8)+'s infinite;box-shadow:0 0 '+(s*2)+'px #00bcd4;';
  pc.appendChild(p);
}
</script>
</body></html>""", height=340, scrolling=False)

    _, mid, _ = st.columns([1.5, 1, 1.5])
    with mid:
        st.markdown(
            "<div style='text-align:center;font-family:Rajdhani,sans-serif;font-size:14px;"
            "color:#80deea;letter-spacing:2px;margin-bottom:10px;'>🔑 &nbsp; ENTER ACCESS PASSWORD</div>",
            unsafe_allow_html=True)
        pw = st.text_input("Password", type="password", placeholder="Enter password…",
                           label_visibility="collapsed", key="pw_input")
        if st.session_state.pw_error:
            st.markdown("<div style='text-align:center;color:#ef5350;font-family:Rajdhani,sans-serif;"
                        "font-size:14px;margin-top:4px;'>❌ Incorrect password. Try again.</div>",
                        unsafe_allow_html=True)
        if st.button("🔓  Unlock NeuroVision", use_container_width=True, key="unlock_btn"):
            if _check_pw(pw):
                st.session_state.authenticated = True
                st.session_state.pw_error = False
                st.rerun()
            else:
                st.session_state.pw_error = True
                st.rerun()

    st.markdown(
        "<div style='text-align:center;font-size:12px;margin-top:10px;"
        "font-family:Rajdhani,sans-serif;color:#4dd0e1;letter-spacing:2px;'>"
        "SRM UNIVERSITY DELHI-NCR, SONEPAT &nbsp;•&nbsp; BTECH COMPUTER SCIENCE</div>",
        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE — INTRO
# ════════════════════════════════════════════════════════════════════════════
def page_intro():
    vid_path = "220232_medium.mp4"
    video_tag = ""
    if os.path.exists(vid_path) and os.path.getsize(vid_path) < 80*1024*1024:
        vb = b64(vid_path)
        if vb:
            video_tag = (
                '<video autoplay muted loop playsinline '
                'style="position:absolute;top:0;left:0;width:100%;height:100%;'
                'object-fit:cover;opacity:0.28;z-index:0;">'
                f'<source src="data:video/mp4;base64,{vb}" type="video/mp4"></video>'
            )

    html = ("""<!DOCTYPE html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:#040810;overflow:hidden;width:100vw;height:100vh;}
.wrap{position:relative;width:100%;height:100vh;display:flex;align-items:center;justify-content:center;overflow:hidden;}
.overlay{position:absolute;inset:0;background:radial-gradient(ellipse at 50% 50%,rgba(0,188,212,0.08),rgba(4,8,16,0.80) 70%);z-index:1;}
.pc{position:absolute;inset:0;z-index:2;overflow:hidden;pointer-events:none;}
.content{position:relative;z-index:10;text-align:center;padding:20px;}
.rw{width:130px;height:130px;margin:0 auto 30px;position:relative;display:flex;align-items:center;justify-content:center;}
.r1{position:absolute;inset:0;border:1.5px solid rgba(0,229,255,0.3);border-radius:50%;animation:spin 20s linear infinite;box-shadow:0 0 30px rgba(0,229,255,0.2),inset 0 0 30px rgba(0,229,255,0.08);}
.r2{position:absolute;inset:10px;border:1px dashed rgba(0,229,255,0.15);border-radius:50%;animation:spin 14s linear infinite reverse;}
.br{font-size:52px;position:relative;z-index:2;animation:pulse 3s ease-in-out infinite;}
.title{font-family:'Orbitron',monospace;font-size:clamp(36px,7vw,96px);font-weight:900;
  background:linear-gradient(135deg,#00e5ff 0%,#4fc3f7 40%,#0288d1 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  letter-spacing:12px;opacity:0;
  animation:titleIn 1.8s cubic-bezier(.22,1,.36,1) .4s forwards,glow 3.5s ease-in-out infinite alternate 3s;
  margin-bottom:14px;line-height:1.1;}
.div{width:200px;height:1.5px;background:linear-gradient(90deg,transparent,#00bcd4,transparent);
  margin:16px auto 24px;opacity:0;animation:fi .9s ease 2.2s forwards;}
.sub{font-family:'Rajdhani',sans-serif;font-size:clamp(12px,1.8vw,19px);color:#80deea;
  letter-spacing:5px;text-transform:uppercase;opacity:0;animation:fu .8s ease 2.6s forwards;margin-bottom:38px;}
.badges{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;opacity:0;animation:fu .8s ease 3.0s forwards;margin-bottom:42px;}
.badge{background:rgba(0,188,212,0.12);border:1px solid rgba(0,188,212,0.3);color:#80deea;
  padding:5px 14px;border-radius:20px;font-family:'Rajdhani',sans-serif;font-size:12px;letter-spacing:1.8px;}
/* water waves at bottom */
.ww{position:absolute;bottom:0;left:0;width:100%;overflow:hidden;height:80px;z-index:5;}
svg.wv{width:200%;}
svg.wv1{animation:mw 7s linear infinite;opacity:.65;}
svg.wv2{animation:mw 11s linear infinite reverse;opacity:.4;position:absolute;bottom:6px;}
.sl{position:absolute;top:0;left:0;width:100%;height:2px;
  background:linear-gradient(90deg,transparent,rgba(0,229,255,0.4),transparent);
  animation:sd 6s linear infinite;z-index:3;pointer-events:none;}
@keyframes titleIn{from{opacity:0;transform:translateY(30px) scale(.95);letter-spacing:40px;}to{opacity:1;transform:translateY(0) scale(1);letter-spacing:12px;}}
@keyframes fi{from{opacity:0;}to{opacity:1;}}
@keyframes fu{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}
@keyframes glow{0%{filter:drop-shadow(0 0 10px rgba(0,229,255,.35));}100%{filter:drop-shadow(0 0 45px rgba(0,229,255,1)) drop-shadow(0 0 90px rgba(0,229,255,.25));}}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes pulse{0%,100%{transform:scale(1);}50%{transform:scale(1.08);}}
@keyframes pf{0%{transform:translateY(110vh) scale(0);opacity:0;}8%{opacity:.7;}92%{opacity:.3;}100%{transform:translateY(-140px) scale(1);opacity:0;}}
@keyframes sd{0%{top:-5%;}100%{top:105%;}}
@keyframes mw{0%{transform:translateX(0);}100%{transform:translateX(-50%);}}
</style></head><body>
<div class="wrap">""" + video_tag + """
<div class="overlay"></div><div class="sl"></div><div class="pc" id="pc"></div>
<div class="content">
  <div class="rw"><div class="r1"></div><div class="r2"></div><div class="br">🧠</div></div>
  <div class="title">NEUROVISION</div>
  <div class="div"></div>
  <div class="sub">AI — Powered Brain Disease Detection System</div>
  <div class="badges">
    <span class="badge">🧩 Alzheimer's</span><span class="badge">⚡ Stroke</span>
    <span class="badge">🔬 Brain Tumor</span><span class="badge">🤖 ResNet50 AI</span>
    <span class="badge">🏥 Clinical Grade</span><span class="badge">🇮🇳 Made in India</span>
  </div>
</div>
<div class="ww">
  <svg class="wv wv1" viewBox="0 0 1440 80" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
    <path fill="rgba(0,188,212,0.38)" d="M0,40 C180,15 360,60 540,40 C720,15 900,60 1080,40 C1260,15 1440,60 1440,40 L1440,80 L0,80 Z"/>
  </svg>
  <svg class="wv wv2" viewBox="0 0 1440 80" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none">
    <path fill="rgba(2,136,209,0.22)" d="M0,50 C200,25 400,65 600,50 C800,25 1000,65 1200,50 C1320,25 1440,55 1440,50 L1440,80 L0,80 Z"/>
  </svg>
</div>
</div>
<script>
const pc=document.getElementById('pc');
const cols=['#00e5ff','#4fc3f7','#00bcd4','#80deea','#0288d1','#b2ebf2'];
for(let i=0;i<70;i++){
  const p=document.createElement('div'),sz=Math.random()*3.5+0.8;
  p.style.cssText='position:absolute;width:'+sz+'px;height:'+sz+'px;background:'
    +cols[Math.floor(Math.random()*cols.length)]+';border-radius:50%;left:'
    +Math.random()*100+'%;animation:pf '+(Math.random()*20+10)+'s linear '
    +(Math.random()*12)+'s infinite;box-shadow:0 0 '+(sz*2.5)+'px #00bcd4;';
  pc.appendChild(p);
}
</script>
</body></html>""")

    st.components.v1.html(html, height=680, scrolling=False)
    st.markdown("<br>", unsafe_allow_html=True)
    _, c2, _ = st.columns([2, 1, 2])
    with c2:
        if st.button("🚀  Enter NeuroVision  →", use_container_width=True):
            st.session_state.page = "main"; st.rerun()
    st.markdown(
        "<div style='text-align:center;font-size:12px;margin-top:8px;"
        "font-family:Rajdhani,sans-serif;color:#4dd0e1;letter-spacing:2px;'>"
        "SRM UNIVERSITY DELHI-NCR, SONEPAT &nbsp;•&nbsp; BTECH COMPUTER SCIENCE</div>",
        unsafe_allow_html=True)
    render_chatbot()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE — MAIN
# ════════════════════════════════════════════════════════════════════════════
def page_main():
    nav_bar()
    brain = b64("1777098010701_image.png")
    itag = (
        f'<img src="data:image/png;base64,{brain}" '
        f'style="width:220px;height:220px;object-fit:cover;border-radius:50%;'
        f'display:block;margin:0 auto 28px;border:2px solid rgba(0,188,212,0.4);'
        f'box-shadow:0 0 60px rgba(0,188,212,0.4),0 0 120px rgba(0,188,212,0.1);">'
    ) if brain else '<div style="font-size:100px;text-align:center;margin-bottom:28px;">🧠</div>'

    st.markdown(
        f"""<div style='background:linear-gradient(180deg,#040810 0%,#061225 50%,#040810 100%);
          padding:60px 40px 50px;text-align:center;position:relative;overflow:hidden;
          border-bottom:1px solid rgba(0,188,212,0.12);'>
          <div style='position:absolute;inset:0;
            background:radial-gradient(ellipse 60% 80% at 50% 50%,rgba(0,188,212,0.06),transparent);
            pointer-events:none;'></div>
          {itag}
          <div style='font-family:Orbitron,monospace;font-size:clamp(24px,4vw,52px);font-weight:900;
            background:linear-gradient(135deg,#00e5ff,#4fc3f7,#0288d1);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;letter-spacing:5px;margin-bottom:12px;'>NEUROVISION</div>
          <div style='font-family:Rajdhani,sans-serif;font-size:clamp(12px,1.6vw,19px);
            color:#80deea;letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;'>
            AI-Powered Brain MRI Disease Detection</div>
          <div style='font-family:Rajdhani,sans-serif;font-size:15px;color:#78909c;
            max-width:700px;margin:0 auto;line-height:1.7;'>
            Upload a brain MRI scan to detect
            <strong style='color:#4fc3f7;'>Alzheimer's</strong>,
            <strong style='color:#ef5350;'>Stroke</strong>, or
            <strong style='color:#66bb6a;'>Brain Tumor</strong>
            using our ResNet50 deep-learning model — then receive personalised medicine and prevention advice.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 40px;'>", unsafe_allow_html=True)
    section_title("About the Conditions","Learn about the three brain conditions NeuroVision can detect")

    c1, c2, c3 = st.columns(3, gap="medium")
    for col, (key, d) in zip([c1,c2,c3], DISEASE_DATA.items()):
        symp = "".join(f"<li>{s}</li>" for s in d["symptoms"])
        with col:
            st.markdown(
                f"""<div style='background:{d["gradient"]};border:1px solid {d["border"]}35;
                  border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 6px 30px {d["glow"]};'>
                  <div style='font-size:40px;text-align:center;margin-bottom:14px;'>{d["icon"]}</div>
                  <div style='font-family:Orbitron,monospace;font-size:15px;font-weight:700;
                    color:{d["color"]};text-align:center;margin-bottom:10px;letter-spacing:1px;'>{d["title"]}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:13px;color:#78909c;
                    text-align:center;margin-bottom:14px;font-style:italic;'>{d["short"]}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:13.5px;color:#90a4ae;
                    line-height:1.65;margin-bottom:16px;'>{d["description"]}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:{d["color"]};
                    font-size:12px;letter-spacing:1.5px;margin-bottom:7px;'>⚠  KEY SYMPTOMS</div>
                  <ul style='font-family:Rajdhani,sans-serif;font-size:13px;color:#78909c;
                    padding-left:17px;line-height:1.9;'>{symp}</ul>
                  <div style='margin-top:16px;padding:12px;background:rgba(0,0,0,0.35);
                    border-radius:10px;text-align:center;'>
                    <div style='font-family:Rajdhani,sans-serif;font-size:11px;color:#78909c;
                      letter-spacing:2px;margin-bottom:6px;'>DANGER LEVEL</div>
                    <div style='font-family:Orbitron,monospace;font-size:13px;
                      color:{d["danger_color"]};font-weight:700;margin-bottom:8px;'>{d["danger_text"]}</div>
                    <div style='background:rgba(255,255,255,0.06);border-radius:8px;height:5px;overflow:hidden;'>
                      <div style='background:{d["danger_color"]};width:{d["danger"]}%;height:100%;
                        border-radius:8px;box-shadow:0 0 8px {d["danger_color"]};'></div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Upload MRI Scan","Drop a brain MRI image (JPG / PNG / JPEG) for instant AI analysis")

    _, up_col, _ = st.columns([1, 2, 1])
    with up_col:
        uploaded = st.file_uploader("Drop MRI scan · JPG / JPEG / PNG",
                                    type=["jpg","jpeg","png"], key="mri_upload")
        if uploaded:
            img = Image.open(uploaded)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image(img, use_container_width=True, caption="Uploaded Brain MRI Scan")
            st.markdown("<br>", unsafe_allow_html=True)
            with st.spinner("🔬  Analysing MRI scan with ResNet50 AI…"):
                cls, conf, probs = predict(img)
            if cls is None:
                st.error("⚠️ Model not found. Ensure neurovision_model.keras is in the same folder as app.py.")
            else:
                d = DISEASE_DATA[cls]
                st.markdown(
                    f"""<div style='background:{d["gradient"]};border:2px solid {d["color"]}55;
                      border-radius:18px;padding:32px;margin:10px 0 24px;
                      box-shadow:0 0 60px {d["glow"]};text-align:center;'>
                      <div style='font-size:56px;margin-bottom:14px;'>{d["icon"]}</div>
                      <div style='font-family:Orbitron,monospace;font-size:11px;color:{d["color"]};
                        letter-spacing:4px;margin-bottom:8px;'>AI DIAGNOSIS RESULT</div>
                      <div style='font-family:Orbitron,monospace;font-size:clamp(18px,3vw,28px);
                        font-weight:900;color:#ffffff;margin-bottom:18px;letter-spacing:2px;'>
                        {d["title"].upper()}</div>
                      <div style='background:rgba(0,0,0,0.45);border-radius:30px;
                        padding:8px 30px;display:inline-block;margin-bottom:14px;'>
                        <span style='font-family:Rajdhani,sans-serif;font-size:21px;
                          font-weight:700;color:#00e5ff;'>Confidence: {conf:.1f}%</span>
                      </div>
                      <div style='font-family:Rajdhani,sans-serif;font-size:14px;
                        color:#78909c;margin-top:8px;'>{d["short"]}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<div style='font-family:Rajdhani,sans-serif;font-size:11px;color:#78909c;"
                            "letter-spacing:2px;margin-bottom:10px;'>CLASS PROBABILITIES</div>",
                            unsafe_allow_html=True)
                for i, cn in enumerate(CLASS_NAMES):
                    dd = DISEASE_DATA[cn]; v = float(probs[i])*100
                    st.markdown(
                        f"""<div style='margin-bottom:9px;'>
                          <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                            <span style='font-family:Rajdhani,sans-serif;font-size:13px;color:#90a4ae;'>{dd["title"]}</span>
                            <span style='font-family:Orbitron,monospace;font-size:11px;color:{dd["color"]};'>{v:.1f}%</span>
                          </div>
                          <div style='background:rgba(255,255,255,0.05);border-radius:6px;height:5px;overflow:hidden;'>
                            <div style='background:{dd["color"]};width:{v:.1f}%;height:100%;
                              border-radius:6px;box-shadow:0 0 5px {dd["color"]};'></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div style='font-family:Orbitron,monospace;font-size:18px;font-weight:700;"
                            "color:#00e5ff;letter-spacing:2px;text-align:center;margin-bottom:22px;'>"
                            "💊  RECOMMENDED TREATMENT &amp; PREVENTION</div>", unsafe_allow_html=True)

                mc, hc = st.columns(2, gap="medium")
                with mc:
                    items = "".join(
                        f"<div style='padding:11px 14px;background:rgba(0,0,0,0.3);border-radius:9px;"
                        f"margin-bottom:8px;border-left:3px solid {d['color']};'>"
                        f"<span style='font-family:Rajdhani,sans-serif;font-size:13.5px;color:#b0bec5;'>💊 {m}</span></div>"
                        for m in d["medicines"])
                    st.markdown(
                        f"""<div style='background:rgba(0,188,212,0.05);border:1px solid rgba(0,188,212,0.18);
                          border-radius:14px;padding:20px;'>
                          <div style='font-family:Rajdhani,sans-serif;font-size:15px;font-weight:700;
                            color:{d["color"]};letter-spacing:1.5px;margin-bottom:14px;'>💊 PRESCRIBED MEDICINES</div>
                          {items}</div>""", unsafe_allow_html=True)
                with hc:
                    items2 = "".join(
                        f"<div style='padding:11px 14px;background:rgba(0,0,0,0.3);border-radius:9px;"
                        f"margin-bottom:8px;border-left:3px solid #66bb6a;'>"
                        f"<span style='font-family:Rajdhani,sans-serif;font-size:13.5px;color:#b0bec5;'>✅ {h}</span></div>"
                        for h in d["habits"])
                    st.markdown(
                        f"""<div style='background:rgba(102,187,106,0.05);border:1px solid rgba(102,187,106,0.18);
                          border-radius:14px;padding:20px;'>
                          <div style='font-family:Rajdhani,sans-serif;font-size:15px;font-weight:700;
                            color:#66bb6a;letter-spacing:1.5px;margin-bottom:14px;'>🥗 HEALTHY HABITS &amp; PREVENTION</div>
                          {items2}</div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                rc, oc = st.columns(2, gap="medium")
                with rc:
                    st.markdown(
                        f"""<div style='background:rgba(0,0,0,0.25);border-radius:12px;padding:20px;
                          border-left:3px solid {d["color"]};'>
                          <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:{d["color"]};
                            font-size:14px;letter-spacing:2px;margin-bottom:10px;'>🩺 REMEDIES &amp; TREATMENT</div>
                          <div style='font-family:Rajdhani,sans-serif;font-size:14px;color:#90a4ae;line-height:1.7;'>
                            {d["remedies"]}</div></div>""", unsafe_allow_html=True)
                with oc:
                    st.markdown(
                        f"""<div style='background:rgba(0,0,0,0.25);border-radius:12px;padding:20px;
                          border-left:3px solid #66bb6a;'>
                          <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:#66bb6a;
                            font-size:14px;letter-spacing:2px;margin-bottom:10px;'>💪 HOW TO OVERCOME</div>
                          <div style='font-family:Rajdhani,sans-serif;font-size:14px;color:#90a4ae;line-height:1.7;'>
                            {d["overcome"]}</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    footer()
    render_chatbot()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE — MEDICAL ASSISTANCE
# ════════════════════════════════════════════════════════════════════════════
def page_medical():
    nav_bar()
    st.markdown(
        """<div style='background:linear-gradient(135deg,#040810,#061225);
          padding:54px 40px 38px;text-align:center;border-bottom:1px solid rgba(0,188,212,0.12);'>
          <div style='font-family:Orbitron,monospace;font-size:clamp(20px,3.5vw,40px);font-weight:900;
            background:linear-gradient(135deg,#00e5ff,#4fc3f7,#0288d1);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text;letter-spacing:4px;margin-bottom:12px;'>MEDICAL ASSISTANCE PANEL</div>
          <div style='font-family:Rajdhani,sans-serif;font-size:15px;color:#78909c;
            max-width:720px;margin:0 auto;line-height:1.7;'>
            Comprehensive medical info on remedies, danger levels, treatments, top Indian hospitals,
            and annual death statistics for Alzheimer's, Stroke, and Brain Tumor.
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='padding:0 40px;'>", unsafe_allow_html=True)
    section_title("Remedies & How to Overcome","Detailed medical guidance for each brain condition")

    for key, d in DISEASE_DATA.items():
        st.markdown(
            f"""<div style='background:{d["gradient"]};border:1px solid {d["border"]}30;
              border-radius:18px;padding:30px;margin-bottom:26px;box-shadow:0 6px 38px {d["glow"]};'>
              <div style='display:flex;align-items:center;gap:16px;margin-bottom:20px;'>
                <div style='font-size:46px;'>{d["icon"]}</div>
                <div>
                  <div style='font-family:Orbitron,monospace;font-size:18px;font-weight:700;
                    color:{d["color"]};letter-spacing:1px;'>{d["title"]}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:13px;color:#78909c;
                    letter-spacing:1px;margin-top:3px;'>Danger Level: &nbsp;
                    <span style='color:{d["danger_color"]};font-weight:700;'>{d["danger_text"]}</span></div>
                </div>
                <div style='margin-left:auto;text-align:right;'>
                  <div style='font-family:Orbitron,monospace;font-size:30px;font-weight:900;
                    color:{d["danger_color"]};'>{d["danger"]}%</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:11px;color:#78909c;
                    letter-spacing:1px;'>SEVERITY INDEX</div>
                </div>
              </div>
              <div style='width:100%;background:rgba(0,0,0,0.3);border-radius:8px;
                height:7px;overflow:hidden;margin-bottom:24px;'>
                <div style='background:{d["danger_color"]};width:{d["danger"]}%;height:100%;
                  border-radius:8px;box-shadow:0 0 12px {d["danger_color"]};'></div>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;'>
                <div style='background:rgba(0,0,0,0.25);border-radius:12px;padding:20px;
                  border-left:3px solid {d["color"]};'>
                  <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:{d["color"]};
                    font-size:13px;letter-spacing:2px;margin-bottom:11px;'>🩺 REMEDIES &amp; TREATMENT</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:14px;color:#90a4ae;line-height:1.75;'>
                    {d["remedies"]}</div>
                </div>
                <div style='background:rgba(0,0,0,0.25);border-radius:12px;padding:20px;
                  border-left:3px solid #66bb6a;'>
                  <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:#66bb6a;
                    font-size:13px;letter-spacing:2px;margin-bottom:11px;'>💪 HOW TO OVERCOME</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:14px;color:#90a4ae;line-height:1.75;'>
                    {d["overcome"]}</div>
                </div>
              </div>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;'>
                <div style='background:rgba(0,0,0,0.2);border-radius:12px;padding:18px;'>
                  <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:{d["color"]};
                    font-size:12px;letter-spacing:2px;margin-bottom:10px;'>💊 KEY MEDICINES</div>
                  {''.join(f"<div style='font-family:Rajdhani,sans-serif;font-size:13px;color:#90a4ae;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>• {m}</div>" for m in d['medicines'])}
                </div>
                <div style='background:rgba(0,0,0,0.2);border-radius:12px;padding:18px;'>
                  <div style='font-family:Rajdhani,sans-serif;font-weight:700;color:#66bb6a;
                    font-size:12px;letter-spacing:2px;margin-bottom:10px;'>🥗 PREVENTION HABITS</div>
                  {''.join(f"<div style='font-family:Rajdhani,sans-serif;font-size:13px;color:#90a4ae;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);'>• {h}</div>" for h in d['habits'])}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    section_title("Top Neurological Hospitals in India",
                  "Leading institutions for Neurology, Neurosurgery & Neuro-Oncology — India only")
    hc = st.columns(3, gap="medium")
    for i, h in enumerate(HOSPITALS):
        with hc[i % 3]:
            st.markdown(
                f"""<div style='background:linear-gradient(145deg,rgba(10,20,40,0.92),rgba(5,10,25,0.96));
                  border:1px solid {h["color"]}38;border-radius:16px;padding:0;margin-bottom:22px;
                  overflow:hidden;box-shadow:0 6px 32px {h["glow"]};'>
                  <div style='background:linear-gradient(135deg,{h["color"]}28,{h["color"]}0e);
                    padding:22px;border-bottom:1px solid {h["color"]}22;text-align:center;'>
                    <div style='font-size:44px;margin-bottom:8px;'>{h["icon"]}</div>
                    <div style='background:{h["color"]}22;border:1px solid {h["color"]}44;
                      color:{h["color"]};font-family:Rajdhani,sans-serif;font-size:11px;letter-spacing:2px;
                      padding:3px 11px;border-radius:12px;display:inline-block;margin-bottom:10px;'>{h["tag"]}</div>
                    <div style='font-family:Orbitron,monospace;font-size:13px;font-weight:700;
                      color:{h["color"]};letter-spacing:0.5px;margin-bottom:4px;'>{h["name"]}</div>
                    <div style='font-family:Rajdhani,sans-serif;font-size:11px;color:#78909c;'>{h["full"]}</div>
                  </div>
                  <div style='padding:16px 18px;'>
                    <div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:9px;'>
                      <span style='color:{h["color"]};font-size:14px;min-width:20px;'>📍</span>
                      <span style='font-family:Rajdhani,sans-serif;font-size:13px;color:#90a4ae;line-height:1.5;'>{h["location"]}</span>
                    </div>
                    <div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:9px;'>
                      <span style='color:{h["color"]};font-size:14px;min-width:20px;'>🔬</span>
                      <span style='font-family:Rajdhani,sans-serif;font-size:12px;color:#90a4ae;line-height:1.5;'>{h["specialty"]}</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:8px;margin-bottom:7px;'>
                      <span style='color:{h["color"]};font-size:14px;min-width:20px;'>📞</span>
                      <span style='font-family:Rajdhani,sans-serif;font-size:13px;color:{h["color"]};font-weight:600;'>{h["phone"]}</span>
                    </div>
                    <div style='display:flex;align-items:center;gap:8px;'>
                      <span style='color:{h["color"]};font-size:14px;min-width:20px;'>🌐</span>
                      <span style='font-family:Rajdhani,sans-serif;font-size:12px;color:#78909c;'>{h["website"]}</span>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    section_title("Annual Brain Disease Deaths in India",
                  "Approximate figures in thousands per year — Source: ICMR / WHO / Published Studies")
    fig = go.Figure()
    for disease, values in INDIA_DEATHS.items():
        color = DEATH_COLORS[disease]; rgb = color.lstrip("#")
        r,g,bv = int(rgb[0:2],16),int(rgb[2:4],16),int(rgb[4:6],16)
        fig.add_trace(go.Scatter(
            x=YEARS,y=values,name=disease,mode="lines+markers",
            line=dict(color=color,width=3),
            marker=dict(size=9,color=color,line=dict(width=2,color="#040810")),
            fill="tozeroy",fillcolor=f"rgba({r},{g},{bv},0.09)",
            hovertemplate=f"<b>{disease}</b><br>Year: %{{x}}<br>Deaths: %{{y}}K<extra></extra>"))
    fig.update_layout(
        paper_bgcolor="#060d1c",plot_bgcolor="#060d1c",
        font=dict(family="Rajdhani, sans-serif",color="#80deea",size=13),
        title=dict(text="Annual Brain Disease Deaths in India (thousands)",
                   font=dict(family="Orbitron, monospace",size=16,color="#00e5ff"),x=0.5,xanchor="center"),
        xaxis=dict(gridcolor="rgba(0,188,212,0.08)",zeroline=False,
                   title=dict(text="Year",font=dict(color="#80deea")),tickfont=dict(color="#80deea")),
        yaxis=dict(gridcolor="rgba(0,188,212,0.08)",zeroline=False,
                   title=dict(text="Deaths (thousands)",font=dict(color="#80deea")),tickfont=dict(color="#80deea")),
        legend=dict(bgcolor="rgba(4,8,16,0.85)",bordercolor="rgba(0,188,212,0.2)",borderwidth=1,
                    font=dict(family="Rajdhani",size=13,color="#80deea")),
        margin=dict(l=60,r=30,t=65,b=60),height=470,hovermode="x unified")
    fig.update_xaxes(showline=True,linecolor="rgba(0,188,212,0.2)")
    fig.update_yaxes(showline=True,linecolor="rgba(0,188,212,0.2)")
    st.plotly_chart(fig, use_container_width=True)

    stats = [
        ("⚡","Stroke","~1.78 Million","#ef5350","deaths / year in India","Leading cause of adult disability"),
        ("🧩","Alzheimer's","~383 Thousand","#4fc3f7","deaths / year in India","5.3M+ Indians living with dementia"),
        ("🔬","Brain Tumor","~36 Thousand","#66bb6a","deaths / year in India","40K+ new cases reported annually"),
    ]
    s1,s2,s3 = st.columns(3,gap="medium")
    for col,(icon,title,num,clr,sub,note) in zip([s1,s2,s3],stats):
        with col:
            st.markdown(
                f"""<div style='background:rgba(10,20,40,0.85);border:1px solid {clr}25;
                  border-radius:14px;padding:22px;text-align:center;
                  margin-bottom:22px;box-shadow:0 4px 22px rgba(0,0,0,0.35);'>
                  <div style='font-size:34px;margin-bottom:8px;'>{icon}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:13px;color:{clr};
                    letter-spacing:2px;margin-bottom:4px;'>{title.upper()}</div>
                  <div style='font-family:Orbitron,monospace;font-size:22px;font-weight:900;
                    color:{clr};margin-bottom:4px;'>{num}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:12px;
                    color:#78909c;margin-bottom:6px;'>{sub}</div>
                  <div style='font-family:Rajdhani,sans-serif;font-size:12px;
                    color:#90a4ae;font-style:italic;'>{note}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    footer()
    render_chatbot()

# ════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    page_password()
else:
    p = st.session_state.page
    if p == "intro":    page_intro()
    elif p == "main":   page_main()
    elif p == "medical":page_medical()