import streamlit as st
import pandas as pd
import joblib
import json

# ---------- Load everything once ----------
model = joblib.load("disease_model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")
desc = pd.read_csv("symptom_Description.csv")
precaution = pd.read_csv("symptom_precaution.csv")
desc['Disease'] = desc['Disease'].str.strip()
precaution['Disease'] = precaution['Disease'].str.strip()

# ---------- Helper functions (same logic as Colab) ----------
def get_disease_info(disease_name):
    disease_name = disease_name.strip()
    desc_row = desc[desc['Disease'] == disease_name]
    description = desc_row['Description'].values[0] if not desc_row.empty else "No description available."

    prec_row = precaution[precaution['Disease'] == disease_name]
    if not prec_row.empty:
        precautions = [prec_row[col].values[0] for col in ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
                       if pd.notnull(prec_row[col].values[0])]
    else:
        precautions = ["No precaution data available."]

    return {"disease": disease_name, "description": description, "precautions": precautions}

def predict_disease_top3(symptom_list):
    input_data = pd.DataFrame(0, index=[0], columns=symptom_columns)
    for symptom in symptom_list:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    disease_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)

    top3 = []
    for disease_name, prob in disease_probs[:3]:
        if prob > 0:
            info = get_disease_info(disease_name)
            info['confidence'] = f"{prob*100:.1f}%"
            top3.append(info)
    return top3

# ---------- Streamlit UI ----------
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Disease Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f4f9ff 0%, #eef7f5 100%);
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #0f766e, #087f8c, #2563eb);
        padding: 35px 40px;
        border-radius: 24px;
        color: white;
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.20);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin: 0;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .hero p {
        font-size: 17px;
        margin-top: 10px;
        opacity: 0.92;
    }

    /* Status badge */
    .status {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 7px 14px;
        border-radius: 30px;
        font-size: 13px;
        margin-bottom: 15px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.95);
        padding: 25px;
        border-radius: 18px;
        border: 1px solid #dbeafe;
        box-shadow: 0 6px 20px rgba(30, 64, 175, 0.08);
        margin-bottom: 20px;
    }

    .card h3 {
        color: #0f766e;
        margin-top: 0;
    }

    .card p {
        color: #475569;
        font-size: 15px;
    }

    /* Metric cards */
    .metric {
        background: white;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    .metric-icon {
        font-size: 30px;
    }

    .metric-title {
        color: #64748b;
        font-size: 13px;
        margin-top: 5px;
    }

    .metric-value {
        color: #0f172a;
        font-size: 22px;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #0f766e, #2563eb);
        color: white;
        font-weight: 600;
        padding: 12px;
        transition: 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(37,99,235,0.25);
    }

    /* Inputs */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div,
    .stMultiSelect div {
        border-radius: 10px !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ecfeff, #f8fafc);
        border-right: 1px solid #dbeafe;
    }

    /* Warning box */
    .medical-warning {
        background: #fff7ed;
        border-left: 5px solid #f97316;
        padding: 15px 18px;
        border-radius: 10px;
        color: #7c2d12;
        margin-top: 20px;
    }

</style>
""", unsafe_allow_html=True)



# ---------------- INFO CARDS ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric">
        <div class="metric-icon">🧠</div>
        <div class="metric-title">AI Analysis</div>
        <div class="metric-value">Smart Prediction</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric">
        <div class="metric-icon">🩺</div>
        <div class="metric-title">Clinical Support</div>
        <div class="metric-value">Symptom Based</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric">
        <div class="metric-icon">📊</div>
        <div class="metric-title">Prediction</div>
        <div class="metric-value">Risk Assessment</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# ---------------- MAIN CONTENT ----------------
left, right = st.columns([1.5, 1])

with left:

    st.markdown("""
    <div class="card">
        <h3>🔍 Patient Assessment</h3>
        <p>
            Enter patient symptoms and basic information to generate
            an AI-assisted prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    name = st.text_input("👤 Patient Name", placeholder="Enter patient name")

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=25
    )

    gender = st.selectbox(
        "⚥ Gender",
        ["Male", "Female", "Other"]
    )
st.markdown(
    "<p style='text-align:center;color:#64748b;margin-top:25px;'>"
    "🩺 Disease Prediction AI • Clinical Decision Support Tool"
    "</p>",
    unsafe_allow_html=True
)
selected_symptoms = st.multiselect(
    "Select symptoms:",
    options=symptom_columns,
    help="Start typing to search symptoms"
)

if st.button("Predict Disease"):
    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        results = predict_disease_top3(selected_symptoms)
        st.subheader("Top Predictions:")
        for r in results:
            with st.expander(f"{r['disease']} — Confidence: {r['confidence']}"):
                st.write("**Description:**", r['description'])
                st.write("**Precautions:**")
                for p in r['precautions']:
                    st.write("- ", p)

    st.markdown("<br>", unsafe_allow_html=True)






with right:

    st.markdown("""
    <div class="card">
        <h3>📋 Patient Summary</h3>
        <p>
            Review the patient's information and selected symptoms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("**Name:**", name if name else "N/A")
    st.write("**Age:**", age)
    st.write("**Gender:**", gender)
    st.write("**Selected Symptoms:**")
    if selected_symptoms:
        for symptom in selected_symptoms:
            st.write("- ", symptom)
    else:
        st.write("No symptoms selected.")


# ---------------- MEDICAL DISCLAIMER ----------------
st.markdown("""
<div class="medical-warning">
    ⚠️ <b>Medical Disclaimer:</b>
    This system is intended for educational and clinical decision-support
    purposes only. It does not replace professional medical diagnosis,
    examination, or treatment.
</div>
""", unsafe_allow_html=True)

