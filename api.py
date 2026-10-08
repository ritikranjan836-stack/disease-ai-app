from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib

# ---------- Load everything once ----------
model = joblib.load("disease_model.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")
desc = pd.read_csv("symptom_Description.csv")
precaution = pd.read_csv("symptom_precaution.csv")
desc['Disease'] = desc['Disease'].str.strip()
precaution['Disease'] = precaution['Disease'].str.strip()

app = FastAPI(title="Disease Prediction API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later, replace * with your website's address
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request format: what the caller must send ----------
class SymptomRequest(BaseModel):
    symptoms: List[str]

# ---------- Same helper functions as before ----------
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

# ---------- The actual API endpoint ----------
@app.post("/predict")
def predict(request: SymptomRequest):
    results = predict_disease_top3(request.symptoms)
    return {
        "input_symptoms": request.symptoms,
        "top_predictions": results,
        "disclaimer": "This is an AI-based educational/clinical decision-support tool. It is NOT a substitute for professional medical diagnosis."
    }

@app.get("/symptoms")
def get_all_symptoms():
    return {"available_symptoms": symptom_columns}