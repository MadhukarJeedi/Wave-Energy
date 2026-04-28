from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model_15.pkl")

# =========================
# INIT APP
# =========================
app = FastAPI()

# =========================
# SELECTED FEATURES (IMPORTANT)
# =========================
FEATURES = [
    'X2','X3','X4','X16','X19','X20',
    'X23','X24','X27','X30','X32',
    'X40','X46','X47','X48'
]

# =========================
# INPUT SCHEMA (ONLY 30 FEATURES)
# =========================
class InputData(BaseModel):
    X2: float
    X3: float
    X4: float
    X16: float
    X19: float
    X20: float
    X23: float
    X24: float
    X27: float
    X32: float
    X40: float
    X46: float
    X47: float
    X48: float

# =========================
# HOME ROUTE
# =========================
@app.get("/")
def home():
    return {"message": "Wave Energy Prediction API Running"}

# =========================
# PREDICTION ROUTE
# =========================
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert to dataframe
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Ensure correct column order
        df = df[FEATURES]

        # Apply same transformation (IMPORTANT)
        df = np.log1p(df)

        # Predict
        prediction = model.predict(df)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        return {"error": str(e)}