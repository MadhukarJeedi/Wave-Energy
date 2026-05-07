from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")

# =========================
# INIT APP
# =========================
app = FastAPI()

# =========================
# INPUT SCHEMA
# =========================
class InputData(BaseModel):
    X1: float; Y1: float
    X2: float; Y2: float
    X3: float; Y3: float
    X4: float; Y4: float
    X5: float; Y5: float
    X6: float; Y6: float
    X7: float; Y7: float
    X8: float; Y8: float
    X9: float; Y9: float
    X10: float; Y10: float
    X11: float; Y11: float
    X12: float; Y12: float
    X13: float; Y13: float
    X14: float; Y14: float
    X15: float; Y15: float
    X16: float; Y16: float
    X17: float; Y17: float
    X18: float; Y18: float
    X19: float; Y19: float
    X20: float; Y20: float
    X21: float; Y21: float
    X22: float; Y22: float
    X23: float; Y23: float
    X24: float; Y24: float
    X25: float; Y25: float
    X26: float; Y26: float
    X27: float; Y27: float
    X28: float; Y28: float
    X29: float; Y29: float
    X30: float; Y30: float
    X31: float; Y31: float
    X32: float; Y32: float
    X33: float; Y33: float
    X34: float; Y34: float
    X35: float; Y35: float
    X36: float; Y36: float
    X37: float; Y37: float
    X38: float; Y38: float
    X39: float; Y39: float
    X40: float; Y40: float
    X41: float; Y41: float
    X42: float; Y42: float
    X43: float; Y43: float
    X44: float; Y44: float
    X45: float; Y45: float
    X46: float; Y46: float
    X47: float; Y47: float
    X48: float; Y48: float
    X49: float; Y49: float

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
        # Convert input to dataframe
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])

        # Reorder columns to match training order
        feature_cols = [
            'X1','Y1','X2','Y2','X3','Y3','X4','Y4','X5','Y5',
            'X6','Y6','X7','Y7','X8','Y8','X9','Y9','X10','Y10',
            'X11','Y11','X12','Y12','X13','Y13','X14','Y14','X15','Y15',
            'X16','Y16','X17','Y17','X18','Y18','X19','Y19','X20','Y20',
            'X21','Y21','X22','Y22','X23','Y23','X24','Y24','X25','Y25',
            'X26','Y26','X27','Y27','X28','Y28','X29','Y29','X30','Y30',
            'X31','Y31','X32','Y32','X33','Y33','X34','Y34','X35','Y35',
            'X36','Y36','X37','Y37','X38','Y38','X39','Y39','X40','Y40',
            'X41','Y41','X42','Y42','X43','Y43','X44','Y44','X45','Y45',
            'X46','Y46','X47','Y47','X48','Y48','X49','Y49'
        ]
        df = df[feature_cols]

        # Apply same transformation (log1p)
        df = np.log1p(df)

        # Predict
        prediction = model.predict(df)[0]

        return {
            "prediction": float(prediction)
        }

    except Exception as e:
        return {"error": str(e)}