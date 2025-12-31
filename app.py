from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn
import pandas as pd  # <-- add this

app = FastAPI(title="Bank Marketing ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dt_model = None
lr_model = None
feature_names = None

class PredictRequest(BaseModel):
    features: list[float]
    model_type: str  # "decision_tree" or "logistic_regression"

def load_models():
    global dt_model, lr_model, feature_names
    dt_model = joblib.load("models/decision_tree_model.joblib")
    lr_model = joblib.load("models/logistic_regression_model.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    print("Loaded feature_names:", feature_names, "len =", len(feature_names))

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {
        "message": "Bank Marketing ML API",
        "models": ["decision_tree", "logistic_regression"],
        "n_features": len(feature_names) if feature_names else None,
    }

@app.post("/predict")
async def predict(req: PredictRequest):
    if dt_model is None or lr_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded yet")

    if req.model_type not in ["decision_tree", "logistic_regression"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'decision_tree' or 'logistic_regression'",
        )

    if len(req.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(feature_names)} features, got {len(req.features)}",
        )

    # Build a DataFrame with correct column names
    x_df = pd.DataFrame([req.features], columns=feature_names)

    if req.model_type == "decision_tree":
        model = dt_model
        model_name = "Decision Tree"
    else:
        model = lr_model
        model_name = "Logistic Regression"

    proba = model.predict_proba(x_df)[0]
    pred = model.predict(x_df)[0]

    return {
        "model": model_name,
        "prediction": int(pred),
        "class_0_probability": float(proba[0]),
        "class_1_probability": float(proba[1]),
        "confidence": float(max(proba)),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
