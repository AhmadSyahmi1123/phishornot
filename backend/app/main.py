from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from backend.app.extract_feature import extract_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load trained XGBoost model
model_path = "backend/app/models/train/output_xgb/xgboost_url_phishing.joblib"
model = joblib.load(model_path)
print("✅ XGBoost model loaded successfully")

# URL normalization
def normalize_url(url: str) -> str:
    # Strip trailing slash only if it's a shallow path (not /a/b/c/)
    return url.rstrip("/") if url.endswith("/") and url.count("/") <= 3 else url

# Define input schema
class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def predict(request: URLRequest):
    try:
        cleaned_url = normalize_url(request.url)

        # Extract features
        features = extract_features(cleaned_url)
        X = np.array(list(features.values())).reshape(1, -1)

        # Predict using XGBoost
        prob = model.predict_proba(X)[0][1]  # probability of class "phishing"
        prediction = int(prob > 0.5)

        status = "phishing" if prediction == 1 else "legitimate"

        return {
            "url": request.url,
            "normalized_url": cleaned_url,
            "is_phishing": status,
            "confidence": float(prob)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
