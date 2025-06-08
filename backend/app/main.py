from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import joblib
import numpy as np
from backend.app.extract_feature import extract_url_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load('backend/app/models/model.pkl')

# Define input schema
class URLRequest(BaseModel):
    url: str

@app.post("/predict")
def predict(request: URLRequest):
    try:
        feature_names = [
        'url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url', 'number_of_digits_in_url',
        'number_of_special_char_in_url', 'number_of_hyphens_in_url', 'number_of_underline_in_url', 
        'number_of_slash_in_url', 'number_of_questionmark_in_url', 'number_of_equal_in_url', 
        'number_of_at_in_url', 'number_of_dollar_in_url', 'number_of_exclamation_in_url', 
        'number_of_hashtag_in_url', 'number_of_percent_in_url', 'domain_length', 'number_of_dots_in_domain',
        'number_of_hyphens_in_domain', 'having_special_characters_in_domain', 'number_of_special_characters_in_domain',
        'having_digits_in_domain', 'number_of_digits_in_domain', 'having_repeated_digits_in_domain',
        'number_of_subdomains', 'having_dot_in_subdomain', 'having_hyphen_in_subdomain', 
        'average_subdomain_length', 'average_number_of_dots_in_subdomain', 'average_number_of_hyphens_in_subdomain',
        'having_special_characters_in_subdomain', 'number_of_special_characters_in_subdomain', 'having_digits_in_subdomain',
        'number_of_digits_in_subdomain', 'having_repeated_digits_in_subdomain', 'having_path', 'path_length',
        'having_query', 'having_fragment', 'having_anchor', 'entropy_of_url', 'entropy_of_domain'
        ]
        
        features = extract_url_features(request.url)
        dmatrix = xgb.DMatrix(np.array(features).reshape(1, -1), feature_names=feature_names)
        prob = model.predict(dmatrix)[0]
        prediction = int(prob > 0.5)
        status = "legitimate" if prediction == 0 else "phishing"

        return {
            "url": request.url,
            "is_phishing": status,
            "confidence": float(prob)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
