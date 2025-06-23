from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from backend.app.extract_feature import extract_features
import torch
import torch.nn as nn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler (if used)
model = joblib.load('backend/app/models/model.pkl')
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.output(x))
        return x
    
# Load scaler
scaler = joblib.load('backend/app/models/scaler.pkl')

# Load model
input_dim =  44
model = Net(input_dim)
model.load_state_dict(torch.load('backend/app/models/model.pth', map_location=torch.device('cpu')))
model.eval()

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
        X_scaled = scaler.transform(X)

        # Convert to tensor
        input_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Predict using PyTorch model
        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()
            prediction = int(prob > 0.5)

        status = "legitimate" if prediction == 1 else "phishing"

        return {
            "url": request.url,
            "normalized_url": cleaned_url,
            "is_phishing": status,
            "confidence": float(prob)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
