import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import joblib
import numpy as np
import pandas as pd

from backend.app.extract_feature import extract_features, normalize_url

OUTPUT_DIR = Path("backend/app/models/train/output_xgb")
DATASET_DIR = Path("backend/app/models/train/datasets")

model = joblib.load(OUTPUT_DIR / "xgboost_url_phishing.joblib")
with open(OUTPUT_DIR / "feature_names.json") as f:
    FEATURE_NAMES = json.load(f)
vectorizer = joblib.load(OUTPUT_DIR / "tfidf_vectorizer.joblib")
with open(OUTPUT_DIR / "test_metrics.json") as f:
    metrics = json.load(f)
THRESHOLD = metrics.get("optimal_threshold", 0.5)
BASE_FEATURE_COUNT = metrics.get("base_feature_count", 48)

# Load test set URLs from the augmented CSV
df = pd.read_csv(DATASET_DIR / "new_augmented_rows.csv")

errors = {"fn": [], "fp": []}
for _, row in df.iterrows():
    url = row["url"]
    true_label = int(row["label"])
    try:
        cleaned = normalize_url(str(url))
        features = extract_features(cleaned)
        vals = np.array([features.get(name, 0.0) for name in FEATURE_NAMES[:BASE_FEATURE_COUNT]])
        tfidf = vectorizer.transform([cleaned]).toarray()[0]
        X = np.concatenate([vals, tfidf]).reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        pred = int(prob > THRESHOLD)
        if pred != true_label:
            if true_label == 1:
                errors["fn"].append((url, prob, {k: features.get(k) for k in ["has_unicode", "has_confusable", "has_mixed_script", "url_length"]}))
            else:
                errors["fp"].append((url, prob, {k: features.get(k) for k in ["has_unicode", "has_confusable", "has_mixed_script", "url_length"]}))
    except:
        pass

print(f"\nFalse negatives (phishing classified as legit): {len(errors['fn'])}")
for url, prob, feats in errors["fn"][:15]:
    print(f"  prob={prob:.3f} feats={feats} url={url[:80]}")

print(f"\nFalse positives (legit classified as phishing): {len(errors['fp'])}")
for url, prob, feats in errors["fp"][:10]:
    print(f"  prob={prob:.3f} feats={feats} url={url[:80]}")