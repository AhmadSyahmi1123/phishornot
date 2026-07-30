"""
Hard-negative mining: find URLs the current model misclassifies and add them to training.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from backend.app.extract_feature import extract_features, normalize_url

OUTPUT_DIR = Path("backend/app/models/train/output_xgb")
DATASET_DIR = Path("backend/app/models/train/datasets")

def main():
    # ── Load model + vectorizer ──
    model = joblib.load(OUTPUT_DIR / "xgboost_url_phishing.joblib")
    with open(OUTPUT_DIR / "feature_names.json") as f:
        FEATURE_NAMES = json.load(f)

    vec_path = OUTPUT_DIR / "tfidf_vectorizer.joblib"
    try:
        vectorizer = joblib.load(vec_path)
    except FileNotFoundError:
        vectorizer = None

    with open(OUTPUT_DIR / "test_metrics.json") as f:
        metrics = json.load(f)
    THRESHOLD = metrics.get("optimal_threshold", 0.5)
    BASE_FEATURE_COUNT = metrics.get("base_feature_count", len(FEATURE_NAMES))
    print(f"Loaded model: threshold={THRESHOLD}, features={len(FEATURE_NAMES)}")

    # ── Load new augmented data ──
    aug_path = DATASET_DIR / "new_augmented_rows.csv"
    if not aug_path.exists():
        print("No augmented data found. Run collect_more_data.py first.")
        return
    df = pd.read_csv(aug_path)
    urls = df["url"].tolist()
    labels = df["label"].tolist()
    print(f"Loaded {len(urls)} URLs to evaluate")

    # ── Predict batch ──
    false_positives = []
    false_negatives = []
    correct = 0

    for i, (url, true_label) in enumerate(zip(urls, labels)):
        try:
            cleaned = normalize_url(url)
            features = extract_features(cleaned)
            vals = np.array([features.get(name, 0.0) for name in FEATURE_NAMES[:BASE_FEATURE_COUNT]])

            if vectorizer is not None:
                tfidf = vectorizer.transform([cleaned]).toarray()[0]
                X = np.concatenate([vals, tfidf]).reshape(1, -1)
            else:
                X = vals.reshape(1, -1)

            prob = model.predict_proba(X)[0][1]
            pred = int(prob > THRESHOLD)

            if pred == 1 and true_label == 0:
                false_positives.append({"url": url, "confidence": float(prob), "label": 0})
            elif pred == 0 and true_label == 1:
                false_negatives.append({"url": url, "confidence": float(prob), "label": 1})
            else:
                correct += 1
        except Exception as e:
            print(f"  Error on {url[:60]}: {e}")

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(urls)} ...")

    print(f"\nResults on new data:")
    print(f"  Correct: {correct}")
    print(f"  False positives: {len(false_positives)}")
    print(f"  False negatives: {len(false_negatives)}")

    # ── Add errors back to training set (weighted) ──
    if false_positives or false_negatives:
        df_errors = pd.DataFrame(false_positives + false_negatives)
        # Duplicate errors 3x to emphasize them
        df_errors_augmented = pd.concat([df_errors] * 3, ignore_index=True)
        df_errors_augmented.to_csv(DATASET_DIR / "hard_negatives.csv", index=False)
        print(f"\nSaved {len(df_errors_augmented)} hard negatives (3x duplicates)")

        # Show top 10 errors
        print("\nTop false positives (legit flagged as phishing):")
        for fp in sorted(false_positives, key=lambda x: x["confidence"], reverse=True)[:10]:
            print(f"  {fp['url'][:80]} (confidence: {fp['confidence']:.3f})")
        print("\nTop false negatives (phishing flagged as legit):")
        for fn in sorted(false_negatives, key=lambda x: -x["confidence"])[:10]:
            print(f"  {fn['url'][:80]} (confidence: {fn['confidence']:.3f})")

        # ── Merge with existing dataset ──
        existing_path = DATASET_DIR / "phishing_legit_dataset.csv"
        df_existing = pd.read_csv(existing_path)
        df_existing["url"] = df_existing["url"].astype(str).str.strip('"\'').str.strip()
        df_errors["url"] = df_errors["url"].astype(str).str.strip('"\'').str.strip()
        df_combined = pd.concat([df_existing, df_errors_augmented], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["url"])
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        df_combined.to_csv(existing_path, index=False)
        print(f"\nMerged {len(df_errors)} unique errors into {existing_path}")
        print(f"  New total: {len(df_combined)} rows")
        print(f"  Distribution: {df_combined['label'].value_counts().to_dict()}")
    else:
        print("\nNo errors found — model is already perfect on the new data!")

if __name__ == "__main__":
    main()