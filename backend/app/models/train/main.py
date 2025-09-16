# train_xgboost.py
import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier, DMatrix, cv
from xgboost.callback import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "backend/app/models/train/datasets/phishing_legit_dataset_with_features.csv"
OUTPUT_DIR = "backend/app/models/train/output_xgb"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# XGBoost training params (a good starting point)
XGB_PARAMS = {
    "n_estimators": 1000,           # large, but early stopping will cap it
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "tree_method": "hist",          # fast; change to "gpu_hist" if using GPU
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "auc"            # we'll use AUC for early stopping
}

# -------------------------
# Helper functions
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column.")
    # Drop url column (we don't train on raw url)
    X = df.drop(columns=[c for c in ["url"] if c in df.columns] + ["label"])
    y = df["label"].astype(int)
    return X, y

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -------------------------
# Main training flow
# -------------------------
def main():
    print("Loading data...")
    X, y = load_data(DATA_PATH)
    feature_names = list(X.columns)
    print(f"Loaded dataset with {X.shape[0]} rows and {len(feature_names)} features.")

    # split: train / temp  (train 70% / temp 30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # split temp into val / test (15% val, 15% test overall)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    # Convert to numpy (xgboost accepts pandas but keep as arrays)
    X_train_np, X_val_np, X_test_np = X_train.values, X_val.values, X_test.values

    # Build model
    print("Initializing XGBoost classifier...")
    model = XGBClassifier(**XGB_PARAMS)

    # Save feature names before converting
    feature_names = X_train.columns.tolist()

    model.fit(X_train_np, y_train.values)

    # Save model and feature names
    model_path = os.path.join(OUTPUT_DIR, "xgboost_url_phishing.joblib")
    joblib.dump(model, model_path)
    save_json(feature_names, os.path.join(OUTPUT_DIR, "feature_names.json"))
    save_json(model.get_params(), os.path.join(OUTPUT_DIR, "xgb_params.json"))
    print(f"Model saved to: {model_path}")

    # Evaluate on test set
    print("Evaluating on test set...")
    y_pred = model.predict(X_test_np)
    y_proba = model.predict_proba(X_test_np)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Legit", "Phish"])

    print("\n--- Test Results ---")
    print(f"Accuracy : {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall   : {rec:.6f}")
    print(f"F1 Score : {f1:.6f}")
    print(f"ROC AUC  : {auc:.6f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # Save metrics
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }
    save_json(metrics, os.path.join(OUTPUT_DIR, "test_metrics.json"))

    # manually assign feature names
    booster = model.get_booster()
    booster.feature_names = feature_names

    # Feature importance (gain)
    print("Calculating & plotting feature importances...")
    importances = model.get_booster().get_score(importance_type="gain")
    # Convert to dataframe for sorted plotting
    imp_df = pd.DataFrame([
        {"feature": f, "importance": importances.get(f, 0.0)}
        for f in model.get_booster().feature_names
    ])
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv"), index=False)

    # Plot top 30 features
    top_n = min(30, len(imp_df))
    plt.figure(figsize=(8, max(4, top_n * 0.25)))
    sns.barplot(x="importance", y="feature", data=imp_df.head(top_n))
    plt.title("XGBoost feature importance (gain) - top {}".format(top_n))
    plt.tight_layout()
    figpath = os.path.join(OUTPUT_DIR, "feature_importance_top.png")
    plt.savefig(figpath, dpi=150)
    plt.close()
    print("Feature importance plot saved to:", figpath)

    print("All done. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
