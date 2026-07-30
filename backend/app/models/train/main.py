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
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

DATA_PATH = "backend/app/models/train/datasets/phishing_legit_dataset_with_features.csv"
OUTPUT_DIR = "backend/app/models/train/output_xgb"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "auc",
}

TFIDF_PARAMS = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "max_features": 500,
    "strip_accents": "unicode",
    "lowercase": True,
}

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_data(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain 'label' column.")
    return df

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)
    print(f"Loaded dataset with {df.shape[0]} rows.")

    base_feature_cols = [c for c in df.columns if c not in ("url", "label", "confidence")]
    has_url_col = "url" in df.columns

    X_base = df[base_feature_cols].values
    y = df["label"].astype(int).values
    urls = df["url"].values if has_url_col else None

    print(f"Base features: {len(base_feature_cols)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    if urls is not None:
        print("Fitting TF-IDF vectorizer on URLs...")
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        X_tfidf = vectorizer.fit_transform(urls).toarray()
        print(f"TF-IDF features: {X_tfidf.shape[1]}")

        tfidf_feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
        X = np.hstack([X_base, X_tfidf])
        all_feature_names = base_feature_cols + tfidf_feature_names
    else:
        X = X_base
        vectorizer = None
        all_feature_names = base_feature_cols

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    print("Training XGBoost...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Save model
    model_path = os.path.join(OUTPUT_DIR, "xgboost_url_phishing.joblib")
    joblib.dump(model, model_path)
    save_json(all_feature_names, os.path.join(OUTPUT_DIR, "feature_names.json"))
    save_json(model.get_params(), os.path.join(OUTPUT_DIR, "xgb_params.json"))
    print(f"Model saved to: {model_path}")

    # Save vectorizer
    if vectorizer is not None:
        vec_path = os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
        joblib.dump(vectorizer, vec_path)
        print(f"Vectorizer saved to: {vec_path}")

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold from validation set
    y_val_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 801)
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        val_pred = (y_val_proba >= t).astype(int)
        f1 = f1_score(y_val, val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    print(f"Optimal threshold: {best_threshold:.4f} (val F1: {best_f1:.6f})")

    y_pred_opt = (y_proba >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_opt)
    prec = precision_score(y_test, y_pred_opt)
    rec = recall_score(y_test, y_pred_opt)
    f1 = f1_score(y_test, y_pred_opt)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred_opt)
    report = classification_report(y_test, y_pred_opt, target_names=["Legit", "Phish"])

    print("\n--- Test Results (optimized threshold) ---")
    print(f"Threshold : {best_threshold:.4f}")
    print(f"Accuracy  : {acc:.6f}")
    print(f"Precision : {prec:.6f}")
    print(f"Recall    : {rec:.6f}")
    print(f"F1 Score  : {f1:.6f}")
    print(f"ROC AUC   : {auc:.6f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "optimal_threshold": float(best_threshold),
        "base_feature_count": len(base_feature_cols),
        "tfidf_feature_count": X_tfidf.shape[1] if urls is not None else 0,
    }
    save_json(metrics, os.path.join(OUTPUT_DIR, "test_metrics.json"))

    # Feature importance
    booster = model.get_booster()
    booster.feature_names = all_feature_names
    importances = booster.get_score(importance_type="gain")
    imp_df = pd.DataFrame([
        {"feature": f, "importance": importances.get(f, 0.0)}
        for f in all_feature_names
    ])
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv"), index=False)

    top_n = min(30, len(imp_df))
    plt.figure(figsize=(10, max(4, top_n * 0.3)))
    sns.barplot(x="importance", y="feature", data=imp_df.head(top_n))
    plt.title(f"XGBoost feature importance (gain) - top {top_n}")
    plt.tight_layout()
    figpath = os.path.join(OUTPUT_DIR, "feature_importance_top.png")
    plt.savefig(figpath, dpi=150)
    plt.close()
    print(f"Feature importance plot saved to: {figpath}")

    print("All done.")

if __name__ == "__main__":
    main()