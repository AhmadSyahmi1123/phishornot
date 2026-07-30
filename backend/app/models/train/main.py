import os
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from extract_feature import extract_features, normalize_url

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_xgb"

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "eval_metric": "logloss",
}

TFIDF_PARAMS = {
    "analyzer": "char",
    "ngram_range": (3, 5),
    "max_features": 200,
    "sublinear_tf": True,
}

RANDOM_STATE = 42


def load_data():
    phishtank_path = DATA_DIR / "phishtank.csv"
    tranco_path = DATA_DIR / "tranco_list.csv"
    openphish_path = DATA_DIR / "openphish.txt"

    urls = []
    labels = []

    if phishtank_path.exists():
        df = pd.read_csv(phishtank_path)
        url_col = None
        for col in ["url", "URL", "phish_url", "phishing_url"]:
            if col in df.columns:
                url_col = col
                break
        if url_col is not None:
            phish_urls = df[url_col].dropna().astype(str).tolist()
            urls.extend(phish_urls)
            labels.extend([1] * len(phish_urls))

    if openphish_path.exists():
        with open(openphish_path, "r") as f:
            phish_urls = [line.strip() for line in f if line.strip()]
        urls.extend(phish_urls)
        labels.extend([1] * len(phish_urls))

    if tranco_path.exists():
        df = pd.read_csv(tranco_path, header=None, nrows=50000)
        cols = df.columns.tolist()
        domain_col = cols[1] if len(cols) > 1 else cols[0]
        domains = []
        for val in df[domain_col].dropna().astype(str):
            if not val.startswith("http"):
                domains.append("https://" + val)
            else:
                domains.append(val)
        urls.extend(domains)
        labels.extend([0] * len(domains))

    if len(urls) == 0:
        print("No data files found. Generating synthetic data...")
        return _generate_synthetic_data()

    df = pd.DataFrame({"url": urls, "label": labels})
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    return df


def _generate_synthetic_data():
    legitimate = [
        "https://www.google.com",
        "https://www.facebook.com",
        "https://www.youtube.com",
        "https://www.amazon.com",
        "https://www.wikipedia.org",
        "https://www.reddit.com",
        "https://www.twitter.com",
        "https://www.instagram.com",
        "https://www.linkedin.com",
        "https://www.github.com",
    ]
    phishing = [
        "http://login-verify-secure.tk/update",
        "http://account-verify.ml/login",
        "http://paypal-secure.ga/confirm",
        "http://bit.ly/3abc12",
        "http://bank-login.xyz/verify",
        "http://secure-login.ga/account",
        "http://apple-id.gq/reset",
        "http://netflix-login.tk/update",
        "http://amazon-verify.top/claim",
        "http://chase-bank.ml/alert",
    ]
    random.seed(RANDOM_STATE)

    legit_urls = legitimate[:]
    phish_urls = phishing[:]

    suffixes = [".com", ".org", ".net", ".io", ".co"]
    for base in ["https://example", "https://test", "https://demo", "https://sample"]:
        for s in suffixes:
            legit_urls.append(base + s)

    words = ["secure", "login", "verify", "account", "bank", "confirm", "update", "alert"]
    suspicious_tlds = ["tk", "ml", "ga", "cf", "gq", "top", "xyz"]
    for word in words:
        for tld in suspicious_tlds[:3]:
            phish_urls.append(f"http://{word}-{word}.{tld}/reset")

    df = pd.DataFrame({
        "url": legit_urls + phish_urls,
        "label": [0] * len(legit_urls) + [1] * len(phish_urls),
    })
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} URLs ({df['label'].sum()} phishing, {len(df) - df['label'].sum()} legitimate)")

    print("Extracting features...")
    feature_list = []
    for url in df["url"]:
        normalized = normalize_url(url)
        feature_list.append(extract_features(normalized))

    X_feat = pd.DataFrame(feature_list)
    y = df["label"].values

    print("Fitting TF-IDF vectorizer on URLs...")
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    urls_clean = df["url"].str.lower().fillna("")
    X_tfidf = vectorizer.fit_transform(urls_clean).toarray()
    print(f"TF-IDF features: {X_tfidf.shape[1]}")

    X = np.hstack([X_feat.values, X_tfidf])
    feature_names = list(X_feat.columns) + [f"tfidf_{i}" for i in range(X_tfidf.shape[1])]
    print(f"Total features: {len(feature_names)}")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train_sub, X_calib, y_train_sub, y_calib = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train_sub.shape[0]}, Calib: {X_calib.shape[0]}, Test: {X_test.shape[0]}")

    print("Training XGBoost...")
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train_sub, y_train_sub)
    train_score = xgb.score(X_train_sub, y_train_sub)
    calib_score = xgb.score(X_calib, y_calib)
    print(f"XGBoost train acc: {train_score:.4f}, calib acc: {calib_score:.4f}")

    print("Calibrating with Platt scaling...")
    calibrator = CalibratedClassifierCV(xgb, method="sigmoid", cv="prefit")
    calibrator.fit(X_calib, y_calib)

    print("Finding optimal threshold via precision-recall...")
    y_proba = calibrator.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
    best_f1 = float(f1_scores[best_idx])
    print(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.6f})")

    y_pred = (y_proba >= best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1:        {f1:.6f}")
    print(f"AUC:       {auc:.6f}")
    print(f"Confusion matrix:\n{cm}")

    print("Saving artifacts...")
    model_path = OUTPUT_DIR / "xgboost_url_phishing.joblib"
    joblib.dump(calibrator, model_path)
    print(f"Calibrated model saved to: {model_path}")

    vec_path = OUTPUT_DIR / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vec_path)
    print(f"TF-IDF vectorizer saved to: {vec_path}")

    feat_path = OUTPUT_DIR / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Feature names saved to: {feat_path}")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(auc),
        "optimal_threshold": float(best_threshold),
        "best_f1_on_pr_curve": best_f1,
        "confusion_matrix": cm.tolist(),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(len(feature_names)),
    }
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    main()
