import os
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, confusion_matrix,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from extract_feature import (
    SUSPICIOUS_TLDS,
    SUSPICIOUS_WORDS,
    extract_features,
    normalize_url,
)

DATA_DIR = Path(__file__).resolve().parent / "datasets"
LEGACY_DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output_xgb"

URL_COLUMN_CANDIDATES = ["url", "URL", "domain", "Domain"]
LABEL_COLUMN_CANDIDATES = ["label", "class", "type"]

MAX_ROWS_PER_FILE = 300000

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.2,
    "reg_lambda": 2.0,
    "random_state": 42,
    "eval_metric": "logloss",
}

RANDOM_STATE = 42


def _label_orientation_hints(urls: pd.Series) -> int:
    """Count phishing-ish signals in a set of URLs (suspicious TLDs, keywords, raw IPs).

    Used to detect datasets whose label column is inverted (label=0 = phishing).
    """
    urls = urls.str.lower()
    tld_hits = urls.str.extract(r"\.([a-z]{2,24})\s*$")[0].isin(SUSPICIOUS_TLDS).sum()
    word_hits = urls.str.contains("|".join(SUSPICIOUS_WORDS), regex=True).sum()
    ip_hits = urls.str.contains(r"//\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", regex=True).sum()
    return int(tld_hits) + int(word_hits) + int(ip_hits)


def _orient_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Force the internal convention label=1 == phishing.

    Several CSVs in datasets/ ship with inverted labels (0=phishing, 1=legit).
    Compare the phishing-signal density per class; if the label=1 class looks
    cleaner than the label=0 class, swap the labels.
    """
    df = df.copy()
    if len(df) < 10 or df["label"].nunique() < 2:
        return df
    hints_0 = _label_orientation_hints(df.loc[df["label"] == 0, "url"])
    hints_1 = _label_orientation_hints(df.loc[df["label"] == 1, "url"])
    if hints_1 < hints_0:
        print("    labels appear inverted (label=1 is the clean class) - flipping")
        df["label"] = 1 - df["label"]
    return df


def _load_single_csv(path: Path) -> pd.DataFrame | None:
    """Parse one CSV into (url, label) rows with label=1 == phishing.

    Tries known url/label column names, skips files that can't be parsed
    (e.g. pre-extracted feature CSVs without a url column).
    """
    try:
        df = pd.read_csv(path, nrows=MAX_ROWS_PER_FILE)
    except Exception:
        return None
    url_col = next((c for c in URL_COLUMN_CANDIDATES if c in df.columns), None)
    if url_col is None:
        return None
    label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)
    if label_col is None:
        return None
    try:
        df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "label"})
        df = df.dropna()
        df["url"] = df["url"].astype(str).str.strip()
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")
    except Exception:
        return None
    df = df[df["label"].isin([0, 1])]
    df["label"] = df["label"].astype(int)
    df = df[df["url"].str.len() > 0]
    df = df[df["url"].str.startswith(("http://", "https://"))]
    if df.empty:
        return None
    return _orient_labels(df)


def load_data():
    frames = []

    for path in sorted(DATA_DIR.glob("*.csv")):
        df = _load_single_csv(path)
        if df is not None:
            print(f"  loaded {path.name}: {len(df)} rows")
            frames.append(df)
        else:
            print(f"  skipped {path.name} (no url/label columns or unparsable)")

    for path in sorted(LEGACY_DATA_DIR.glob("*.csv")):
        df = _load_single_csv(path)
        if df is not None:
            print(f"  loaded {path.name}: {len(df)} rows")
            frames.append(df)

    openphish_path = LEGACY_DATA_DIR / "openphish.txt"
    if openphish_path.exists():
        with open(openphish_path, "r", encoding="utf-8", errors="ignore") as f:
            phish_urls = [line.strip() for line in f if line.strip()]
        if phish_urls:
            frames.append(pd.DataFrame({"url": phish_urls, "label": [1] * len(phish_urls)}))
            print(f"  loaded openphish.txt: {len(phish_urls)} rows")

    tranco_path = LEGACY_DATA_DIR / "tranco_list.csv"
    if tranco_path.exists():
        try:
            df = pd.read_csv(tranco_path, header=None, nrows=50000)
            cols = df.columns.tolist()
            domain_col = cols[1] if len(cols) > 1 else cols[0]
            domains = []
            for val in df[domain_col].dropna().astype(str):
                if not val.startswith("http"):
                    domains.append("https://" + val)
                else:
                    domains.append(val)
            frames.append(pd.DataFrame({"url": domains, "label": [0] * len(domains)}))
            print(f"  loaded tranco_list.csv: {len(domains)} rows")
        except Exception:
            pass

    if len(frames) == 0:
        print("No data files found. Generating synthetic data...")
        return _generate_synthetic_data()

    df = pd.concat(frames, ignore_index=True)
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

    X = X_feat.values
    feature_names = list(X_feat.columns)
    print(f"Total features: {len(feature_names)}")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print("Training XGBoost...")
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_train, y_train)
    train_score = xgb.score(X_train, y_train)
    print(f"XGBoost train acc: {train_score:.4f}")

    print("Finding optimal threshold via precision-recall...")
    y_proba = xgb.predict_proba(X_test)[:, 1]
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
    joblib.dump(xgb, model_path)
    print(f"Model saved to: {model_path}")

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
        "base_feature_count": int(X_feat.shape[1]),
        "tfidf_feature_count": 0,
    }
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    print("Done.")


if __name__ == "__main__":
    main()
