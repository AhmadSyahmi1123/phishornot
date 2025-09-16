import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("backend/app/models/train/datasets/phishing_legit_dataset_with_features.csv")

# Features & Labels
X = df.drop(columns=["url", "label"])
y = df["label"]

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for LR, SVM, MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(kernel="linear", probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"
    ),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300, random_state=42)
}

# Train & Evaluate
for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    if name in ["Logistic Regression", "SVM", "MLP (Neural Net)"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:  # Tree-based models don't need scaling
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    print(f"{name} Results:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Legit", "Phish"]))