import pandas as pd
import joblib
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lime
import lime.lime_tabular

df = pd.read_csv('backend/app/models/train/datasets/dataset_with_features.csv')

X=df.drop(['Type'], axis=1)
y=df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert y_train and y_test to numeric
y_train_numeric = y_train.astype(int)
y_test_numeric = y_test.astype(int)

pos = sum(y_train_numeric)
neg = len(y_train_numeric) - pos
scale = neg / pos

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    eta=0.1,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=42,
    scale_pos_weight=scale,
    n_estimators=863  # match your boosting rounds
)

model.fit(X_train_scaled, y_train_numeric)

# Predict on test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns.tolist(),
    class_names=['Legitimate', 'Phishing'],
    mode='classification'
)

i = 0
exp = explainer.explain_instance(X_test_scaled[i], model.predict_proba, num_features=10)
exp.save_to_file('lime_explanation.html')

joblib.dump(model, 'backend/app/models/model.pkl')
