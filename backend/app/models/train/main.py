import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('backend/app/models/train/datasets/Dataset.csv')

X=df.drop(['Type'], axis=1)
y=df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')

# Standardize the data to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(y)

# Convert y_train and y_test to numeric
y_train_numeric = y_train.astype(int)
y_test_numeric = y_test.astype(int)

# Create DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'binary:logistic',  # binary classification
    'eval_metric': 'logloss',        # evaluation metric
    'eta': 0.1,                     # learning rate
    'max_depth': 10,
    'seed': 42,
    'verbosity': 1
}

epoch = 1000
bst = xgb.train(params, dtrain, num_boost_round=epoch)
# Predict on test set
y_pred_prob = bst.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(bst, 'backend/app/models/model.pkl')
