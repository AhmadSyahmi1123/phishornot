import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('backend/app/models/train/datasets/dataset_with_features.csv')
X = df.drop(['Type'], axis=1)
y = df['Type']

# Convert target to numeric
y = pd.to_numeric(y, errors='coerce').astype(int)

# Prepare DMatrix for xgb.cv
dtrain = xgb.DMatrix(X, label=y)

# Parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.1,
    'seed': 42,
    'verbosity': 1
}

# Try different max_depth values
max_depths = [9]

for depth in max_depths:
    params['max_depth'] = depth
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=1000,
        nfold=5,               # 5-fold cross-validation
        early_stopping_rounds=20,
        metrics="logloss",
        as_pandas=True,
        seed=42
    )
    best_iteration = cv_results['test-logloss-mean'].idxmin()
    best_logloss = cv_results['test-logloss-mean'].min()
    print(f"max_depth={depth} | Best iteration: {best_iteration} | CV logloss: {best_logloss:.5f}")
