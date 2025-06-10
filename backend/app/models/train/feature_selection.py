from sklearn.feature_selection import SelectKBest, f_classif, chi2, f_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('backend/app/models/train/datasets/Dataset.csv')

X=df.drop(['Type'], axis=1)
y=df['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Compare performance
# Without feature selection
rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X_train, y_train)
pred1 = rf1.predict(X_test)

# With feature selection
rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X_train_selected, y_train)
pred2 = rf2.predict(X_test_selected)

print(f"Accuracy without selection: {accuracy_score(y_test, pred1):.3f}")
print(f"Accuracy with selection: {accuracy_score(y_test, pred2):.3f}")

# Get selected feature indices
selected_indices = selector.get_support(indices=True)
print("Selected feature indices:", selected_indices)

# Get feature scores
feature_scores = selector.scores_
print("Feature scores:", feature_scores)

# If you have feature names
feature_names = X.columns
selected_features = [feature_names[i] for i in selected_indices]
print("Selected features:", selected_features)