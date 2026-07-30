import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import pandas as pd

from backend.app.extract_feature import extract_features

df = pd.read_csv(
    "backend/app/models/train/datasets/phishing_legit_dataset.csv",
    dtype={"label": int},
)
print(df.head())

df = df[df["url"].apply(lambda x: isinstance(x, str))].copy()

features_df = df["url"].apply(extract_features).apply(pd.Series)

df_combined = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

output_path = "backend/app/models/train/datasets/phishing_legit_dataset_with_features.csv"
df_combined.to_csv(output_path, index=False)

print(f"Saved {output_path} with {features_df.shape[1]} features.")