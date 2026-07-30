import pandas as pd

existing = pd.read_csv("backend/app/models/train/datasets/phishing_legit_dataset.csv")
hard = pd.read_csv("backend/app/models/train/datasets/hard_negatives.csv")

existing["url"] = existing["url"].astype(str).str.strip('"\'').str.strip()
hard["url"] = hard["url"].astype(str).str.strip('"\'').str.strip()

combined = pd.concat([existing, hard], ignore_index=True)
combined = combined.drop_duplicates(subset=["url"])
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
combined.to_csv("backend/app/models/train/datasets/phishing_legit_dataset.csv", index=False)

print(f"Existing: {len(existing)}, Hard negatives: {len(hard)}, Merged (deduped): {len(combined)}")
print(f"Distribution: {combined['label'].value_counts().to_dict()}")