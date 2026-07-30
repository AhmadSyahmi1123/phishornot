import pandas as pd

imp = pd.read_csv("backend/app/models/train/output_xgb/xgb_feature_importance.csv")
print("Top 20 features by importance:")
for _, row in imp.head(20).iterrows():
    print(f'  {row["feature"]}: {row["importance"]:.4f}')

print("\nHomograph features:")
for feature in ["has_unicode", "has_mixed_script", "has_confusable", "entropy_of_subdomain"]:
    idx = imp[imp["feature"] == feature].index.tolist()
    if idx:
        row = imp.loc[idx[0]]
        print(f'  {feature}: rank {idx[0]+1}, importance {row["importance"]:.4f}')
    else:
        print(f'  {feature}: NOT FOUND')

print(f"\nTotal features: {len(imp)}")
print(f"Zero importance: {(imp['importance'] == 0).sum()}")