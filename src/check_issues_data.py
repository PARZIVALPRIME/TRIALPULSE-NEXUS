"""Check patient_issues.parquet columns"""
import pandas as pd

df = pd.read_parquet("data/processed/analytics/patient_issues.parquet")

print("PATIENT ISSUES COLUMNS:")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for col in sorted(df.columns):
    print(f"  - {col}")

print("\n\nSAMPLE (first 3 rows, key columns):")
key_cols = [c for c in df.columns if 'issue' in c.lower() or 'has_' in c.lower() or c in ['patient_key', 'study_id', 'site_id']]
if key_cols:
    print(df[key_cols[:10]].head(3))