"""Debug script to check patient_issues.parquet columns."""
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_PROCESSED

issues_path = DATA_PROCESSED / 'analytics' / 'patient_issues.parquet'
df = pd.read_parquet(issues_path)

print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print("\nAll columns:")
for col in sorted(df.columns):
    print(f"  {col}")

print("\n\nColumns starting with 'has_':")
has_cols = [c for c in df.columns if c.startswith('has_')]
for col in has_cols:
    count = df[col].sum() if df[col].dtype == bool else (df[col] > 0).sum()
    print(f"  {col}: {count:,}")

print("\n\nColumns with 'issue' in name:")
issue_cols = [c for c in df.columns if 'issue' in c.lower()]
for col in issue_cols:
    print(f"  {col}")