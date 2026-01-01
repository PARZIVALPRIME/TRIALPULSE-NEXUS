"""
TRIALPULSE NEXUS 10X - Column Diagnostic Script
Check what columns exist in UPR and fix metrics engine mapping
"""

import pandas as pd
from pathlib import Path

# Load the segmented UPR
upr_path = Path(r"D:\trialpulse_nexus\data\processed\segments\unified_patient_record_segmented.parquet")
df = pd.read_parquet(upr_path)

print("=" * 70)
print("UPR COLUMN DIAGNOSTIC")
print("=" * 70)
print(f"\nTotal Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")

print("\n" + "=" * 70)
print("ALL COLUMNS IN UPR:")
print("=" * 70)
for i, col in enumerate(sorted(df.columns)):
    # Get non-null count and sample values
    non_null = df[col].notna().sum()
    dtype = df[col].dtype
    
    # Get sample non-null value
    sample = ""
    if non_null > 0:
        sample_val = df[col].dropna().iloc[0] if non_null > 0 else ""
        sample = f" | Sample: {sample_val}"[:50]
    
    print(f"{i+1:3}. {col:50} | {dtype:15} | Non-null: {non_null:>6}{sample}")

print("\n" + "=" * 70)
print("COLUMNS BY CATEGORY:")
print("=" * 70)

# Categorize columns
categories = {
    'Identifiers': ['patient_key', 'study_id', 'site_id', 'subject_id'],
    'Status': [c for c in df.columns if 'status' in c.lower()],
    'Visit': [c for c in df.columns if 'visit' in c.lower()],
    'Query': [c for c in df.columns if 'query' in c.lower() or 'queries' in c.lower()],
    'SDV': [c for c in df.columns if 'sdv' in c.lower()],
    'Signature': [c for c in df.columns if 'sign' in c.lower() or 'esign' in c.lower()],
    'SAE': [c for c in df.columns if 'sae' in c.lower()],
    'Coding': [c for c in df.columns if 'cod' in c.lower() or 'meddra' in c.lower() or 'whodrug' in c.lower()],
    'Lab': [c for c in df.columns if 'lab' in c.lower()],
    'Pages': [c for c in df.columns if 'page' in c.lower()],
    'CRF': [c for c in df.columns if 'crf' in c.lower()],
    'EDRR': [c for c in df.columns if 'edrr' in c.lower()],
    'Inactivated': [c for c in df.columns if 'inactivat' in c.lower()],
    'Issues': [c for c in df.columns if 'issue' in c.lower()],
    'Missing': [c for c in df.columns if 'missing' in c.lower()],
    'DQI/Metrics': [c for c in df.columns if 'dqi' in c.lower() or 'score' in c.lower() or 'rate' in c.lower()],
    'Risk/Tier': [c for c in df.columns if 'risk' in c.lower() or 'tier' in c.lower() or 'cohort' in c.lower()],
}

for category, cols in categories.items():
    if cols:
        print(f"\n{category}:")
        for col in cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"  - {col} (non-null: {non_null:,})")

print("\n" + "=" * 70)
print("NUMERIC COLUMNS WITH VALUES > 0:")
print("=" * 70)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    if df[col].sum() > 0:
        total = df[col].sum()
        mean = df[col].mean()
        max_val = df[col].max()
        print(f"  {col}: sum={total:,.0f}, mean={mean:.2f}, max={max_val:.0f}")