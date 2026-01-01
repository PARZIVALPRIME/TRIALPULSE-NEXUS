# src/check_columns.py
"""Quick script to check available columns in UPR"""

import pandas as pd
from pathlib import Path

# Load UPR
data_dir = Path('data/processed')
upr_path = data_dir / 'upr' / 'unified_patient_record.parquet'
if not upr_path.exists():
    upr_path = data_dir / 'segments' / 'unified_patient_record_segmented.parquet'

df = pd.read_parquet(upr_path)
print(f"Total columns: {len(df.columns)}")
print(f"Total rows: {len(df)}")

# Print all columns grouped by pattern
print("\n" + "="*60)
print("COLUMNS BY CATEGORY")
print("="*60)

categories = {
    'SAE DM': 'sae_dm',
    'SAE Safety': 'sae_safety', 
    'Queries': 'quer',
    'CRFs/SDV': 'crf',
    'Signatures': 'sign',
    'MedDRA': 'meddra',
    'WHODrug': 'whodrug',
    'Visits': 'visit',
    'Pages': 'page',
    'Lab': 'lab',
    'EDRR': 'edrr',
    'Inactivated': 'inactiv'
}

for cat_name, pattern in categories.items():
    cols = [c for c in df.columns if pattern.lower() in c.lower()]
    print(f"\n{cat_name} ({len(cols)} columns):")
    for c in cols[:10]:  # Show first 10
        sample = df[c].dropna().head(3).tolist()
        print(f"  {c}: {sample}")
    if len(cols) > 10:
        print(f"  ... and {len(cols)-10} more")

# Save full column list
col_df = pd.DataFrame({'column': df.columns})
col_df.to_csv('data/processed/upr_columns.csv', index=False)
print(f"\nFull column list saved to: data/processed/upr_columns.csv")