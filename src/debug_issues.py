# src/debug_issues.py
"""Debug script to check actual column values"""

import pandas as pd
from pathlib import Path

# Load data
data_dir = Path('data/processed')
upr_path = data_dir / 'segments' / 'unified_patient_record_segmented.parquet'
df = pd.read_parquet(upr_path)

print("="*70)
print("DEBUGGING ISSUE DETECTOR DATA")
print("="*70)

# 1. Check SAE columns
print("\n1. SAE COLUMNS:")
sae_cols = [c for c in df.columns if 'sae' in c.lower()]
for col in sae_cols:
    non_zero = (df[col].fillna(0) > 0).sum()
    print(f"  {col}: non-zero={non_zero}, max={df[col].max()}, sample={df[col].dropna().head(3).tolist()}")

# 2. Check Query columns
print("\n2. QUERY COLUMNS:")
query_cols = [c for c in df.columns if 'quer' in c.lower()]
for col in query_cols:
    non_zero = (df[col].fillna(0) > 0).sum()
    print(f"  {col}: non-zero={non_zero}, max={df[col].max()}")

# 3. Check SDV columns
print("\n3. SDV COLUMNS:")
sdv_cols = [c for c in df.columns if 'sdv' in c.lower() or 'verif' in c.lower()]
for col in sdv_cols:
    non_zero = (df[col].fillna(0) > 0).sum()
    print(f"  {col}: non-zero={non_zero}, max={df[col].max()}")

# 4. Check Signature columns
print("\n4. SIGNATURE COLUMNS:")
sign_cols = [c for c in df.columns if 'sign' in c.lower()]
for col in sign_cols:
    non_zero = (df[col].fillna(0) > 0).sum()
    print(f"  {col}: non-zero={non_zero}, max={df[col].max()}")

# 5. Check overdue columns
print("\n5. OVERDUE COLUMNS:")
overdue_cols = [c for c in df.columns if 'overdue' in c.lower() or 'greater' in c.lower() or 'days' in c.lower()]
for col in overdue_cols:
    if df[col].dtype in ['float64', 'int64']:
        non_zero = (df[col].fillna(0) > 0).sum()
        print(f"  {col}: non-zero={non_zero}, max={df[col].max()}")

# 6. Look for any query aging columns
print("\n6. ALL NUMERIC COLUMNS WITH 'day' or 'age':")
for col in df.columns:
    if ('day' in col.lower() or 'age' in col.lower()) and df[col].dtype in ['float64', 'int64']:
        non_zero = (df[col].fillna(0) > 0).sum()
        print(f"  {col}: non-zero={non_zero}")

# 7. Check if there's a pending calculation issue
print("\n7. SAE PENDING CALCULATION CHECK:")
if 'sae_dm_sae_dm_total' in df.columns and 'sae_dm_sae_dm_completed' in df.columns:
    total = df['sae_dm_sae_dm_total'].fillna(0)
    completed = df['sae_dm_sae_dm_completed'].fillna(0)
    pending_calc = total - completed
    print(f"  SAE DM: Total>0: {(total>0).sum()}, Completed>0: {(completed>0).sum()}")
    print(f"  SAE DM: Pending (total-completed)>0: {(pending_calc>0).sum()}")
    
if 'sae_safety_sae_safety_total' in df.columns and 'sae_safety_sae_safety_completed' in df.columns:
    total = df['sae_safety_sae_safety_total'].fillna(0)
    completed = df['sae_safety_sae_safety_completed'].fillna(0)
    pending_calc = total - completed
    print(f"  SAE Safety: Total>0: {(total>0).sum()}, Completed>0: {(completed>0).sum()}")
    print(f"  SAE Safety: Pending (total-completed)>0: {(pending_calc>0).sum()}")

# 8. Check the pending columns directly
print("\n8. DIRECT PENDING COLUMN VALUES:")
if 'sae_dm_sae_dm_pending' in df.columns:
    pending = df['sae_dm_sae_dm_pending'].fillna(0)
    print(f"  sae_dm_sae_dm_pending: >0 count={( pending>0).sum()}, unique={pending.unique()[:10]}")
    
if 'sae_safety_sae_safety_pending' in df.columns:
    pending = df['sae_safety_sae_safety_pending'].fillna(0)
    print(f"  sae_safety_sae_safety_pending: >0 count={(pending>0).sum()}, unique={pending.unique()[:10]}")

print("\n" + "="*70)