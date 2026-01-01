# src/verify_sdv.py
"""Quick verification of SDV logic"""

import pandas as pd
from pathlib import Path

df = pd.read_parquet('data/processed/segments/unified_patient_record_segmented.parquet')

print("SDV COLUMNS ANALYSIS:")
print("="*60)

# Check SDV column
sdv_require = df['crfs_require_verification_sdv'].fillna(0)
print(f"\ncrfs_require_verification_sdv:")
print(f"  Non-zero: {(sdv_require > 0).sum():,} ({(sdv_require > 0).mean()*100:.1f}%)")
print(f"  Mean: {sdv_require.mean():.1f}")
print(f"  Max: {sdv_require.max():.0f}")
print(f"  Distribution: {sdv_require.describe()}")

# Check if there's a "completed" column
print(f"\n  Sample values:")
print(df[['crfs_require_verification_sdv', 'crfs_signed', 'crfs_frozen', 'crfs_locked']].head(20))

# Check relationship with other columns
print(f"\nPatients with SDV require > 0 AND signed CRFs:")
has_sdv_req = sdv_require > 0
has_signed = df['crfs_signed'].fillna(0) > 0
print(f"  SDV require > 0: {has_sdv_req.sum():,}")
print(f"  Signed > 0: {has_signed.sum():,}")
print(f"  Both: {(has_sdv_req & has_signed).sum():,}")

# The question: should SDV incomplete be "require > 0" or something else?
# If nearly everyone has require > 0, it's probably not the right interpretation