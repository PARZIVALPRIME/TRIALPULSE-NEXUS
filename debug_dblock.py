"""
Debug DB Lock data to find why ready count is 0
"""
import pandas as pd
from pathlib import Path

data_dir = Path("D:/trialpulse_nexus/data/processed")
dblock_path = data_dir / "analytics" / "patient_dblock_status.parquet"

print("=" * 70)
print("DB LOCK DIAGNOSTIC")
print("=" * 70)

df = pd.read_parquet(dblock_path)
print(f"Total rows: {len(df):,}")

# Find all dblock-related columns
dblock_cols = [c for c in df.columns if 'dblock' in c.lower()]
print(f"\nDB Lock columns ({len(dblock_cols)}):")
for col in dblock_cols:
    print(f"  - {col}")

# Check for ready/eligible columns
print("\n--- Checking key columns ---")

ready_candidates = ['dblock_ready', 'dblock_is_ready', 'is_ready', 'ready']
eligible_candidates = ['dblock_eligible', 'dblock_is_eligible', 'is_eligible', 'eligible']

for col in ready_candidates:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts())

for col in eligible_candidates:
    if col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts())

# Check dblock_status
if 'dblock_status' in df.columns:
    print("\ndblock_status:")
    print(df['dblock_status'].value_counts())

# Check tier columns
tier_cols = [c for c in df.columns if 'tier' in c.lower() and 'dblock' in c.lower()]
print(f"\nTier columns: {tier_cols}")
for col in tier_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

# Check if tier ready exists
if 'dblock_tier' in df.columns:
    print("\ndblock_tier value counts:")
    print(df['dblock_tier'].value_counts())

# Check pass columns
pass_cols = [c for c in df.columns if 'pass' in c.lower() and 'dblock' in c.lower()]
print(f"\nPass columns: {pass_cols}")

# What makes a patient ready?
print("\n--- Readiness Logic Check ---")
# If we have tier column
if 'dblock_tier' in df.columns:
    ready_tiers = df['dblock_tier'] == 'Ready'
    print(f"Patients with dblock_tier='Ready': {ready_tiers.sum():,}")

# Check blocker counts
if 'dblock_blocker_count' in df.columns:
    print(f"\nBlocker count distribution:")
    print(df['dblock_blocker_count'].value_counts().head(10))
    zero_blockers = (df['dblock_blocker_count'] == 0).sum()
    print(f"Patients with 0 blockers: {zero_blockers:,}")

# Check critical blockers
if 'dblock_critical_count' in df.columns:
    print(f"\nCritical blocker distribution:")
    print(df['dblock_critical_count'].value_counts().head(10))
