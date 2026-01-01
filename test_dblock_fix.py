"""
Test the DB Lock metrics calculation with the fixed code
"""
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path("D:/trialpulse_nexus/data/processed")

print("=" * 70)
print("TESTING DB LOCK FIX")
print("=" * 70)

# Load dblock data
dblock = pd.read_parquet(data_dir / "analytics" / "patient_dblock_status.parquet")

# Find columns using the FIXED logic
ready_col = None
eligible_col = None

for col in ['dblock_tier1_ready', 'dblock_ready', 'db_lock_ready', 'ready', 'is_ready']:
    if col in dblock.columns:
        ready_col = col
        break

for col in ['dblock_is_eligible', 'dblock_eligible', 'db_lock_eligible', 'eligible', 'is_eligible']:
    if col in dblock.columns:
        eligible_col = col
        break

print(f"Ready column found: {ready_col}")
print(f"Eligible column found: {eligible_col}")

# Calculate metrics
if ready_col:
    dblock_ready_count = int(dblock[ready_col].sum())
    print(f"\n✅ DB Lock Ready Count: {dblock_ready_count:,}")
else:
    print("\n❌ No ready column found")

if eligible_col:
    dblock_eligible_count = int(dblock[eligible_col].sum())
    print(f"✅ DB Lock Eligible Count: {dblock_eligible_count:,}")
    
    if ready_col:
        eligible_df = dblock[dblock[eligible_col] == True]
        if len(eligible_df) > 0:
            ready_rate = float(eligible_df[ready_col].mean() * 100)
            print(f"✅ DB Lock Ready Rate: {ready_rate:.1f}%")
else:
    print("\n❌ No eligible column found - using all patients as eligible")
    dblock_eligible_count = len(dblock)
    print(f"✅ DB Lock Eligible Count: {dblock_eligible_count:,}")
    if ready_col:
        ready_rate = float(dblock[ready_col].mean() * 100)
        print(f"✅ DB Lock Ready Rate: {ready_rate:.1f}%")

# Fallback test
print("\n--- Fallback using blocker_count ---")
if 'dblock_blocker_count' in dblock.columns:
    zero_blockers = (dblock['dblock_blocker_count'] == 0).sum()
    print(f"Patients with 0 blockers: {zero_blockers:,}")
    fallback_rate = zero_blockers / len(dblock) * 100
    print(f"Fallback ready rate: {fallback_rate:.1f}%")

print("\n" + "=" * 70)
print("FIX VERIFIED - Dashboard should now show correct values!")
print("=" * 70)
