# src/debug_dm_hub_columns.py
"""
Debug script to check column names in patient_issues.parquet
"""

import pandas as pd
from pathlib import Path

def check_columns():
    print("\n" + "="*60)
    print("DEBUG: Checking patient_issues.parquet columns")
    print("="*60 + "\n")
    
    # Load patient issues
    path = Path("data/processed/analytics/patient_issues.parquet")
    
    if not path.exists():
        print(f"❌ File not found: {path}")
        return
    
    df = pd.read_parquet(path)
    
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns\n")
    
    print("ALL COLUMNS:")
    print("-" * 40)
    for i, col in enumerate(sorted(df.columns)):
        dtype = df[col].dtype
        if dtype == bool:
            true_count = df[col].sum()
            print(f"  {i+1:2}. {col:<40} {dtype} (True: {true_count:,})")
        elif dtype in ['int64', 'float64']:
            non_zero = (df[col] > 0).sum() if df[col].notna().any() else 0
            print(f"  {i+1:2}. {col:<40} {dtype} (>0: {non_zero:,})")
        else:
            print(f"  {i+1:2}. {col:<40} {dtype}")
    
    print("\n" + "-" * 40)
    print("COLUMNS WITH 'issue' IN NAME:")
    print("-" * 40)
    issue_cols = [c for c in df.columns if 'issue' in c.lower()]
    for col in issue_cols:
        dtype = df[col].dtype
        if dtype == bool:
            true_count = df[col].sum()
            print(f"  {col:<40} {dtype} (True: {true_count:,})")
        elif dtype in ['int64', 'float64']:
            non_zero = (df[col] > 0).sum()
            print(f"  {col:<40} {dtype} (>0: {non_zero:,})")
        else:
            print(f"  {col:<40} {dtype}")
    
    print("\n" + "-" * 40)
    print("COLUMNS WITH 'has_' PREFIX:")
    print("-" * 40)
    has_cols = [c for c in df.columns if c.startswith('has_')]
    for col in has_cols:
        dtype = df[col].dtype
        if dtype == bool:
            true_count = df[col].sum()
            print(f"  {col:<40} {dtype} (True: {true_count:,})")
        elif dtype in ['int64', 'float64']:
            non_zero = (df[col] > 0).sum()
            print(f"  {col:<40} {dtype} (>0: {non_zero:,})")
    
    print("\n" + "-" * 40)
    print("COLUMNS WITH 'count' IN NAME:")
    print("-" * 40)
    count_cols = [c for c in df.columns if 'count' in c.lower()]
    for col in count_cols:
        dtype = df[col].dtype
        total = df[col].sum() if dtype in ['int64', 'float64'] else 'N/A'
        print(f"  {col:<40} {dtype} (Total: {total:,})")
    
    # Also check UPR
    print("\n" + "="*60)
    print("DEBUG: Checking unified_patient_record.parquet columns")
    print("="*60 + "\n")
    
    upr_path = Path("data/processed/upr/unified_patient_record.parquet")
    if upr_path.exists():
        upr = pd.read_parquet(upr_path)
        print(f"✅ Loaded {len(upr):,} rows, {len(upr.columns)} columns\n")
        
        # Look for issue-related columns
        print("POTENTIAL ISSUE COLUMNS IN UPR:")
        print("-" * 40)
        keywords = ['query', 'sign', 'sdv', 'sae', 'missing', 'broken', 'pending']
        for col in sorted(upr.columns):
            if any(kw in col.lower() for kw in keywords):
                dtype = upr[col].dtype
                if dtype in ['int64', 'float64']:
                    non_zero = (upr[col] > 0).sum()
                    total = upr[col].sum()
                    print(f"  {col:<45} (>0: {non_zero:,}, sum: {total:,.0f})")

if __name__ == "__main__":
    check_columns()