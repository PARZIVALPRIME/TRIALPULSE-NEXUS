"""
Diagnose dashboard data files to check column names.
"""

import pandas as pd
from pathlib import Path

def main():
    data_dir = Path("D:/trialpulse_nexus/data/processed")
    
    print("=" * 70)
    print("TRIALPULSE NEXUS 10X - Dashboard Data Diagnostic")
    print("=" * 70)
    
    # Check each analytics file
    files_to_check = [
        ("upr/unified_patient_record.parquet", "UPR"),
        ("analytics/patient_dqi_enhanced.parquet", "DQI"),
        ("analytics/patient_clean_status.parquet", "Clean Status"),
        ("analytics/patient_dblock_status.parquet", "DB Lock Status"),
        ("analytics/patient_issues.parquet", "Issues"),
    ]
    
    for file_path, name in files_to_check:
        full_path = data_dir / file_path
        print(f"\n{'=' * 70}")
        print(f"FILE: {name}")
        print(f"Path: {full_path}")
        print("=" * 70)
        
        if not full_path.exists():
            print("❌ FILE NOT FOUND")
            continue
        
        try:
            df = pd.read_parquet(full_path)
            print(f"✅ Loaded: {len(df):,} rows x {len(df.columns)} columns")
            print(f"\nColumns ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                dtype = df[col].dtype
                non_null = df[col].notna().sum()
                print(f"  {i+1:2}. {col:40} | {str(dtype):15} | {non_null:,} non-null")
            
            # Special checks for DB Lock
            if "dblock" in name.lower():
                print(f"\n--- DB Lock Specific Checks ---")
                
                # Check for ready columns
                ready_cols = [c for c in df.columns if 'ready' in c.lower()]
                print(f"Columns with 'ready': {ready_cols}")
                
                # Check for eligible columns
                eligible_cols = [c for c in df.columns if 'eligible' in c.lower()]
                print(f"Columns with 'eligible': {eligible_cols}")
                
                # Check for status columns
                status_cols = [c for c in df.columns if 'status' in c.lower()]
                print(f"Columns with 'status': {status_cols}")
                
                # Show sample values
                if ready_cols:
                    col = ready_cols[0]
                    print(f"\nSample of '{col}':")
                    print(df[col].value_counts().head(10))
                
                if eligible_cols:
                    col = eligible_cols[0]
                    print(f"\nSample of '{col}':")
                    print(df[col].value_counts().head(10))
            
            # Special checks for Clean Status
            if "clean" in name.lower():
                print(f"\n--- Clean Status Specific Checks ---")
                tier_cols = [c for c in df.columns if 'tier' in c.lower() or 'clean' in c.lower()]
                print(f"Tier/Clean columns: {tier_cols}")
                
                for col in tier_cols[:5]:
                    if df[col].dtype == 'bool' or df[col].dtype == 'object':
                        print(f"\n'{col}' value counts:")
                        print(df[col].value_counts().head())
                    else:
                        print(f"\n'{col}' stats: mean={df[col].mean():.3f}, sum={df[col].sum():,}")
            
            # Special checks for Issues
            if "issues" in name.lower():
                print(f"\n--- Issues Specific Checks ---")
                issue_cols = [c for c in df.columns if any(x in c.lower() for x in ['issue', 'sdv', 'query', 'signature', 'sae', 'missing'])]
                print(f"Issue-related columns: {issue_cols[:15]}...")
                
                if 'total_issues' in df.columns:
                    print(f"\ntotal_issues stats:")
                    print(f"  Sum: {df['total_issues'].sum():,}")
                    print(f"  Mean: {df['total_issues'].mean():.2f}")
                    print(f"  Patients with issues: {(df['total_issues'] > 0).sum():,}")
                
                if 'priority' in df.columns:
                    print(f"\npriority value counts:")
                    print(df['priority'].value_counts())
                    
        except Exception as e:
            print(f"❌ Error loading: {e}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()