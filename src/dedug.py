"""
Debug script to identify DB Lock column names
"""

import pandas as pd
from pathlib import Path

def debug_dblock_columns():
    """Find the actual DB Lock column names in the data"""
    
    print("\n" + "="*70)
    print("DEBUG: DB Lock Column Detection")
    print("="*70 + "\n")
    
    base_path = Path("data/processed")
    
    # Check all relevant files
    files_to_check = [
        ("UPR", base_path / "upr" / "unified_patient_record.parquet"),
        ("DB Lock Status", base_path / "analytics" / "patient_dblock_status.parquet"),
        ("Clean Status", base_path / "analytics" / "patient_clean_status.parquet"),
        ("Patient Issues", base_path / "analytics" / "patient_issues.parquet"),
    ]
    
    for name, path in files_to_check:
        print(f"\n{'='*50}")
        print(f"FILE: {name}")
        print(f"Path: {path}")
        print(f"{'='*50}")
        
        if path.exists():
            df = pd.read_parquet(path)
            print(f"Shape: {df.shape}")
            print(f"\nAll Columns ({len(df.columns)}):")
            
            # Group columns by likely category
            dblock_cols = [c for c in df.columns if 'lock' in c.lower() or 'dblock' in c.lower() or 'ready' in c.lower()]
            clean_cols = [c for c in df.columns if 'clean' in c.lower() or 'tier' in c.lower()]
            status_cols = [c for c in df.columns if 'status' in c.lower()]
            
            if dblock_cols:
                print(f"\n🔒 DB Lock related columns: {dblock_cols}")
                for col in dblock_cols:
                    print(f"   - {col}: dtype={df[col].dtype}, unique={df[col].nunique()}")
                    if df[col].dtype == object:
                        print(f"     Values: {df[col].value_counts().head(5).to_dict()}")
                    elif df[col].dtype == bool:
                        print(f"     True: {df[col].sum()}, False: {(~df[col]).sum()}")
                    else:
                        print(f"     Sum: {df[col].sum()}, Mean: {df[col].mean():.2f}")
            
            if clean_cols:
                print(f"\n✅ Clean related columns: {clean_cols}")
                for col in clean_cols[:5]:  # Limit output
                    print(f"   - {col}: dtype={df[col].dtype}")
            
            if status_cols:
                print(f"\n📊 Status columns: {status_cols}")
                for col in status_cols[:5]:
                    print(f"   - {col}: dtype={df[col].dtype}")
                    if df[col].dtype == object:
                        print(f"     Values: {df[col].value_counts().head(5).to_dict()}")
            
            # Show first few column names
            print(f"\nFirst 20 columns: {list(df.columns[:20])}")
            
            # Check for Site_1 specifically
            if 'site_id' in df.columns:
                site1 = df[df['site_id'] == 'Site_1']
                print(f"\nSite_1 data: {len(site1)} rows")
                
                if dblock_cols and len(site1) > 0:
                    for col in dblock_cols:
                        if df[col].dtype == bool:
                            ready_count = site1[col].sum()
                        elif df[col].dtype == object:
                            ready_count = site1[col].isin(['Ready', 'ready', 'READY', 'Tier 1 Ready']).sum()
                        else:
                            ready_count = (site1[col] == 1).sum()
                        print(f"   Site_1 {col} ready: {ready_count}")
        else:
            print(f"❌ File not found!")
    
    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    debug_dblock_columns()