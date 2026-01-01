"""
TRIALPULSE NEXUS 10X - Trial State Model Debug Script

Use this to diagnose issues with state loading and transitions.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd


def debug_data_files():
    """Check availability and structure of data files"""
    print("=" * 70)
    print("TRIAL STATE MODEL - DATA FILE DIAGNOSTICS")
    print("=" * 70)
    
    data_dir = Path("data/processed")
    
    # Check directories
    print("\n📁 Directory Structure:")
    dirs_to_check = [
        data_dir,
        data_dir / "upr",
        data_dir / "analytics",
        data_dir / "cleaned",
        data_dir / "segments",
    ]
    
    for d in dirs_to_check:
        exists = "✅" if d.exists() else "❌"
        print(f"   {exists} {d}")
    
    # Check key files
    print("\n📄 Key Data Files:")
    files_to_check = [
        ("UPR", data_dir / "upr" / "unified_patient_record.parquet"),
        ("Patient DQI", data_dir / "analytics" / "patient_dqi_enhanced.parquet"),
        ("Clean Status", data_dir / "analytics" / "patient_clean_status.parquet"),
        ("DB Lock Status", data_dir / "analytics" / "patient_dblock_status.parquet"),
        ("Patient Issues", data_dir / "analytics" / "patient_issues.parquet"),
        ("Site Benchmarks", data_dir / "analytics" / "site_benchmarks.parquet"),
        ("Cascade Analysis", data_dir / "analytics" / "patient_cascade_analysis.parquet"),
    ]
    
    for name, path in files_to_check:
        if path.exists():
            df = pd.read_parquet(path)
            print(f"   ✅ {name}: {len(df):,} rows, {len(df.columns)} cols")
        else:
            print(f"   ❌ {name}: NOT FOUND at {path}")
    
    # Examine UPR columns
    print("\n📊 UPR Column Analysis:")
    upr_path = data_dir / "upr" / "unified_patient_record.parquet"
    if upr_path.exists():
        df = pd.read_parquet(upr_path)
        
        # Key columns
        key_cols = [
            'patient_key', 'study_id', 'site_id', 'subject_id',
            'subject_status', 'subject_status_clean',
            'risk_level', 'total_open_queries', 'crfs_never_signed'
        ]
        
        print("   Key columns:")
        for col in key_cols:
            if col in df.columns:
                sample = df[col].dropna().head(3).tolist()
                print(f"      ✅ {col}: {sample}")
            else:
                print(f"      ❌ {col}: NOT FOUND")
        
        # Status distribution
        status_col = 'subject_status_clean' if 'subject_status_clean' in df.columns else 'subject_status'
        if status_col in df.columns:
            print(f"\n   Status Distribution ({status_col}):")
            for status, count in df[status_col].value_counts().head(10).items():
                print(f"      {status}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Examine Patient Issues columns
    print("\n📊 Patient Issues Column Analysis:")
    issues_path = data_dir / "analytics" / "patient_issues.parquet"
    if issues_path.exists():
        df = pd.read_parquet(issues_path)
        
        # Issue columns
        issue_cols = [c for c in df.columns if c.startswith('issue_')]
        count_cols = [c for c in df.columns if c.startswith('count_')]
        
        print(f"   Issue flag columns ({len(issue_cols)}):")
        for col in issue_cols[:5]:
            true_count = df[col].sum() if df[col].dtype == bool else (df[col] == True).sum()
            print(f"      {col}: {true_count:,} patients")
        
        print(f"\n   Count columns ({len(count_cols)}):")
        for col in count_cols[:5]:
            total = df[col].sum()
            print(f"      {col}: {total:,} total")
    
    # Examine Site Benchmarks
    print("\n📊 Site Benchmarks Analysis:")
    bench_path = data_dir / "analytics" / "site_benchmarks.parquet"
    if bench_path.exists():
        df = pd.read_parquet(bench_path)
        
        print(f"   Sites: {len(df):,}")
        print(f"   Columns: {list(df.columns)[:10]}...")
        
        # Key metrics
        for col in ['mean_dqi', 'dqi_score', 'composite_score', 'tier2_clean_rate']:
            if col in df.columns:
                print(f"   {col}: mean={df[col].mean():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]")


def debug_state_loading():
    """Debug state loading process"""
    print("\n" + "=" * 70)
    print("TRIAL STATE MODEL - STATE LOADING DEBUG")
    print("=" * 70)
    
    from src.simulation.trial_state_model import (
        TrialStateModel, reset_trial_state_model,
        PatientStatus, SiteStatus
    )
    
    reset_trial_state_model()
    model = TrialStateModel(Path("data/processed"))
    
    print("\n🔄 Loading state...")
    try:
        snapshot = model.load_from_data()
        
        print("\n✅ State loaded successfully!")
        print(f"   Snapshot ID: {snapshot.snapshot_id}")
        print(f"   Timestamp: {snapshot.timestamp}")
        print(f"   Checksum: {snapshot.checksum}")
        
        # Entity counts
        print("\n📦 Entities:")
        print(f"   Patients: {len(snapshot.patients):,}")
        print(f"   Sites: {len(snapshot.sites):,}")
        print(f"   Studies: {len(snapshot.studies):,}")
        print(f"   Issues: {len(snapshot.issues):,}")
        print(f"   Resources: {len(snapshot.resources):,}")
        
        # Sample patient
        if snapshot.patients:
            print("\n👤 Sample Patient:")
            patient = list(snapshot.patients.values())[0]
            print(f"   Key: {patient.patient_key}")
            print(f"   Study: {patient.study_id}")
            print(f"   Site: {patient.site_id}")
            print(f"   Status: {patient.status.value}")
            print(f"   DQI: {patient.dqi_score:.1f}")
            print(f"   Tier1 Clean: {patient.tier1_clean}")
            print(f"   Tier2 Clean: {patient.tier2_clean}")
            print(f"   DB Lock Ready: {patient.db_lock_ready}")
            print(f"   Total Issues: {patient.total_issues}")
            print(f"   Open Queries: {patient.open_queries}")
        
        # Patient status distribution
        print("\n📊 Patient Status Distribution:")
        status_counts = {}
        for patient in snapshot.patients.values():
            status = patient.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            print(f"   {status}: {count:,} ({count/len(snapshot.patients)*100:.1f}%)")
        
        # Sample site
        if snapshot.sites:
            print("\n🏥 Sample Site:")
            site = list(snapshot.sites.values())[0]
            print(f"   ID: {site.site_id}")
            print(f"   Study: {site.study_id}")
            print(f"   Status: {site.status.value}")
            print(f"   Patients: {site.patient_count}")
            print(f"   Mean DQI: {site.mean_dqi:.1f}")
            print(f"   Clean Rate: {site.tier2_clean_rate:.1%}")
        
        # Metrics
        print("\n📈 Aggregate Metrics:")
        for key, value in snapshot.metrics.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in list(value.items())[:5]:
                    print(f"      {k}: {v:,}")
            elif isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"\n❌ State loading failed: {e}")
        import traceback
        traceback.print_exc()


def debug_constraints():
    """Debug constraint checking"""
    print("\n" + "=" * 70)
    print("TRIAL STATE MODEL - CONSTRAINT DEBUG")
    print("=" * 70)
    
    from src.simulation.trial_state_model import (
        get_trial_state_model, ConstraintType
    )
    
    model = get_trial_state_model(Path("data/processed"))
    
    if not model.current_state:
        model.load_from_data()
    
    print("\n📋 Defined Constraints:")
    for cid, constraint in model.constraints.items():
        print(f"   {constraint.constraint_type.value.upper()}: {constraint.name}")
        print(f"      {constraint.description}")
    
    print("\n🔍 Checking Constraints...")
    violations = model.check_constraints()
    
    hard = [v for v in violations if v.constraint_type == ConstraintType.HARD]
    soft = [v for v in violations if v.constraint_type == ConstraintType.SOFT]
    advisory = [v for v in violations if v.constraint_type == ConstraintType.ADVISORY]
    
    print(f"\n📊 Violation Summary:")
    print(f"   🔴 HARD violations: {len(hard)}")
    print(f"   🟡 SOFT violations: {len(soft)}")
    print(f"   🔵 ADVISORY violations: {len(advisory)}")
    
    if hard:
        print(f"\n🔴 Sample HARD Violations:")
        for v in hard[:5]:
            print(f"   - {v.constraint_name}: {v.message[:60]}...")
    
    if soft:
        print(f"\n🟡 Sample SOFT Violations:")
        for v in soft[:5]:
            print(f"   - {v.constraint_name}: {v.message[:60]}...")


def debug_transitions():
    """Debug transition rules"""
    print("\n" + "=" * 70)
    print("TRIAL STATE MODEL - TRANSITION RULES DEBUG")
    print("=" * 70)
    
    from src.simulation.trial_state_model import get_trial_state_model
    
    model = get_trial_state_model(Path("data/processed"))
    
    print("\n📋 Defined Transition Rules:")
    for rule_id, rule in model.transition_rules.items():
        print(f"\n   📌 {rule.name}")
        print(f"      Type: {rule.transition_type.value}")
        print(f"      From: {rule.from_states}")
        print(f"      To: {rule.to_states}")
        print(f"      Side Effects: {rule.side_effects}")
        print(f"      Validators: {rule.validators}")
        print(f"      Requires Approval: {rule.requires_approval}")


if __name__ == "__main__":
    debug_data_files()
    debug_state_loading()
    debug_constraints()
    debug_transitions()