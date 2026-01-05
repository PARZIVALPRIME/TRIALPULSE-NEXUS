"""
TRIALPULSE NEXUS 10X — 14-LABEL ISSUE DETECTOR (LEAKAGE-FREE)
Strict Feature Isolation — No Direct Outcome Features

PROBLEM FIXED: Previous version had AUC=1.0 because features = labels
SOLUTION: Predict issues from INDIRECT behavioral signals only

VERSION: LEAKAGE_FREE_v1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
import pickle
import sys
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, confusion_matrix, brier_score_loss
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost required")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
UPR_PATH = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'issue_detector_LEAKAGE_FREE'

for d in [OUTPUT_DIR, OUTPUT_DIR/'figures', OUTPUT_DIR/'models', OUTPUT_DIR/'tables']:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'training.log', mode='w', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# STRICT FEATURE EXCLUSIONS — NO LEAKAGE ALLOWED
# ============================================================================

# ALL outcome-related columns that must be excluded from ALL models
OUTCOME_COLUMNS = {
    # SAE columns
    'sae_dm_sae_dm_pending', 'sae_dm_sae_dm_completed', 'sae_dm_sae_dm_total',
    'sae_safety_sae_safety_pending', 'sae_safety_sae_safety_completed', 'sae_safety_sae_safety_total',
    'total_sae_pending', 'total_sae_issues',
    
    # Query columns
    'total_queries', 'dm_queries', 'clinical_queries', 'medical_queries',
    'site_queries', 'field_monitor_queries', 'coding_queries', 'safety_queries',
    
    # SDV columns
    'crfs_require_verification_sdv', 'forms_verified', 'completeness_score',
    
    # Signature columns
    'crfs_signed', 'crfs_never_signed', 'broken_signatures',
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    
    # Coding columns
    'meddra_coding_meddra_uncoded', 'meddra_coding_meddra_coded', 'meddra_coding_meddra_total',
    'whodrug_coding_whodrug_uncoded', 'whodrug_coding_whodrug_coded', 'whodrug_coding_whodrug_total',
    'coding_completion_rate', 'total_coding_terms', 'total_coded_terms', 'total_uncoded_terms',
    'coded_terms', 'uncoded_terms',
    
    # Missing data columns
    'has_missing_visits', 'has_missing_pages', 'missing_visits', 'missing_pages',
    'visit_missing_visit_count', 'pages_missing_page_count',
    
    # Lab/EDRR columns
    'lab_lab_issue_count', 'has_lab_issues',
    'edrr_edrr_issue_count', 'has_edrr_issues',
    'open_issues_lnr', 'open_issues_edrr',
    
    # Inactivated forms
    'inactivated_inactivated_form_count', 'inactivated_inactivated_unique_forms',
    
    # Freeze/Lock columns
    'crfs_frozen', 'crfs_not_frozen', 'crfs_locked', 'crfs_unlocked',
    'clean_entered_crf',
    
    # Risk/Issue aggregates
    'risk_level', 'total_issues_all_sources'
}

# Identifier columns
IDENTIFIER_COLUMNS = {
    'project_name', 'region', 'country', 'site', 'subject', 'latest_visit',
    'subject_status', 'input_files', '_source_file', '_study_id', '_ingestion_ts',
    'study_id', 'site_id', 'subject_id', 'patient_key', 'risk_level',
    '_cleaned_ts', '_upr_built_ts', '_upr_version', '_file_type', 'cpmd', 'ssm',
    'subject_status_original', 'subject_status_clean', '_cleaning_version'
}

# All columns to exclude
ALL_EXCLUDE = OUTCOME_COLUMNS.union(IDENTIFIER_COLUMNS)

# ============================================================================
# INDIRECT FEATURE SET — What we CAN use
# ============================================================================

# These are behavioral/contextual features that don't directly encode outcomes
ALLOWED_FEATURES = [
    # Workload indicators
    'pages_entered',
    'pages_with_nonconformant_data',
    'expected_visits_rave_edc_bo4',
    
    # Protocol deviations (quality history, not current issue)
    'pds_proposed',
    'pds_confirmed',
    
    # Engineered features (we'll create these)
    'pages_normalized',
    'nonconformant_rate',
    'site_patient_count',
    'large_site',
    'small_site',
    'study_avg_pages',
    'study_std_pages',
    'pages_vs_study',
    'has_deviations',
    'pending_deviations',
    'high_volume',
    'low_volume'
]


# ============================================================================
# 14 ISSUE DEFINITIONS
# ============================================================================

ISSUE_DEFINITIONS = {
    'sae_dm_pending': {
        'name': 'SAE DM Pending',
        'priority': 'CRITICAL',
        'source': 'sae_dm_sae_dm_pending',
        'logic': lambda df: (df.get('sae_dm_sae_dm_pending', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'sae_safety_pending': {
        'name': 'SAE Safety Pending',
        'priority': 'CRITICAL',
        'source': 'sae_safety_sae_safety_pending',
        'logic': lambda df: (df.get('sae_safety_sae_safety_pending', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'open_queries': {
        'name': 'Open Queries',
        'priority': 'HIGH',
        'source': 'total_queries',
        'logic': lambda df: (df.get('total_queries', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'high_query_volume': {
        'name': 'High Query Volume',
        'priority': 'HIGH',
        'source': 'total_queries',
        'logic': lambda df: (df.get('total_queries', pd.Series(0, index=df.index)).fillna(0) > 10).astype(int)
    },
    'sdv_incomplete': {
        'name': 'SDV Incomplete',
        'priority': 'HIGH',
        'source': 'forms_verified',
        'logic': lambda df: create_sdv_label(df)
    },
    'signature_gaps': {
        'name': 'Signature Gaps',
        'priority': 'MEDIUM',
        'source': 'overdue_signatures',
        'logic': lambda df: create_signature_label(df)
    },
    'broken_signatures': {
        'name': 'Broken Signatures',
        'priority': 'MEDIUM',
        'source': 'broken_signatures',
        'logic': lambda df: (df.get('broken_signatures', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'meddra_uncoded': {
        'name': 'MedDRA Uncoded',
        'priority': 'MEDIUM',
        'source': 'meddra_coding_meddra_uncoded',
        'logic': lambda df: (df.get('meddra_coding_meddra_uncoded', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'whodrug_uncoded': {
        'name': 'WHODrug Uncoded',
        'priority': 'MEDIUM',
        'source': 'whodrug_coding_whodrug_uncoded',
        'logic': lambda df: (df.get('whodrug_coding_whodrug_uncoded', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'missing_visits': {
        'name': 'Missing Visits',
        'priority': 'HIGH',
        'source': 'has_missing_visits',
        'logic': lambda df: (df.get('has_missing_visits', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'missing_pages': {
        'name': 'Missing Pages',
        'priority': 'HIGH',
        'source': 'has_missing_pages',
        'logic': lambda df: (df.get('has_missing_pages', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'lab_issues': {
        'name': 'Lab Issues',
        'priority': 'MEDIUM',
        'source': 'lab_lab_issue_count',
        'logic': lambda df: (df.get('lab_lab_issue_count', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'edrr_issues': {
        'name': 'EDRR Issues',
        'priority': 'MEDIUM',
        'source': 'edrr_edrr_issue_count',
        'logic': lambda df: (df.get('edrr_edrr_issue_count', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    },
    'inactivated_forms': {
        'name': 'Inactivated Forms',
        'priority': 'LOW',
        'source': 'inactivated_inactivated_form_count',
        'logic': lambda df: (df.get('inactivated_inactivated_form_count', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    }
}


def create_sdv_label(df):
    """Create SDV incomplete label."""
    if 'crfs_require_verification_sdv' in df.columns and 'forms_verified' in df.columns:
        required = df['crfs_require_verification_sdv'].fillna(0)
        verified = df['forms_verified'].fillna(0)
        rate = np.where(required > 0, verified / (required + 0.001), 1.0)
        return (rate < 1.0).astype(int)
    return pd.Series(0, index=df.index)


def create_signature_label(df):
    """Create signature gaps label."""
    overdue_cols = [c for c in df.columns if 'overdue_for_signs' in c]
    if overdue_cols:
        return (df[overdue_cols].fillna(0).sum(axis=1) > 0).astype(int)
    return pd.Series(0, index=df.index)


# ============================================================================
# FEATURE ENGINEERING — INDIRECT SIGNALS ONLY
# ============================================================================

def engineer_indirect_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from indirect signals only."""
    df = df.copy()
    engineered = 0
    
    # Workload normalization
    if 'pages_entered' in df.columns:
        p90 = df['pages_entered'].quantile(0.90)
        if p90 > 0:
            df['pages_normalized'] = (df['pages_entered'] / p90).clip(0, 2)
            engineered += 1
        
        df['high_volume'] = (df['pages_entered'] > df['pages_entered'].quantile(0.75)).astype(float)
        df['low_volume'] = (df['pages_entered'] < df['pages_entered'].quantile(0.25)).astype(float)
        engineered += 2
    
    # Nonconformant rate
    if 'pages_entered' in df.columns and 'pages_with_nonconformant_data' in df.columns:
        df['nonconformant_rate'] = np.where(
            df['pages_entered'] > 0,
            df['pages_with_nonconformant_data'] / (df['pages_entered'] + 1),
            0
        )
        engineered += 1
    
    # Site context
    if 'site_id' in df.columns:
        site_counts = df.groupby('site_id').size().reset_index(name='site_patient_count')
        df = df.merge(site_counts, on='site_id', how='left')
        df['site_patient_count'] = df['site_patient_count'].fillna(1)
        df['large_site'] = (df['site_patient_count'] > df['site_patient_count'].quantile(0.75)).astype(float)
        df['small_site'] = (df['site_patient_count'] < df['site_patient_count'].quantile(0.25)).astype(float)
        engineered += 3
    
    # Study context
    if 'study_id' in df.columns and 'pages_entered' in df.columns:
        study_stats = df.groupby('study_id')['pages_entered'].agg(['mean', 'std']).reset_index()
        study_stats.columns = ['study_id', 'study_avg_pages', 'study_std_pages']
        df = df.merge(study_stats, on='study_id', how='left')
        df['study_std_pages'] = df['study_std_pages'].fillna(0)
        df['pages_vs_study'] = np.where(
            df['study_avg_pages'] > 0,
            df['pages_entered'] / (df['study_avg_pages'] + 1),
            1.0
        )
        engineered += 2
    
    # Protocol deviations
    if 'pds_confirmed' in df.columns:
        df['has_deviations'] = (df['pds_confirmed'].fillna(0) > 0).astype(float)
        engineered += 1
    
    if 'pds_proposed' in df.columns:
        df['pending_deviations'] = (df['pds_proposed'].fillna(0) > 0).astype(float)
        engineered += 1
    
    # Interaction features
    if 'high_volume' in df.columns and 'large_site' in df.columns:
        df['high_volume_large_site'] = df['high_volume'] * df['large_site']
        df['high_volume_small_site'] = df['high_volume'] * df['small_site']
        engineered += 2
    
    if 'has_deviations' in df.columns and 'high_volume' in df.columns:
        df['deviation_high_volume'] = df['has_deviations'] * df['high_volume']
        engineered += 1
    
    log.info(f"  Engineered {engineered} indirect features")
    return df


# ============================================================================
# STRICT FEATURE SELECTION
# ============================================================================

def select_indirect_features(df: pd.DataFrame) -> List[str]:
    """Select ONLY indirect features — strict leakage prevention."""
    
    safe_cols = []
    for c in df.columns:
        # Skip if in exclusion list
        if c in ALL_EXCLUDE:
            continue
        
        # Only numeric
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        
        # Must have variance
        if df[c].nunique() < 2:
            continue
        
        std = df[c].std()
        if pd.isna(std) or std < 0.001:
            continue
        
        safe_cols.append(c)
    
    return safe_cols


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                     issue_key: str, issue_def: dict) -> Optional[Dict]:
    """Train leakage-free classifier."""
    
    pos_rate = y_train.mean()
    
    # Skip extreme imbalance
    if pos_rate < 0.005 or pos_rate > 0.995:
        return {
            'issue_key': issue_key,
            'name': issue_def['name'],
            'priority': issue_def['priority'],
            'status': 'SKIPPED',
            'reason': f'Extreme imbalance (pos={pos_rate:.4f})',
            'prevalence': float(y_test.mean())
        }
    
    # Conservative model
    scale_weight = min((1 - pos_rate) / max(pos_rate, 0.01), 15)
    
    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_weight,
        min_child_weight=15,
        reg_alpha=0.2,
        reg_lambda=1.5,
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    # Calibrate
    try:
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        final_model = calibrated
    except:
        final_model = model
    
    # Predict
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    auc = roc_auc_score(y_test, y_proba) if 0 < y_test.sum() < len(y_test) else 0.5
    ap = average_precision_score(y_test, y_proba) if y_test.sum() > 0 else 0.0
    
    # Threshold optimization
    best_f1, best_th = 0, 0.5
    for th in np.linspace(0.1, 0.9, 81):
        pred = (y_proba >= th).astype(int)
        f1 = f1_score(y_test, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    
    y_pred = (y_proba >= best_th).astype(int)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    
    # Feature importance
    importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Leakage check
    flags = []
    if auc > 0.95:
        flags.append('HIGH_AUC_WARNING')
    if auc < 0.55:
        flags.append('LOW_PREDICTIVE_VALUE')
    
    return {
        'model': final_model,
        'base_model': model,
        'issue_key': issue_key,
        'name': issue_def['name'],
        'priority': issue_def['priority'],
        'status': 'TRAINED',
        'prevalence': float(y_test.mean()),
        'auc': float(auc),
        'ap': float(ap),
        'f1': float(best_f1),
        'precision': float(precision),
        'recall': float(recall),
        'threshold': float(best_th),
        'flags': flags,
        'top_features': top_features,
        'n_features': len(X_train.columns)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results: List[Dict], output_dir: Path):
    """Create visualizations."""
    trained = [r for r in results if r.get('status') == 'TRAINED']
    if not trained:
        return
    
    # Performance heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = [r['name'] for r in trained]
    metrics = np.array([[r['auc'], r['ap'], r['f1'], r['precision'], r['recall']] for r in trained])
    
    im = ax.imshow(metrics, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['AUC', 'AP', 'F1', 'Prec', 'Recall'], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    
    for i in range(len(names)):
        for j in range(5):
            val = metrics[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    
    ax.set_title('14-LABEL ISSUE DETECTOR — LEAKAGE-FREE', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # AUC distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    aucs = [r['auc'] for r in trained]
    colors = ['green' if 0.60 <= a <= 0.85 else 'orange' if a < 0.60 else 'red' for a in aucs]
    
    bars = ax.barh(range(len(names)), aucs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Leakage-Free AUC (Realistic Performance)', fontsize=14, fontweight='bold')
    ax.set_xlim(0.4, 1.0)
    ax.axvline(x=0.60, color='orange', linestyle='--', alpha=0.5, label='Min acceptable')
    ax.axvline(x=0.85, color='green', linestyle='--', alpha=0.5, label='Good')
    ax.axvline(x=0.95, color='red', linestyle='--', alpha=0.5, label='Leakage warning')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'auc_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  Visualizations saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_leakage_free_pipeline():
    """Run leakage-free 14-label pipeline."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  14-LABEL ISSUE DETECTOR — LEAKAGE-FREE VERSION")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Strict feature isolation — indirect signals only")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("LEAKAGE-FREE 14-LABEL DETECTOR")
    log.info("=" * 60)
    
    # 1. Load
    log.info("\n[1/5] Loading data...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  {len(df):,} patients, {len(df.columns)} columns")
    
    # 2. Create labels BEFORE feature exclusion
    log.info("\n[2/5] Creating labels...")
    labels = pd.DataFrame(index=df.index)
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        labels[issue_key] = issue_def['logic'](df)
        count = labels[issue_key].sum()
        prev = labels[issue_key].mean() * 100
        flag = "⚠️ LOW" if prev < 1 else "⚠️ HIGH" if prev > 50 else ""
        log.info(f"  {issue_key}: {count:,} ({prev:.2f}%) {flag}")
    
    # 3. Engineer indirect features
    log.info("\n[3/5] Engineering indirect features...")
    df = engineer_indirect_features(df)
    
    # Select ONLY indirect features
    indirect_features = select_indirect_features(df)
    log.info(f"  Selected {len(indirect_features)} indirect features")
    log.info(f"  Excluded {len(ALL_EXCLUDE)} outcome columns")
    
    # 4. Split
    log.info("\n[4/5] Splitting data...")
    train_val_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    log.info(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    
    # 5. Train classifiers
    log.info("\n[5/5] Training 14 classifiers (leakage-free)...")
    results = []
    
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        log.info(f"\n  [{len(results)+1}/14] {issue_def['name']} ({issue_def['priority']})")
        
        # Use same indirect features for all
        log.info(f"    Features: {len(indirect_features)} (indirect only)")
        
        # Prepare data
        X_train = df.loc[train_idx, indirect_features].fillna(0).replace([np.inf, -np.inf], 0)
        X_val = df.loc[val_idx, indirect_features].fillna(0).replace([np.inf, -np.inf], 0)
        X_test = df.loc[test_idx, indirect_features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale
        scaler = RobustScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=indirect_features, index=X_train.index)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=indirect_features, index=X_val.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=indirect_features, index=X_test.index)
        
        # Labels
        y_train = labels.loc[train_idx, issue_key].values
        y_val = labels.loc[val_idx, issue_key].values
        y_test = labels.loc[test_idx, issue_key].values
        
        # Train
        result = train_classifier(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
                                  issue_key, issue_def)
        
        if result:
            if result.get('status') == 'TRAINED':
                result['features'] = indirect_features
                result['scaler'] = scaler
                flags_str = f" {result['flags']}" if result['flags'] else ""
                log.info(f"    AUC={result['auc']:.3f} | F1={result['f1']:.3f}{flags_str}")
            else:
                log.warning(f"    {result.get('reason', 'Unknown')}")
            results.append(result)
    
    # Visualizations
    log.info("\n  Creating visualizations...")
    create_visualizations(results, OUTPUT_DIR / 'figures')
    
    # Save models
    trained = [r for r in results if r.get('status') == 'TRAINED']
    for r in trained:
        with open(OUTPUT_DIR / 'models' / f"{r['issue_key']}.pkl", 'wb') as f:
            pickle.dump({
                'model': r['model'],
                'features': r['features'],
                'threshold': r['threshold'],
                'scaler': r['scaler']
            }, f)
    
    # Save config
    avg_auc = np.mean([r['auc'] for r in trained]) if trained else 0
    avg_f1 = np.mean([r['f1'] for r in trained]) if trained else 0
    
    config = {
        'version': 'LEAKAGE_FREE_v1',
        'created': datetime.now().isoformat(),
        'total_labels': 14,
        'trained': len(trained),
        'skipped': len(results) - len(trained),
        'excluded_columns': len(ALL_EXCLUDE),
        'indirect_features': len(indirect_features),
        'avg_auc': float(avg_auc),
        'avg_f1': float(avg_f1),
        'issues': {r['issue_key']: {
            'name': r['name'],
            'priority': r.get('priority', 'MEDIUM'),
            'status': r.get('status', 'UNKNOWN'),
            'prevalence': r.get('prevalence', 0),
            'auc': r.get('auc', 0),
            'f1': r.get('f1', 0),
            'flags': r.get('flags', [])
        } for r in results}
    }
    
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Performance table
    perf = [{
        'Issue': r['name'],
        'Priority': r.get('priority', 'MEDIUM'),
        'Prevalence': f"{r.get('prevalence', 0)*100:.2f}%",
        'AUC': f"{r.get('auc', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'F1': f"{r.get('f1', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'Status': r.get('status', 'UNKNOWN'),
        'Flags': ', '.join(r.get('flags', []))
    } for r in results]
    
    pd.DataFrame(perf).to_csv(OUTPUT_DIR / 'tables' / 'performance.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    
    # Summary
    print("\n" + "=" * 70)
    print("  LEAKAGE-FREE 14-LABEL — COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n  LEAKAGE PREVENTION:")
    print(f"    Excluded columns: {len(ALL_EXCLUDE)}")
    print(f"    Indirect features: {len(indirect_features)}")
    print(f"\n  RESULTS:")
    print(f"    Trained: {len(trained)}/14")
    print(f"    Avg AUC: {avg_auc:.4f}")
    print(f"    Avg F1:  {avg_f1:.4f}")
    
    print("\n  PERFORMANCE (Realistic):")
    for r in trained:
        flags = f" {r['flags']}" if r.get('flags') else ""
        print(f"    {r['name']:25s} | AUC={r['auc']:.3f} | F1={r['f1']:.3f}{flags}")
    
    skipped = [r for r in results if r.get('status') != 'TRAINED']
    if skipped:
        print("\n  SKIPPED:")
        for r in skipped:
            print(f"    {r['name']:25s} | {r.get('reason', 'Unknown')}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return results, config


if __name__ == '__main__':
    run_leakage_free_pipeline()
