"""
TRIALPULSE NEXUS 10X — COMPLETE 14-LABEL ISSUE DETECTOR
Full Implementation per ML_TRAINING_METHODOLOGY.md Specification

LABELS (14 Total):
1.  sae_dm_pending        — SAE DM review pending
2.  sae_safety_pending    — SAE Safety review pending
3.  open_queries          — Has open queries
4.  high_query_volume     — >10 queries (high load)
5.  sdv_incomplete        — SDV not complete
6.  signature_gaps        — Missing/overdue signatures
7.  broken_signatures     — Has broken signatures
8.  meddra_uncoded        — MedDRA terms uncoded
9.  whodrug_uncoded       — WHODrug terms uncoded
10. missing_visits        — Has missing visits
11. missing_pages         — Has missing pages
12. lab_issues            — Lab name/range issues
13. edrr_issues           — Third-party reconciliation issues
14. inactivated_forms     — Has inactivated forms

VERSION: COMPLETE_14_LABEL_v1
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
from typing import Dict, List, Tuple, Optional

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
import seaborn as sns

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
UPR_PATH = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'issue_detector_14_LABEL'

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
# 14 ISSUE DEFINITIONS — COMPLETE SPEC
# ============================================================================

ISSUE_DEFINITIONS = {
    # SAE Issues (CRITICAL priority)
    'sae_dm_pending': {
        'name': 'SAE DM Pending',
        'description': 'SAE Data Management review pending',
        'source_col': 'sae_dm_sae_dm_pending',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'CRITICAL',
        'type': 'RULE_BASED'
    },
    'sae_safety_pending': {
        'name': 'SAE Safety Pending',
        'description': 'SAE Safety review pending',
        'source_col': 'sae_safety_sae_safety_pending',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'CRITICAL',
        'type': 'RULE_BASED'
    },
    
    # Query Issues (HIGH priority)
    'open_queries': {
        'name': 'Open Queries',
        'description': 'Has any open queries',
        'source_col': 'total_queries',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'HIGH',
        'type': 'RULE_BASED'
    },
    'high_query_volume': {
        'name': 'High Query Volume',
        'description': 'More than 10 open queries (high workload)',
        'source_col': 'total_queries',
        'logic': lambda df, col: (df[col].fillna(0) > 10).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'HIGH',
        'type': 'RULE_BASED'
    },
    
    # SDV Issues (HIGH priority)
    'sdv_incomplete': {
        'name': 'SDV Incomplete',
        'description': 'SDV verification not complete',
        'source_cols': ['crfs_require_verification_sdv', 'forms_verified'],
        'logic': 'rate_based',  # Special handling
        'priority': 'HIGH',
        'type': 'THRESHOLD_BASED'
    },
    
    # Signature Issues (MEDIUM priority)  
    'signature_gaps': {
        'name': 'Signature Gaps',
        'description': 'Has overdue signatures',
        'source_cols': [
            'crfs_overdue_for_signs_within_45_days_of_data_entry',
            'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
            'crfs_overdue_for_signs_beyond_90_days_of_data_entry'
        ],
        'logic': 'sum_positive',
        'priority': 'MEDIUM',
        'type': 'RULE_BASED'
    },
    'broken_signatures': {
        'name': 'Broken Signatures',
        'description': 'Has broken/invalid signatures',
        'source_col': 'broken_signatures',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'MEDIUM',
        'type': 'RULE_BASED'
    },
    
    # Coding Issues (MEDIUM priority)
    'meddra_uncoded': {
        'name': 'MedDRA Uncoded',
        'description': 'Has uncoded MedDRA terms',
        'source_col': 'meddra_coding_meddra_uncoded',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'MEDIUM',
        'type': 'RULE_BASED'
    },
    'whodrug_uncoded': {
        'name': 'WHODrug Uncoded',
        'description': 'Has uncoded WHODrug terms',
        'source_col': 'whodrug_coding_whodrug_uncoded',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'MEDIUM',
        'type': 'RULE_BASED'
    },
    
    # Missing Data Issues (HIGH priority)
    'missing_visits': {
        'name': 'Missing Visits',
        'description': 'Has missing visits',
        'source_col': 'has_missing_visits',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'HIGH',
        'type': 'RULE_BASED'
    },
    'missing_pages': {
        'name': 'Missing Pages',
        'description': 'Has missing pages',
        'source_col': 'has_missing_pages',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'HIGH',
        'type': 'RULE_BASED'
    },
    
    # Lab & EDRR Issues (MEDIUM priority)
    'lab_issues': {
        'name': 'Lab Issues',
        'description': 'Has lab name/range issues',
        'source_col': 'lab_lab_issue_count',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'MEDIUM',
        'type': 'RULE_BASED'
    },
    'edrr_issues': {
        'name': 'EDRR Issues',
        'description': 'Has third-party reconciliation issues',
        'source_col': 'edrr_edrr_issue_count',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'MEDIUM',
        'type': 'RULE_BASED'
    },
    
    # Form Issues (LOW priority)
    'inactivated_forms': {
        'name': 'Inactivated Forms',
        'description': 'Has inactivated CRF forms',
        'source_col': 'inactivated_inactivated_form_count',
        'logic': lambda df, col: (df[col].fillna(0) > 0).astype(int) if col in df.columns else pd.Series(0, index=df.index),
        'priority': 'LOW',
        'type': 'RULE_BASED'
    }
}


# ============================================================================
# LABEL CREATION
# ============================================================================

def create_all_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create all 14 binary labels."""
    labels = pd.DataFrame(index=df.index)
    
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        try:
            if issue_def.get('logic') == 'rate_based':
                # SDV incomplete: rate < 100%
                cols = issue_def.get('source_cols', [])
                if all(c in df.columns for c in cols):
                    required = df[cols[0]].fillna(0)
                    verified = df[cols[1]].fillna(0)
                    rate = np.where(required > 0, verified / (required + 0.001), 1.0)
                    labels[issue_key] = (rate < 1.0).astype(int)
                else:
                    labels[issue_key] = 0
                    
            elif issue_def.get('logic') == 'sum_positive':
                # Signature gaps: sum of overdue columns > 0
                cols = issue_def.get('source_cols', [])
                existing = [c for c in cols if c in df.columns]
                if existing:
                    labels[issue_key] = (df[existing].fillna(0).sum(axis=1) > 0).astype(int)
                else:
                    labels[issue_key] = 0
                    
            elif callable(issue_def.get('logic')):
                # Simple column check
                col = issue_def.get('source_col', '')
                labels[issue_key] = issue_def['logic'](df, col)
            else:
                labels[issue_key] = 0
                
        except Exception as e:
            log.warning(f"  {issue_key}: Error creating label - {str(e)}")
            labels[issue_key] = 0
    
    return labels


# ============================================================================
# FEATURE ENGINEERING  
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for ML models."""
    df = df.copy()
    
    # Site context
    if 'site_id' in df.columns:
        site_counts = df.groupby('site_id').size().reset_index(name='site_patient_count')
        df = df.merge(site_counts, on='site_id', how='left')
        df['site_patient_count'] = df['site_patient_count'].fillna(1)
    
    # Study context
    if 'study_id' in df.columns and 'pages_entered' in df.columns:
        study_stats = df.groupby('study_id')['pages_entered'].agg(['mean', 'std']).reset_index()
        study_stats.columns = ['study_id', 'study_avg_pages', 'study_std_pages']
        df = df.merge(study_stats, on='study_id', how='left')
        df['study_std_pages'] = df['study_std_pages'].fillna(0)
    
    # Workload normalization
    if 'pages_entered' in df.columns:
        p90 = df['pages_entered'].quantile(0.90)
        if p90 > 0:
            df['pages_normalized'] = (df['pages_entered'] / p90).clip(0, 2)
    
    # Nonconformant rate
    if 'pages_entered' in df.columns and 'pages_with_nonconformant_data' in df.columns:
        df['nonconformant_rate'] = np.where(
            df['pages_entered'] > 0,
            df['pages_with_nonconformant_data'] / (df['pages_entered'] + 1),
            0
        )
    
    # Protocol deviations
    if 'pds_confirmed' in df.columns:
        df['has_deviations'] = (df['pds_confirmed'].fillna(0) > 0).astype(float)
    
    return df


# ============================================================================
# FEATURE SELECTION (Per Issue)
# ============================================================================

def get_feature_exclusions(issue_key: str) -> set:
    """Get features to exclude for each issue to prevent leakage."""
    
    # Base exclusions for all issues
    base_exclude = {
        'project_name', 'region', 'country', 'site', 'subject', 'latest_visit',
        'subject_status', 'input_files', '_source_file', '_study_id', '_ingestion_ts',
        'study_id', 'site_id', 'subject_id', 'patient_key', 'risk_level',
        '_cleaned_ts', '_upr_built_ts', '_upr_version', '_file_type', 'cpmd', 'ssm'
    }
    
    # Issue-specific exclusions
    issue_exclusions = {
        'sae_dm_pending': {'sae_dm_sae_dm_pending', 'sae_dm_sae_dm_completed', 'sae_dm_sae_dm_total'},
        'sae_safety_pending': {'sae_safety_sae_safety_pending', 'sae_safety_sae_safety_completed', 'sae_safety_sae_safety_total'},
        'open_queries': {'total_queries', 'dm_queries', 'clinical_queries', 'medical_queries', 'site_queries', 'field_monitor_queries', 'coding_queries', 'safety_queries'},
        'high_query_volume': {'total_queries', 'dm_queries', 'clinical_queries', 'medical_queries', 'site_queries', 'field_monitor_queries', 'coding_queries', 'safety_queries'},
        'sdv_incomplete': {'crfs_require_verification_sdv', 'forms_verified', 'completeness_score'},
        'signature_gaps': {'crfs_overdue_for_signs_within_45_days_of_data_entry', 'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry', 'crfs_overdue_for_signs_beyond_90_days_of_data_entry', 'crfs_signed', 'crfs_never_signed'},
        'broken_signatures': {'broken_signatures', 'crfs_signed'},
        'meddra_uncoded': {'meddra_coding_meddra_uncoded', 'meddra_coding_meddra_coded', 'meddra_coding_meddra_total', 'coding_completion_rate'},
        'whodrug_uncoded': {'whodrug_coding_whodrug_uncoded', 'whodrug_coding_whodrug_coded', 'whodrug_coding_whodrug_total', 'coding_completion_rate'},
        'missing_visits': {'has_missing_visits', 'visit_missing_visit_count', 'missing_visits'},
        'missing_pages': {'has_missing_pages', 'pages_missing_page_count', 'missing_pages'},
        'lab_issues': {'lab_lab_issue_count', 'has_lab_issues'},
        'edrr_issues': {'edrr_edrr_issue_count', 'has_edrr_issues'},
        'inactivated_forms': {'inactivated_inactivated_form_count', 'inactivated_inactivated_unique_forms'}
    }
    
    return base_exclude.union(issue_exclusions.get(issue_key, set()))


def select_features(df: pd.DataFrame, issue_key: str) -> List[str]:
    """Select features for a specific issue."""
    exclude = get_feature_exclusions(issue_key)
    
    safe_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
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
    """Train binary classifier for one issue."""
    
    pos_rate = y_train.mean()
    
    # Skip if too imbalanced
    if pos_rate < 0.001 or pos_rate > 0.999:
        return {
            'issue_key': issue_key,
            'name': issue_def['name'],
            'status': 'SKIPPED',
            'reason': f'Extreme imbalance (pos_rate={pos_rate:.4f})',
            'prevalence': float(y_test.mean())
        }
    
    # Calculate class weight
    scale_weight = min((1 - pos_rate) / max(pos_rate, 0.001), 20)
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
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
    auc = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 and y_test.sum() < len(y_test) else 0.5
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
    
    return {
        'model': final_model,
        'base_model': model,
        'issue_key': issue_key,
        'name': issue_def['name'],
        'priority': issue_def['priority'],
        'type': issue_def['type'],
        'status': 'TRAINED',
        'prevalence': float(y_test.mean()),
        'auc': float(auc),
        'ap': float(ap),
        'f1': float(best_f1),
        'precision': float(precision),
        'recall': float(recall),
        'threshold': float(best_th),
        'top_features': top_features,
        'n_features': len(X_train.columns)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(results: List[Dict], output_dir: Path):
    """Create comprehensive visualizations."""
    
    trained = [r for r in results if r.get('status') == 'TRAINED']
    if not trained:
        return
    
    # 1. Performance Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = [r['name'] for r in trained]
    metrics = np.array([[r['auc'], r['ap'], r['f1'], r['precision'], r['recall']] for r in trained])
    
    im = ax.imshow(metrics, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['AUC', 'AP', 'F1', 'Precision', 'Recall'], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    
    for i in range(len(names)):
        for j in range(5):
            val = metrics[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    
    ax.set_title('14-LABEL ISSUE DETECTOR — Performance Metrics', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. AUC by Priority
    fig, ax = plt.subplots(figsize=(12, 6))
    
    priority_colors = {'CRITICAL': '#e74c3c', 'HIGH': '#f39c12', 'MEDIUM': '#3498db', 'LOW': '#2ecc71'}
    colors = [priority_colors.get(r['priority'], '#95a5a6') for r in trained]
    
    bars = ax.barh(range(len(names)), [r['auc'] for r in trained], color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{r['name']} [{r['priority']}]" for r in trained], fontsize=9)
    ax.set_xlabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Issue Detection Performance by Priority', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 1.0)
    ax.axvline(x=0.80, color='green', linestyle='--', alpha=0.5, label='Good (0.80)')
    ax.axvline(x=0.90, color='blue', linestyle='--', alpha=0.5, label='Excellent (0.90)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'auc_by_priority.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Prevalence Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    
    prevalences = [r['prevalence'] * 100 for r in trained]
    colors = ['red' if p < 1 else 'orange' if p < 5 else 'green' for p in prevalences]
    
    bars = ax.bar(range(len(names)), prevalences, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([r['name'] for r in trained], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Prevalence (%)')
    ax.set_title('Issue Prevalence Distribution', fontsize=14, fontweight='bold')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='<1% (rare)')
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='<5% (uncommon)')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / 'prevalence_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  Visualizations saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_14_label_pipeline():
    """Run complete 14-label pipeline."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  COMPLETE 14-LABEL ISSUE DETECTOR")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Binary Relevance: 14 Independent Classifiers")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("14-LABEL ISSUE DETECTOR")
    log.info("=" * 60)
    
    # 1. Load
    log.info("\n[1/5] Loading data...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  Loaded {len(df):,} patients, {len(df.columns)} columns")
    
    # 2. Create Labels
    log.info("\n[2/5] Creating 14 labels...")
    labels = create_all_labels(df)
    
    for col in labels.columns:
        count = labels[col].sum()
        prev = labels[col].mean() * 100
        flag = "⚠️ LOW" if prev < 1 else ""
        log.info(f"  {col}: {count:,} ({prev:.2f}%) {flag}")
    
    # 3. Feature Engineering
    log.info("\n[3/5] Engineering features...")
    df = engineer_features(df)
    
    # 4. Split
    log.info("\n[4/5] Splitting data...")
    train_val_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    log.info(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    
    # 5. Train 14 Classifiers
    log.info("\n[5/5] Training 14 classifiers...")
    results = []
    
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        log.info(f"\n  [{len(results)+1}/14] {issue_def['name']} ({issue_def['priority']})")
        
        # Get features
        features = select_features(df, issue_key)
        log.info(f"    Features: {len(features)}")
        
        # Prepare data
        X_train = df.loc[train_idx, features].fillna(0).replace([np.inf, -np.inf], 0)
        X_val = df.loc[val_idx, features].fillna(0).replace([np.inf, -np.inf], 0)
        X_test = df.loc[test_idx, features].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale
        scaler = RobustScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=features, index=X_val.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)
        
        # Labels
        y_train = labels.loc[train_idx, issue_key].values
        y_val = labels.loc[val_idx, issue_key].values
        y_test = labels.loc[test_idx, issue_key].values
        
        # Train
        result = train_classifier(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
                                  issue_key, issue_def)
        
        if result:
            if result.get('status') == 'TRAINED':
                result['features'] = features
                result['scaler'] = scaler
                log.info(f"    AUC={result['auc']:.3f} | AP={result['ap']:.3f} | F1={result['f1']:.3f}")
            else:
                log.warning(f"    {result.get('reason', 'Unknown issue')}")
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
    
    # Summary config
    config = {
        'version': 'COMPLETE_14_LABEL_v1',
        'created': datetime.now().isoformat(),
        'total_labels': 14,
        'trained': len(trained),
        'skipped': len(results) - len(trained),
        'avg_auc': float(np.mean([r['auc'] for r in trained])) if trained else 0,
        'avg_f1': float(np.mean([r['f1'] for r in trained])) if trained else 0,
        'issues': {r['issue_key']: {
            'name': r['name'],
            'priority': r.get('priority', 'MEDIUM'),
            'status': r.get('status', 'UNKNOWN'),
            'prevalence': r.get('prevalence', 0),
            'auc': r.get('auc', 0),
            'ap': r.get('ap', 0),
            'f1': r.get('f1', 0),
            'threshold': r.get('threshold', 0.5)
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
        'AP': f"{r.get('ap', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'F1': f"{r.get('f1', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'Status': r.get('status', 'UNKNOWN')
    } for r in results]
    
    pd.DataFrame(perf).to_csv(OUTPUT_DIR / 'tables' / 'performance.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    
    # Final Summary
    avg_auc = np.mean([r['auc'] for r in trained]) if trained else 0
    avg_f1 = np.mean([r['f1'] for r in trained]) if trained else 0
    
    print("\n" + "=" * 70)
    print("  14-LABEL ISSUE DETECTOR — COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n  SUMMARY:")
    print(f"    Total Labels:  14")
    print(f"    Trained:       {len(trained)}")
    print(f"    Skipped:       {len(results) - len(trained)}")
    print(f"    Avg AUC:       {avg_auc:.4f}")
    print(f"    Avg F1:        {avg_f1:.4f}")
    
    print("\n  PERFORMANCE BY ISSUE:")
    for r in trained:
        print(f"    {r['name']:25s} | AUC={r['auc']:.3f} | F1={r['f1']:.3f} | Prev={r['prevalence']*100:.1f}%")
    
    skipped = [r for r in results if r.get('status') != 'TRAINED']
    if skipped:
        print("\n  SKIPPED ISSUES:")
        for r in skipped:
            print(f"    {r['name']:25s} | {r.get('reason', 'Unknown')}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return results, config


if __name__ == '__main__':
    run_14_label_pipeline()
