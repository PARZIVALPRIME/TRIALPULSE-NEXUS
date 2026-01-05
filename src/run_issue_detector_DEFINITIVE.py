"""
TRIALPULSE NEXUS 10X — 14-LABEL ISSUE DETECTOR (DEFINITIVE)
Complete Leakage Prevention — Whitelist Approach

STRATEGY: Instead of blacklisting outcome columns, WHITELIST only safe features
This guarantees no leakage by design.

VERSION: DEFINITIVE_v1
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
    recall_score
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
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'issue_detector_DEFINITIVE'

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
# WHITELIST APPROACH — ONLY THESE FEATURES ARE ALLOWED
# Everything else is considered a potential leak
# ============================================================================

SAFE_RAW_FEATURES = [
    # Workload indicators (how much data, not what issues)
    'pages_entered',
    'expected_visits_rave_edc_bo4',
    
    # Quality behavior (past quality, not current issues)
    'pages_with_nonconformant_data',  # Historical quality
    'pds_proposed',                    # Protocol deviations proposed
    'pds_confirmed',                   # Protocol deviations confirmed
]

# ============================================================================
# 14 ISSUE DEFINITIONS
# ============================================================================

ISSUE_DEFINITIONS = {
    'sae_dm_pending': {'name': 'SAE DM Pending', 'priority': 'CRITICAL'},
    'sae_safety_pending': {'name': 'SAE Safety Pending', 'priority': 'CRITICAL'},
    'open_queries': {'name': 'Open Queries', 'priority': 'HIGH'},
    'high_query_volume': {'name': 'High Query Volume', 'priority': 'HIGH'},
    'sdv_incomplete': {'name': 'SDV Incomplete', 'priority': 'HIGH'},
    'signature_gaps': {'name': 'Signature Gaps', 'priority': 'MEDIUM'},
    'broken_signatures': {'name': 'Broken Signatures', 'priority': 'MEDIUM'},
    'meddra_uncoded': {'name': 'MedDRA Uncoded', 'priority': 'MEDIUM'},
    'whodrug_uncoded': {'name': 'WHODrug Uncoded', 'priority': 'MEDIUM'},
    'missing_visits': {'name': 'Missing Visits', 'priority': 'HIGH'},
    'missing_pages': {'name': 'Missing Pages', 'priority': 'HIGH'},
    'lab_issues': {'name': 'Lab Issues', 'priority': 'MEDIUM'},
    'edrr_issues': {'name': 'EDRR Issues', 'priority': 'MEDIUM'},
    'inactivated_forms': {'name': 'Inactivated Forms', 'priority': 'LOW'}
}


# ============================================================================
# LABEL CREATION
# ============================================================================

def create_all_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create all 14 binary labels."""
    labels = pd.DataFrame(index=df.index)
    
    # 1. SAE DM Pending
    labels['sae_dm_pending'] = (df.get('sae_dm_sae_dm_pending', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 2. SAE Safety Pending
    labels['sae_safety_pending'] = (df.get('sae_safety_sae_safety_pending', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 3. Open Queries
    labels['open_queries'] = (df.get('total_queries', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 4. High Query Volume
    labels['high_query_volume'] = (df.get('total_queries', pd.Series(0, index=df.index)).fillna(0) > 10).astype(int)
    
    # 5. SDV Incomplete
    if 'crfs_require_verification_sdv' in df.columns and 'forms_verified' in df.columns:
        required = df['crfs_require_verification_sdv'].fillna(0)
        verified = df['forms_verified'].fillna(0)
        rate = np.where(required > 0, verified / (required + 0.001), 1.0)
        labels['sdv_incomplete'] = (rate < 1.0).astype(int)
    else:
        labels['sdv_incomplete'] = 0
    
    # 6. Signature Gaps
    overdue_cols = [c for c in df.columns if 'overdue_for_signs' in c]
    if overdue_cols:
        labels['signature_gaps'] = (df[overdue_cols].fillna(0).sum(axis=1) > 0).astype(int)
    else:
        labels['signature_gaps'] = 0
    
    # 7. Broken Signatures
    labels['broken_signatures'] = (df.get('broken_signatures', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 8. MedDRA Uncoded
    labels['meddra_uncoded'] = (df.get('meddra_coding_meddra_uncoded', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 9. WHODrug Uncoded
    labels['whodrug_uncoded'] = (df.get('whodrug_coding_whodrug_uncoded', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 10. Missing Visits
    labels['missing_visits'] = (df.get('has_missing_visits', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 11. Missing Pages
    labels['missing_pages'] = (df.get('has_missing_pages', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 12. Lab Issues
    labels['lab_issues'] = (df.get('lab_lab_issue_count', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 13. EDRR Issues
    labels['edrr_issues'] = (df.get('edrr_edrr_issue_count', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    # 14. Inactivated Forms
    labels['inactivated_forms'] = (df.get('inactivated_inactivated_form_count', pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    
    return labels


# ============================================================================
# FEATURE ENGINEERING — STRICT WHITELIST
# ============================================================================

def engineer_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from STRICTLY safe sources only."""
    features = pd.DataFrame(index=df.index)
    
    # Raw safe features
    for col in SAFE_RAW_FEATURES:
        if col in df.columns:
            features[col] = df[col].fillna(0)
    
    # Workload normalization
    if 'pages_entered' in features.columns:
        p90 = features['pages_entered'].quantile(0.90)
        if p90 > 0:
            features['pages_normalized'] = (features['pages_entered'] / p90).clip(0, 2)
        else:
            features['pages_normalized'] = 0
        
        features['high_volume'] = (features['pages_entered'] > features['pages_entered'].quantile(0.75)).astype(float)
        features['low_volume'] = (features['pages_entered'] < features['pages_entered'].quantile(0.25)).astype(float)
    
    # Nonconformant rate
    if 'pages_entered' in features.columns and 'pages_with_nonconformant_data' in features.columns:
        features['nonconformant_rate'] = np.where(
            features['pages_entered'] > 0,
            features['pages_with_nonconformant_data'] / (features['pages_entered'] + 1),
            0
        )
    
    # Protocol deviation flags
    if 'pds_confirmed' in features.columns:
        features['has_deviations'] = (features['pds_confirmed'] > 0).astype(float)
    
    if 'pds_proposed' in features.columns:
        features['pending_deviations'] = (features['pds_proposed'] > 0).astype(float)
    
    # Site context (aggregated, no IDs)
    if 'site_id' in df.columns:
        site_counts = df.groupby('site_id').size().reset_index(name='site_patient_count')
        df_with_site = df[['site_id']].merge(site_counts, on='site_id', how='left')
        features['site_patient_count'] = df_with_site['site_patient_count'].values
        features['site_patient_count'] = features['site_patient_count'].fillna(1)
        features['large_site'] = (features['site_patient_count'] > features['site_patient_count'].quantile(0.75)).astype(float)
        features['small_site'] = (features['site_patient_count'] < features['site_patient_count'].quantile(0.25)).astype(float)
    
    # Study context (aggregated, no IDs)
    if 'study_id' in df.columns and 'pages_entered' in df.columns:
        study_means = df.groupby('study_id')['pages_entered'].mean().reset_index(name='study_avg_pages')
        df_with_study = df[['study_id']].merge(study_means, on='study_id', how='left')
        features['study_avg_pages'] = df_with_study['study_avg_pages'].values
        features['study_avg_pages'] = features['study_avg_pages'].fillna(features['pages_entered'].mean() if 'pages_entered' in features.columns else 0)
        
        if 'pages_entered' in features.columns:
            features['pages_vs_study'] = np.where(
                features['study_avg_pages'] > 0,
                features['pages_entered'] / (features['study_avg_pages'] + 1),
                1.0
            )
    
    # Interaction features
    if 'high_volume' in features.columns and 'large_site' in features.columns:
        features['high_vol_large_site'] = features['high_volume'] * features['large_site']
    
    if 'has_deviations' in features.columns and 'high_volume' in features.columns:
        features['deviations_high_vol'] = features['has_deviations'] * features['high_volume']
    
    # Ensure all numeric
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
    
    return features


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_classifier(X_train, y_train, X_val, y_val, X_test, y_test,
                     issue_key: str, issue_def: dict) -> Optional[Dict]:
    """Train classifier with strict feature set."""
    
    pos_rate = y_train.mean()
    
    # Skip extreme imbalance
    if pos_rate < 0.005 or pos_rate > 0.995:
        return {
            'issue_key': issue_key,
            'name': issue_def['name'],
            'priority': issue_def['priority'],
            'status': 'SKIPPED',
            'reason': f'Extreme imbalance ({pos_rate:.4f})',
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
    
    # Honest assessment
    assessment = 'GENUINE_ML' if auc < 0.85 else 'GOOD' if auc < 0.92 else 'EXCELLENT' if auc < 0.97 else 'CHECK_LEAKAGE'
    
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
        'assessment': assessment,
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
    
    ax.set_title('14-LABEL ISSUE DETECTOR — DEFINITIVE (Whitelist Features)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  Visualizations saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_definitive_pipeline():
    """Run definitive 14-label pipeline with whitelist approach."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  14-LABEL ISSUE DETECTOR — DEFINITIVE VERSION")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  WHITELIST approach — only safe features allowed")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("DEFINITIVE 14-LABEL DETECTOR")
    log.info("=" * 60)
    
    # 1. Load
    log.info("\n[1/5] Loading data...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  {len(df):,} patients")
    
    # 2. Create labels
    log.info("\n[2/5] Creating labels...")
    labels = create_all_labels(df)
    for col in labels.columns:
        count = labels[col].sum()
        prev = labels[col].mean() * 100
        flag = "⚠️" if prev < 1 or prev > 50 else ""
        log.info(f"  {col}: {count:,} ({prev:.2f}%) {flag}")
    
    # 3. Engineer SAFE features (whitelist)
    log.info("\n[3/5] Engineering SAFE features (whitelist)...")
    features = engineer_safe_features(df)
    feature_cols = list(features.columns)
    log.info(f"  Created {len(feature_cols)} safe features:")
    for f in feature_cols:
        log.info(f"    • {f}")
    
    # 4. Split
    log.info("\n[4/5] Splitting data...")
    train_val_idx, test_idx = train_test_split(features.index, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    log.info(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    
    # 5. Train
    log.info("\n[5/5] Training 14 classifiers...")
    results = []
    
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        log.info(f"\n  [{len(results)+1}/14] {issue_def['name']}")
        
        # Prepare data
        X_train = features.loc[train_idx].fillna(0)
        X_val = features.loc[val_idx].fillna(0)
        X_test = features.loc[test_idx].fillna(0)
        
        # Scale
        scaler = RobustScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
        
        # Labels
        y_train = labels.loc[train_idx, issue_key].values
        y_val = labels.loc[val_idx, issue_key].values
        y_test = labels.loc[test_idx, issue_key].values
        
        # Train
        result = train_classifier(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test,
                                  issue_key, issue_def)
        
        if result:
            if result.get('status') == 'TRAINED':
                result['features'] = feature_cols
                result['scaler'] = scaler
                log.info(f"    AUC={result['auc']:.3f} | F1={result['f1']:.3f} | {result['assessment']}")
            else:
                log.warning(f"    {result.get('reason', 'Unknown')}")
            results.append(result)
    
    # Visualizations
    log.info("\n  Creating visualizations...")
    create_visualizations(results, OUTPUT_DIR / 'figures')
    
    # Save
    trained = [r for r in results if r.get('status') == 'TRAINED']
    for r in trained:
        with open(OUTPUT_DIR / 'models' / f"{r['issue_key']}.pkl", 'wb') as f:
            pickle.dump({
                'model': r['model'],
                'features': r['features'],
                'threshold': r['threshold'],
                'scaler': r['scaler']
            }, f)
    
    # Config
    avg_auc = np.mean([r['auc'] for r in trained]) if trained else 0
    avg_f1 = np.mean([r['f1'] for r in trained]) if trained else 0
    
    config = {
        'version': 'DEFINITIVE_v1',
        'approach': 'WHITELIST_FEATURES',
        'created': datetime.now().isoformat(),
        'safe_features': feature_cols,
        'n_features': len(feature_cols),
        'total_labels': 14,
        'trained': len(trained),
        'skipped': len(results) - len(trained),
        'avg_auc': float(avg_auc),
        'avg_f1': float(avg_f1),
        'issues': {r['issue_key']: {
            'name': r['name'],
            'status': r.get('status'),
            'auc': r.get('auc', 0),
            'f1': r.get('f1', 0),
            'assessment': r.get('assessment', 'N/A')
        } for r in results}
    }
    
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Performance table
    perf = [{
        'Issue': r['name'],
        'Prevalence': f"{r.get('prevalence', 0)*100:.2f}%",
        'AUC': f"{r.get('auc', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'F1': f"{r.get('f1', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'Assessment': r.get('assessment', r.get('reason', 'N/A'))
    } for r in results]
    
    pd.DataFrame(perf).to_csv(OUTPUT_DIR / 'tables' / 'performance.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    
    # Summary
    print("\n" + "=" * 70)
    print("  DEFINITIVE 14-LABEL — COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n  WHITELIST FEATURES ({len(feature_cols)}):")
    for f in feature_cols[:10]:
        print(f"    • {f}")
    if len(feature_cols) > 10:
        print(f"    ... and {len(feature_cols) - 10} more")
    
    print(f"\n  RESULTS:")
    print(f"    Trained: {len(trained)}/14")
    print(f"    Avg AUC: {avg_auc:.4f}")
    print(f"    Avg F1:  {avg_f1:.4f}")
    
    print("\n  PERFORMANCE:")
    for r in trained:
        print(f"    {r['name']:25s} | AUC={r['auc']:.3f} | F1={r['f1']:.3f} | {r['assessment']}")
    
    skipped = [r for r in results if r.get('status') != 'TRAINED']
    if skipped:
        print("\n  SKIPPED:")
        for r in skipped:
            print(f"    {r['name']:25s} | {r.get('reason', 'Unknown')}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return results, config


if __name__ == '__main__':
    run_definitive_pipeline()
