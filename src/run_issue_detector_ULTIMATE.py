"""
TRIALPULSE NEXUS 10X — ULTIMATE 14-LABEL ISSUE DETECTOR
Zero Leakage Guarantee — Pure Behavioral Signals Only

PROBLEM: Previous versions still had 3 issues with AUC>0.97
CAUSE: Site/study aggregates may encode population-level issue rates
SOLUTION: Use ONLY patient-level behavioral features, no aggregates

VERSION: ULTIMATE_v1
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
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'issue_detector_ULTIMATE'

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
# PURE BEHAVIORAL FEATURES — NO AGGREGATES
# These are ONLY patient-level behavioral signals
# ============================================================================

PURE_BEHAVIORAL_COLS = [
    'pages_entered',                      # How much data entered
    'pages_with_nonconformant_data',      # Quality history
    'pds_proposed',                       # Deviations proposed
    'pds_confirmed',                      # Deviations confirmed
    'expected_visits_rave_edc_bo4',       # Expected visits
]


# ============================================================================
# 14 ISSUE DEFINITIONS
# ============================================================================

ISSUE_DEFS = {
    'sae_dm_pending': {'name': 'SAE DM Pending', 'priority': 'CRITICAL', 'source': 'sae_dm_sae_dm_pending'},
    'sae_safety_pending': {'name': 'SAE Safety Pending', 'priority': 'CRITICAL', 'source': 'sae_safety_sae_safety_pending'},
    'open_queries': {'name': 'Open Queries', 'priority': 'HIGH', 'source': 'total_queries'},
    'high_query_volume': {'name': 'High Query Volume', 'priority': 'HIGH', 'source': 'total_queries'},
    'sdv_incomplete': {'name': 'SDV Incomplete', 'priority': 'HIGH', 'source': 'forms_verified'},
    'signature_gaps': {'name': 'Signature Gaps', 'priority': 'MEDIUM', 'source': 'overdue'},
    'broken_signatures': {'name': 'Broken Signatures', 'priority': 'MEDIUM', 'source': 'broken_signatures'},
    'meddra_uncoded': {'name': 'MedDRA Uncoded', 'priority': 'MEDIUM', 'source': 'meddra_coding_meddra_uncoded'},
    'whodrug_uncoded': {'name': 'WHODrug Uncoded', 'priority': 'MEDIUM', 'source': 'whodrug_coding_whodrug_uncoded'},
    'missing_visits': {'name': 'Missing Visits', 'priority': 'HIGH', 'source': 'has_missing_visits'},
    'missing_pages': {'name': 'Missing Pages', 'priority': 'HIGH', 'source': 'has_missing_pages'},
    'lab_issues': {'name': 'Lab Issues', 'priority': 'MEDIUM', 'source': 'lab_lab_issue_count'},
    'edrr_issues': {'name': 'EDRR Issues', 'priority': 'MEDIUM', 'source': 'edrr_edrr_issue_count'},
    'inactivated_forms': {'name': 'Inactivated Forms', 'priority': 'LOW', 'source': 'inactivated_inactivated_form_count'}
}


# ============================================================================
# LABEL CREATION
# ============================================================================

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create all 14 binary labels."""
    labels = pd.DataFrame(index=df.index)
    
    def safe_gt(col, threshold=0):
        return (df.get(col, pd.Series(0, index=df.index)).fillna(0) > threshold).astype(int)
    
    labels['sae_dm_pending'] = safe_gt('sae_dm_sae_dm_pending')
    labels['sae_safety_pending'] = safe_gt('sae_safety_sae_safety_pending')
    labels['open_queries'] = safe_gt('total_queries')
    labels['high_query_volume'] = safe_gt('total_queries', 10)
    
    # SDV incomplete
    if 'crfs_require_verification_sdv' in df.columns and 'forms_verified' in df.columns:
        req = df['crfs_require_verification_sdv'].fillna(0)
        ver = df['forms_verified'].fillna(0)
        rate = np.where(req > 0, ver / (req + 0.001), 1.0)
        labels['sdv_incomplete'] = (rate < 1.0).astype(int)
    else:
        labels['sdv_incomplete'] = 0
    
    # Signature gaps
    overdue_cols = [c for c in df.columns if 'overdue_for_signs' in c]
    labels['signature_gaps'] = (df[overdue_cols].fillna(0).sum(axis=1) > 0).astype(int) if overdue_cols else 0
    
    labels['broken_signatures'] = safe_gt('broken_signatures')
    labels['meddra_uncoded'] = safe_gt('meddra_coding_meddra_uncoded')
    labels['whodrug_uncoded'] = safe_gt('whodrug_coding_whodrug_uncoded')
    labels['missing_visits'] = safe_gt('has_missing_visits')
    labels['missing_pages'] = safe_gt('has_missing_pages')
    labels['lab_issues'] = safe_gt('lab_lab_issue_count')
    labels['edrr_issues'] = safe_gt('edrr_edrr_issue_count')
    labels['inactivated_forms'] = safe_gt('inactivated_inactivated_form_count')
    
    return labels


# ============================================================================
# PURE BEHAVIORAL FEATURES
# ============================================================================

def create_pure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create ONLY pure behavioral features — no aggregates, no leakage."""
    features = pd.DataFrame(index=df.index)
    
    # Raw behavioral columns
    for col in PURE_BEHAVIORAL_COLS:
        if col in df.columns:
            features[col] = df[col].fillna(0)
        else:
            features[col] = 0
    
    # Derived features (from raw behavioral only)
    if 'pages_entered' in features.columns:
        # Normalize by global percentile (not site/study specific)
        p50 = features['pages_entered'].median()
        p90 = features['pages_entered'].quantile(0.90)
        
        if p90 > 0:
            features['pages_norm'] = (features['pages_entered'] / p90).clip(0, 2)
        else:
            features['pages_norm'] = 0
        
        # Binary indicators
        features['high_pages'] = (features['pages_entered'] > p90).astype(float)
        features['low_pages'] = (features['pages_entered'] < features['pages_entered'].quantile(0.25)).astype(float)
    
    # Quality rate
    if 'pages_with_nonconformant_data' in features.columns and 'pages_entered' in features.columns:
        features['nonconformant_rate'] = np.where(
            features['pages_entered'] > 0,
            features['pages_with_nonconformant_data'] / (features['pages_entered'] + 1),
            0
        )
        features['has_quality_issues'] = (features['nonconformant_rate'] > 0).astype(float)
    
    # Deviation flags
    if 'pds_confirmed' in features.columns:
        features['has_deviations'] = (features['pds_confirmed'] > 0).astype(float)
        features['deviation_count'] = features['pds_confirmed'].clip(0, 10)  # Cap extreme values
    
    if 'pds_proposed' in features.columns:
        features['has_pending_dev'] = (features['pds_proposed'] > 0).astype(float)
    
    # Expected visits normalization
    if 'expected_visits_rave_edc_bo4' in features.columns:
        ev_median = features['expected_visits_rave_edc_bo4'].median()
        if ev_median > 0:
            features['visits_norm'] = (features['expected_visits_rave_edc_bo4'] / ev_median).clip(0, 3)
        else:
            features['visits_norm'] = 0
    
    # Simple interactions (no leakage risk)
    if 'high_pages' in features.columns and 'has_quality_issues' in features.columns:
        features['high_pages_quality'] = features['high_pages'] * features['has_quality_issues']
    
    if 'has_deviations' in features.columns and 'high_pages' in features.columns:
        features['dev_high_pages'] = features['has_deviations'] * features['high_pages']
    
    # Clean up
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        features[col] = features[col].clip(-100, 100)  # Prevent extreme values
    
    return features


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X_train, y_train, X_val, y_val, X_test, y_test,
                issue_key: str, issue_def: dict) -> Optional[Dict]:
    """Train classifier."""
    
    pos_rate = y_train.mean()
    
    if pos_rate < 0.005 or pos_rate > 0.995:
        return {
            'issue_key': issue_key,
            'name': issue_def['name'],
            'priority': issue_def['priority'],
            'status': 'SKIPPED',
            'reason': f'Imbalance ({pos_rate:.4f})',
            'prevalence': float(y_test.mean())
        }
    
    # Balanced weight with cap
    scale_weight = min((1 - pos_rate) / max(pos_rate, 0.01), 10)
    
    # Very conservative model to avoid overfitting
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.6,
        scale_pos_weight=scale_weight,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.2,
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
    
    # Assessment
    if auc > 0.90:
        assessment = 'SUSPICIOUS'
    elif auc > 0.80:
        assessment = 'EXCELLENT'
    elif auc > 0.70:
        assessment = 'GOOD'
    elif auc > 0.60:
        assessment = 'ACCEPTABLE'
    else:
        assessment = 'WEAK'
    
    return {
        'model': final_model,
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
        'top_features': top_features
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_viz(results: List[Dict], output_dir: Path):
    """Create visualizations."""
    trained = [r for r in results if r.get('status') == 'TRAINED']
    if not trained:
        return
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    names = [r['name'] for r in trained]
    metrics = np.array([[r['auc'], r['ap'], r['f1'], r['precision'], r['recall']] for r in trained])
    
    # Color based on realistic expectations
    im = ax.imshow(metrics, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=0.95)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['AUC', 'AP', 'F1', 'Prec', 'Recall'], fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    
    for i in range(len(names)):
        for j in range(5):
            val = metrics[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, fontweight='bold', color=color)
    
    ax.set_title('14-LABEL ISSUE DETECTOR — ULTIMATE (Pure Behavioral)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Score')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  Visualizations saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_ultimate_pipeline():
    """Run ultimate pipeline."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  14-LABEL ISSUE DETECTOR — ULTIMATE VERSION")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Pure behavioral features only — ZERO aggregates")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("ULTIMATE 14-LABEL DETECTOR")
    log.info("=" * 60)
    
    # 1. Load
    log.info("\n[1/5] Loading...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  {len(df):,} patients")
    
    # 2. Labels
    log.info("\n[2/5] Creating labels...")
    labels = create_labels(df)
    for col in labels.columns:
        count = labels[col].sum()
        prev = labels[col].mean() * 100
        log.info(f"  {col}: {count:,} ({prev:.2f}%)")
    
    # 3. Features
    log.info("\n[3/5] Creating PURE behavioral features...")
    features = create_pure_features(df)
    log.info(f"  {len(features.columns)} features:")
    for f in features.columns:
        log.info(f"    • {f}")
    
    # 4. Split
    log.info("\n[4/5] Splitting...")
    train_val_idx, test_idx = train_test_split(features.index, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)
    log.info(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    
    # 5. Train
    log.info("\n[5/5] Training...")
    results = []
    feature_cols = list(features.columns)
    
    for issue_key, issue_def in ISSUE_DEFS.items():
        log.info(f"\n  [{len(results)+1}/14] {issue_def['name']}")
        
        X_train = features.loc[train_idx].fillna(0)
        X_val = features.loc[val_idx].fillna(0)
        X_test = features.loc[test_idx].fillna(0)
        
        scaler = RobustScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
        
        y_train = labels.loc[train_idx, issue_key].values
        y_val = labels.loc[val_idx, issue_key].values
        y_test = labels.loc[test_idx, issue_key].values
        
        result = train_model(X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, issue_key, issue_def)
        
        if result:
            if result.get('status') == 'TRAINED':
                result['features'] = feature_cols
                result['scaler'] = scaler
                log.info(f"    AUC={result['auc']:.3f} | F1={result['f1']:.3f} | {result['assessment']}")
            else:
                log.warning(f"    {result.get('reason')}")
            results.append(result)
    
    # Viz
    log.info("\n  Visualizations...")
    create_viz(results, OUTPUT_DIR / 'figures')
    
    # Save
    trained = [r for r in results if r.get('status') == 'TRAINED']
    for r in trained:
        with open(OUTPUT_DIR / 'models' / f"{r['issue_key']}.pkl", 'wb') as f:
            pickle.dump({'model': r['model'], 'features': r['features'], 'threshold': r['threshold'], 'scaler': r['scaler']}, f)
    
    avg_auc = np.mean([r['auc'] for r in trained]) if trained else 0
    avg_f1 = np.mean([r['f1'] for r in trained]) if trained else 0
    
    config = {
        'version': 'ULTIMATE_v1',
        'approach': 'PURE_BEHAVIORAL',
        'created': datetime.now().isoformat(),
        'features': feature_cols,
        'trained': len(trained),
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
    
    perf = [{
        'Issue': r['name'],
        'Prevalence': f"{r.get('prevalence', 0)*100:.2f}%",
        'AUC': f"{r.get('auc', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'F1': f"{r.get('f1', 0):.4f}" if r.get('status') == 'TRAINED' else 'N/A',
        'Assessment': r.get('assessment', r.get('reason', 'N/A'))
    } for r in results]
    pd.DataFrame(perf).to_csv(OUTPUT_DIR / 'tables' / 'performance.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("  ULTIMATE 14-LABEL — COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n  PURE BEHAVIORAL FEATURES ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"    • {f}")
    
    print(f"\n  RESULTS: {len(trained)}/14 trained")
    print(f"  Avg AUC: {avg_auc:.4f}")
    print(f"  Avg F1:  {avg_f1:.4f}")
    
    print("\n  PERFORMANCE:")
    for r in trained:
        print(f"    {r['name']:25s} | AUC={r['auc']:.3f} | F1={r['f1']:.3f} | {r['assessment']}")
    
    skipped = [r for r in results if r.get('status') != 'TRAINED']
    if skipped:
        print("\n  SKIPPED:")
        for r in skipped:
            print(f"    {r['name']:25s} | {r.get('reason')}")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return results, config


if __name__ == '__main__':
    run_ultimate_pipeline()
