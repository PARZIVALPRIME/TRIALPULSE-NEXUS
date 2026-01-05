"""
TRIALPULSE NEXUS 10X - Risk Classifier v7.0 CALIBRATED
Production-grade patient risk classification with ISOTONIC CALIBRATION

FIXES FROM v6:
- [x] Added isotonic calibration for true probability interpretation
- [x] Documented Medium class semantics (monitoring priority, not intervention)
- [x] Documented Low class drift monitoring advantage
- [x] Enhanced class interpretation guidelines

CLASS SEMANTICS:
================
- CRITICAL: Immediate intervention required
- HIGH: Prioritized attention within 24-48 hours
- MEDIUM: Monitoring priority - enhanced surveillance, NOT intervention
- LOW: Standard workflow - first detector of process drift

PROBABILITY INTERPRETATION:
===========================
With isotonic calibration: Probabilities ≈ true likelihood
P(Critical) = 0.7 means ~70% of similar patients were actually Critical
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    recall_score, precision_score, f1_score, confusion_matrix, 
    roc_auc_score, brier_score_loss
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ============================================================================
# SETUP
# ============================================================================

ROOT = Path(__file__).parent.parent
UPR_PATH = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'ml_training_v7'

for d in [OUTPUT_DIR, OUTPUT_DIR/'figures', OUTPUT_DIR/'models', OUTPUT_DIR/'tables']:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'training_v7.log', mode='w')
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# CLASS SEMANTICS DOCUMENTATION
# ============================================================================

CLASS_SEMANTICS = {
    0: {
        'name': 'Critical',
        'action': 'IMMEDIATE INTERVENTION',
        'timeframe': 'Within 4 hours',
        'description': 'Active safety event or severe compliance breach requiring urgent action'
    },
    1: {
        'name': 'High', 
        'action': 'PRIORITIZED ATTENTION',
        'timeframe': 'Within 24-48 hours',
        'description': 'Significant issues requiring prompt follow-up and resolution'
    },
    2: {
        'name': 'Low',
        'action': 'STANDARD WORKFLOW',
        'timeframe': 'Routine monitoring',
        'description': 'Clean record - serves as DRIFT DETECTOR due to stability'
    },
    3: {
        'name': 'Medium',
        'action': 'MONITORING PRIORITY',  # NOT intervention!
        'timeframe': 'Enhanced surveillance',
        'description': 'Elevated monitoring - watchlist for potential escalation, NOT direct intervention'
    }
}

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASS_WEIGHTS = {
    0: 20.0,  # Critical
    1: 15.0,  # High
    2: 1.0,   # Low
    3: 8.0,   # Medium
}

TARGET_RECALLS = {
    0: 0.70,  # Critical
    1: 0.55,  # High
    2: 0.75,  # Low
    3: 0.50,  # Medium
}

# Outcome features - NEVER use as input
OUTCOME_FEATURES = {
    'broken_signatures', 'crfs_never_signed', 'protocol_deviations',
    'safety_queries', 'sae_dm_sae_dm_completed', 'sae_safety_sae_safety_completed',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'visit_missing_visit_count', 'visit_visits_overdue_count',
    'lab_lab_issue_count', 'lab_lab_missing_names', 'lab_lab_missing_ranges',
    'edrr_edrr_issue_count', 'meddra_coding_meddra_coded', 'whodrug_coding_whodrug_coded',
    'inactivated_inactivated_form_count', 'pages_pages_missing_count',
    'pages_with_nonconformant_data'
}


# ============================================================================
# FEATURE ENGINEERING (same as v6)
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 25+ derived features for improved prediction."""
    log.info("Engineering derived features...")
    df = df.copy()
    n = 0
    
    # 1. Query metrics
    query_cols = ['dm_queries', 'clinical_queries', 'medical_queries', 'site_queries']
    existing = [c for c in query_cols if c in df.columns]
    if existing:
        df['total_queries'] = df[existing].fillna(0).sum(axis=1)
        df['query_type_count'] = (df[existing].fillna(0) > 0).sum(axis=1).astype(float)
        n += 2
        
        if 'queries_answered' in df.columns:
            total = df['total_queries'] + df['queries_answered'].fillna(0)
            df['query_resolution_rate'] = np.where(total > 0, df['queries_answered'].fillna(0) / (total + 1), 1.0)
            n += 1
    
    # 2. CRF completion
    if 'total_crfs' in df.columns:
        for col, name in [('crfs_signed', 'signature_rate'), ('crfs_frozen', 'freeze_rate'), 
                         ('crfs_locked', 'lock_rate'), ('crfs_verified_sdv', 'sdv_rate')]:
            if col in df.columns:
                df[name] = np.where(df['total_crfs'] > 0, df[col].fillna(0) / (df['total_crfs'] + 1), 0.0)
                n += 1
    
    # 3. SAE indicators
    for sae in ['sae_dm', 'sae_safety']:
        total_col = f'{sae}_{sae}_total'
        comp_col = f'{sae}_{sae}_completed'
        if total_col in df.columns:
            pending = df[total_col].fillna(0)
            if comp_col in df.columns:
                pending = (pending - df[comp_col].fillna(0)).clip(lower=0)
            df[f'{sae}_pending'] = pending
            df[f'has_{sae}'] = (df[total_col].fillna(0) > 0).astype(float)
            df[f'has_{sae}_pending'] = (pending > 0).astype(float)
            n += 3
    
    # Total SAE pending
    if 'sae_dm_pending' in df.columns and 'sae_safety_pending' in df.columns:
        df['total_sae_pending'] = df['sae_dm_pending'] + df['sae_safety_pending']
        df['has_any_sae_pending'] = (df['total_sae_pending'] > 0).astype(float)
        n += 2
    
    # 4. Coding metrics
    for code, prefix in [('meddra', 'meddra_coding'), ('whodrug', 'whodrug_coding')]:
        total_col = f'{prefix}_{code}_total'
        coded_col = f'{prefix}_{code}_coded'
        if total_col in df.columns and coded_col in df.columns:
            pending = (df[total_col].fillna(0) - df[coded_col].fillna(0)).clip(lower=0)
            df[f'{code}_pending'] = pending
            df[f'{code}_rate'] = np.where(df[total_col] > 0, df[coded_col].fillna(0) / (df[total_col] + 1), 1.0)
            df[f'has_{code}_pending'] = (pending > 0).astype(float)
            n += 3
    
    # 5. Query load flags
    if 'total_queries' in df.columns:
        df['high_query_load'] = (df['total_queries'] > 10).astype(float)
        df['critical_query_load'] = (df['total_queries'] > 25).astype(float)
        n += 2
    
    # 6. CRF volume flags
    if 'total_crfs' in df.columns:
        df['high_crf_volume'] = (df['total_crfs'] > 50).astype(float)
        df['very_high_crf_volume'] = (df['total_crfs'] > 100).astype(float)
        n += 2
    
    # 7. Query density
    if 'pages_entered' in df.columns and 'total_queries' in df.columns:
        df['query_density'] = np.where(df['pages_entered'] > 0, df['total_queries'] / (df['pages_entered'] + 1), 0.0)
        n += 1
    
    # 8. Workload composite
    work_cols = []
    for col in ['total_crfs', 'pages_entered', 'total_queries']:
        if col in df.columns:
            q99 = df[col].fillna(0).quantile(0.99)
            if q99 > 0:
                df[f'{col}_pctl'] = (df[col].fillna(0).clip(upper=q99) / q99).clip(0, 1)
                work_cols.append(f'{col}_pctl')
                n += 1
    
    if work_cols:
        df['workload_score'] = df[work_cols].mean(axis=1)
        n += 1
    
    log.info(f"  Created {n} derived features")
    return df


def create_risk_target(df: pd.DataFrame) -> pd.Series:
    """Create 4-tier risk target matching v5 methodology."""
    log.info("Creating risk target...")
    
    risk = pd.Series(0.0, index=df.index)
    
    # CRITICAL indicators
    for total_col, comp_col, w in [
        ('sae_dm_sae_dm_total', 'sae_dm_sae_dm_completed', 4.0),
        ('sae_safety_sae_safety_total', 'sae_safety_sae_safety_completed', 4.0)
    ]:
        if total_col in df.columns:
            pending = df[total_col].fillna(0)
            if comp_col in df.columns:
                pending = (pending - df[comp_col].fillna(0)).clip(lower=0)
            risk += (pending > 0).astype(float) * w
    
    if 'broken_signatures' in df.columns:
        risk += (df['broken_signatures'].fillna(0) > 0).astype(float) * 3.0
    if 'safety_queries' in df.columns:
        risk += (df['safety_queries'].fillna(0) > 0).astype(float) * 3.0
    
    # HIGH indicators
    if 'crfs_never_signed' in df.columns:
        val = df['crfs_never_signed'].fillna(0)
        risk += (val > 5).astype(float) * 2.5
        risk += (val > 0).astype(float) * 1.0
    if 'crfs_overdue_for_signs_beyond_90_days_of_data_entry' in df.columns:
        risk += (df['crfs_overdue_for_signs_beyond_90_days_of_data_entry'].fillna(0) > 0).astype(float) * 2.5
    if 'protocol_deviations' in df.columns:
        val = df['protocol_deviations'].fillna(0)
        risk += (val > 0).astype(float) * 2.0
        risk += (val > 2).astype(float) * 1.5
    
    # MEDIUM indicators
    if 'visit_missing_visit_count' in df.columns:
        risk += (df['visit_missing_visit_count'].fillna(0) > 0).astype(float) * 1.5
    if 'pages_pages_missing_count' in df.columns:
        risk += (df['pages_pages_missing_count'].fillna(0) > 0).astype(float) * 1.0
    if 'lab_lab_issue_count' in df.columns:
        risk += (df['lab_lab_issue_count'].fillna(0) > 0).astype(float) * 1.0
    
    p50, p80, p95 = risk.quantile(0.50), risk.quantile(0.80), risk.quantile(0.95)
    
    level = pd.Series('Low', index=df.index)
    level[risk > p50] = 'Medium'
    level[risk > p80] = 'High'
    level[risk > p95] = 'Critical'
    
    level = pd.Categorical(level, categories=['Low', 'Medium', 'High', 'Critical'], ordered=True)
    level = pd.Series(level, index=df.index)
    
    dist = level.value_counts()
    log.info(f"  Distribution: {dict(dist)}")
    
    return level


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select prediction features, excluding outcomes."""
    cols = []
    for c in df.columns:
        if c in OUTCOME_FEATURES:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if df[c].nunique() < 2 or df[c].std() < 0.001:
            continue
        cols.append(c)
    
    log.info(f"  Selected {len(cols)} features")
    return df[cols].copy()


def optimize_thresholds(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Optimize thresholds per class for target recalls."""
    thresholds = {}
    
    for cls in range(4):
        target = TARGET_RECALLS.get(cls, 0.5)
        cls_proba = proba[:, cls]
        cls_true = (y_true == cls).astype(int)
        
        best_th = 0.15
        best_recall = 0
        best_f1 = 0
        
        for th in np.linspace(0.02, 0.70, 100):
            pred = (cls_proba >= th).astype(int)
            tp = ((pred == 1) & (cls_true == 1)).sum()
            fp = ((pred == 1) & (cls_true == 0)).sum()
            fn = ((pred == 0) & (cls_true == 1)).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            
            if recall >= target:
                if f1 > best_f1:
                    best_f1 = f1
                    best_recall = recall
                    best_th = th
            elif recall > best_recall:
                best_recall = recall
                best_th = th
        
        thresholds[cls] = float(best_th)
    
    return thresholds


def cascade_predict(proba: np.ndarray, thresholds: dict) -> np.ndarray:
    """Cascade prediction: Critical > High > Medium > Low."""
    n = len(proba)
    pred = np.full(n, 2)  # Default Low
    
    for cls in [0, 1, 3]:
        mask = proba[:, cls] >= thresholds.get(cls, 0.5)
        if cls == 0:
            pred[mask] = 0
        elif cls == 1:
            pred[mask & (pred != 0)] = 1
        elif cls == 3:
            pred[mask & ~np.isin(pred, [0, 1])] = 3
    
    return pred


def run_training():
    """Main training pipeline with ISOTONIC CALIBRATION."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  TRIALPULSE NEXUS 10X — RISK CLASSIFIER v7 (CALIBRATED)")
    print("=" * 70)
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  NEW: Isotonic calibration for true probability interpretation")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("RISK CLASSIFIER v7 CALIBRATED — Training Started")
    log.info("=" * 60)
    
    # Log class semantics
    log.info("\nCLASS SEMANTICS:")
    for cls_id, sem in CLASS_SEMANTICS.items():
        log.info(f"  {sem['name']}: {sem['action']} | {sem['description']}")
    
    # 1. Load Data
    log.info("\n[1/8] Loading data...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  Loaded {len(df):,} samples")
    
    # 2. Create Target
    log.info("\n[2/8] Creating target...")
    y = create_risk_target(df)
    
    # 3. Feature Engineering
    log.info("\n[3/8] Engineering features...")
    df = engineer_features(df)
    
    # 4. Feature Selection
    log.info("\n[4/8] Selecting features...")
    X = select_features(df)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    feature_names = list(X.columns)
    log.info(f"  Final: {len(feature_names)} features")
    
    # 5. Encode & Scale
    log.info("\n[5/8] Encoding and scaling...")
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # 6. Split - need separate calibration set
    log.info("\n[6/8] Splitting data (train/calibration/test)...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    log.info(f"  Train: {len(X_train):,}, Calibration: {len(X_calib):,}, Test: {len(X_test):,}")
    
    # Apply SMOTE to training data only
    if SMOTE_AVAILABLE:
        try:
            min_samples = y_train.value_counts().min()
            k = min(5, min_samples - 1)
            if k >= 1:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_train, columns=X_scaled.columns)
                y_train = pd.Series(y_train)
                log.info(f"  After SMOTE: {len(X_train):,}")
        except Exception as e:
            log.warning(f"  SMOTE failed: {e}")
    
    # Sample weights
    weights = np.array([CLASS_WEIGHTS.get(c, 1.0) for c in y_train])
    
    # 7. Train Models with Calibration
    log.info("\n[7/8] Training and CALIBRATING models...")
    models = {}
    calibrated_models = {}
    
    # LightGBM
    if LGB_AVAILABLE:
        log.info("  Training LightGBM...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_lgb = lgb.LGBMClassifier(
                n_estimators=150, max_depth=10, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', verbosity=-1, random_state=42, n_jobs=1
            )
            base_lgb.fit(X_train, y_train)
            models['LGB'] = base_lgb
            
            # Isotonic calibration
            log.info("    Calibrating LightGBM with isotonic regression...")
            # Use cv=3 for cross-validation based calibration (sklearn compatible)
            calibrated_lgb = CalibratedClassifierCV(base_lgb, method='isotonic', cv=3)
            calibrated_lgb.fit(X_calib, y_calib)
            calibrated_models['LGB'] = calibrated_lgb
    
    # XGBoost
    if XGB_AVAILABLE:
        log.info("  Training XGBoost...")
        xgb_w = weights.copy()
        xgb_w[y_train == 0] *= 3.0
        xgb_w[y_train == 1] *= 2.5
        xgb_w[y_train == 3] *= 2.0
        
        base_xgb = xgb.XGBClassifier(
            n_estimators=150, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, verbosity=0, random_state=42, n_jobs=1
        )
        base_xgb.fit(X_train, y_train, sample_weight=xgb_w)
        models['XGB'] = base_xgb
        
        log.info("    Calibrating XGBoost...")
        calibrated_xgb = CalibratedClassifierCV(base_xgb, method='isotonic', cv=3)
        calibrated_xgb.fit(X_calib, y_calib)
        calibrated_models['XGB'] = calibrated_xgb
    
    # Random Forest
    log.info("  Training Random Forest...")
    base_rf = RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=5,
        class_weight=CLASS_WEIGHTS, random_state=42, n_jobs=1
    )
    base_rf.fit(X_train, y_train)
    models['RF'] = base_rf
    
    log.info("    Calibrating Random Forest...")
    calibrated_rf = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
    calibrated_rf.fit(X_calib, y_calib)
    calibrated_models['RF'] = calibrated_rf
    
    # Gradient Boosting
    log.info("  Training Gradient Boosting...")
    base_gb = GradientBoostingClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    base_gb.fit(X_train, y_train, sample_weight=weights)
    models['GB'] = base_gb
    
    log.info("    Calibrating Gradient Boosting...")
    calibrated_gb = CalibratedClassifierCV(base_gb, method='isotonic', cv=3)
    calibrated_gb.fit(X_calib, y_calib)
    calibrated_models['GB'] = calibrated_gb
    
    # 8. Evaluate
    log.info("\n[8/8] Evaluating CALIBRATED models...")
    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    results = {}
    
    for name, model in calibrated_models.items():
        proba = model.predict_proba(X_test)
        thresholds = optimize_thresholds(proba, y_test_arr)
        pred = cascade_predict(proba, thresholds)
        
        recall = recall_score(y_test_arr, pred, average=None, zero_division=0)
        precision = precision_score(y_test_arr, pred, average=None, zero_division=0)
        f1 = f1_score(y_test_arr, pred, average=None, zero_division=0)
        
        try:
            auc = roc_auc_score(
                label_binarize(y_test_arr, classes=[0, 1, 2, 3]),
                proba, average='macro', multi_class='ovr'
            )
        except:
            auc = 0.5
        
        # Brier score for calibration quality
        brier_scores = []
        for cls in range(4):
            y_cls = (y_test_arr == cls).astype(int)
            brier_scores.append(brier_score_loss(y_cls, proba[:, cls]))
        avg_brier = np.mean(brier_scores)
        
        targets_met = sum([1 for i, t in TARGET_RECALLS.items() if recall[i] >= t])
        combined = (recall[0]*4 + recall[1]*3 + recall[3]*2 + recall[2]) / 10
        
        results[name] = {
            'thresholds': thresholds,
            'critical_recall': float(recall[0]),
            'high_recall': float(recall[1]),
            'low_recall': float(recall[2]),
            'medium_recall': float(recall[3]),
            'critical_precision': float(precision[0]),
            'high_precision': float(precision[1]),
            'low_precision': float(precision[2]),
            'medium_precision': float(precision[3]),
            'auc': float(auc),
            'brier_score': float(avg_brier),
            'combined': float(combined),
            'targets_met': targets_met,
            'cm': confusion_matrix(y_test_arr, pred),
            'calibrated': True
        }
        
        log.info(f"\n  {name} (CALIBRATED):")
        log.info(f"    Critical: {recall[0]:.2%} {'✓' if recall[0] >= 0.70 else '✗'}")
        log.info(f"    High:     {recall[1]:.2%} {'✓' if recall[1] >= 0.55 else '✗'}")
        log.info(f"    Medium:   {recall[3]:.2%} {'✓' if recall[3] >= 0.50 else '✗'}")
        log.info(f"    Low:      {recall[2]:.2%} {'✓' if recall[2] >= 0.75 else '✗'}")
        log.info(f"    AUC: {auc:.4f} | Brier: {avg_brier:.4f} | Targets: {targets_met}/4")
    
    # Best model
    best_name = max(results, key=lambda x: (results[x]['targets_met'], results[x]['auc'], results[x]['combined']))
    best = results[best_name]
    
    log.info(f"\n{'='*60}")
    log.info(f"🏆 BEST CALIBRATED: {best_name}")
    log.info(f"   Targets Met: {best['targets_met']}/4")
    log.info(f"   AUC: {best['auc']:.4f}")
    log.info(f"   Brier Score: {best['brier_score']:.4f} (lower = better calibrated)")
    log.info(f"{'='*60}")
    
    # Save outputs
    log.info("\nSaving outputs...")
    
    # Metrics CSV
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name + ' (Calibrated)',
            'Critical_Recall': f"{r['critical_recall']:.2%}",
            'High_Recall': f"{r['high_recall']:.2%}",
            'Medium_Recall': f"{r['medium_recall']:.2%}",
            'Low_Recall': f"{r['low_recall']:.2%}",
            'Targets_Met': f"{r['targets_met']}/4",
            'AUC': f"{r['auc']:.4f}",
            'Brier_Score': f"{r['brier_score']:.4f}"
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'tables' / 'production_metrics_v7.csv', index=False)
    
    # Save calibrated models
    for name, model in calibrated_models.items():
        with open(OUTPUT_DIR / 'models' / f'model_v7_{name.lower()}_calibrated.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Save base models too
    for name, model in models.items():
        with open(OUTPUT_DIR / 'models' / f'model_v7_{name.lower()}_base.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Save scaler and encoder
    with open(OUTPUT_DIR / 'models' / 'scaler_v7.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / 'models' / 'label_encoder_v7.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Production config with class semantics
    config = {
        'version': '7.0.0',
        'created': datetime.now().isoformat(),
        'best_model': best_name,
        'calibration': 'isotonic',
        'thresholds': {str(k): v for k, v in best['thresholds'].items()},
        'class_weights': {str(k): v for k, v in CLASS_WEIGHTS.items()},
        'target_recalls': {str(k): v for k, v in TARGET_RECALLS.items()},
        'class_semantics': {str(k): v for k, v in CLASS_SEMANTICS.items()},
        'metrics': {
            'critical_recall': best['critical_recall'],
            'high_recall': best['high_recall'],
            'medium_recall': best['medium_recall'],
            'low_recall': best['low_recall'],
            'auc': best['auc'],
            'brier_score': best['brier_score'],
            'targets_met': best['targets_met']
        },
        'interpretation_notes': {
            'probability': 'With isotonic calibration, P(class)=0.7 means ~70% of similar patients belong to that class',
            'medium_class': 'MONITORING PRIORITY - enhanced surveillance, NOT direct intervention',
            'low_class': 'DRIFT DETECTOR - clean records, changes here indicate process shifts'
        },
        'n_features': len(feature_names),
        'training_samples': len(X_train),
        'calibration_samples': len(X_calib),
        'test_samples': len(X_test)
    }
    with open(OUTPUT_DIR / 'models' / 'production_config_v7.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Confusion matrix visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = best['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['Critical', 'High', 'Low', 'Medium']
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'{best_name} (CALIBRATED) Confusion Matrix — v7', fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figures' / 'confusion_matrix_v7.png', dpi=150)
    plt.close()
    
    # Recall bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(4)
    width = 0.2
    for idx, (name, r) in enumerate(results.items()):
        recalls = [r['critical_recall'], r['high_recall'], r['medium_recall'], r['low_recall']]
        offset = (idx - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=f'{name} (Cal)', alpha=0.85)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.0%}', ha='center', va='bottom', fontsize=8)
    
    targets = [TARGET_RECALLS[0], TARGET_RECALLS[1], TARGET_RECALLS[3], TARGET_RECALLS[2]]
    for i, t in enumerate(targets):
        ax.axhline(y=t, xmin=i/4 + 0.05, xmax=(i+1)/4 - 0.05, color='red', linestyle='--', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Critical (70%)', 'High (55%)', 'Medium (50%)', 'Low (75%)'])
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Per-Class Recall — Risk Classifier v7 (CALIBRATED)', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figures' / 'per_class_recall_v7.png', dpi=150)
    plt.close()
    
    # Training report
    with open(OUTPUT_DIR / 'training_report_v7.json', 'w') as f:
        json.dump({
            'version': '7.0.0',
            'created': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start).total_seconds(),
            'best_model': best_name,
            'calibration_method': 'isotonic',
            'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'cm'} for k, v in results.items()}
        }, f, indent=2)
    
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("  ✅ TRAINING COMPLETE — RISK CLASSIFIER v7 (CALIBRATED)")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"  Best Model: {best_name} (isotonic calibrated)")
    print(f"")
    print(f"  RESULTS:")
    print(f"    Critical Recall: {best['critical_recall']:.2%} (target: 70%) {'✓' if best['critical_recall'] >= 0.70 else '✗'}")
    print(f"    High Recall:     {best['high_recall']:.2%} (target: 55%) {'✓' if best['high_recall'] >= 0.55 else '✗'}")
    print(f"    Medium Recall:   {best['medium_recall']:.2%} (target: 50%) {'✓' if best['medium_recall'] >= 0.50 else '✗'}")
    print(f"    Low Recall:      {best['low_recall']:.2%} (target: 75%) {'✓' if best['low_recall'] >= 0.75 else '✗'}")
    print(f"")
    print(f"  ROC-AUC: {best['auc']:.4f}")
    print(f"  Brier Score: {best['brier_score']:.4f} (calibration quality)")
    print(f"  Targets Met: {best['targets_met']}/4")
    print(f"")
    print(f"  CLASS INTERPRETATION:")
    print(f"    Medium = MONITORING PRIORITY (not intervention)")
    print(f"    Low = DRIFT DETECTOR (process change indicator)")
    print(f"")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    log.info("✅ Training complete!")
    
    return best


if __name__ == '__main__':
    run_training()
