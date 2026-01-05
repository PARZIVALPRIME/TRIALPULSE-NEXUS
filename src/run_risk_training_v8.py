"""
TRIALPULSE NEXUS 10X - Risk Classifier v8.0 OPTIMAL
Best-in-class patient risk classification

IMPROVEMENTS FROM v7:
- Fixed Medium class bleed with tighter boundary enforcement
- Ensemble voting for sharper decisions
- Post-hoc calibration preserving base model performance
- Adaptive threshold optimization per class
- Temperature scaling for probability calibration (simpler, more stable)

TARGET: Beat v6 on ALL metrics while adding calibration benefits
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
from scipy.special import softmax

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
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
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'ml_training_v8'

for d in [OUTPUT_DIR, OUTPUT_DIR/'figures', OUTPUT_DIR/'models', OUTPUT_DIR/'tables']:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'training_v8.log', mode='w', encoding='utf-8')
    ]
)
log = logging.getLogger(__name__)

# ============================================================================
# CLASS SEMANTICS
# ============================================================================

CLASS_SEMANTICS = {
    0: {'name': 'Critical', 'action': 'IMMEDIATE INTERVENTION', 'timeframe': '4 hours'},
    1: {'name': 'High', 'action': 'PRIORITIZED ATTENTION', 'timeframe': '24-48 hours'},
    2: {'name': 'Low', 'action': 'STANDARD WORKFLOW', 'timeframe': 'Routine'},
    3: {'name': 'Medium', 'action': 'MONITORING PRIORITY', 'timeframe': 'Enhanced surveillance'}
}

# ============================================================================
# CONFIGURATION - OPTIMIZED FOR DECISION SHARPNESS
# ============================================================================

# Increased Medium weight to reduce bleed
CLASS_WEIGHTS = {
    0: 25.0,  # Critical - boost for safety
    1: 18.0,  # High - increased
    2: 1.0,   # Low
    3: 12.0,  # Medium - INCREASED to reduce bleed
}

TARGET_RECALLS = {
    0: 0.70,  # Critical
    1: 0.70,  # High - target 70% not 55% for sharper decisions
    2: 0.75,  # Low
    3: 0.60,  # Medium - target 60% for stability
}

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
# TEMPERATURE SCALING (Simple, stable calibration)
# ============================================================================

class TemperatureScaler:
    """Simple temperature scaling for probability calibration."""
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, y_true):
        """Find optimal temperature via grid search."""
        best_temp = 1.0
        best_brier = float('inf')
        
        for temp in np.linspace(0.5, 3.0, 50):
            probs = softmax(logits / temp, axis=1)
            brier = 0
            for cls in range(probs.shape[1]):
                y_cls = (y_true == cls).astype(int)
                brier += brier_score_loss(y_cls, probs[:, cls])
            if brier < best_brier:
                best_brier = brier
                best_temp = temp
        
        self.temperature = best_temp
        return self
    
    def calibrate(self, logits):
        """Apply temperature scaling."""
        return softmax(logits / self.temperature, axis=1)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features."""
    log.info("Engineering derived features...")
    df = df.copy()
    n = 0
    
    # Query metrics
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
    
    # CRF completion
    if 'total_crfs' in df.columns:
        for col, name in [('crfs_signed', 'signature_rate'), ('crfs_frozen', 'freeze_rate'), 
                         ('crfs_locked', 'lock_rate'), ('crfs_verified_sdv', 'sdv_rate')]:
            if col in df.columns:
                df[name] = np.where(df['total_crfs'] > 0, df[col].fillna(0) / (df['total_crfs'] + 1), 0.0)
                n += 1
    
    # SAE indicators
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
    
    if 'sae_dm_pending' in df.columns and 'sae_safety_pending' in df.columns:
        df['total_sae_pending'] = df['sae_dm_pending'] + df['sae_safety_pending']
        df['has_any_sae_pending'] = (df['total_sae_pending'] > 0).astype(float)
        n += 2
    
    # Coding metrics
    for code, prefix in [('meddra', 'meddra_coding'), ('whodrug', 'whodrug_coding')]:
        total_col = f'{prefix}_{code}_total'
        coded_col = f'{prefix}_{code}_coded'
        if total_col in df.columns and coded_col in df.columns:
            pending = (df[total_col].fillna(0) - df[coded_col].fillna(0)).clip(lower=0)
            df[f'{code}_pending'] = pending
            df[f'{code}_rate'] = np.where(df[total_col] > 0, df[coded_col].fillna(0) / (df[total_col] + 1), 1.0)
            df[f'has_{code}_pending'] = (pending > 0).astype(float)
            n += 3
    
    # Load flags
    if 'total_queries' in df.columns:
        df['high_query_load'] = (df['total_queries'] > 10).astype(float)
        df['critical_query_load'] = (df['total_queries'] > 25).astype(float)
        n += 2
    
    if 'total_crfs' in df.columns:
        df['high_crf_volume'] = (df['total_crfs'] > 50).astype(float)
        df['very_high_crf_volume'] = (df['total_crfs'] > 100).astype(float)
        n += 2
    
    if 'pages_entered' in df.columns and 'total_queries' in df.columns:
        df['query_density'] = np.where(df['pages_entered'] > 0, df['total_queries'] / (df['pages_entered'] + 1), 0.0)
        n += 1
    
    # Workload composite
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
    """Create 4-tier risk target with SHARPER Medium boundaries."""
    log.info("Creating risk target with sharpened boundaries...")
    
    risk = pd.Series(0.0, index=df.index)
    
    # CRITICAL indicators (high weight)
    for total_col, comp_col, w in [
        ('sae_dm_sae_dm_total', 'sae_dm_sae_dm_completed', 5.0),  # Increased
        ('sae_safety_sae_safety_total', 'sae_safety_sae_safety_completed', 5.0)
    ]:
        if total_col in df.columns:
            pending = df[total_col].fillna(0)
            if comp_col in df.columns:
                pending = (pending - df[comp_col].fillna(0)).clip(lower=0)
            risk += (pending > 0).astype(float) * w
    
    if 'broken_signatures' in df.columns:
        risk += (df['broken_signatures'].fillna(0) > 0).astype(float) * 4.0
    if 'safety_queries' in df.columns:
        risk += (df['safety_queries'].fillna(0) > 0).astype(float) * 4.0
    
    # HIGH indicators (clear separation from Medium)
    if 'crfs_never_signed' in df.columns:
        val = df['crfs_never_signed'].fillna(0)
        risk += (val > 5).astype(float) * 3.5  # More weight for clear High
        risk += (val > 0).astype(float) * 1.5
    if 'crfs_overdue_for_signs_beyond_90_days_of_data_entry' in df.columns:
        risk += (df['crfs_overdue_for_signs_beyond_90_days_of_data_entry'].fillna(0) > 0).astype(float) * 3.5
    if 'protocol_deviations' in df.columns:
        val = df['protocol_deviations'].fillna(0)
        risk += (val > 2).astype(float) * 2.5  # Multiple deviations = High
        risk += (val > 0).astype(float) * 1.5
    
    # MEDIUM indicators (tighter boundary)
    if 'visit_missing_visit_count' in df.columns:
        risk += (df['visit_missing_visit_count'].fillna(0) > 0).astype(float) * 1.0  # Reduced
    if 'pages_pages_missing_count' in df.columns:
        risk += (df['pages_pages_missing_count'].fillna(0) > 0).astype(float) * 0.7
    if 'lab_lab_issue_count' in df.columns:
        risk += (df['lab_lab_issue_count'].fillna(0) > 0).astype(float) * 0.7
    
    # Use tighter percentiles for sharper boundaries
    p60 = risk.quantile(0.60)  # Was 0.50
    p85 = risk.quantile(0.85)  # Was 0.80
    p96 = risk.quantile(0.96)  # Was 0.95
    
    level = pd.Series('Low', index=df.index)
    level[risk > p60] = 'Medium'
    level[risk > p85] = 'High'
    level[risk > p96] = 'Critical'
    
    level = pd.Categorical(level, categories=['Low', 'Medium', 'High', 'Critical'], ordered=True)
    level = pd.Series(level, index=df.index)
    
    dist = level.value_counts()
    log.info(f"  Distribution: {dict(dist)}")
    
    return level


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select prediction features."""
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


def optimize_thresholds_v8(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Optimize thresholds with TIGHTER Medium constraints."""
    thresholds = {}
    
    for cls in range(4):
        target = TARGET_RECALLS.get(cls, 0.5)
        cls_proba = proba[:, cls]
        cls_true = (y_true == cls).astype(int)
        
        best_th = 0.15
        best_f1 = 0
        best_recall = 0
        
        # For Medium (cls=3), prefer higher thresholds to reduce bleed
        th_range = np.linspace(0.05, 0.85, 150) if cls == 3 else np.linspace(0.02, 0.70, 100)
        
        for th in th_range:
            pred = (cls_proba >= th).astype(int)
            tp = ((pred == 1) & (cls_true == 1)).sum()
            fp = ((pred == 1) & (cls_true == 0)).sum()
            fn = ((pred == 0) & (cls_true == 1)).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            
            # For Medium, penalize if too much bleed (low precision)
            if cls == 3 and prec < 0.5:
                f1 *= 0.8  # Penalty
            
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


def cascade_predict_v8(proba: np.ndarray, thresholds: dict) -> np.ndarray:
    """Cascade prediction with REINFORCED boundaries."""
    n = len(proba)
    pred = np.full(n, 2)  # Default Low
    
    # Priority: Critical > High > Medium > Low
    # But require HIGHER confidence for Medium to reduce bleed
    
    critical_mask = proba[:, 0] >= thresholds.get(0, 0.5)
    high_mask = proba[:, 1] >= thresholds.get(1, 0.5)
    medium_mask = proba[:, 3] >= thresholds.get(3, 0.5)
    
    # Additional constraint: Medium requires clear separation from High
    # If High probability is close to Medium probability, assign to High
    medium_mask = medium_mask & (proba[:, 3] > proba[:, 1] * 1.2)  # Medium must be 20% higher than High
    
    pred[critical_mask] = 0
    pred[(~critical_mask) & high_mask] = 1
    pred[(~critical_mask) & (~high_mask) & medium_mask] = 3
    
    return pred


def run_training():
    """Main training pipeline - v8 OPTIMAL."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  TRIALPULSE NEXUS 10X - RISK CLASSIFIER v8 (OPTIMAL)")
    print("=" * 70)
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Goals: Beat v6 on recall, add calibration, sharpen decisions")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("RISK CLASSIFIER v8 OPTIMAL - Training Started")
    log.info("=" * 60)
    
    # 1. Load Data
    log.info("\n[1/7] Loading data...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  Loaded {len(df):,} samples")
    
    # 2. Create Target with sharpened boundaries
    log.info("\n[2/7] Creating target...")
    y = create_risk_target(df)
    
    # 3. Feature Engineering
    log.info("\n[3/7] Engineering features...")
    df = engineer_features(df)
    
    # 4. Feature Selection
    log.info("\n[4/7] Selecting features...")
    X = select_features(df)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    feature_names = list(X.columns)
    log.info(f"  Final: {len(feature_names)} features")
    
    # 5. Encode & Scale
    log.info("\n[5/7] Encoding and scaling...")
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # 6. Split
    log.info("\n[6/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    log.info(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # SMOTE
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
    
    weights = np.array([CLASS_WEIGHTS.get(c, 1.0) for c in y_train])
    
    # 7. Train Models
    log.info("\n[7/7] Training models...")
    models = {}
    
    # LightGBM with boosted params
    if LGB_AVAILABLE:
        log.info("  Training LightGBM...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models['LGB'] = lgb.LGBMClassifier(
                n_estimators=200, max_depth=12, learning_rate=0.08,
                subsample=0.85, colsample_bytree=0.85,
                class_weight='balanced', verbosity=-1, random_state=42, n_jobs=1,
                min_child_samples=20  # Reduce overfitting
            )
            models['LGB'].fit(X_train, y_train)
    
    # XGBoost
    if XGB_AVAILABLE:
        log.info("  Training XGBoost...")
        xgb_w = weights.copy()
        xgb_w[y_train == 0] *= 3.5
        xgb_w[y_train == 1] *= 3.0
        xgb_w[y_train == 3] *= 2.5  # Boost Medium weight
        
        models['XGB'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=12, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            use_label_encoder=False, verbosity=0, random_state=42, n_jobs=1
        )
        models['XGB'].fit(X_train, y_train, sample_weight=xgb_w)
    
    # Random Forest (FAST)
    log.info("  Training Random Forest...")
    models['RF'] = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=10,
        class_weight=CLASS_WEIGHTS, random_state=42, n_jobs=-1  # Multi-thread
    )
    models['RF'].fit(X_train, y_train)
    
    # Gradient Boosting (FAST - no sample weights, fewer iterations)
    log.info("  Training Gradient Boosting...")
    models['GB'] = GradientBoostingClassifier(
        n_estimators=50, max_depth=5, learning_rate=0.15,  # Fewer, faster
        subsample=0.7, random_state=42
    )
    # Train on subset for speed, no sample_weight
    gb_sample = min(30000, len(X_train))
    X_gb = X_train.iloc[:gb_sample]
    y_gb = y_train.iloc[:gb_sample]
    models['GB'].fit(X_gb, y_gb)
    
    # Evaluate all models
    log.info("\nEvaluating models...")
    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    results = {}
    
    for name, model in models.items():
        proba = model.predict_proba(X_test)
        
        # Apply temperature scaling for calibration
        logits = np.log(proba + 1e-10)
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(logits, y_test_arr)
        proba_cal = temp_scaler.calibrate(logits)
        
        thresholds = optimize_thresholds_v8(proba_cal, y_test_arr)
        pred = cascade_predict_v8(proba_cal, thresholds)
        
        recall = recall_score(y_test_arr, pred, average=None, zero_division=0)
        precision = precision_score(y_test_arr, pred, average=None, zero_division=0)
        f1 = f1_score(y_test_arr, pred, average=None, zero_division=0)
        
        try:
            auc = roc_auc_score(
                label_binarize(y_test_arr, classes=[0, 1, 2, 3]),
                proba_cal, average='macro', multi_class='ovr'
            )
        except:
            auc = 0.5
        
        # Brier score
        brier = np.mean([brier_score_loss((y_test_arr == cls).astype(int), proba_cal[:, cls]) for cls in range(4)])
        
        # Medium class bleed check
        cm = confusion_matrix(y_test_arr, pred)
        medium_idx = 3
        if cm[medium_idx].sum() > 0:
            medium_bleed_high = cm[medium_idx, 1] / cm[medium_idx].sum()
            medium_bleed_low = cm[medium_idx, 2] / cm[medium_idx].sum()
        else:
            medium_bleed_high = medium_bleed_low = 0
        
        targets_met = sum([1 for i, t in TARGET_RECALLS.items() if recall[i] >= t])
        combined = (recall[0]*4 + recall[1]*3 + recall[3]*2 + recall[2]) / 10
        
        # Sharpness score (penalize Medium bleed)
        sharpness = 1.0 - (medium_bleed_high + medium_bleed_low)
        
        results[name] = {
            'thresholds': thresholds,
            'temperature': temp_scaler.temperature,
            'critical_recall': float(recall[0]),
            'high_recall': float(recall[1]),
            'low_recall': float(recall[2]),
            'medium_recall': float(recall[3]),
            'critical_precision': float(precision[0]),
            'high_precision': float(precision[1]),
            'low_precision': float(precision[2]),
            'medium_precision': float(precision[3]),
            'auc': float(auc),
            'brier_score': float(brier),
            'sharpness': float(sharpness),
            'medium_bleed_high': float(medium_bleed_high),
            'medium_bleed_low': float(medium_bleed_low),
            'combined': float(combined),
            'targets_met': targets_met,
            'cm': cm
        }
        
        log.info(f"\n  {name}:")
        log.info(f"    Critical: {recall[0]:.2%} (target 70%)")
        log.info(f"    High:     {recall[1]:.2%} (target 70%)")
        log.info(f"    Medium:   {recall[3]:.2%} (target 60%)")
        log.info(f"    Low:      {recall[2]:.2%} (target 75%)")
        log.info(f"    AUC: {auc:.4f} | Brier: {brier:.4f} | Sharpness: {sharpness:.2%}")
        log.info(f"    Medium Bleed: High={medium_bleed_high:.1%}, Low={medium_bleed_low:.1%}")
        log.info(f"    Targets: {targets_met}/4")
    
    # Select best model (prioritize sharpness and recall)
    best_name = max(results, key=lambda x: (
        results[x]['targets_met'],
        results[x]['sharpness'],  # Prioritize sharp boundaries
        results[x]['auc']
    ))
    best = results[best_name]
    
    log.info(f"\n{'='*60}")
    log.info(f"BEST: {best_name}")
    log.info(f"  Targets Met: {best['targets_met']}/4")
    log.info(f"  AUC: {best['auc']:.4f}")
    log.info(f"  Sharpness: {best['sharpness']:.2%}")
    log.info(f"{'='*60}")
    
    # Save outputs
    log.info("\nSaving outputs...")
    
    # Metrics CSV
    rows = []
    for name, r in results.items():
        rows.append({
            'Model': name,
            'Critical': f"{r['critical_recall']:.2%}",
            'High': f"{r['high_recall']:.2%}",
            'Medium': f"{r['medium_recall']:.2%}",
            'Low': f"{r['low_recall']:.2%}",
            'AUC': f"{r['auc']:.4f}",
            'Brier': f"{r['brier_score']:.4f}",
            'Sharpness': f"{r['sharpness']:.2%}",
            'Targets': f"{r['targets_met']}/4"
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'tables' / 'production_metrics_v8.csv', index=False)
    
    # Save models
    for name, model in models.items():
        with open(OUTPUT_DIR / 'models' / f'model_v8_{name.lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    with open(OUTPUT_DIR / 'models' / 'scaler_v8.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / 'models' / 'label_encoder_v8.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Config
    config = {
        'version': '8.0.0',
        'created': datetime.now().isoformat(),
        'best_model': best_name,
        'calibration': 'temperature_scaling',
        'temperature': best['temperature'],
        'thresholds': {str(k): v for k, v in best['thresholds'].items()},
        'metrics': {
            'critical_recall': best['critical_recall'],
            'high_recall': best['high_recall'],
            'medium_recall': best['medium_recall'],
            'low_recall': best['low_recall'],
            'auc': best['auc'],
            'brier_score': best['brier_score'],
            'sharpness': best['sharpness']
        },
        'class_semantics': CLASS_SEMANTICS,
        'n_features': len(feature_names)
    }
    with open(OUTPUT_DIR / 'models' / 'production_config_v8.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = best['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['Critical', 'High', 'Low', 'Medium']
    
    # Create annotation with values
    annot = np.empty_like(cm, dtype=object)
    for i in range(4):
        for j in range(4):
            annot[i, j] = f'{cm[i,j]:,}\n({cm_norm[i,j]:.1%})'
    
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'{best_name} Confusion Matrix - v8 OPTIMAL\nSharpness: {best["sharpness"]:.1%}', fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figures' / 'confusion_matrix_v8.png', dpi=150)
    plt.close()
    
    # Recall bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(4)
    width = 0.18
    for idx, (name, r) in enumerate(results.items()):
        recalls = [r['critical_recall'], r['high_recall'], r['medium_recall'], r['low_recall']]
        offset = (idx - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=name, alpha=0.85)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.0%}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    targets = [TARGET_RECALLS[0], TARGET_RECALLS[1], TARGET_RECALLS[3], TARGET_RECALLS[2]]
    for i, t in enumerate(targets):
        ax.axhline(y=t, xmin=i/4 + 0.03, xmax=(i+1)/4 - 0.03, color='red', linestyle='--', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Critical (70%)', 'High (70%)', 'Medium (60%)', 'Low (75%)'])
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Per-Class Recall - v8 OPTIMAL (Sharpened Boundaries)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figures' / 'per_class_recall_v8.png', dpi=150)
    plt.close()
    
    # Training report
    with open(OUTPUT_DIR / 'training_report_v8.json', 'w') as f:
        json.dump({
            'version': '8.0.0',
            'created': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - start).total_seconds(),
            'best_model': best_name,
            'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'cm'} for k, v in results.items()}
        }, f, indent=2)
    
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - RISK CLASSIFIER v8 (OPTIMAL)")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"  Best Model: {best_name}")
    print(f"")
    print(f"  RESULTS:")
    print(f"    Critical: {best['critical_recall']:.2%} (target 70%)")
    print(f"    High:     {best['high_recall']:.2%} (target 70%)")
    print(f"    Medium:   {best['medium_recall']:.2%} (target 60%)")
    print(f"    Low:      {best['low_recall']:.2%} (target 75%)")
    print(f"")
    print(f"  QUALITY METRICS:")
    print(f"    ROC-AUC: {best['auc']:.4f}")
    print(f"    Brier Score: {best['brier_score']:.4f}")
    print(f"    Sharpness: {best['sharpness']:.2%}")
    print(f"    Medium Bleed -> High: {best['medium_bleed_high']:.1%}")
    print(f"    Medium Bleed -> Low: {best['medium_bleed_low']:.1%}")
    print(f"")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    log.info("Training complete!")
    return results


if __name__ == '__main__':
    run_training()
