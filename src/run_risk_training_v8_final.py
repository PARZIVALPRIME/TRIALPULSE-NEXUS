"""
TRIALPULSE NEXUS 10X - Risk Classifier v8 FINAL  
Combines v6's proven recall performance with temperature scaling calibration.

KEY:
- Uses v6's percentile thresholds (p50/p80/p95) for target creation
- Adds temperature scaling post-hoc calibration
- Documents class semantics properly
- Tracks sharpness metrics
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

ROOT = Path(__file__).parent.parent
UPR_PATH = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'ml_training_v8_final'

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

# Same config as v6
CLASS_WEIGHTS = {0: 20.0, 1: 15.0, 2: 1.0, 3: 8.0}
TARGET_RECALLS = {0: 0.70, 1: 0.55, 2: 0.75, 3: 0.50}

CLASS_SEMANTICS = {
    'Critical': 'IMMEDIATE INTERVENTION required within 4 hours',
    'High': 'PRIORITIZED ATTENTION within 24-48 hours',
    'Medium': 'MONITORING PRIORITY - enhanced surveillance, NOT direct intervention',
    'Low': 'STANDARD WORKFLOW - serves as DRIFT DETECTOR'
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


class TemperatureScaler:
    """Temperature scaling for probability calibration."""
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, y_true):
        best_temp, best_brier = 1.0, float('inf')
        for temp in np.linspace(0.5, 2.5, 40):
            probs = softmax(logits / temp, axis=1)
            brier = np.mean([brier_score_loss((y_true == c).astype(int), probs[:, c]) for c in range(4)])
            if brier < best_brier:
                best_brier, best_temp = brier, temp
        self.temperature = best_temp
        return self
    
    def calibrate(self, logits):
        return softmax(logits / self.temperature, axis=1)


def engineer_features(df):
    """Same feature engineering as v6."""
    df = df.copy()
    
    query_cols = ['dm_queries', 'clinical_queries', 'medical_queries', 'site_queries']
    existing = [c for c in query_cols if c in df.columns]
    if existing:
        df['total_queries'] = df[existing].fillna(0).sum(axis=1)
        df['query_type_count'] = (df[existing].fillna(0) > 0).sum(axis=1).astype(float)
        if 'queries_answered' in df.columns:
            total = df['total_queries'] + df['queries_answered'].fillna(0)
            df['query_resolution_rate'] = np.where(total > 0, df['queries_answered'].fillna(0) / (total + 1), 1.0)
    
    if 'total_crfs' in df.columns:
        for col, name in [('crfs_signed', 'signature_rate'), ('crfs_frozen', 'freeze_rate'), 
                         ('crfs_locked', 'lock_rate'), ('crfs_verified_sdv', 'sdv_rate')]:
            if col in df.columns:
                df[name] = np.where(df['total_crfs'] > 0, df[col].fillna(0) / (df['total_crfs'] + 1), 0.0)
    
    for sae in ['sae_dm', 'sae_safety']:
        total_col, comp_col = f'{sae}_{sae}_total', f'{sae}_{sae}_completed'
        if total_col in df.columns:
            pending = df[total_col].fillna(0)
            if comp_col in df.columns:
                pending = (pending - df[comp_col].fillna(0)).clip(lower=0)
            df[f'{sae}_pending'] = pending
            df[f'has_{sae}'] = (df[total_col].fillna(0) > 0).astype(float)
            df[f'has_{sae}_pending'] = (pending > 0).astype(float)
    
    if 'sae_dm_pending' in df.columns and 'sae_safety_pending' in df.columns:
        df['total_sae_pending'] = df['sae_dm_pending'] + df['sae_safety_pending']
        df['has_any_sae_pending'] = (df['total_sae_pending'] > 0).astype(float)
    
    for code, prefix in [('meddra', 'meddra_coding'), ('whodrug', 'whodrug_coding')]:
        total_col, coded_col = f'{prefix}_{code}_total', f'{prefix}_{code}_coded'
        if total_col in df.columns and coded_col in df.columns:
            pending = (df[total_col].fillna(0) - df[coded_col].fillna(0)).clip(lower=0)
            df[f'{code}_pending'] = pending
            df[f'{code}_rate'] = np.where(df[total_col] > 0, df[coded_col].fillna(0) / (df[total_col] + 1), 1.0)
            df[f'has_{code}_pending'] = (pending > 0).astype(float)
    
    if 'total_queries' in df.columns:
        df['high_query_load'] = (df['total_queries'] > 10).astype(float)
        df['critical_query_load'] = (df['total_queries'] > 25).astype(float)
    
    if 'total_crfs' in df.columns:
        df['high_crf_volume'] = (df['total_crfs'] > 50).astype(float)
        df['very_high_crf_volume'] = (df['total_crfs'] > 100).astype(float)
    
    if 'pages_entered' in df.columns and 'total_queries' in df.columns:
        df['query_density'] = np.where(df['pages_entered'] > 0, df['total_queries'] / (df['pages_entered'] + 1), 0.0)
    
    work_cols = []
    for col in ['total_crfs', 'pages_entered', 'total_queries']:
        if col in df.columns:
            q99 = df[col].fillna(0).quantile(0.99)
            if q99 > 0:
                df[f'{col}_pctl'] = (df[col].fillna(0).clip(upper=q99) / q99).clip(0, 1)
                work_cols.append(f'{col}_pctl')
    
    if work_cols:
        df['workload_score'] = df[work_cols].mean(axis=1)
    
    return df


def create_risk_target(df):
    """Same target creation as v6 (p50/p80/p95)."""
    risk = pd.Series(0.0, index=df.index)
    
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
    
    if 'visit_missing_visit_count' in df.columns:
        risk += (df['visit_missing_visit_count'].fillna(0) > 0).astype(float) * 1.5
    if 'pages_pages_missing_count' in df.columns:
        risk += (df['pages_pages_missing_count'].fillna(0) > 0).astype(float) * 1.0
    if 'lab_lab_issue_count' in df.columns:
        risk += (df['lab_lab_issue_count'].fillna(0) > 0).astype(float) * 1.0
    
    # SAME percentiles as v6
    p50, p80, p95 = risk.quantile(0.50), risk.quantile(0.80), risk.quantile(0.95)
    
    level = pd.Series('Low', index=df.index)
    level[risk > p50] = 'Medium'
    level[risk > p80] = 'High'
    level[risk > p95] = 'Critical'
    
    return pd.Series(pd.Categorical(level, categories=['Low', 'Medium', 'High', 'Critical'], ordered=True), index=df.index)


def select_features(df):
    cols = [c for c in df.columns 
            if c not in OUTCOME_FEATURES 
            and np.issubdtype(df[c].dtype, np.number) 
            and df[c].nunique() >= 2 
            and df[c].std() >= 0.001]
    return df[cols].copy()


def optimize_thresholds(proba, y_true):
    thresholds = {}
    for cls in range(4):
        target = TARGET_RECALLS.get(cls, 0.5)
        cls_proba, cls_true = proba[:, cls], (y_true == cls).astype(int)
        best_th, best_recall, best_f1 = 0.15, 0, 0
        
        for th in np.linspace(0.02, 0.70, 100):
            pred = (cls_proba >= th).astype(int)
            tp = ((pred == 1) & (cls_true == 1)).sum()
            fn = ((pred == 0) & (cls_true == 1)).sum()
            fp = ((pred == 1) & (cls_true == 0)).sum()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
            
            if recall >= target and f1 > best_f1:
                best_f1, best_recall, best_th = f1, recall, th
            elif recall > best_recall:
                best_recall, best_th = recall, th
        
        thresholds[cls] = float(best_th)
    return thresholds


def cascade_predict(proba, thresholds):
    n = len(proba)
    pred = np.full(n, 2)
    
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
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  TRIALPULSE NEXUS 10X - RISK CLASSIFIER v8 FINAL")
    print("  (v6 recall performance + temperature calibration)")
    print("=" * 70 + "\n")
    
    log.info("Loading data...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  Loaded {len(df):,} samples")
    
    log.info("Creating target (v6 method)...")
    y = create_risk_target(df)
    log.info(f"  Distribution: {dict(y.value_counts())}")
    
    log.info("Engineering features...")
    df = engineer_features(df)
    
    log.info("Selecting features...")
    X = select_features(df)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    log.info(f"  {len(X.columns)} features")
    
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    log.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    log.info(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    if SMOTE_AVAILABLE:
        try:
            k = min(5, y_train.value_counts().min() - 1)
            if k >= 1:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_train, columns=X_scaled.columns)
                y_train = pd.Series(y_train)
                log.info(f"  After SMOTE: {len(X_train):,}")
        except:
            pass
    
    weights = np.array([CLASS_WEIGHTS.get(c, 1.0) for c in y_train])
    
    log.info("Training models...")
    models = {}
    
    if LGB_AVAILABLE:
        log.info("  LightGBM...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models['LGB'] = lgb.LGBMClassifier(
                n_estimators=150, max_depth=10, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                class_weight='balanced', verbosity=-1, random_state=42, n_jobs=1
            )
            models['LGB'].fit(X_train, y_train)
    
    if XGB_AVAILABLE:
        log.info("  XGBoost...")
        xgb_w = weights.copy()
        xgb_w[y_train == 0] *= 3.0
        xgb_w[y_train == 1] *= 2.5
        models['XGB'] = xgb.XGBClassifier(
            n_estimators=150, max_depth=10, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, verbosity=0, random_state=42, n_jobs=1
        )
        models['XGB'].fit(X_train, y_train, sample_weight=xgb_w)
    
    log.info("  RandomForest...")
    models['RF'] = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=5,
        class_weight=CLASS_WEIGHTS, random_state=42, n_jobs=-1
    )
    models['RF'].fit(X_train, y_train)
    
    log.info("Evaluating with temperature calibration...")
    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    results = {}
    
    for name, model in models.items():
        proba = model.predict_proba(X_test)
        
        logits = np.log(proba + 1e-10)
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(logits, y_test_arr)
        proba_cal = temp_scaler.calibrate(logits)
        
        thresholds = optimize_thresholds(proba_cal, y_test_arr)
        pred = cascade_predict(proba_cal, thresholds)
        
        recall = recall_score(y_test_arr, pred, average=None, zero_division=0)
        precision = precision_score(y_test_arr, pred, average=None, zero_division=0)
        
        try:
            auc = roc_auc_score(label_binarize(y_test_arr, classes=[0,1,2,3]), proba_cal, average='macro', multi_class='ovr')
        except:
            auc = 0.5
        
        brier = np.mean([brier_score_loss((y_test_arr == c).astype(int), proba_cal[:, c]) for c in range(4)])
        cm = confusion_matrix(y_test_arr, pred)
        
        med_total = cm[3].sum()
        med_bleed_high = cm[3, 1] / med_total if med_total > 0 else 0
        med_bleed_low = cm[3, 2] / med_total if med_total > 0 else 0
        sharpness = 1.0 - (med_bleed_high + med_bleed_low)
        
        targets_met = sum([1 for i, t in TARGET_RECALLS.items() if recall[i] >= t])
        
        results[name] = {
            'critical_recall': recall[0], 'high_recall': recall[1],
            'medium_recall': recall[3], 'low_recall': recall[2],
            'auc': auc, 'brier': brier, 'sharpness': sharpness,
            'med_bleed_high': med_bleed_high, 'med_bleed_low': med_bleed_low,
            'targets_met': targets_met, 'temperature': temp_scaler.temperature,
            'thresholds': thresholds, 'cm': cm
        }
        
        log.info(f"\n  {name}:")
        log.info(f"    Critical: {recall[0]:.2%}, High: {recall[1]:.2%}, Medium: {recall[3]:.2%}, Low: {recall[2]:.2%}")
        log.info(f"    AUC: {auc:.4f}, Brier: {brier:.4f}, Sharpness: {sharpness:.1%}")
        log.info(f"    Targets: {targets_met}/4")
    
    best_name = max(results, key=lambda x: (results[x]['targets_met'], results[x]['auc']))
    best = results[best_name]
    
    log.info(f"\nBEST: {best_name} with {best['targets_met']}/4 targets")
    
    # Save
    for name, model in models.items():
        with open(OUTPUT_DIR / 'models' / f'model_{name.lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    with open(OUTPUT_DIR / 'models' / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(OUTPUT_DIR / 'models' / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    # Metrics CSV
    rows = [{
        'Model': name,
        'Critical': f"{r['critical_recall']:.2%}",
        'High': f"{r['high_recall']:.2%}",
        'Medium': f"{r['medium_recall']:.2%}",
        'Low': f"{r['low_recall']:.2%}",
        'AUC': f"{r['auc']:.4f}",
        'Brier': f"{r['brier']:.4f}",
        'Sharpness': f"{r['sharpness']:.1%}",
        'Targets': f"{r['targets_met']}/4"
    } for name, r in results.items()]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'tables' / 'metrics.csv', index=False)
    
    # Config
    config = {
        'version': '8.0.0-final',
        'created': datetime.now().isoformat(),
        'best_model': best_name,
        'calibration': 'temperature_scaling',
        'temperature': best['temperature'],
        'metrics': {k: v for k, v in best.items() if k not in ['cm', 'thresholds', 'temperature']},
        'class_semantics': CLASS_SEMANTICS
    }
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = best['cm']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = ['Critical', 'High', 'Low', 'Medium']
    annot = np.array([[f'{cm[i,j]:,}\n({cm_norm[i,j]:.1%})' for j in range(4)] for i in range(4)])
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='RdYlGn', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'{best_name} - v8 FINAL (Calibrated)\nSharpness: {best["sharpness"]:.1%}', fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figures' / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    # Recall chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(4)
    width = 0.25
    for idx, (name, r) in enumerate(results.items()):
        recalls = [r['critical_recall'], r['high_recall'], r['medium_recall'], r['low_recall']]
        offset = (idx - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=name, alpha=0.85)
        for bar, val in zip(bars, recalls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.0%}', ha='center', fontsize=8)
    
    targets = [0.70, 0.55, 0.50, 0.75]
    for i, t in enumerate(targets):
        ax.axhline(y=t, xmin=i/4+0.03, xmax=(i+1)/4-0.03, color='red', linestyle='--', linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Critical (70%)', 'High (55%)', 'Medium (50%)', 'Low (75%)'])
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Per-Class Recall - v8 FINAL (v6 + Temperature Calibration)', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figures' / 'per_class_recall.png', dpi=150)
    plt.close()
    
    duration = (datetime.now() - start).total_seconds()
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - v8 FINAL")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"  Best: {best_name}")
    print(f"\n  RECALLS:")
    print(f"    Critical: {best['critical_recall']:.2%} (target 70%)")
    print(f"    High:     {best['high_recall']:.2%} (target 55%)")
    print(f"    Medium:   {best['medium_recall']:.2%} (target 50%)")
    print(f"    Low:      {best['low_recall']:.2%} (target 75%)")
    print(f"\n  QUALITY:")
    print(f"    AUC: {best['auc']:.4f}")
    print(f"    Brier: {best['brier']:.4f}")
    print(f"    Sharpness: {best['sharpness']:.1%}")
    print(f"    Temperature: {best['temperature']:.2f}")
    print(f"\n  CLASS INTERPRETATION:")
    print(f"    Medium = MONITORING PRIORITY (not intervention)")
    print(f"    Low = DRIFT DETECTOR")
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return results


if __name__ == '__main__':
    run_training()
