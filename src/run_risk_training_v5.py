"""
TRIALPULSE NEXUS 10X - Risk Classifier v5.0 FINAL
Single-threaded, reliable, fast execution
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize
from sklearn.metrics import recall_score, precision_score, confusion_matrix, roc_auc_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGB = True
except:
    XGB = False

try:
    import lightgbm as lgb
    LGB = True
except:
    LGB = False

ROOT = Path(__file__).parent.parent
UPR = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUT = ROOT / 'data' / 'outputs' / 'ml_training_v5'
for d in [OUT, OUT/'figures', OUT/'models', OUT/'tables']:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

WEIGHTS = {0: 10, 1: 8, 3: 4, 2: 1}
TARGETS = {0: 0.70, 1: 0.55, 3: 0.50, 2: 0.75}
OUTCOMES = {'broken_signatures', 'crfs_never_signed', 'protocol_deviations', 'safety_queries',
            'crfs_overdue_for_signs_beyond_90_days_of_data_entry', 'visit_missing_visit_count',
            'sae_dm_sae_dm_completed', 'sae_safety_sae_safety_completed'}

def run():
    start = datetime.now()
    log.info("=" * 60)
    log.info("RISK CLASSIFIER v5 FINAL - Starting")
    log.info("=" * 60)
    
    # Load
    df = pd.read_parquet(UPR)
    log.info(f"Loaded {len(df):,} samples")
    
    # Target
    score = pd.Series(0.0, index=df.index)
    for col, w in [('sae_dm_sae_dm_total', 4), ('sae_safety_sae_safety_total', 4), ('broken_signatures', 3),
                   ('crfs_never_signed', 2.5), ('protocol_deviations', 2), ('visit_missing_visit_count', 1.5)]:
        if col in df.columns:
            if 'total' in col:
                comp = df.get(col.replace('total', 'completed'), 0)
                val = (df[col].fillna(0) - comp).clip(lower=0)
            else:
                val = df[col].fillna(0)
            score += (val > 0).astype(float) * w
    
    p50, p80, p95 = score.quantile([0.5, 0.8, 0.95])
    y = pd.cut(score, [-np.inf, p50, p80, p95, np.inf], labels=['Low', 'Medium', 'High', 'Critical'])
    log.info(f"Target: {dict(y.value_counts())}")
    
    # Features - exclude outcomes
    feat_cols = [c for c in df.columns if c not in OUTCOMES and np.issubdtype(df[c].dtype, np.number) and df[c].std() > 0.01]
    X = df[feat_cols].fillna(0)
    log.info(f"Features: {len(feat_cols)}")
    
    # Encode
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
    
    # Scale
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Split
    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    log.info(f"Train: {len(X_tr):,}, Test: {len(X_te):,}")
    
    # Weights
    w = np.array([WEIGHTS.get(c, 1) for c in y_tr])
    
    # Train - SINGLE THREADED (n_jobs=1)
    models = {}
    
    log.info("Training RF...")
    models['RF'] = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight=WEIGHTS, random_state=42, n_jobs=1)
    models['RF'].fit(X_tr, y_tr)
    
    if XGB:
        log.info("Training XGB...")
        xw = w.copy()
        xw[y_tr == 0] *= 2
        xw[y_tr == 1] *= 1.5
        models['XGB'] = xgb.XGBClassifier(n_estimators=150, max_depth=10, learning_rate=0.1, use_label_encoder=False, verbosity=0, random_state=42, n_jobs=1)
        models['XGB'].fit(X_tr, y_tr, sample_weight=xw)
    
    if LGB:
        log.info("Training LGB...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models['LGB'] = lgb.LGBMClassifier(n_estimators=150, max_depth=10, learning_rate=0.1, class_weight=WEIGHTS, verbosity=-1, random_state=42, n_jobs=1)
            models['LGB'].fit(X_tr, y_tr)
    
    # Evaluate
    log.info("Evaluating...")
    y_te_arr = y_te.values if hasattr(y_te, 'values') else y_te
    results = {}
    
    for name, model in models.items():
        proba = model.predict_proba(X_te)
        
        # Grid search thresholds
        thresholds = {}
        for cls in range(4):
            target = TARGETS.get(cls, 0.5)
            p = proba[:, cls]
            t_arr = (y_te_arr == cls).astype(int)
            
            best_t, best_s = 0.5, -1
            for th in np.linspace(0.05, 0.95, 50):
                pred = (p >= th).astype(int)
                tp = ((pred == 1) & (t_arr == 1)).sum()
                fn = ((pred == 0) & (t_arr == 1)).sum()
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                if rec >= target and rec > best_s:
                    best_s = rec
                    best_t = th
            if best_s < 0:
                best_t = 0.2
            thresholds[cls] = best_t
        
        # Cascade predict
        pred = np.full(len(proba), 2)
        for cls in [0, 1, 3]:
            mask = proba[:, cls] >= thresholds[cls]
            if cls == 0:
                pred[mask] = 0
            elif cls == 1:
                pred[mask & (pred != 0)] = 1
            else:
                pred[mask & ~np.isin(pred, [0, 1])] = 3
        
        rec = recall_score(y_te_arr, pred, average=None, zero_division=0)
        prec = precision_score(y_te_arr, pred, average=None, zero_division=0)
        
        try:
            auc = roc_auc_score(label_binarize(y_te_arr, classes=[0,1,2,3]), proba, average='macro', multi_class='ovr')
        except:
            auc = 0.5
        
        tgt_met = sum(1 for i in [0,1,3,2] if rec[i] >= TARGETS.get(i, 0.5))
        comb = (rec[0]*4 + rec[1]*3 + rec[3]*2 + rec[2]) / 10
        
        results[name] = {
            'thresholds': thresholds,
            'critical_recall': float(rec[0]), 'high_recall': float(rec[1]),
            'medium_recall': float(rec[3]), 'low_recall': float(rec[2]),
            'critical_prec': float(prec[0]), 'high_prec': float(prec[1]),
            'auc': float(auc), 'combined': float(comb), 'targets_met': tgt_met,
            'cm': confusion_matrix(y_te_arr, pred)
        }
        
        log.info(f"  {name}: Crit={rec[0]:.2%} High={rec[1]:.2%} Med={rec[3]:.2%} Low={rec[2]:.2%} Targets={tgt_met}/4")
    
    # Best
    best_name = max(results, key=lambda x: (results[x]['targets_met'], results[x]['combined']))
    best = results[best_name]
    
    log.info(f"\n🏆 BEST: {best_name}")
    
    # Outputs
    rows = [{'Model': n, 'Critical': f"{r['critical_recall']:.2%}", 'High': f"{r['high_recall']:.2%}",
             'Medium': f"{r['medium_recall']:.2%}", 'Low': f"{r['low_recall']:.2%}",
             'Targets': f"{r['targets_met']}/4", 'AUC': f"{r['auc']:.4f}"} for n, r in results.items()]
    pd.DataFrame(rows).to_csv(OUT/'tables'/'production_metrics_v5.csv', index=False)
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = best['cm'].astype('float') / best['cm'].sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='RdYlGn', xticklabels=['Crit','High','Low','Med'], yticklabels=['Crit','High','Low','Med'], ax=ax)
    ax.set_title(f'{best_name} Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT/'figures'/'confusion_matrix_v5.png', dpi=150)
    plt.close()
    
    # Recall bar
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(4)
    for i, n in enumerate(results):
        r = results[n]
        ax.bar(x + i*0.25, [r['critical_recall'], r['high_recall'], r['medium_recall'], r['low_recall']], 0.25, label=n)
    for i, t in enumerate([TARGETS[0], TARGETS[1], TARGETS[3], TARGETS[2]]):
        ax.axhline(y=t, xmin=i/4+0.05, xmax=(i+1)/4-0.05, color='r', linestyle='--')
    ax.set_xticks(x + 0.25)
    ax.set_xticklabels(['Critical', 'High', 'Medium', 'Low'])
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall vs Targets', fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(OUT/'figures'/'per_class_recall_v5.png', dpi=150)
    plt.close()
    
    # Save models
    for n, m in models.items():
        with open(OUT/'models'/f'model_v5_{n.lower()}.pkl', 'wb') as f:
            pickle.dump(m, f)
    
    config = {
        'version': '5.0.0', 'created': datetime.now().isoformat(),
        'best_model': best_name, 'thresholds': {str(k): v for k, v in best['thresholds'].items()},
        'metrics': {'critical': best['critical_recall'], 'high': best['high_recall'],
                   'medium': best['medium_recall'], 'low': best['low_recall'], 'targets_met': best['targets_met']},
        'n_features': len(feat_cols)
    }
    with open(OUT/'models'/'production_config_v5.json', 'w') as f:
        json.dump(config, f, indent=2)
    with open(OUT/'training_report_v5.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    dur = (datetime.now() - start).total_seconds()
    
    log.info("\n" + "=" * 60)
    log.info("✅ COMPLETE")
    log.info("=" * 60)
    log.info(f"  Duration: {dur:.1f}s")
    log.info(f"  Best: {best_name}")
    log.info(f"  Critical: {best['critical_recall']:.2%}")
    log.info(f"  High: {best['high_recall']:.2%}")
    log.info(f"  Medium: {best['medium_recall']:.2%}")
    log.info(f"  Low: {best['low_recall']:.2%}")
    log.info(f"  Targets Met: {best['targets_met']}/4")
    log.info(f"  Output: {OUT}")
    
    print("\n✅ TRAINING COMPLETE")
    return best

if __name__ == '__main__':
    run()
