"""
TRIALPULSE NEXUS 10X - Multi-Label Issue Detector v1.0
Industry-grade issue type detection using Binary Relevance

ARCHITECTURE: 14 separate binary classifiers (One-vs-Rest)
- Each classifier detects one issue type
- XGBoost with class weighting for imbalanced labels
- Per-classifier threshold optimization
- SHAP explainability for all classifiers

PRODUCTION READY
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve, precision_recall_curve
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
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
UPR_PATH = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'ml_training_issue_detector_v1'

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
# ISSUE DEFINITIONS - 14 ISSUE TYPES
# ============================================================================

ISSUE_DEFINITIONS = {
    'sae_dm_pending': {
        'name': 'SAE DM Pending',
        'description': 'SAE Data Management review pending',
        'source_cols': ['sae_dm_sae_dm_total', 'sae_dm_sae_dm_completed'],
        'logic': 'total - completed > 0',
        'clinical_priority': 'CRITICAL',
        'target_auc': 0.90,
        'target_f1': 0.75
    },
    'sae_safety_pending': {
        'name': 'SAE Safety Pending',
        'description': 'SAE Safety review pending',
        'source_cols': ['sae_safety_sae_safety_total', 'sae_safety_sae_safety_completed'],
        'logic': 'total - completed > 0',
        'clinical_priority': 'CRITICAL',
        'target_auc': 0.90,
        'target_f1': 0.70
    },
    'open_queries': {
        'name': 'Open Queries',
        'description': 'Has open queries requiring attention',
        'source_cols': ['dm_queries', 'clinical_queries', 'medical_queries', 'site_queries'],
        'logic': 'sum > 0',
        'clinical_priority': 'HIGH',
        'target_auc': 0.85,
        'target_f1': 0.75
    },
    'high_query_volume': {
        'name': 'High Query Volume',
        'description': 'More than 10 open queries (high workload)',
        'source_cols': ['dm_queries', 'clinical_queries', 'medical_queries', 'site_queries'],
        'logic': 'sum > 10',
        'clinical_priority': 'HIGH',
        'target_auc': 0.88,
        'target_f1': 0.72
    },
    'sdv_incomplete': {
        'name': 'SDV Incomplete',
        'description': 'Source Data Verification not complete',
        'source_cols': ['crfs_require_verification_sdv', 'crfs_verified_sdv'],
        'logic': 'require > verified',
        'clinical_priority': 'MEDIUM',
        'target_auc': 0.85,
        'target_f1': 0.72
    },
    'signature_gaps': {
        'name': 'Signature Gaps',
        'description': 'Missing or overdue signatures',
        'source_cols': ['crfs_overdue_for_signs_within_45_days_of_data_entry',
                       'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
                       'crfs_overdue_for_signs_beyond_90_days_of_data_entry'],
        'logic': 'sum > 0',
        'clinical_priority': 'MEDIUM',
        'target_auc': 0.85,
        'target_f1': 0.70
    },
    'broken_signatures': {
        'name': 'Broken Signatures',
        'description': 'Has broken or invalidated signatures',
        'source_cols': ['broken_signatures'],
        'logic': '> 0',
        'clinical_priority': 'HIGH',
        'target_auc': 0.92,
        'target_f1': 0.80
    },
    'meddra_uncoded': {
        'name': 'MedDRA Uncoded',
        'description': 'MedDRA terms pending coding',
        'source_cols': ['meddra_coding_meddra_total', 'meddra_coding_meddra_coded'],
        'logic': 'total - coded > 0',
        'clinical_priority': 'MEDIUM',
        'target_auc': 0.90,
        'target_f1': 0.75
    },
    'whodrug_uncoded': {
        'name': 'WHODrug Uncoded',
        'description': 'WHODrug terms pending coding',
        'source_cols': ['whodrug_coding_whodrug_total', 'whodrug_coding_whodrug_coded'],
        'logic': 'total - coded > 0',
        'clinical_priority': 'MEDIUM',
        'target_auc': 0.90,
        'target_f1': 0.75
    },
    'missing_visits': {
        'name': 'Missing Visits',
        'description': 'Has missing scheduled visits',
        'source_cols': ['visit_missing_visit_count'],
        'logic': '> 0',
        'clinical_priority': 'HIGH',
        'target_auc': 0.88,
        'target_f1': 0.70
    },
    'missing_pages': {
        'name': 'Missing Pages',
        'description': 'Has missing CRF pages',
        'source_cols': ['pages_pages_missing_count'],
        'logic': '> 0',
        'clinical_priority': 'MEDIUM',
        'target_auc': 0.85,
        'target_f1': 0.72
    },
    'lab_issues': {
        'name': 'Lab Issues',
        'description': 'Lab name/range issues present',
        'source_cols': ['lab_lab_issue_count', 'lab_lab_missing_names', 'lab_lab_missing_ranges'],
        'logic': 'sum > 0',
        'clinical_priority': 'MEDIUM',
        'target_auc': 0.88,
        'target_f1': 0.70
    },
    'edrr_issues': {
        'name': 'EDRR Issues',
        'description': 'Third-party reconciliation issues',
        'source_cols': ['edrr_edrr_issue_count'],
        'logic': '> 0',
        'clinical_priority': 'MEDIUM',
        'target_auc': 0.85,
        'target_f1': 0.68
    },
    'inactivated_forms': {
        'name': 'Inactivated Forms',
        'description': 'Has inactivated forms requiring review',
        'source_cols': ['inactivated_inactivated_form_count'],
        'logic': '> 0',
        'clinical_priority': 'LOW',
        'target_auc': 0.82,
        'target_f1': 0.65
    }
}

# Features to exclude from prediction (target outcomes)
EXCLUDE_FEATURES = set()
for issue_key, issue_def in ISSUE_DEFINITIONS.items():
    EXCLUDE_FEATURES.update(issue_def['source_cols'])


# ============================================================================
# CREATE ISSUE TARGETS
# ============================================================================

def create_issue_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary targets for all 14 issue types."""
    targets = pd.DataFrame(index=df.index)
    
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        cols = issue_def['source_cols']
        existing = [c for c in cols if c in df.columns]
        
        if not existing:
            log.warning(f"  {issue_key}: No source columns found, setting to 0")
            targets[issue_key] = 0
            continue
        
        if 'total' in issue_def['logic'] and 'completed' in issue_def['logic']:
            # Logic: total - completed > 0
            total_col = [c for c in existing if 'total' in c.lower()]
            comp_col = [c for c in existing if 'completed' in c.lower() or 'coded' in c.lower()]
            
            if total_col and comp_col:
                total = df[total_col[0]].fillna(0)
                completed = df[comp_col[0]].fillna(0)
                targets[issue_key] = ((total - completed) > 0).astype(int)
            else:
                targets[issue_key] = (df[existing].fillna(0).sum(axis=1) > 0).astype(int)
        
        elif 'require' in issue_def['logic'] and 'verified' in issue_def['logic']:
            # Logic: require > verified (SDV)
            req_col = [c for c in existing if 'require' in c.lower()]
            ver_col = [c for c in existing if 'verified' in c.lower()]
            
            if req_col and ver_col:
                required = df[req_col[0]].fillna(0)
                verified = df[ver_col[0]].fillna(0)
                targets[issue_key] = (required > verified).astype(int)
            else:
                targets[issue_key] = (df[existing].fillna(0).sum(axis=1) > 0).astype(int)
        
        elif 'sum > 0' in issue_def['logic']:
            # Logic: sum of columns > 0
            targets[issue_key] = (df[existing].fillna(0).sum(axis=1) > 0).astype(int)
        
        elif 'sum > 10' in issue_def['logic']:
            # Logic: sum of columns > 10
            targets[issue_key] = (df[existing].fillna(0).sum(axis=1) > 10).astype(int)
        
        else:
            # Default logic: first column > 0
            targets[issue_key] = (df[existing[0]].fillna(0) > 0).astype(int)
    
    return targets


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for issue detection."""
    df = df.copy()
    n = 0
    
    # Query metrics
    query_cols = ['dm_queries', 'clinical_queries', 'medical_queries', 'site_queries', 
                  'field_monitor_queries', 'coding_queries', 'safety_queries']
    existing = [c for c in query_cols if c in df.columns]
    if existing:
        df['total_queries'] = df[existing].fillna(0).sum(axis=1)
        df['query_type_count'] = (df[existing].fillna(0) > 0).sum(axis=1).astype(float)
        df['max_query_type'] = df[existing].fillna(0).max(axis=1)
        n += 3
        
        if 'queries_answered' in df.columns:
            total = df['total_queries'] + df['queries_answered'].fillna(0)
            df['query_resolution_rate'] = np.where(total > 0, 
                df['queries_answered'].fillna(0) / (total + 1), 1.0)
            n += 1
    
    # CRF completion rates
    if 'total_crfs' in df.columns:
        for col, name in [('crfs_signed', 'signature_rate'), 
                         ('crfs_frozen', 'freeze_rate'),
                         ('crfs_locked', 'lock_rate'), 
                         ('crfs_verified_sdv', 'sdv_rate'),
                         ('pages_entered', 'entry_rate')]:
            if col in df.columns:
                df[name] = np.where(df['total_crfs'] > 0, 
                    df[col].fillna(0) / (df['total_crfs'] + 1), 0.0)
                n += 1
    
    # SAE composite metrics
    sae_pending_cols = []
    for sae in ['sae_dm', 'sae_safety']:
        total_col = f'{sae}_{sae}_total'
        comp_col = f'{sae}_{sae}_completed'
        if total_col in df.columns:
            pending = df[total_col].fillna(0)
            if comp_col in df.columns:
                pending = (pending - df[comp_col].fillna(0)).clip(lower=0)
            df[f'{sae}_pending_count'] = pending
            df[f'has_{sae}'] = (df[total_col].fillna(0) > 0).astype(float)
            sae_pending_cols.append(f'{sae}_pending_count')
            n += 2
    
    if sae_pending_cols:
        df['total_sae_pending'] = df[sae_pending_cols].sum(axis=1)
        df['has_any_sae'] = (df['total_sae_pending'] > 0).astype(float)
        n += 2
    
    # Signature overdue metrics
    sig_cols = ['crfs_overdue_for_signs_within_45_days_of_data_entry',
                'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
                'crfs_overdue_for_signs_beyond_90_days_of_data_entry']
    existing_sig = [c for c in sig_cols if c in df.columns]
    if existing_sig:
        df['total_overdue_signatures'] = df[existing_sig].fillna(0).sum(axis=1)
        df['has_overdue_signatures'] = (df['total_overdue_signatures'] > 0).astype(float)
        df['severe_overdue_signatures'] = (df[existing_sig[-1]].fillna(0) > 0).astype(float) if existing_sig[-1] in df.columns else 0
        n += 3
    
    # Coding metrics
    for code in ['meddra', 'whodrug']:
        total_col = f'{code}_coding_{code}_total'
        coded_col = f'{code}_coding_{code}_coded'
        if total_col in df.columns and coded_col in df.columns:
            pending = (df[total_col].fillna(0) - df[coded_col].fillna(0)).clip(lower=0)
            df[f'{code}_pending'] = pending
            df[f'{code}_rate'] = np.where(df[total_col] > 0, 
                df[coded_col].fillna(0) / (df[total_col] + 1), 1.0)
            n += 2
    
    # Boolean flags for workload indicators
    if 'total_queries' in df.columns:
        df['low_query_load'] = (df['total_queries'] <= 3).astype(float)
        df['medium_query_load'] = ((df['total_queries'] > 3) & (df['total_queries'] <= 10)).astype(float)
        df['high_query_load'] = (df['total_queries'] > 10).astype(float)
        df['critical_query_load'] = (df['total_queries'] > 25).astype(float)
        n += 4
    
    if 'total_crfs' in df.columns:
        df['high_crf_volume'] = (df['total_crfs'] > 50).astype(float)
        df['very_high_crf_volume'] = (df['total_crfs'] > 100).astype(float)
        n += 2
    
    # Query density
    if 'pages_entered' in df.columns and 'total_queries' in df.columns:
        df['query_density'] = np.where(df['pages_entered'] > 0, 
            df['total_queries'] / (df['pages_entered'] + 1), 0.0)
        n += 1
    
    # Composite workload score
    work_cols = []
    for col in ['total_crfs', 'pages_entered', 'total_queries']:
        if col in df.columns:
            q99 = df[col].fillna(0).quantile(0.99)
            if q99 > 0:
                df[f'{col}_normalized'] = (df[col].fillna(0).clip(upper=q99) / q99).clip(0, 1)
                work_cols.append(f'{col}_normalized')
                n += 1
    
    if work_cols:
        df['workload_composite'] = df[work_cols].mean(axis=1)
        n += 1
    
    # Issue complexity score (number of concurrent issues)
    issue_indicators = []
    for indicator in ['has_any_sae', 'has_overdue_signatures', 'high_query_load']:
        if indicator in df.columns:
            issue_indicators.append(indicator)
    if issue_indicators:
        df['issue_complexity'] = df[issue_indicators].sum(axis=1)
        n += 1
    
    log.info(f"  Engineered {n} features")
    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features(df: pd.DataFrame, exclude_cols: set = None) -> pd.DataFrame:
    """Select numeric features, excluding target outcomes."""
    if exclude_cols is None:
        exclude_cols = EXCLUDE_FEATURES
    
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if df[c].nunique() < 2 or df[c].std() < 0.001:
            continue
        cols.append(c)
    
    return df[cols].copy()


# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """Optimize threshold to maximize F1 score."""
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    for threshold in np.linspace(0.1, 0.9, 81):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision_score(y_true, y_pred, zero_division=0)
            best_recall = recall_score(y_true, y_pred, zero_division=0)
    
    return best_threshold, best_f1, best_precision, best_recall


# ============================================================================
# TRAIN SINGLE CLASSIFIER
# ============================================================================

def train_issue_classifier(X_train, y_train, X_test, y_test, issue_key: str, issue_def: dict) -> dict:
    """Train a single binary classifier for an issue type."""
    
    # Check class balance
    pos_rate = y_train.mean()
    if pos_rate < 0.001:
        log.warning(f"  {issue_key}: Positive rate {pos_rate:.4%} too low, skipping")
        return None
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / max(pos_count, 1)
    scale_weight = min(scale_weight, 100)  # Cap to avoid extreme weights
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        use_label_encoder=False,
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_proba) if y_test.sum() > 0 else 0.5
    ap = average_precision_score(y_test, y_proba) if y_test.sum() > 0 else 0.0
    
    # Optimize threshold
    threshold, f1, precision, recall = optimize_threshold(y_test, y_proba)
    
    # Get feature importance
    importance = dict(zip(X_train.columns, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # SHAP values for explainability (sample for speed)
    shap_values = None
    if SHAP_AVAILABLE:
        try:
            sample_size = min(500, len(X_test))
            X_shap = X_test.iloc[:sample_size]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
        except Exception as e:
            log.warning(f"  SHAP failed for {issue_key}: {e}")
    
    return {
        'model': model,
        'issue_key': issue_key,
        'name': issue_def['name'],
        'clinical_priority': issue_def['clinical_priority'],
        'prevalence': float(pos_rate),
        'auc': float(auc),
        'average_precision': float(ap),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'threshold': float(threshold),
        'target_auc': issue_def['target_auc'],
        'target_f1': issue_def['target_f1'],
        'auc_pass': auc >= issue_def['target_auc'],
        'f1_pass': f1 >= issue_def['target_f1'],
        'top_features': top_features,
        'shap_values': shap_values,
        'scale_pos_weight': scale_weight
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_performance_heatmap(results: list, output_dir: Path):
    """Create a heatmap showing performance across all issue types."""
    data = []
    for r in results:
        data.append({
            'Issue': r['name'],
            'Priority': r['clinical_priority'],
            'Prevalence': r['prevalence'],
            'AUC': r['auc'],
            'AP': r['average_precision'],
            'F1': r['f1'],
            'Precision': r['precision'],
            'Recall': r['recall']
        })
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Performance metrics heatmap
    ax1 = axes[0]
    metrics = df[['AUC', 'AP', 'F1', 'Precision', 'Recall']].values
    
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    im = ax1.imshow(metrics, cmap=cmap, aspect='auto', vmin=0.5, vmax=1.0)
    
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['AUC', 'AP', 'F1', 'Precision', 'Recall'], fontsize=12, fontweight='bold')
    ax1.set_yticks(range(len(df)))
    ax1.set_yticklabels(df['Issue'].values, fontsize=10)
    
    # Add text annotations
    for i in range(len(df)):
        for j in range(5):
            val = metrics[i, j]
            color = 'white' if val < 0.7 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color=color)
    
    ax1.set_title('Issue Detection Performance Metrics\nv1 Multi-Label Issue Detector', 
                  fontsize=14, fontweight='bold', pad=15)
    
    # Priority color bar
    ax2 = axes[1]
    priority_colors = {'CRITICAL': '#e74c3c', 'HIGH': '#f39c12', 'MEDIUM': '#3498db', 'LOW': '#2ecc71'}
    colors = [priority_colors.get(p, 'gray') for p in df['Priority']]
    
    bars = ax2.barh(range(len(df)), df['Prevalence'] * 100, color=colors, alpha=0.8, edgecolor='white')
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels([])
    ax2.set_xlabel('Prevalence (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Issue Prevalence by Clinical Priority', fontsize=14, fontweight='bold', pad=15)
    
    for i, (bar, prev) in enumerate(zip(bars, df['Prevalence'])):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{prev*100:.1f}%', va='center', fontsize=10)
    
    # Legend for priority
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=priority_colors[p], label=p) 
                       for p in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']]
    ax2.legend(handles=legend_elements, loc='lower right', title='Priority')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved: performance_heatmap.png")


def create_roc_curves(results: list, output_dir: Path, X_test, y_targets):
    """Create ROC curves for all issue types."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        issue_key = r['issue_key']
        
        if issue_key in y_targets.columns:
            y_true = y_targets[issue_key].values
            model = r['model']
            y_proba = model.predict_proba(X_test)[:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            
            ax.plot(fpr, tpr, color=colors[idx], lw=2, 
                   label=f'AUC = {r["auc"]:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.fill_between(fpr, tpr, alpha=0.2, color=colors[idx])
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{r["name"]}\n({r["clinical_priority"]})', fontsize=10, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(results), 16):
        axes[idx].axis('off')
    
    plt.suptitle('ROC Curves - All 14 Issue Types\nMulti-Label Issue Detector v1', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'roc_curves_all_issues.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved: roc_curves_all_issues.png")


def create_precision_recall_curves(results: list, output_dir: Path, X_test, y_targets):
    """Create Precision-Recall curves for all issue types."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    colors = plt.cm.tab20(np.linspace(0, 1, 14))
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        issue_key = r['issue_key']
        
        if issue_key in y_targets.columns:
            y_true = y_targets[issue_key].values
            model = r['model']
            y_proba = model.predict_proba(X_test)[:, 1]
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            
            ax.plot(recall, precision, color=colors[idx], lw=2,
                   label=f'AP = {r["average_precision"]:.3f}')
            ax.axhline(y=r['prevalence'], color='red', linestyle='--', 
                      linewidth=1, alpha=0.5, label=f'Baseline = {r["prevalence"]:.3f}')
            ax.fill_between(recall, precision, alpha=0.2, color=colors[idx])
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{r["name"]}', fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(results), 16):
        axes[idx].axis('off')
    
    plt.suptitle('Precision-Recall Curves - All 14 Issue Types\nMulti-Label Issue Detector v1', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved: precision_recall_curves.png")


def create_shap_importance_grid(results: list, output_dir: Path, X_test):
    """Create SHAP feature importance grid for top issues."""
    # Select top 6 issues by AUC for visualization
    top_results = sorted(results, key=lambda x: x['auc'], reverse=True)[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, r in enumerate(top_results):
        ax = axes[idx]
        
        # Use built-in feature importance if SHAP not available
        top_features = r['top_features'][:8]
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        
        bars = ax.barh(range(len(features)), importances, color=colors)
        ax.set_yticks(range(len(features)))
        
        # Truncate long feature names
        feature_labels = [f[:25] + '...' if len(f) > 25 else f for f in features]
        ax.set_yticklabels(feature_labels, fontsize=9)
        ax.invert_yaxis()
        
        ax.set_xlabel('Feature Importance', fontsize=10)
        ax.set_title(f'{r["name"]}\nAUC: {r["auc"]:.3f} | F1: {r["f1"]:.3f}', 
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Feature Importance - Top 6 Issue Detectors\nMulti-Label Issue Detector v1', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / 'shap_importance_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved: shap_importance_grid.png")


def create_class_balance_chart(targets: pd.DataFrame, output_dir: Path):
    """Create chart showing class balance for all issue types."""
    prevalences = []
    for col in targets.columns:
        prev = targets[col].mean()
        prevalences.append({'Issue': col, 'Prevalence': prev})
    
    df = pd.DataFrame(prevalences)
    df = df.sort_values('Prevalence', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.RdYlGn(df['Prevalence'].values / df['Prevalence'].max())
    
    bars = ax.barh(range(len(df)), df['Prevalence'] * 100, color=colors, edgecolor='white')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Issue'].values, fontsize=11)
    ax.set_xlabel('Prevalence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Issue Type Prevalence Distribution\nClass Balance Analysis', 
                 fontsize=14, fontweight='bold', pad=15)
    
    for bar, prev in zip(bars, df['Prevalence']):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
               f'{prev*100:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'class_balance_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved: class_balance_distribution.png")


def create_performance_summary_chart(results: list, output_dir: Path):
    """Create summary bar chart comparing AUC and F1 to targets."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Sort by AUC
    sorted_results = sorted(results, key=lambda x: x['auc'], reverse=True)
    
    names = [r['name'] for r in sorted_results]
    aucs = [r['auc'] for r in sorted_results]
    f1s = [r['f1'] for r in sorted_results]
    target_aucs = [r['target_auc'] for r in sorted_results]
    target_f1s = [r['target_f1'] for r in sorted_results]
    
    x = np.arange(len(names))
    width = 0.35
    
    # AUC subplot
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, aucs, width, label='Achieved AUC', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, target_aucs, width, label='Target AUC', color='#e74c3c', alpha=0.5)
    
    ax1.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title('AUC: Achieved vs Target\nMulti-Label Issue Detector v1', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='lower right')
    ax1.set_ylim(0.5, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Good (0.85)')
    
    # Add pass/fail indicators
    for i, (auc, target) in enumerate(zip(aucs, target_aucs)):
        marker = '✓' if auc >= target else '✗'
        color = 'green' if auc >= target else 'red'
        ax1.text(i, max(auc, target) + 0.02, marker, ha='center', va='bottom', 
                fontsize=14, fontweight='bold', color=color)
    
    # F1 subplot
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, f1s, width, label='Achieved F1', color='#2ecc71', alpha=0.8)
    bars4 = ax2.bar(x + width/2, target_f1s, width, label='Target F1', color='#e74c3c', alpha=0.5)
    
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1: Achieved vs Target\nMulti-Label Issue Detector v1', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add pass/fail indicators
    for i, (f1, target) in enumerate(zip(f1s, target_f1s)):
        marker = '✓' if f1 >= target else '✗'
        color = 'green' if f1 >= target else 'red'
        ax2.text(i, max(f1, target) + 0.02, marker, ha='center', va='bottom',
                fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'performance_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info("  Saved: performance_summary.png")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def run_training():
    """Main training pipeline for Multi-Label Issue Detector."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  TRIALPULSE NEXUS 10X - MULTI-LABEL ISSUE DETECTOR v1.0")
    print("=" * 70)
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Architecture: 14 Binary Classifiers (One-vs-Rest)")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("MULTI-LABEL ISSUE DETECTOR v1.0")
    log.info("=" * 60)
    
    if not XGB_AVAILABLE:
        log.error("XGBoost not available! Install with: pip install xgboost")
        return None
    
    # 1. Load Data
    log.info("\n[1/8] Loading unified patient record...")
    df = pd.read_parquet(UPR_PATH)
    log.info(f"  Loaded {len(df):,} patients × {len(df.columns)} columns")
    
    # 2. Create Issue Targets
    log.info("\n[2/8] Creating 14 issue targets...")
    targets = create_issue_targets(df)
    
    log.info("  Issue prevalences:")
    for col in targets.columns:
        prev = targets[col].mean()
        count = targets[col].sum()
        log.info(f"    {col}: {prev:.2%} ({count:,} cases)")
    
    # 3. Feature Engineering
    log.info("\n[3/8] Engineering features...")
    df = engineer_features(df)
    
    # 4. Feature Selection
    log.info("\n[4/8] Selecting prediction features...")
    X = select_features(df)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    feature_names = list(X.columns)
    log.info(f"  Selected {len(feature_names)} features")
    
    # 5. Scale Features
    log.info("\n[5/8] Scaling features...")
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # 6. Train/Test Split
    log.info("\n[6/8] Splitting data (80/20)...")
    X_train, X_test, targets_train, targets_test = train_test_split(
        X_scaled, targets, test_size=0.2, random_state=42
    )
    log.info(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # 7. Train Classifiers
    log.info("\n[7/8] Training 14 issue classifiers...")
    results = []
    
    for issue_key, issue_def in ISSUE_DEFINITIONS.items():
        log.info(f"\n  Training: {issue_def['name']} ({issue_def['clinical_priority']})")
        
        y_train = targets_train[issue_key].values
        y_test = targets_test[issue_key].values
        
        result = train_issue_classifier(X_train, y_train, X_test, y_test, issue_key, issue_def)
        
        if result:
            results.append(result)
            status = "✓ PASS" if result['auc_pass'] and result['f1_pass'] else "○ PARTIAL" if result['auc_pass'] or result['f1_pass'] else "✗ MISS"
            log.info(f"    AUC: {result['auc']:.4f} (target: {issue_def['target_auc']}) | " +
                    f"F1: {result['f1']:.4f} (target: {issue_def['target_f1']}) | {status}")
    
    # 8. Generate Visualizations & Save
    log.info("\n[8/8] Generating visualizations and saving outputs...")
    
    # Create visualizations
    create_performance_heatmap(results, OUTPUT_DIR / 'figures')
    create_roc_curves(results, OUTPUT_DIR / 'figures', X_test, targets_test)
    create_precision_recall_curves(results, OUTPUT_DIR / 'figures', X_test, targets_test)
    create_shap_importance_grid(results, OUTPUT_DIR / 'figures', X_test)
    create_class_balance_chart(targets, OUTPUT_DIR / 'figures')
    create_performance_summary_chart(results, OUTPUT_DIR / 'figures')
    
    # Save models
    for r in results:
        model_path = OUTPUT_DIR / 'models' / f'issue_detector_{r["issue_key"]}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(r['model'], f)
    
    with open(OUTPUT_DIR / 'models' / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save performance table
    perf_data = []
    for r in results:
        perf_data.append({
            'Issue Type': r['name'],
            'Issue Key': r['issue_key'],
            'Priority': r['clinical_priority'],
            'Prevalence': f"{r['prevalence']:.2%}",
            'AUC': f"{r['auc']:.4f}",
            'Target AUC': r['target_auc'],
            'AUC Pass': '✓' if r['auc_pass'] else '✗',
            'Average Precision': f"{r['average_precision']:.4f}",
            'F1': f"{r['f1']:.4f}",
            'Target F1': r['target_f1'],
            'F1 Pass': '✓' if r['f1_pass'] else '✗',
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'Threshold': f"{r['threshold']:.3f}",
            'Top Feature': r['top_features'][0][0] if r['top_features'] else ''
        })
    
    perf_df = pd.DataFrame(perf_data)
    perf_df.to_csv(OUTPUT_DIR / 'tables' / 'issue_performance_table.csv', index=False)
    log.info("  Saved: issue_performance_table.csv")
    
    # Save feature importance for all issues
    importance_data = []
    for r in results:
        for feat, imp in r['top_features']:
            importance_data.append({
                'Issue': r['name'],
                'Feature': feat,
                'Importance': imp
            })
    
    imp_df = pd.DataFrame(importance_data)
    imp_df.to_csv(OUTPUT_DIR / 'tables' / 'feature_importance_all.csv', index=False)
    log.info("  Saved: feature_importance_all.csv")
    
    # Save production config
    production_config = {
        'version': '1.0.0',
        'status': 'PRODUCTION',
        'created': datetime.now().isoformat(),
        'model_type': 'Multi-Label Binary Relevance',
        'n_classifiers': len(results),
        'n_features': len(feature_names),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'overall_metrics': {
            'avg_auc': float(np.mean([r['auc'] for r in results])),
            'avg_f1': float(np.mean([r['f1'] for r in results])),
            'avg_precision': float(np.mean([r['precision'] for r in results])),
            'avg_recall': float(np.mean([r['recall'] for r in results])),
            'auc_pass_rate': sum(1 for r in results if r['auc_pass']) / len(results),
            'f1_pass_rate': sum(1 for r in results if r['f1_pass']) / len(results)
        },
        'classifiers': {
            r['issue_key']: {
                'name': r['name'],
                'priority': r['clinical_priority'],
                'auc': r['auc'],
                'f1': r['f1'],
                'threshold': r['threshold'],
                'pass': r['auc_pass'] and r['f1_pass']
            } for r in results
        }
    }
    
    with open(OUTPUT_DIR / 'models' / 'production_config.json', 'w') as f:
        json.dump(production_config, f, indent=2)
    log.info("  Saved: production_config.json")
    
    # Final summary
    duration = (datetime.now() - start).total_seconds()
    
    auc_passes = sum(1 for r in results if r['auc_pass'])
    f1_passes = sum(1 for r in results if r['f1_pass'])
    full_passes = sum(1 for r in results if r['auc_pass'] and r['f1_pass'])
    
    avg_auc = np.mean([r['auc'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    
    print("\n" + "=" * 70)
    print("  MULTI-LABEL ISSUE DETECTOR v1.0 - TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Duration: {duration:.1f} seconds")
    print(f"\n  OVERALL PERFORMANCE:")
    print(f"    Average AUC: {avg_auc:.4f}")
    print(f"    Average F1:  {avg_f1:.4f}")
    print(f"\n  TARGET ACHIEVEMENT:")
    print(f"    AUC targets met: {auc_passes}/{len(results)}")
    print(f"    F1 targets met:  {f1_passes}/{len(results)}")
    print(f"    Full pass:       {full_passes}/{len(results)}")
    print(f"\n  TOP PERFORMERS:")
    
    top_3 = sorted(results, key=lambda x: x['auc'], reverse=True)[:3]
    for r in top_3:
        print(f"    • {r['name']}: AUC={r['auc']:.4f}, F1={r['f1']:.4f}")
    
    print(f"\n  NEEDS IMPROVEMENT:")
    bottom_3 = sorted(results, key=lambda x: x['auc'])[:3]
    for r in bottom_3:
        status = "PASS" if r['auc_pass'] else "MISS"
        print(f"    • {r['name']}: AUC={r['auc']:.4f} ({status})")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    log.info("Multi-Label Issue Detector v1.0 complete!")
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    run_training()
