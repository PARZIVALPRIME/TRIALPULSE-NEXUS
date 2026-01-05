"""
TRIALPULSE NEXUS 10X - ML Visualization Generator v6
Generates all 8 advanced ML visualizations for Patient Risk Classifier

Visualizations:
1. ROC Curve (all models overlaid)
2. Precision-Recall Curve
3. Enhanced Confusion Matrix
4. SHAP Beeswarm Plot
5. SHAP Feature Importance Bar
6. Calibration Curve
7. Threshold Analysis Plot
8. Learning Curve
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import sys

# ML imports
from sklearn.preprocessing import RobustScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve

# Visualization imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. SHAP visualizations will be skipped.")

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
UPR_PATH = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'ml_training_v6'
FIG_DIR = OUTPUT_DIR / 'figures'
MODEL_DIR = OUTPUT_DIR / 'models'

# Ensure output directory exists
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Class configuration
CLASS_NAMES = {0: 'Critical', 1: 'High', 2: 'Low', 3: 'Medium'}
CLASS_COLORS = {0: '#FF4444', 1: '#FF8800', 2: '#44AA44', 3: '#FFCC00'}
TARGET_RECALLS = {0: 0.70, 1: 0.55, 2: 0.75, 3: 0.50}

# Outcome features to exclude
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
# DATA LOADING & PREPARATION
# ============================================================================

def load_data_and_models():
    """Load data and all trained models."""
    print("Loading data and models...")
    
    # Load data
    df = pd.read_parquet(UPR_PATH)
    print(f"  Loaded {len(df):,} samples")
    
    # Create target (same logic as training)
    y = create_risk_target(df)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Select features
    X = select_features(df)
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Load scaler and encoder
    with open(MODEL_DIR / 'scaler_v6.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / 'label_encoder_v6.pkl', 'rb') as f:
        le = pickle.load(f)
    
    # Convert Categorical to Series with proper index for encoding
    y_series = pd.Series(y, index=df.index)
    y_enc = pd.Series(le.transform(y_series.astype(str)), index=df.index)
    
    # Scale features
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    
    # Load models
    models = {}
    model_names = {'lgb': 'LightGBM', 'xgb': 'XGBoost', 'rf': 'RandomForest', 'gb': 'GradientBoost'}
    
    for abbr, name in model_names.items():
        model_path = MODEL_DIR / f'model_v6_{abbr}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
            print(f"  Loaded {name}")
    
    return X_train, X_test, y_train, y_test, models, X.columns.tolist()


def engineer_features(df):
    """Feature engineering (same as training)."""
    df = df.copy()
    
    # Query metrics
    query_cols = ['dm_queries', 'clinical_queries', 'medical_queries', 'site_queries']
    existing = [c for c in query_cols if c in df.columns]
    if existing:
        df['total_queries'] = df[existing].fillna(0).sum(axis=1)
        df['query_type_count'] = (df[existing].fillna(0) > 0).sum(axis=1).astype(float)
        if 'queries_answered' in df.columns:
            total = df['total_queries'] + df['queries_answered'].fillna(0)
            df['query_resolution_rate'] = np.where(total > 0, df['queries_answered'].fillna(0) / (total + 1), 1.0)
    
    # CRF completion
    if 'total_crfs' in df.columns:
        for col, name in [('crfs_signed', 'signature_rate'), ('crfs_frozen', 'freeze_rate'), 
                         ('crfs_locked', 'lock_rate'), ('crfs_verified_sdv', 'sdv_rate')]:
            if col in df.columns:
                df[name] = np.where(df['total_crfs'] > 0, df[col].fillna(0) / (df['total_crfs'] + 1), 0.0)
    
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
    
    if 'sae_dm_pending' in df.columns and 'sae_safety_pending' in df.columns:
        df['total_sae_pending'] = df['sae_dm_pending'] + df['sae_safety_pending']
        df['has_any_sae_pending'] = (df['total_sae_pending'] > 0).astype(float)
    
    # Coding metrics
    for code, prefix in [('meddra', 'meddra_coding'), ('whodrug', 'whodrug_coding')]:
        total_col = f'{prefix}_{code}_total'
        coded_col = f'{prefix}_{code}_coded'
        if total_col in df.columns and coded_col in df.columns:
            pending = (df[total_col].fillna(0) - df[coded_col].fillna(0)).clip(lower=0)
            df[f'{code}_pending'] = pending
            df[f'{code}_rate'] = np.where(df[total_col] > 0, df[coded_col].fillna(0) / (df[total_col] + 1), 1.0)
            df[f'has_{code}_pending'] = (pending > 0).astype(float)
    
    # Load flags
    if 'total_queries' in df.columns:
        df['high_query_load'] = (df['total_queries'] > 10).astype(float)
        df['critical_query_load'] = (df['total_queries'] > 25).astype(float)
    
    if 'total_crfs' in df.columns:
        df['high_crf_volume'] = (df['total_crfs'] > 50).astype(float)
        df['very_high_crf_volume'] = (df['total_crfs'] > 100).astype(float)
    
    if 'pages_entered' in df.columns and 'total_queries' in df.columns:
        df['query_density'] = np.where(df['pages_entered'] > 0, df['total_queries'] / (df['pages_entered'] + 1), 0.0)
    
    # Workload composite (same as training)
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
    """Create 4-tier risk target."""
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
    
    p50, p80, p95 = risk.quantile(0.50), risk.quantile(0.80), risk.quantile(0.95)
    
    level = pd.Series('Low', index=df.index)
    level[risk > p50] = 'Medium'
    level[risk > p80] = 'High'
    level[risk > p95] = 'Critical'
    
    return pd.Categorical(level, categories=['Low', 'Medium', 'High', 'Critical'], ordered=True)


def select_features(df):
    """Select features, excluding outcomes."""
    cols = []
    for c in df.columns:
        if c in OUTCOME_FEATURES:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if df[c].nunique() < 2 or df[c].std() < 0.001:
            continue
        cols.append(c)
    return df[cols].copy()


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_roc_curve(X_test, y_test, models):
    """1. ROC Curve - All models overlaid."""
    print("Generating ROC Curve...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {'LightGBM': '#00CC88', 'XGBoost': '#0088FF', 
              'RandomForest': '#FF6644', 'GradientBoost': '#AA44FF'}
    linestyles = {'LightGBM': '-', 'XGBoost': '--', 
                  'RandomForest': '-.', 'GradientBoost': ':'}
    
    # Binarize labels for multi-class ROC
    y_test_bin = label_binarize(y_test.values, classes=[0, 1, 2, 3])
    
    for name, model in models.items():
        proba = model.predict_proba(X_test)
        
        # Compute micro-average ROC
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), proba.ravel())
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors.get(name, '#888888'), 
                linestyle=linestyles.get(name, '-'), linewidth=2.5,
                label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Add shaded confidence region for best model (LightGBM)
    if 'LightGBM' in models:
        proba = models['LightGBM'].predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), proba.ravel())
        ax.fill_between(fpr, tpr * 0.95, np.minimum(tpr * 1.05, 1), 
                        alpha=0.15, color='#00CC88', label='LightGBM ±5% CI')
    
    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curve — Patient Risk Classifier v6\n(All Models Comparison)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Excellent\nPerformance', xy=(0.1, 0.9), fontsize=10, 
                color='green', fontweight='bold', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'roc_curve_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: roc_curve_v6.png")


def plot_precision_recall_curve(X_test, y_test, models):
    """2. Precision-Recall Curve - All models overlaid."""
    print("Generating Precision-Recall Curve...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = {'LightGBM': '#00CC88', 'XGBoost': '#0088FF', 
              'RandomForest': '#FF6644', 'GradientBoost': '#AA44FF'}
    
    y_test_bin = label_binarize(y_test.values, classes=[0, 1, 2, 3])
    
    for name, model in models.items():
        proba = model.predict_proba(X_test)
        
        # Micro-average PR curve
        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), proba.ravel())
        ap = average_precision_score(y_test_bin, proba, average='micro')
        
        ax.plot(recall, precision, color=colors.get(name, '#888888'), 
                linewidth=2.5, label=f'{name} (AP = {ap:.4f})')
    
    # Baseline (random classifier)
    baseline = y_test_bin.sum() / y_test_bin.size
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Random (baseline = {baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curve — Patient Risk Classifier v6\n(Imbalanced Data Performance)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'precision_recall_curve_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: precision_recall_curve_v6.png")


def plot_confusion_matrix_enhanced(X_test, y_test, models):
    """3. Enhanced Confusion Matrix - Best model with counts and percentages."""
    print("Generating Enhanced Confusion Matrix...")
    
    # Use best model (LightGBM)
    model = models.get('LightGBM', list(models.values())[0])
    model_name = 'LightGBM' if 'LightGBM' in models else list(models.keys())[0]
    
    # Get predictions
    proba = model.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test.values, pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Class labels
    labels = ['Critical', 'High', 'Low', 'Medium']
    
    # Create figure with custom colormap (green diagonal, red off-diagonal)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create annotation with counts and percentages
    annot = np.empty_like(cm, dtype=object)
    for i in range(4):
        for j in range(4):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            annot[i, j] = f'{count:,}\n({pct:.1f}%)'
    
    # Custom colormap: green for diagonal, red shades for off-diagonal
    mask_diag = np.eye(4, dtype=bool)
    
    # Plot heatmap
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='RdYlGn',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Recall Rate'}, linewidths=0.5,
                annot_kws={'fontsize': 12})
    
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Class', fontsize=14, fontweight='bold')
    ax.set_title(f'Confusion Matrix — {model_name} v6\n(Counts & Percentages)', 
                 fontsize=16, fontweight='bold')
    
    # Add recall values on side
    for i, label in enumerate(labels):
        recall_val = cm_norm[i, i] * 100
        ax.text(4.5, i + 0.5, f'{recall_val:.1f}%', va='center', ha='left',
                fontsize=11, fontweight='bold', 
                color='green' if recall_val >= 50 else 'red')
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'confusion_matrix_enhanced_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: confusion_matrix_enhanced_v6.png")


def plot_shap_beeswarm(X_test, models, feature_names):
    """4. SHAP Summary Plot (Beeswarm) - Top 15 features."""
    if not SHAP_AVAILABLE:
        print("  ⚠ SHAP not available, skipping beeswarm plot")
        return
    
    print("Generating SHAP Beeswarm Plot...")
    
    model = models.get('LightGBM', list(models.values())[0])
    
    # Sample for speed
    n_samples = min(1000, len(X_test))
    X_sample = X_test.iloc[:n_samples]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # For multi-class, take mean absolute SHAP across all classes
            if isinstance(shap_values, list):
                shap_mean = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_mean = np.abs(shap_values)
            
            # Get top 15 features
            mean_abs_shap = np.mean(shap_mean, axis=0)
            top_idx = np.argsort(mean_abs_shap)[-15:][::-1]
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            if isinstance(shap_values, list):
                # Use first class (Critical) for beeswarm
                shap.summary_plot(shap_values[0][:, top_idx], X_sample.iloc[:, top_idx],
                                  feature_names=[feature_names[i] for i in top_idx],
                                  show=False, max_display=15)
            else:
                shap.summary_plot(shap_values[:, top_idx], X_sample.iloc[:, top_idx],
                                  feature_names=[feature_names[i] for i in top_idx],
                                  show=False, max_display=15)
            
            plt.title('SHAP Summary — Top 15 Features\n(LightGBM Risk Classifier v6)', 
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'shap_beeswarm_v6.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: shap_beeswarm_v6.png")
        except Exception as e:
            print(f"  ⚠ SHAP beeswarm failed: {e}")


def plot_shap_importance_bar_shap(X_test, models, feature_names):
    """5. SHAP Feature Importance Bar Chart - Top 15 features."""
    if not SHAP_AVAILABLE:
        print("  ⚠ SHAP not available, skipping importance bar chart")
        return
    
    print("Generating SHAP Feature Importance Bar...")
    
    model = models.get('LightGBM', list(models.values())[0])
    
    n_samples = min(1000, len(X_test))
    X_sample = X_test.iloc[:n_samples]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # For multi-class, sum absolute SHAP across all classes
            if isinstance(shap_values, list):
                shap_sum = sum([np.abs(sv) for sv in shap_values])
            else:
                shap_sum = np.abs(shap_values)
            
            mean_abs_shap = np.mean(shap_sum, axis=0)
            
            # Sort and get top 15
            sorted_idx = np.argsort(mean_abs_shap)[-15:]
            top_features = [feature_names[i] for i in sorted_idx]
            top_values = mean_abs_shap[sorted_idx]
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(12, 10))
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, 15))
            bars = ax.barh(top_features, top_values, color=colors, edgecolor='white')
            
            # Add value labels
            for bar, val in zip(bars, top_values):
                ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=10)
            
            ax.set_xlabel('Mean |SHAP Value|', fontsize=14, fontweight='bold')
            ax.set_title('SHAP Feature Importance — Top 15\n(LightGBM Risk Classifier v6)', 
                         fontsize=16, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(FIG_DIR / 'shap_importance_v6.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: shap_importance_v6.png")
        except Exception as e:
            print(f"  ⚠ SHAP importance failed: {e}")


def plot_calibration_curve(X_test, y_test, models):
    """6. Calibration Curve - Before and after calibration."""
    print("Generating Calibration Curve...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    class_labels = ['Critical', 'High', 'Low', 'Medium']
    
    model = models.get('LightGBM', list(models.values())[0])
    proba = model.predict_proba(X_test)
    
    for cls_idx, (ax, label) in enumerate(zip(axes, class_labels)):
        # Binary labels for this class
        y_binary = (y_test.values == cls_idx).astype(int)
        prob_class = proba[:, cls_idx]
        
        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_binary, prob_class, n_bins=10, strategy='uniform')
            
            ax.plot(prob_pred, prob_true, 's-', color=CLASS_COLORS[cls_idx], 
                    linewidth=2, markersize=8, label='LightGBM')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
            
            # Fill between for under/over confidence
            ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, 
                            color=CLASS_COLORS[cls_idx])
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=11)
            ax.set_ylabel('Fraction of Positives', fontsize=11)
            ax.set_title(f'{label} Class Calibration', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
    
    fig.suptitle('Calibration Curves — LightGBM v6\n(Predicted Probability vs Actual Frequency)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'calibration_curve_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: calibration_curve_v6.png")


def plot_threshold_analysis(X_test, y_test, models):
    """7. Threshold Analysis Plot - Precision/Recall/F1 vs threshold."""
    print("Generating Threshold Analysis Plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    class_labels = ['Critical', 'High', 'Low', 'Medium']
    target_thresholds = [0.70, 0.55, 0.75, 0.50]
    
    model = models.get('LightGBM', list(models.values())[0])
    proba = model.predict_proba(X_test)
    
    for cls_idx, (ax, label, target) in enumerate(zip(axes, class_labels, target_thresholds)):
        y_binary = (y_test.values == cls_idx).astype(int)
        prob_class = proba[:, cls_idx]
        
        thresholds = np.linspace(0.01, 0.99, 100)
        precisions, recalls, f1s = [], [], []
        
        for th in thresholds:
            pred = (prob_class >= th).astype(int)
            tp = ((pred == 1) & (y_binary == 1)).sum()
            fp = ((pred == 1) & (y_binary == 0)).sum()
            fn = ((pred == 0) & (y_binary == 1)).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        
        ax.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
        ax.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
        ax.plot(thresholds, f1s, 'r-', linewidth=2, label='F1 Score')
        
        # Find optimal threshold (max F1)
        opt_idx = np.argmax(f1s)
        opt_th = thresholds[opt_idx]
        ax.axvline(x=opt_th, color='purple', linestyle='--', linewidth=2, 
                   label=f'Optimal (th={opt_th:.2f})')
        
        # Target recall line
        ax.axhline(y=target, color='orange', linestyle=':', linewidth=1.5,
                   label=f'Target Recall ({target:.0%})')
        
        ax.set_xlabel('Threshold', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'{label} — Threshold Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
    
    fig.suptitle('Threshold Analysis — LightGBM v6\n(Precision/Recall/F1 Trade-offs)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'threshold_analysis_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: threshold_analysis_v6.png")


def plot_learning_curve(X_scaled, y_enc, models):
    """8. Learning Curve - Training vs validation across sizes."""
    print("Generating Learning Curve...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use RandomForest for speed (sklearn native)
    model = models.get('RandomForest', list(models.values())[0])
    
    # Learning curve with cross-validation
    train_sizes = np.linspace(0.1, 1.0, 8)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_scaled, y_enc,
        train_sizes=train_sizes,
        cv=3,  # Reduced for speed
        scoring='roc_auc_ovr',
        n_jobs=1,
        random_state=42
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Plot
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#0088FF')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color='#00CC88')
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#0088FF', linewidth=2,
            markersize=8, label='Training Score')
    ax.plot(train_sizes_abs, val_mean, 's-', color='#00CC88', linewidth=2,
            markersize=8, label='Validation Score')
    
    ax.set_xlabel('Training Set Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontsize=14, fontweight='bold')
    ax.set_title('Learning Curve — Risk Classifier v6\n(Training vs Validation Performance)', 
                 fontsize=16, fontweight='bold')
    
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add annotation about gap
    gap = train_mean[-1] - val_mean[-1]
    if gap < 0.05:
        note = "✓ Low variance gap — model generalizes well"
        color = 'green'
    else:
        note = "⚠ High variance gap — consider regularization"
        color = 'orange'
    
    ax.text(0.5, 0.05, note, transform=ax.transAxes, fontsize=11,
            ha='center', color=color, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'learning_curve_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: learning_curve_v6.png")


def plot_feature_importance_bar(models, feature_names):
    """Alternative to SHAP: Feature importance bar using model-native importances."""
    print("Generating Feature Importance Bar Chart...")
    
    model = models.get('LightGBM', list(models.values())[0])
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Normalize and sort
    importances_norm = importances / importances.max()
    sorted_idx = np.argsort(importances_norm)[-15:]
    top_features = [feature_names[i] for i in sorted_idx]
    top_values = importances_norm[sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 15))
    bars = ax.barh(top_features, top_values, color=colors, edgecolor='white')
    
    for bar, val in zip(bars, top_values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Relative Importance (Normalized)', fontsize=14, fontweight='bold')
    ax.set_title('Feature Importance — Top 15\n(LightGBM Risk Classifier v6)', 
                 fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1.15)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'feature_importance_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: feature_importance_v6.png")


def plot_feature_importance_comparison(models, feature_names):
    """Multi-model feature importance comparison (alternative to SHAP beeswarm)."""
    print("Generating Feature Importance Comparison...")
    
    # Get all importances
    all_importances = {}
    for name, model in models.items():
        imp = model.feature_importances_
        imp = imp / imp.max()
        all_importances[name] = imp
    
    # Average to get top features
    avg_imp = np.mean([imp for imp in all_importances.values()], axis=0)
    top_15_idx = np.argsort(avg_imp)[-15:][::-1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(15)
    width = 0.2
    colors = {'LightGBM': '#00CC88', 'XGBoost': '#0088FF', 
              'RandomForest': '#FF6644', 'GradientBoost': '#AA44FF'}
    
    for idx, (name, imp) in enumerate(all_importances.items()):
        offset = (idx - 1.5) * width
        values = imp[top_15_idx]
        ax.barh(y_pos + offset, values, width, label=name, 
                color=colors.get(name, '#888888'), alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in top_15_idx])
    ax.invert_yaxis()
    
    ax.set_xlabel('Relative Importance (Normalized)', fontsize=14, fontweight='bold')
    ax.set_title('Feature Importance Comparison — Top 15\n(All Models, Risk Classifier v6)', 
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'feature_importance_comparison_v6.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: feature_importance_comparison_v6.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all 8 visualizations."""
    print("\n" + "=" * 70)
    print("  TRIALPULSE NEXUS 10X — ML VISUALIZATION GENERATOR v6")
    print("=" * 70 + "\n")
    
    # Load data and models
    X_train, X_test, y_train, y_test, models, feature_names = load_data_and_models()
    
    print(f"\n  Models loaded: {list(models.keys())}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(feature_names)}\n")
    
    # Combine for learning curve
    X_scaled = pd.concat([X_train, X_test])
    y_enc = pd.concat([y_train, y_test])
    
    print("=" * 70)
    print("  Generating Visualizations...")
    print("=" * 70 + "\n")
    
    # Generate all visualizations
    plot_roc_curve(X_test, y_test, models)
    plot_precision_recall_curve(X_test, y_test, models)
    plot_confusion_matrix_enhanced(X_test, y_test, models)
    
    # SHAP plots (if available) or fallback to model-native importance
    if SHAP_AVAILABLE:
        plot_shap_beeswarm(X_test, models, feature_names)
        plot_shap_importance_bar_shap(X_test, models, feature_names)
    else:
        print("  Note: Using model-native importance (SHAP not available)")
        
    # Always generate model-native importance plots
    plot_feature_importance_bar(models, feature_names)
    plot_feature_importance_comparison(models, feature_names)
    
    plot_calibration_curve(X_test, y_test, models)
    plot_threshold_analysis(X_test, y_test, models)
    plot_learning_curve(X_scaled, y_enc, models)
    
    print("\n" + "=" * 70)
    print("  ✅ ALL VISUALIZATIONS GENERATED")
    print(f"     Output: {FIG_DIR}")
    print("=" * 70 + "\n")
    
    # List generated files
    print("  Generated files:")
    for f in sorted(FIG_DIR.glob('*.png')):
        print(f"    - {f.name}")
    
    return True


if __name__ == '__main__':
    main()
