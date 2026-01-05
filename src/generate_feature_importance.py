"""
Generate Feature Importance visualizations using model-native importances.
Alternative to SHAP when SHAP is unavailable or too slow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / 'data' / 'outputs' / 'ml_training_v6' / 'models'
FIG_DIR = ROOT / 'data' / 'outputs' / 'ml_training_v6' / 'figures'
UPR_PATH = ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'

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


def get_feature_names():
    """Get feature names from data."""
    df = pd.read_parquet(UPR_PATH)
    df = engineer_features(df)
    X = select_features(df)
    return list(X.columns)


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
    
    # Workload composite
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


def plot_feature_importance_bar():
    """Create feature importance bar chart using model-native importances."""
    print("Generating Feature Importance Bar Chart...")
    
    feature_names = get_feature_names()
    
    # Load LightGBM model
    with open(MODEL_DIR / 'model_v6_lgb.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    
    # Get feature importances
    importances = lgb_model.feature_importances_
    
    # Sort and get top 15
    sorted_idx = np.argsort(importances)[-15:]
    top_features = [feature_names[i] for i in sorted_idx]
    top_values = importances[sorted_idx]
    
    # Normalize
    top_values = top_values / top_values.max()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 15))
    bars = ax.barh(top_features, top_values, color=colors, edgecolor='white')
    
    # Add value labels
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


def plot_feature_importance_comparison():
    """Compare feature importances across all models (beeswarm alternative)."""
    print("Generating Feature Importance Comparison (Beeswarm Alternative)...")
    
    feature_names = get_feature_names()
    
    # Load all models
    models = {}
    model_files = {'LightGBM': 'lgb', 'XGBoost': 'xgb', 'RandomForest': 'rf', 'GradientBoost': 'gb'}
    
    for name, abbr in model_files.items():
        model_path = MODEL_DIR / f'model_v6_{abbr}.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[name] = pickle.load(f)
    
    # Get all importances
    all_importances = {}
    for name, model in models.items():
        imp = model.feature_importances_
        # Normalize
        imp = imp / imp.max()
        all_importances[name] = imp
    
    # Average across models to get top features
    avg_imp = np.mean([imp for imp in all_importances.values()], axis=0)
    top_15_idx = np.argsort(avg_imp)[-15:][::-1]
    
    # Create multi-model comparison plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_pos = np.arange(15)
    width = 0.2
    colors = {'LightGBM': '#00CC88', 'XGBoost': '#0088FF', 
              'RandomForest': '#FF6644', 'GradientBoost': '#AA44FF'}
    
    for idx, (name, imp) in enumerate(all_importances.items()):
        offset = (idx - 1.5) * width
        values = imp[top_15_idx]
        ax.barh(y_pos + offset, values, width, label=name, 
                color=colors[name], alpha=0.8)
    
    # Set labels
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


def main():
    print("\n" + "=" * 60)
    print("  Feature Importance Visualizations")
    print("=" * 60 + "\n")
    
    plot_feature_importance_bar()
    plot_feature_importance_comparison()
    
    print("\n" + "=" * 60)
    print("  ✅ Feature Importance Plots Generated")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
