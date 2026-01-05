"""
TRIALPULSE NEXUS 10X — RESOLUTION TIME PREDICTOR
Quantile Regression for Issue Resolution Duration

OBJECTIVE: Predict days until issue resolution
TARGET: resolution_days = (response_date - open_date).days
OUTPUT: Point estimate + 80% prediction interval

TRAINING DATA: Resolved issues only
PREDICTION: For open issues

VERSION: RESOLUTION_TIME_v1
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
import os
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost required")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / 'data' / 'raw'
OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'resolution_time_predictor'

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

# Quantiles for prediction intervals
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_query_data() -> pd.DataFrame:
    """Load query data from all studies."""
    all_queries = []
    
    for study_folder in os.listdir(RAW_DIR):
        folder_path = RAW_DIR / study_folder
        if not folder_path.is_dir():
            continue
        
        for f in os.listdir(folder_path):
            if 'EDC_Metrics' in f and f.endswith('.xlsx'):
                try:
                    path = folder_path / f
                    df = pd.read_excel(path, sheet_name='Query Report - Cumulative')
                    df['_source_study'] = study_folder
                    all_queries.append(df)
                except Exception as e:
                    pass
    
    if not all_queries:
        return pd.DataFrame()
    
    queries_df = pd.concat(all_queries, ignore_index=True)
    return queries_df


def prepare_resolution_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare resolved and open issue datasets."""
    
    # Convert dates
    df['Query Open Date'] = pd.to_datetime(df['Query Open Date'], errors='coerce')
    df['Query Response Date'] = pd.to_datetime(df['Query Response Date'], errors='coerce')
    
    # Calculate resolution days for resolved queries
    df['resolution_days'] = (df['Query Response Date'] - df['Query Open Date']).dt.days
    
    # Split into resolved (for training) and open (for prediction)
    resolved_mask = (df['Query Status'] == 'Answered') & (df['resolution_days'].notna()) & (df['resolution_days'] >= 0)
    resolved = df[resolved_mask].copy()
    
    open_mask = df['Query Status'] == 'Open'
    open_issues = df[open_mask].copy()
    
    return resolved, open_issues


# ============================================================================
# FEATURE ENGINEERING (Leakage-Free)
# ============================================================================

# FORBIDDEN FEATURES (post-resolution information)
FORBIDDEN_FEATURES = {
    'Query Response Date',
    'Query Status',
    'resolution_days',
    '# Days Since Response',
    'action_status',
    'review_status',
    'closed',
    'resolved'
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer leakage-free features available at issue creation time."""
    features = pd.DataFrame(index=df.index)
    
    # Issue metadata
    if 'Action Owner' in df.columns:
        features['owner_dm'] = (df['Action Owner'].astype(str).str.lower().str.contains('dm|data', na=False)).astype(int)
        features['owner_site'] = (df['Action Owner'].astype(str).str.lower().str.contains('site', na=False)).astype(int)
        features['owner_cra'] = (df['Action Owner'].astype(str).str.lower().str.contains('cra', na=False)).astype(int)
    
    if 'Marking Group Name' in df.columns:
        # Extract marking group type
        features['marking_dm'] = (df['Marking Group Name'].astype(str).str.lower().str.contains('dm|data', na=False)).astype(int)
        features['marking_clinical'] = (df['Marking Group Name'].astype(str).str.lower().str.contains('clinical', na=False)).astype(int)
        features['marking_safety'] = (df['Marking Group Name'].astype(str).str.lower().str.contains('safety', na=False)).astype(int)
    
    if 'Form' in df.columns:
        # Form complexity proxy
        features['form_ae'] = (df['Form'].astype(str).str.lower().str.contains('ae|adverse', na=False)).astype(int)
        features['form_lab'] = (df['Form'].astype(str).str.lower().str.contains('lab', na=False)).astype(int)
        features['form_vital'] = (df['Form'].astype(str).str.lower().str.contains('vital', na=False)).astype(int)
        features['form_med'] = (df['Form'].astype(str).str.lower().str.contains('med|drug', na=False)).astype(int)
    
    # Time context (from issue creation, NOT resolution)
    if 'Query Open Date' in df.columns:
        open_date = pd.to_datetime(df['Query Open Date'], errors='coerce')
        features['open_day_of_week'] = open_date.dt.dayofweek.fillna(0)
        features['open_hour'] = open_date.dt.hour.fillna(12)
        features['open_month'] = open_date.dt.month.fillna(6)
        features['open_is_weekend'] = (features['open_day_of_week'] >= 5).astype(int)
        features['open_is_end_of_month'] = (open_date.dt.day >= 25).astype(int).fillna(0)
    
    # Site context (historical, not outcome-based)
    if 'Site Number' in df.columns and 'Query Open Date' in df.columns:
        # Site query volume (count of queries per site - available at issue creation)
        site_counts = df.groupby('Site Number').size().reset_index(name='site_query_count')
        df_with_site = df[['Site Number']].merge(site_counts, on='Site Number', how='left')
        features['site_query_count'] = df_with_site['site_query_count'].values
        features['site_query_count'] = features['site_query_count'].fillna(1)
        features['high_volume_site'] = (features['site_query_count'] > features['site_query_count'].median()).astype(int)
    
    # Study context
    if 'Study' in df.columns:
        study_counts = df.groupby('Study').size().reset_index(name='study_query_count')
        df_with_study = df[['Study']].merge(study_counts, on='Study', how='left')
        features['study_query_count'] = df_with_study['study_query_count'].values
        features['study_query_count'] = features['study_query_count'].fillna(1)
    
    # Log # (complexity proxy)
    if 'Log #' in df.columns:
        features['has_log'] = df['Log #'].notna().astype(int)
    
    # Clean up
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
    
    return features


# ============================================================================
# QUANTILE REGRESSION MODELS
# ============================================================================

def train_quantile_model(X_train, y_train, quantile: float) -> xgb.XGBRegressor:
    """Train a single quantile model."""
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='reg:quantileerror',
        quantile_alpha=quantile,
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    
    return model


def train_all_quantile_models(X_train, y_train) -> Dict[float, xgb.XGBRegressor]:
    """Train models for all quantiles."""
    models = {}
    
    for q in QUANTILES:
        log.info(f"    Training quantile {q}")
        models[q] = train_quantile_model(X_train, y_train, q)
    
    return models


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(y_true, predictions: Dict[float, np.ndarray]) -> Dict:
    """Evaluate quantile predictions."""
    
    median_pred = predictions[0.5]
    lower_pred = predictions[0.1]
    upper_pred = predictions[0.9]
    
    # Accuracy metrics
    mae = mean_absolute_error(y_true, median_pred)
    rmse = np.sqrt(mean_squared_error(y_true, median_pred))
    
    # Interval coverage (target: 80%)
    in_interval = (y_true >= lower_pred) & (y_true <= upper_pred)
    coverage = in_interval.mean()
    
    # Interval width
    interval_width = (upper_pred - lower_pred).mean()
    
    # Sharpness (narrower is better, but must maintain coverage)
    sharpness = interval_width / (y_true.std() + 1)
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'coverage': float(coverage),
        'interval_width': float(interval_width),
        'sharpness': float(sharpness),
        'median_prediction': float(median_pred.mean()),
        'actual_mean': float(y_true.mean()),
        'n_samples': len(y_true)
    }


def run_red_flag_checks(metrics: Dict) -> List[str]:
    """Run red flag detection."""
    flags = []
    
    if metrics['mae'] < 1.0:
        flags.append('SUSPICIOUS: MAE < 1 day (too good)')
    
    if metrics['interval_width'] < 2.0:
        flags.append('SUSPICIOUS: Interval width < 2 days (too narrow)')
    
    if metrics['coverage'] > 0.95:
        flags.append('WARNING: Coverage > 95% (too conservative)')
    
    if metrics['coverage'] < 0.70:
        flags.append('WARNING: Coverage < 70% (underconfident)')
    
    if metrics['sharpness'] > 2.0:
        flags.append('WARNING: Sharpness too high (wide intervals)')
    
    return flags


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(y_test, predictions: Dict[float, np.ndarray], 
                          metrics: Dict, output_dir: Path):
    """Create evaluation visualizations."""
    
    median_pred = predictions[0.5]
    lower_pred = predictions[0.1]
    upper_pred = predictions[0.9]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(y_test, median_pred, alpha=0.3, s=20)
    max_val = max(y_test.max(), median_pred.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual Resolution Days')
    ax1.set_ylabel('Predicted Resolution Days')
    ax1.set_title(f'Predicted vs Actual\nMAE={metrics["mae"]:.1f}, RMSE={metrics["rmse"]:.1f}')
    ax1.legend()
    
    # 2. Prediction Interval Coverage
    ax2 = axes[0, 1]
    in_interval = (y_test >= lower_pred) & (y_test <= upper_pred)
    ax2.bar(['In Interval', 'Outside'], [in_interval.sum(), (~in_interval).sum()], 
            color=['green', 'red'], alpha=0.7)
    ax2.set_title(f'80% Prediction Interval Coverage\nActual: {metrics["coverage"]:.1%}')
    ax2.set_ylabel('Count')
    
    # 3. Error Distribution
    ax3 = axes[1, 0]
    errors = y_test - median_pred
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', label='Zero error')
    ax3.set_xlabel('Prediction Error (days)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution')
    ax3.legend()
    
    # 4. Interval Width by Actual Duration
    ax4 = axes[1, 1]
    interval_widths = upper_pred - lower_pred
    ax4.scatter(y_test, interval_widths, alpha=0.3, s=20)
    ax4.set_xlabel('Actual Resolution Days')
    ax4.set_ylabel('Prediction Interval Width')
    ax4.set_title(f'Interval Width\nMean: {metrics["interval_width"]:.1f} days')
    
    plt.suptitle('RESOLUTION TIME PREDICTOR — Evaluation', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / 'evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    log.info("  Visualizations saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_resolution_time_pipeline():
    """Run resolution time prediction pipeline."""
    start = datetime.now()
    
    print("\n" + "=" * 70)
    print("  RESOLUTION TIME PREDICTOR")
    print("=" * 70)
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Quantile Regression for Issue Resolution Duration")
    print("=" * 70 + "\n")
    
    log.info("=" * 60)
    log.info("RESOLUTION TIME PREDICTOR")
    log.info("=" * 60)
    
    # ======================================================================
    # STEP 1: Load Data
    # ======================================================================
    log.info("\n[1/6] Loading query data...")
    queries = load_query_data()
    
    if len(queries) == 0:
        log.error("  No query data found!")
        return None, None
    
    log.info(f"  Loaded {len(queries):,} total queries")
    log.info(f"  Status distribution:")
    for status, count in queries['Query Status'].value_counts().items():
        log.info(f"    {status}: {count:,}")
    
    # ======================================================================
    # STEP 2: Prepare Data
    # ======================================================================
    log.info("\n[2/6] Preparing resolution data...")
    resolved, open_issues = prepare_resolution_data(queries)
    
    log.info(f"  Resolved queries (training): {len(resolved):,}")
    log.info(f"  Open queries (prediction):   {len(open_issues):,}")
    
    if len(resolved) < 100:
        log.warning("  Insufficient resolved queries for reliable training!")
    
    # Resolution time statistics
    resolution_stats = resolved['resolution_days'].describe()
    log.info(f"\n  Resolution Time Statistics:")
    log.info(f"    Mean:   {resolution_stats['mean']:.1f} days")
    log.info(f"    Median: {resolution_stats['50%']:.1f} days")
    log.info(f"    Std:    {resolution_stats['std']:.1f} days")
    log.info(f"    Min:    {resolution_stats['min']:.1f} days")
    log.info(f"    Max:    {resolution_stats['max']:.1f} days")
    
    # ======================================================================
    # STEP 3: Feature Engineering
    # ======================================================================
    log.info("\n[3/6] Engineering features (leakage-free)...")
    X = engineer_features(resolved)
    y = resolved['resolution_days'].values
    
    log.info(f"  Created {len(X.columns)} features:")
    for col in X.columns:
        log.info(f"    • {col}")
    
    # ======================================================================
    # STEP 4: Train/Test Split
    # ======================================================================
    log.info("\n[4/6] Splitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    log.info(f"  Train: {len(X_train):,}")
    log.info(f"  Val:   {len(X_val):,}")
    log.info(f"  Test:  {len(X_test):,}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X.columns, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    # ======================================================================
    # STEP 5: Train Quantile Models
    # ======================================================================
    log.info("\n[5/6] Training quantile models...")
    models = train_all_quantile_models(X_train_s.values, y_train)
    
    # Get predictions
    predictions = {}
    for q in QUANTILES:
        predictions[q] = models[q].predict(X_test_s.values)
    
    # ======================================================================
    # STEP 6: Evaluate
    # ======================================================================
    log.info("\n[6/6] Evaluating...")
    metrics = evaluate_predictions(y_test, predictions)
    
    log.info(f"\n  METRICS:")
    log.info(f"    MAE:           {metrics['mae']:.2f} days")
    log.info(f"    RMSE:          {metrics['rmse']:.2f} days")
    log.info(f"    Coverage:      {metrics['coverage']:.1%}")
    log.info(f"    Interval Width: {metrics['interval_width']:.1f} days")
    
    # Red flag checks
    red_flags = run_red_flag_checks(metrics)
    if red_flags:
        log.warning("\n  RED FLAGS:")
        for flag in red_flags:
            log.warning(f"    {flag}")
    else:
        log.info("\n  No red flags detected ✓")
    
    metrics['red_flags'] = red_flags
    metrics['passed'] = len(red_flags) == 0
    
    # Visualizations
    create_visualizations(y_test, predictions, metrics, OUTPUT_DIR / 'figures')
    
    # ======================================================================
    # Save Models
    # ======================================================================
    log.info("\n  Saving models...")
    
    for q, model in models.items():
        with open(OUTPUT_DIR / 'models' / f'quantile_{q:.2f}.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    with open(OUTPUT_DIR / 'models' / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Feature importance (convert numpy floats to Python floats)
    importance = {k: float(v) for k, v in zip(X.columns, models[0.5].feature_importances_)}
    top_features = [(k, float(v)) for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)]
    
    # Config
    config = {
        'version': 'RESOLUTION_TIME_v1',
        'created': datetime.now().isoformat(),
        'quantiles': QUANTILES,
        'n_training_samples': len(X_train),
        'n_test_samples': len(X_test),
        'features': list(X.columns),
        'metrics': metrics,
        'top_features': top_features[:5],
        'resolution_stats': {
            'mean': float(resolution_stats['mean']),
            'median': float(resolution_stats['50%']),
            'std': float(resolution_stats['std']),
            'min': float(resolution_stats['min']),
            'max': float(resolution_stats['max'])
        }
    }
    
    with open(OUTPUT_DIR / 'models' / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Performance table
    perf_df = pd.DataFrame([{
        'Metric': 'MAE',
        'Value': f"{metrics['mae']:.2f}",
        'Unit': 'days'
    }, {
        'Metric': 'RMSE',
        'Value': f"{metrics['rmse']:.2f}",
        'Unit': 'days'
    }, {
        'Metric': 'Coverage (80% target)',
        'Value': f"{metrics['coverage']:.1%}",
        'Unit': ''
    }, {
        'Metric': 'Interval Width',
        'Value': f"{metrics['interval_width']:.1f}",
        'Unit': 'days'
    }])
    
    perf_df.to_csv(OUTPUT_DIR / 'tables' / 'performance.csv', index=False)
    
    duration = (datetime.now() - start).total_seconds()
    
    # Summary
    status = "PASSED" if metrics['passed'] else "FLAGGED"
    
    print("\n" + "=" * 70)
    print(f"  RESOLUTION TIME PREDICTOR — {status}")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s")
    print(f"\n  DATA:")
    print(f"    Resolved queries: {len(resolved):,}")
    print(f"    Features:         {len(X.columns)}")
    print(f"\n  METRICS:")
    print(f"    MAE:            {metrics['mae']:.2f} days")
    print(f"    RMSE:           {metrics['rmse']:.2f} days")
    print(f"    Coverage:       {metrics['coverage']:.1%} (target: 80%)")
    print(f"    Interval Width: {metrics['interval_width']:.1f} days")
    
    if red_flags:
        print(f"\n  RED FLAGS ({len(red_flags)}):")
        for flag in red_flags:
            print(f"    • {flag}")
    else:
        print(f"\n  No red flags ✓")
    
    print(f"\n  SAMPLE PREDICTION:")
    sample_idx = 0
    print(f"    Actual:     {y_test[sample_idx]:.0f} days")
    print(f"    Predicted:  {predictions[0.5][sample_idx]:.0f} days")
    print(f"    80% Range:  {predictions[0.1][sample_idx]:.0f} - {predictions[0.9][sample_idx]:.0f} days")
    
    print(f"\n  Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")
    
    return metrics, config


if __name__ == '__main__':
    run_resolution_time_pipeline()
