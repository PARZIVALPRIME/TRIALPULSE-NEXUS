"""
TRIALPULSE NEXUS 10X - Anomaly Detection Engine v2.0 LITE
==========================================================
Memory-optimized version for systems with limited RAM.
Uses sampling and lightweight algorithms.
"""

import json
import logging
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyConfigLite:
    """Lightweight configuration for anomaly detection."""
    contamination: float = 0.05
    if_n_estimators: int = 100
    pca_components: int = 10
    weight_isolation_forest: float = 0.5
    weight_reconstruction: float = 0.3
    weight_zscore: float = 0.2
    severity_critical_percentile: float = 99
    severity_high_percentile: float = 95
    severity_medium_percentile: float = 90
    random_state: int = 42


FEATURE_WHITELIST = [
    'total_open_queries', 'dm_queries', 'clinical_queries',
    'total_crfs', 'crfs_signed', 'crfs_never_signed',
    'total_crfs_requiring_sdv', 'crfs_source_data_verified',
    'visit_missing_visit_count', 'pages_missing_page_count',
    'sae_dm_sae_dm_total', 'sae_safety_sae_safety_total',
    'meddra_total_terms', 'meddra_coded_terms',
]


class AnomalyDetectorLite:
    """Lightweight anomaly detector - memory efficient."""
    
    def __init__(self, config: Optional[AnomalyConfigLite] = None):
        self.config = config or AnomalyConfigLite()
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.feature_names = []
        self.fit_stats = {}
        self.is_fitted = False
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in FEATURE_WHITELIST if f in df.columns]
        X = df[available].copy()
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.feature_names = list(X.columns)
        return X
    
    def fit(self, df: pd.DataFrame) -> 'AnomalyDetectorLite':
        logger.info("=" * 60)
        logger.info("FITTING ANOMALY DETECTOR LITE")
        logger.info("=" * 60)
        
        X = self._prepare_features(df)
        n_samples = len(X)
        logger.info(f"Data: {n_samples:,} samples, {len(self.feature_names)} features")
        
        # Scale
        self.scalers['robust'] = RobustScaler()
        X_scaled = self.scalers['robust'].fit_transform(X)
        
        # Store stats for Z-scores
        self.fit_stats['means'] = X.mean().to_dict()
        self.fit_stats['stds'] = X.std().to_dict()
        
        # 1. Isolation Forest
        logger.info("\n1. Training Isolation Forest...")
        self.models['if'] = IsolationForest(
            n_estimators=self.config.if_n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        self.models['if'].fit(X_scaled)
        if_scores = self.models['if'].decision_function(X_scaled)
        logger.info(f"   IF done. Range: [{if_scores.min():.4f}, {if_scores.max():.4f}]")
        
        # 2. PCA Reconstruction
        logger.info("\n2. Training PCA Reconstruction...")
        n_comp = min(self.config.pca_components, len(self.feature_names) - 1)
        self.models['pca'] = PCA(n_components=n_comp, random_state=self.config.random_state)
        X_pca = self.models['pca'].fit_transform(X_scaled)
        X_recon = self.models['pca'].inverse_transform(X_pca)
        recon_errors = np.mean((X_scaled - X_recon) ** 2, axis=1)
        logger.info(f"   PCA done. Variance explained: {self.models['pca'].explained_variance_ratio_.sum():.2%}")
        
        # 3. Z-score
        logger.info("\n3. Computing Z-scores...")
        z_scores = np.zeros_like(X.values, dtype=float)
        for i, col in enumerate(X.columns):
            mean = self.fit_stats['means'][col]
            std = self.fit_stats['stds'].get(col, 1) or 1
            z_scores[:, i] = (X[col].values - mean) / std
        max_z = np.abs(z_scores).max(axis=1)
        
        # Compute ensemble
        ensemble = self._compute_ensemble(if_scores, recon_errors, max_z)
        
        # Set thresholds
        self.thresholds['critical'] = float(np.percentile(ensemble, self.config.severity_critical_percentile))
        self.thresholds['high'] = float(np.percentile(ensemble, self.config.severity_high_percentile))
        self.thresholds['medium'] = float(np.percentile(ensemble, self.config.severity_medium_percentile))
        
        self.is_fitted = True
        logger.info("\n✅ Fitting complete!")
        return self
    
    def _compute_ensemble(self, if_scores, recon_errors, z_scores):
        def norm(s, invert=False):
            mn, mx = s.min(), s.max()
            if mx - mn < 1e-10: return np.zeros_like(s)
            n = (s - mn) / (mx - mn)
            return 1 - n if invert else n
        
        if_norm = norm(if_scores, invert=True)
        recon_norm = norm(recon_errors)
        z_norm = norm(z_scores)
        
        return (self.config.weight_isolation_forest * if_norm +
                self.config.weight_reconstruction * recon_norm +
                self.config.weight_zscore * z_norm)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        logger.info("Computing predictions...")
        X = self._prepare_features(df)
        X_scaled = self.scalers['robust'].transform(X)
        
        # IF scores
        if_scores = self.models['if'].decision_function(X_scaled)
        
        # Reconstruction
        X_pca = self.models['pca'].transform(X_scaled)
        X_recon = self.models['pca'].inverse_transform(X_pca)
        recon_errors = np.mean((X_scaled - X_recon) ** 2, axis=1)
        
        # Z-scores
        z_scores = np.zeros_like(X.values, dtype=float)
        for i, col in enumerate(X.columns):
            mean = self.fit_stats['means'].get(col, 0)
            std = self.fit_stats['stds'].get(col, 1) or 1
            z_scores[:, i] = (X[col].values - mean) / std
        max_z = np.abs(z_scores).max(axis=1)
        
        # Ensemble
        ensemble = self._compute_ensemble(if_scores, recon_errors, max_z)
        
        # Severity
        severity = []
        for s in ensemble:
            if s >= self.thresholds['critical']:
                severity.append('Critical')
            elif s >= self.thresholds['high']:
                severity.append('High')
            elif s >= self.thresholds['medium']:
                severity.append('Medium')
            else:
                severity.append('Normal')
        
        results = pd.DataFrame({
            'patient_key': df['patient_key'].values if 'patient_key' in df.columns else df.index,
            'study_id': df['study_id'].values if 'study_id' in df.columns else 'Unknown',
            'site_id': df['site_id'].values if 'site_id' in df.columns else 'Unknown',
            'anomaly_score': ensemble,
            'score_if': -if_scores,
            'score_recon': recon_errors,
            'score_zscore': max_z,
            'is_anomaly': ensemble >= self.thresholds['medium'],
            'severity': severity,
        })
        
        logger.info(f"Anomalies: {results['is_anomaly'].sum():,} ({results['is_anomaly'].mean():.1%})")
        return results
    
    def aggregate_to_site(self, results: pd.DataFrame) -> pd.DataFrame:
        site_agg = results.groupby(['study_id', 'site_id']).agg({
            'patient_key': 'count',
            'anomaly_score': ['mean', 'max'],
            'is_anomaly': 'sum',
        }).reset_index()
        site_agg.columns = ['study_id', 'site_id', 'patients', 'avg_score', 'max_score', 'anomaly_count']
        site_agg['anomaly_rate'] = site_agg['anomaly_count'] / site_agg['patients']
        site_agg['severity'] = site_agg.apply(
            lambda r: 'Critical' if r['anomaly_rate'] > 0.2 else ('High' if r['anomaly_rate'] > 0.1 else 'Normal'), 
            axis=1
        )
        return site_agg


def run_lite(upr_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Run lightweight anomaly detection."""
    logger.info("=" * 70)
    logger.info("ANOMALY DETECTION LITE v2.0")
    logger.info("=" * 70)
    
    start = datetime.now()
    
    # Create output
    v2_dir = Path(output_dir) / 'v2'
    v2_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"\nLoading {upr_path}...")
    df = pd.read_parquet(upr_path)
    logger.info(f"Loaded {len(df):,} patients")
    
    # Train and predict
    detector = AnomalyDetectorLite()
    detector.fit(df)
    patient_results = detector.predict(df)
    site_results = detector.aggregate_to_site(patient_results)
    
    # Save
    logger.info("\nSaving results...")
    patient_results.to_parquet(v2_dir / 'patient_anomalies_v2.parquet', index=False)
    site_results.to_csv(v2_dir / 'site_anomalies_v2.csv', index=False)
    
    high_sev = patient_results[patient_results['severity'].isin(['Critical', 'High'])]
    high_sev.to_csv(v2_dir / 'high_severity_v2.csv', index=False)
    
    # Summary
    sev_dist = patient_results['severity'].value_counts().to_dict()
    summary = {
        'version': '2.0-lite',
        'total_patients': len(df),
        'anomalies': int(patient_results['is_anomaly'].sum()),
        'anomaly_rate': float(patient_results['is_anomaly'].mean()),
        'severity': {k: int(v) for k, v in sev_dist.items()},
        'sites': len(site_results),
        'critical_sites': int((site_results['severity'] == 'Critical').sum()),
        'duration': (datetime.now() - start).total_seconds(),
    }
    
    with open(v2_dir / 'summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save model
    with open(v2_dir / 'model_lite.pkl', 'wb') as f:
        pickle.dump({
            'models': detector.models,
            'scalers': detector.scalers,
            'thresholds': detector.thresholds,
            'feature_names': detector.feature_names,
            'fit_stats': detector.fit_stats,
        }, f)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Anomalies: {summary['anomalies']:,} ({summary['anomaly_rate']:.1%})")
    logger.info(f"Critical: {sev_dist.get('Critical', 0)}, High: {sev_dist.get('High', 0)}")
    logger.info(f"Duration: {summary['duration']:.1f}s")
    
    return summary


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import DATA_PROCESSED
    
    run_lite(
        DATA_PROCESSED / 'upr' / 'unified_patient_record.parquet',
        DATA_PROCESSED / 'analytics'
    )
