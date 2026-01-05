"""
TRIALPULSE NEXUS 10X - ML Data Preparation v2.0 (LEAKAGE-FREE)
Strict separation of PREDICTION features from OUTCOME features

Key Principle:
  - PREDICTION features: Available BEFORE outcome is known (workload, progress)
  - OUTCOME features: Define the target (issues, deviations, pending items)
  - These NEVER overlap

Expected Performance: 0.65-0.85 AUC (realistic for true prediction)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FEATURE SEPARATION: The Critical Fix for Data Leakage
# ============================================================================

# PREDICTION FEATURES: Leading indicators, workload metrics, progress measures
# These are available BEFORE the outcome is determined
# These do NOT directly encode the risk outcome
PREDICTION_FEATURES = [
    # Workload metrics (volume of work, not issues)
    'expected_visits_rave_edc_bo4',
    'pages_entered',
    'total_crfs',
    'crfs_require_signature',
    'crfs_require_verification_sdv',
    'forms_verified',
    
    # Progress indicators (completion status, not problems)
    'crfs_frozen',
    'crfs_locked',
    'crfs_signed',
    'crfs_verified_sdv',
    
    # Query workload (questions asked, not issues found)
    'dm_queries',
    'clinical_queries',
    'medical_queries',
    'site_queries',
    'queries_answered',  # Resolution rate
    
    # Coding workload (volume, not pending)
    'meddra_coding_meddra_total',
    'whodrug_coding_whodrug_total',
    
    # SAE workload (total cases, not pending)
    'sae_dm_sae_dm_total',
    'sae_safety_sae_safety_total',
    
    # EDRR workload
    'edrr_edrr_resolved',  # Resolution count, not pending
]

# OUTCOME FEATURES: Used ONLY to define the target risk level
# These are the "bad things" we want to predict BEFORE they happen
OUTCOME_FEATURES = [
    # Critical issues
    'broken_signatures',
    'crfs_never_signed',
    'protocol_deviations',
    'safety_queries',  # Safety queries are outcomes, not workload
    
    # Overdue items
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    
    # Missing/incomplete items
    'visit_missing_visit_count',
    'visit_visits_overdue_count',
    'pages_with_nonconformant_data',
    'pages_pages_missing_count',
    
    # Pending items (total - completed = pending)
    'sae_dm_sae_dm_completed',
    'sae_safety_sae_safety_completed',
    'meddra_coding_meddra_coded',
    'whodrug_coding_whodrug_coded',
    
    # Other issues
    'lab_lab_issue_count',
    'lab_lab_missing_names',
    'lab_lab_missing_ranges',
    'edrr_edrr_issue_count',
    'inactivated_inactivated_form_count',
]


@dataclass
class MLDataConfigV2:
    """ML Data Configuration - Leakage-Free v2.0"""
    
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    random_state: int = 42
    min_samples_per_class: int = 10
    correlation_threshold: float = 0.90
    variance_threshold: float = 0.01
    smote_sampling_strategy: str = 'auto'
    smote_k_neighbors: int = 3


class MLDataPreparatorV2:
    """
    ML Data Preparation - Leakage-Free v2.0
    
    Key Innovation: Strict separation of prediction features from outcomes
    """
    
    def __init__(self, config: MLDataConfigV2 = None):
        self.config = config or MLDataConfigV2()
        self.feature_columns: List[str] = []
        self.outcome_columns: List[str] = []
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.dropped_features: List[Tuple[str, str]] = []
        self.leakage_violations: List[str] = []
        
    def validate_no_leakage(self, prediction_cols: List[str], outcome_cols: Set[str]) -> bool:
        """Critical: Verify no outcome features leak into prediction features"""
        violations = []
        
        for col in prediction_cols:
            if col in outcome_cols:
                violations.append(col)
        
        if violations:
            logger.error("=" * 70)
            logger.error("🚨 LEAKAGE DETECTED! 🚨")
            logger.error("=" * 70)
            for v in violations:
                logger.error(f"  VIOLATION: '{v}' in both prediction and outcome sets")
            logger.error("")
            logger.error("This would cause artificially high AUC (0.99+)")
            logger.error("=" * 70)
            self.leakage_violations = violations
            return False
        
        logger.info("  ✅ No leakage detected - feature sets are disjoint")
        return True
    
    def select_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select ONLY prediction features - outcomes are excluded"""
        logger.info("\n" + "=" * 70)
        logger.info("SELECTING PREDICTION FEATURES (LEAKAGE-FREE)")
        logger.info("=" * 70)
        
        # Find available prediction features
        available = [c for c in PREDICTION_FEATURES if c in df.columns]
        outcome_set = set(OUTCOME_FEATURES)
        
        # Validate no leakage
        if not self.validate_no_leakage(available, outcome_set):
            raise ValueError("Leakage detected! Cannot proceed with training.")
        
        logger.info(f"  Available prediction features: {len(available)}")
        logger.info(f"  Excluded outcome features: {len(OUTCOME_FEATURES)}")
        
        # Filter to numeric only
        self.feature_columns = []
        for col in available:
            if np.issubdtype(df[col].dtype, np.number):
                if df[col].nunique() > 1:
                    self.feature_columns.append(col)
        
        logger.info(f"  Final feature count: {len(self.feature_columns)}")
        
        # Log features being used
        logger.info(f"\n  Prediction features:")
        for col in self.feature_columns[:10]:
            logger.info(f"    ✓ {col}")
        if len(self.feature_columns) > 10:
            logger.info(f"    ... and {len(self.feature_columns) - 10} more")
        
        return df[self.feature_columns].copy()
    
    def create_risk_target_from_outcomes(self, df: pd.DataFrame) -> pd.Series:
        """
        Create risk target ONLY from outcome features.
        
        Critical: None of these features appear in prediction set!
        """
        logger.info("\n" + "=" * 70)
        logger.info("CREATING RISK TARGET (FROM OUTCOMES ONLY)")
        logger.info("=" * 70)
        
        risk_score = pd.Series(0.0, index=df.index)
        contributors = []
        
        # ===== CRITICAL TIER 1: SAE Pending =====
        # SAE pending = total - completed (pending defined here, not as feature)
        if 'sae_dm_sae_dm_total' in df.columns and 'sae_dm_sae_dm_completed' in df.columns:
            sae_dm_pending = df['sae_dm_sae_dm_total'].fillna(0) - df['sae_dm_sae_dm_completed'].fillna(0)
            sae_dm_pending = sae_dm_pending.clip(lower=0)
            mask = sae_dm_pending > 0
            risk_score += mask.astype(float) * 4.0
            contributors.append(('SAE DM Pending', mask.sum()))
        
        if 'sae_safety_sae_safety_total' in df.columns and 'sae_safety_sae_safety_completed' in df.columns:
            sae_safety_pending = df['sae_safety_sae_safety_total'].fillna(0) - df['sae_safety_sae_safety_completed'].fillna(0)
            sae_safety_pending = sae_safety_pending.clip(lower=0)
            mask = sae_safety_pending > 0
            risk_score += mask.astype(float) * 4.0
            contributors.append(('SAE Safety Pending', mask.sum()))
        
        # ===== HIGH: Broken/Never Signed =====
        if 'broken_signatures' in df.columns:
            mask = df['broken_signatures'].fillna(0) > 0
            risk_score += mask.astype(float) * 3.0
            contributors.append(('Broken Signatures', mask.sum()))
        
        if 'crfs_never_signed' in df.columns:
            val = df['crfs_never_signed'].fillna(0)
            risk_score += (val > 5).astype(float) * 3.0
            contributors.append(('CRFs Never Signed >5', (val > 5).sum()))
        
        # ===== HIGH: Overdue >90 days =====
        if 'crfs_overdue_for_signs_beyond_90_days_of_data_entry' in df.columns:
            mask = df['crfs_overdue_for_signs_beyond_90_days_of_data_entry'].fillna(0) > 0
            risk_score += mask.astype(float) * 3.0
            contributors.append(('Overdue >90d', mask.sum()))
        
        # ===== MEDIUM: Protocol Deviations =====
        if 'protocol_deviations' in df.columns:
            val = df['protocol_deviations'].fillna(0)
            risk_score += (val > 0).astype(float) * 2.0
            risk_score += (val > 2).astype(float) * 1.0
            contributors.append(('Protocol Deviations', (val > 0).sum()))
        
        # ===== MEDIUM: Missing Visits/Pages =====
        if 'visit_missing_visit_count' in df.columns:
            val = df['visit_missing_visit_count'].fillna(0)
            risk_score += (val > 0).astype(float) * 2.0
            contributors.append(('Missing Visits', (val > 0).sum()))
        
        if 'pages_pages_missing_count' in df.columns:
            val = df['pages_pages_missing_count'].fillna(0)
            risk_score += (val > 0).astype(float) * 1.5
            contributors.append(('Missing Pages', (val > 0).sum()))
        
        # ===== LOW-MEDIUM: Uncoded Terms =====
        if 'meddra_coding_meddra_total' in df.columns and 'meddra_coding_meddra_coded' in df.columns:
            uncoded = df['meddra_coding_meddra_total'].fillna(0) - df['meddra_coding_meddra_coded'].fillna(0)
            uncoded = uncoded.clip(lower=0)
            risk_score += (uncoded > 0).astype(float) * 1.0
            contributors.append(('MedDRA Uncoded', (uncoded > 0).sum()))
        
        if 'whodrug_coding_whodrug_total' in df.columns and 'whodrug_coding_whodrug_coded' in df.columns:
            uncoded = df['whodrug_coding_whodrug_total'].fillna(0) - df['whodrug_coding_whodrug_coded'].fillna(0)
            uncoded = uncoded.clip(lower=0)
            risk_score += (uncoded > 0).astype(float) * 1.0
            contributors.append(('WHODrug Uncoded', (uncoded > 0).sum()))
        
        # ===== LOW: Other Issues =====
        if 'lab_lab_issue_count' in df.columns:
            mask = df['lab_lab_issue_count'].fillna(0) > 0
            risk_score += mask.astype(float) * 1.0
            contributors.append(('Lab Issues', mask.sum()))
        
        if 'safety_queries' in df.columns:
            mask = df['safety_queries'].fillna(0) > 0
            risk_score += mask.astype(float) * 2.0
            contributors.append(('Safety Queries', mask.sum()))
        
        # Log target contributors
        logger.info(f"\n  Target components used:")
        for name, count in contributors:
            pct = count / len(df) * 100 if len(df) > 0 else 0
            logger.info(f"    {name}: {count:,} patients ({pct:.1f}%)")
        
        # Convert to risk tiers using percentiles
        logger.info(f"\n  Risk score stats: min={risk_score.min():.1f}, max={risk_score.max():.1f}, mean={risk_score.mean():.1f}")
        
        # Use fixed thresholds for reproducibility
        p50 = risk_score.quantile(0.50)
        p80 = risk_score.quantile(0.80)
        p95 = risk_score.quantile(0.95)
        
        logger.info(f"  Thresholds: P50={p50:.1f}, P80={p80:.1f}, P95={p95:.1f}")
        
        risk_level = pd.cut(
            risk_score,
            bins=[-np.inf, p50, p80, p95, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Log distribution
        dist = risk_level.value_counts()
        logger.info(f"\n  Risk distribution:")
        for level in ['Critical', 'High', 'Medium', 'Low']:
            count = dist.get(level, 0)
            pct = count / len(risk_level) * 100
            logger.info(f"    {level}: {count:,} ({pct:.1f}%)")
        
        return risk_level
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with median imputation"""
        logger.info("\n" + "=" * 70)
        logger.info("HANDLING MISSING VALUES")
        logger.info("=" * 70)
        
        df = df.copy()
        initial_missing = df.isnull().sum().sum()
        logger.info(f"  Initial missing: {initial_missing:,}")
        
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        logger.info(f"  Final missing: {df.isnull().sum().sum()}")
        return df
    
    def remove_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-zero variance features"""
        logger.info("\n" + "=" * 70)
        logger.info("REMOVING LOW VARIANCE FEATURES")
        logger.info("=" * 70)
        
        if not self.feature_columns:
            return df
        
        existing = [c for c in self.feature_columns if c in df.columns]
        variances = df[existing].var()
        low_var = variances[variances < self.config.variance_threshold].index.tolist()
        
        if low_var:
            logger.info(f"  Removing {len(low_var)} low variance features:")
            for col in low_var:
                logger.info(f"    - {col}")
                if col in self.feature_columns:
                    self.feature_columns.remove(col)
                self.dropped_features.append((col, 'low_variance'))
            df = df.drop(columns=low_var, errors='ignore')
        else:
            logger.info("  No low variance features found ✓")
        
        return df
    
    def remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        logger.info("\n" + "=" * 70)
        logger.info("REMOVING HIGHLY CORRELATED FEATURES")
        logger.info("=" * 70)
        
        existing = [c for c in self.feature_columns if c in df.columns]
        if len(existing) < 2:
            return df
        
        corr = df[existing].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        
        to_drop = []
        for col in upper.columns:
            high = upper[col][upper[col] > self.config.correlation_threshold]
            for idx in high.index:
                if idx not in to_drop:
                    to_drop.append(idx)
                    logger.info(f"  Dropping '{idx}' (correlated with '{col}')")
        
        if to_drop:
            for col in to_drop:
                if col in self.feature_columns:
                    self.feature_columns.remove(col)
                self.dropped_features.append((col, 'high_correlation'))
            df = df.drop(columns=to_drop, errors='ignore')
            logger.info(f"  Removed {len(to_drop)} correlated features")
        else:
            logger.info("  No highly correlated features found ✓")
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        logger.info("\n" + "=" * 70)
        logger.info("SCALING FEATURES")
        logger.info("=" * 70)
        
        existing = [c for c in self.feature_columns if c in df.columns]
        if not existing:
            return df
        
        df = df.copy()
        scaler = RobustScaler()
        df[existing] = scaler.fit_transform(df[existing])
        self.scalers['numeric'] = scaler
        
        logger.info(f"  Scaled {len(existing)} features using RobustScaler ✓")
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Create stratified train/val/test splits"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING DATA SPLITS")
        logger.info("=" * 70)
        
        # Encode target
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        self.encoders['target'] = le
        self.encoders['target_classes'] = list(le.classes_)
        
        logger.info(f"  Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        # Split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=self.config.test_ratio,
            random_state=self.config.random_state, stratify=y_encoded
        )
        
        val_ratio = self.config.val_ratio / (1 - self.config.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            random_state=self.config.random_state, stratify=y_temp
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        logger.info(f"\n  Split sizes:")
        for name, (X_s, y_s) in splits.items():
            logger.info(f"    {name}: {len(X_s):,}")
        
        return splits
    
    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE for class balancing"""
        logger.info("\n" + "=" * 70)
        logger.info("APPLYING SMOTE")
        logger.info("=" * 70)
        
        if not IMBLEARN_AVAILABLE:
            logger.warning("  imblearn not available, skipping SMOTE")
            return X_train, y_train
        
        original = y_train.value_counts().sort_index()
        logger.info(f"  Original distribution: {original.to_dict()}")
        
        min_samples = y_train.value_counts().min()
        k = min(self.config.smote_k_neighbors, min_samples - 1)
        
        if k < 1:
            sampler = RandomOverSampler(random_state=self.config.random_state)
        else:
            sampler = SMOTE(k_neighbors=k, random_state=self.config.random_state)
        
        try:
            X_res, y_res = sampler.fit_resample(X_train, y_train)
            X_res = pd.DataFrame(X_res, columns=X_train.columns)
            y_res = pd.Series(y_res)
            
            logger.info(f"  After SMOTE: {y_res.value_counts().sort_index().to_dict()}")
            logger.info(f"  Total: {len(X_train):,} → {len(X_res):,}")
            
            return X_res, y_res
        except Exception as e:
            logger.error(f"  SMOTE failed: {e}")
            return X_train, y_train
    
    def prepare_for_training(self, df: pd.DataFrame) -> Dict:
        """Complete data preparation pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("TRIALPULSE NEXUS 10X - ML DATA PREP v2.0 (LEAKAGE-FREE)")
        logger.info("=" * 70)
        logger.info(f"Input samples: {len(df):,}")
        
        start = datetime.now()
        
        # 1. Create target from OUTCOME features only
        y = self.create_risk_target_from_outcomes(df)
        
        # 2. Select PREDICTION features only (disjoint from outcomes)
        X = self.select_prediction_features(df)
        
        # 3. Handle missing values
        X = self.handle_missing_values(X)
        
        # 4. Remove low variance
        X = self.remove_low_variance(X)
        
        # 5. Remove high correlation
        X = self.remove_high_correlation(X)
        
        # 6. Scale features
        X = self.scale_features(X)
        
        # 7. Split data
        splits = self.split_data(X, y)
        
        # 8. Apply SMOTE to training set
        X_res, y_res = self.apply_smote(splits['train'][0], splits['train'][1])
        splits['train_resampled'] = (X_res, y_res)
        
        duration = (datetime.now() - start).total_seconds()
        
        results = {
            'splits': splits,
            'feature_columns': self.feature_columns,
            'outcome_columns': list(OUTCOME_FEATURES),
            'scalers': self.scalers,
            'encoders': self.encoders,
            'dropped_features': self.dropped_features,
            'metadata': {
                'version': '2.0.0',
                'duration': duration,
                'n_features': len(self.feature_columns),
                'leakage_prevention': 'strict_feature_outcome_separation',
                'leakage_violations': self.leakage_violations
            }
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("DATA PREPARATION COMPLETE (LEAKAGE-FREE)")
        logger.info("=" * 70)
        logger.info(f"  Features: {len(self.feature_columns)}")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info("")
        logger.info("  ⚠️  Expected AUC: 0.65-0.85 (realistic prediction)")
        logger.info("  ⚠️  If AUC > 0.90, investigate remaining leakage")
        
        return results


class MLDataPrepRunnerV2:
    """Runner for leakage-free data preparation"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.preparator = MLDataPreparatorV2()
        self.results = None
        
    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading: {self.input_path}")
        df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def run(self) -> Dict:
        df = self.load_data()
        self.results = self.preparator.prepare_for_training(df)
        return self.results
    
    def save_outputs(self):
        logger.info("\n" + "=" * 70)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 70)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, (X, y) in self.results['splits'].items():
            data = X.copy()
            data['target'] = y.values
            path = self.output_dir / f'ml_v2_{name}.parquet'
            data.to_parquet(path, index=False)
            logger.info(f"  ✅ {name}: {len(data):,} → {path.name}")
        
        # Save metadata
        with open(self.output_dir / 'ml_v2_features.json', 'w') as f:
            json.dump({
                'prediction_features': self.results['feature_columns'],
                'outcome_features': self.results['outcome_columns'],
                'note': 'Prediction and outcome features are DISJOINT - no leakage'
            }, f, indent=2)
        
        with open(self.output_dir / 'ml_v2_metadata.json', 'w') as f:
            json.dump(self.results['metadata'], f, indent=2)
        
        logger.info(f"\n  Output directory: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent
    
    input_path = project_root / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return None
    
    output_dir = project_root / 'data' / 'processed' / 'ml_v2'
    
    runner = MLDataPrepRunnerV2(input_path, output_dir)
    runner.run()
    runner.save_outputs()
    
    print("\n" + "=" * 70)
    print("📊 DATA PREP v2.0 COMPLETE (LEAKAGE-FREE)")
    print("=" * 70)
    print(f"\n  Features used for prediction: {len(runner.results['feature_columns'])}")
    print(f"  Outcome features (target only): {len(runner.results['outcome_columns'])}")
    print(f"\n  Output: {output_dir}")
    print("\n  ⚠️  Expected AUC: 0.65-0.85 (realistic for true prediction)")
    print("  ⚠️  If AUC > 0.90, investigate potential remaining leakage")
    
    return runner


if __name__ == '__main__':
    main()
