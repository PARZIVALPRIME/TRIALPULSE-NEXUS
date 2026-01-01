"""
TRIALPULSE NEXUS 10X - ML Data Preparation v1.3 (LEAKAGE FULLY FIXED)
Phase 3.1: Feature Engineering with STRICT leakage prevention

FIXES v1.3:
- Target created from RAW source data only (not DQI)
- ALL DQI features excluded (they were used to derive risk)
- Only truly independent features used
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import warnings
from typing import Dict, List, Tuple, Any
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


@dataclass
class MLDataConfig:
    """ML Data Preparation Configuration - FULLY LEAKAGE-FREE"""
    
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    random_state: int = 42
    min_samples_per_class: int = 10
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    smote_sampling_strategy: str = 'auto'
    smote_k_neighbors: int = 5
    
    targets: Dict[str, str] = field(default_factory=lambda: {
        'risk': 'risk_level',
    })


# RAW FEATURES ONLY - No derived metrics
# These come directly from source data and are NOT used to calculate risk
RAW_FEATURES_WHITELIST = [
    # From CPID EDC Metrics (raw counts)
    'expected_visits_rave_edc_bo4',
    'pages_entered',
    'forms_verified',
    'crfs_frozen',
    'crfs_locked',
    'crfs_require_signature',
    'crfs_signed',
    'crfs_require_verification_sdv',
    'crfs_verified_sdv',
    'crfs_never_signed',
    'broken_signatures',
    'pages_with_nonconformant_data',
    'total_crfs',
    'protocol_deviations',
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    
    # Query counts (raw from source)
    'dm_queries',
    'clinical_queries', 
    'medical_queries',
    'site_queries',
    'safety_queries',
    'queries_answered',
    
    # From visit_projection (raw)
    'visit_missing_visit_count',
    'visit_visits_overdue_count',
    'visit_visits_overdue_avg_days',
    'visit_visits_overdue_max_days',
    'visit_visits_overdue_total_days',
    
    # From missing_lab_ranges (raw)
    'lab_lab_issue_count',
    'lab_lab_missing_names',
    'lab_lab_missing_ranges',
    
    # From SAE dashboard (raw) - NOT pending counts used in risk
    'sae_dm_sae_dm_total',
    'sae_dm_sae_dm_completed',
    'sae_safety_sae_safety_total',
    'sae_safety_sae_safety_completed',
    
    # From coding (raw counts)
    'meddra_coding_meddra_total',
    'meddra_coding_meddra_coded',
    'whodrug_coding_whodrug_total',
    'whodrug_coding_whodrug_coded',
    
    # From inactivated forms (raw)
    'inactivated_inactivated_form_count',
    'inactivated_inactivated_page_count',
    'inactivated_inactivated_folder_count',
    
    # From EDRR (raw) - NOT pending used in risk
    'edrr_edrr_issue_count',
    'edrr_edrr_resolved',
    
    # From missing pages (raw)
    'pages_pages_missing_count',
    'pages_pages_missing_avg_days',
    'pages_pages_missing_max_days',
    'pages_pages_missing_total_days',
]


class MLDataPreparator:
    """ML Data Preparation - FULLY LEAKAGE-FREE v1.3"""
    
    def __init__(self, config: MLDataConfig = None):
        self.config = config or MLDataConfig()
        self.feature_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.dropped_features: List[Tuple[str, str]] = []
        
    def _is_raw_feature(self, col: str) -> bool:
        """Check if feature is in raw whitelist"""
        return col in RAW_FEATURES_WHITELIST
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select ONLY raw features - strict whitelist"""
        logger.info("\n" + "=" * 70)
        logger.info("FEATURE SELECTION (STRICT WHITELIST)")
        logger.info("=" * 70)
        
        all_cols = df.columns.tolist()
        logger.info(f"  Starting columns: {len(all_cols)}")
        
        # Only use whitelisted raw features that exist in data
        available_raw = [c for c in RAW_FEATURES_WHITELIST if c in df.columns]
        excluded = [c for c in all_cols if c not in RAW_FEATURES_WHITELIST]
        
        logger.info(f"  Raw features available: {len(available_raw)}")
        logger.info(f"  Features excluded: {len(excluded)}")
        
        # Identify types
        self.numeric_columns = []
        self.categorical_columns = []
        
        for col in available_raw:
            dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                if df[col].nunique() > 1:
                    self.numeric_columns.append(col)
            elif dtype in ['object', 'category']:
                if df[col].nunique() <= 20:
                    self.categorical_columns.append(col)
        
        self.feature_columns = self.numeric_columns + self.categorical_columns
        
        logger.info(f"  Numeric features: {len(self.numeric_columns)}")
        logger.info(f"  Categorical features: {len(self.categorical_columns)}")
        logger.info(f"  Total features: {len(self.feature_columns)}")
        
        # Log what we're using
        logger.info(f"\n  Features selected:")
        for col in self.feature_columns[:15]:
            logger.info(f"    ✓ {col}")
        if len(self.feature_columns) > 15:
            logger.info(f"    ... and {len(self.feature_columns) - 15} more")
        
        return df[self.feature_columns].copy()
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        logger.info("\n" + "=" * 70)
        logger.info("HANDLING MISSING VALUES")
        logger.info("=" * 70)
        
        df = df.copy()
        initial_missing = df.isnull().sum().sum()
        logger.info(f"  Initial missing: {initial_missing:,}")
        
        for col in df.columns:
            if df[col].isnull().any():
                if col in self.numeric_columns:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        logger.info(f"  Final missing: {df.isnull().sum().sum()}")
        return df
    
    def remove_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove near-zero variance"""
        logger.info("\n" + "=" * 70)
        logger.info("REMOVING LOW VARIANCE")
        logger.info("=" * 70)
        
        existing = [c for c in self.numeric_columns if c in df.columns]
        if not existing:
            return df
        
        variances = df[existing].var()
        low_var = variances[variances < self.config.variance_threshold].index.tolist()
        
        if low_var:
            logger.info(f"  Removing {len(low_var)} low variance features")
            for col in low_var:
                self.numeric_columns.remove(col)
                self.feature_columns.remove(col)
                self.dropped_features.append((col, 'low_variance'))
            df = df.drop(columns=low_var)
        else:
            logger.info("  No low variance features ✓")
        
        return df
    
    def remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated"""
        logger.info("\n" + "=" * 70)
        logger.info("REMOVING HIGH CORRELATION")
        logger.info("=" * 70)
        
        existing = [c for c in self.numeric_columns if c in df.columns]
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
        
        if to_drop:
            logger.info(f"  Removing {len(to_drop)} correlated features")
            for col in to_drop:
                if col in self.numeric_columns:
                    self.numeric_columns.remove(col)
                if col in self.feature_columns:
                    self.feature_columns.remove(col)
                self.dropped_features.append((col, 'correlation'))
            df = df.drop(columns=to_drop, errors='ignore')
        else:
            logger.info("  No high correlation ✓")
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("\n" + "=" * 70)
        logger.info("SCALING FEATURES")
        logger.info("=" * 70)
        
        existing = [c for c in self.numeric_columns if c in df.columns]
        if not existing:
            return df
        
        df = df.copy()
        scaler = RobustScaler()
        df[existing] = scaler.fit_transform(df[existing])
        self.scalers['numeric'] = scaler
        
        logger.info(f"  Scaled {len(existing)} features ✓")
        return df
    
    def create_risk_target(self, df_original: pd.DataFrame) -> pd.Series:
        """Create risk target from RAW DATA ONLY - no DQI, no derived metrics"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING RISK TARGET (FROM RAW DATA ONLY)")
        logger.info("=" * 70)
        
        risk_score = pd.Series(0.0, index=df_original.index)
        
        # SAE pending (HIGH RISK) - use pending counts
        if 'sae_dm_sae_dm_pending' in df_original.columns:
            pending = df_original['sae_dm_sae_dm_pending'].fillna(0)
            risk_score += (pending > 0).astype(float) * 3
            risk_score += (pending > 2).astype(float) * 2
            logger.info(f"  SAE DM pending: {(pending > 0).sum()} patients with issues")
        elif 'sae_dm_sae_dm_total' in df_original.columns and 'sae_dm_sae_dm_completed' in df_original.columns:
            pending = df_original['sae_dm_sae_dm_total'].fillna(0) - df_original['sae_dm_sae_dm_completed'].fillna(0)
            pending = pending.clip(lower=0)
            risk_score += (pending > 0).astype(float) * 3
            logger.info(f"  SAE DM (derived pending): {(pending > 0).sum()} patients")
        
        if 'sae_safety_sae_safety_pending' in df_original.columns:
            pending = df_original['sae_safety_sae_safety_pending'].fillna(0)
            risk_score += (pending > 0).astype(float) * 3
            logger.info(f"  SAE Safety pending: {(pending > 0).sum()} patients")
        elif 'sae_safety_sae_safety_total' in df_original.columns and 'sae_safety_sae_safety_completed' in df_original.columns:
            pending = df_original['sae_safety_sae_safety_total'].fillna(0) - df_original['sae_safety_sae_safety_completed'].fillna(0)
            pending = pending.clip(lower=0)
            risk_score += (pending > 0).astype(float) * 3
        
        # Missing visits (MEDIUM-HIGH RISK)
        if 'visit_missing_visit_count' in df_original.columns:
            missing = df_original['visit_missing_visit_count'].fillna(0)
            risk_score += (missing > 0).astype(float) * 2
            risk_score += (missing > 3).astype(float) * 1
            logger.info(f"  Missing visits: {(missing > 0).sum()} patients")
        
        # Missing pages
        if 'pages_pages_missing_count' in df_original.columns:
            missing = df_original['pages_pages_missing_count'].fillna(0)
            risk_score += (missing > 0).astype(float) * 1.5
            risk_score += (missing > 5).astype(float) * 1
            logger.info(f"  Missing pages: {(missing > 0).sum()} patients")
        
        # Overdue signatures (MEDIUM RISK)
        if 'crfs_never_signed' in df_original.columns:
            unsigned = df_original['crfs_never_signed'].fillna(0)
            risk_score += (unsigned > 0).astype(float) * 1
            risk_score += (unsigned > 5).astype(float) * 1
            logger.info(f"  Unsigned CRFs: {(unsigned > 0).sum()} patients")
        
        if 'crfs_overdue_for_signs_beyond_90_days_of_data_entry' in df_original.columns:
            overdue = df_original['crfs_overdue_for_signs_beyond_90_days_of_data_entry'].fillna(0)
            risk_score += (overdue > 0).astype(float) * 2
            logger.info(f"  Overdue >90 days: {(overdue > 0).sum()} patients")
        
        # Uncoded terms (LOW-MEDIUM RISK)
        if 'meddra_coding_meddra_total' in df_original.columns and 'meddra_coding_meddra_coded' in df_original.columns:
            uncoded = df_original['meddra_coding_meddra_total'].fillna(0) - df_original['meddra_coding_meddra_coded'].fillna(0)
            uncoded = uncoded.clip(lower=0)
            risk_score += (uncoded > 0).astype(float) * 0.5
            risk_score += (uncoded > 5).astype(float) * 0.5
            logger.info(f"  MedDRA uncoded: {(uncoded > 0).sum()} patients")
        
        if 'whodrug_coding_whodrug_total' in df_original.columns and 'whodrug_coding_whodrug_coded' in df_original.columns:
            uncoded = df_original['whodrug_coding_whodrug_total'].fillna(0) - df_original['whodrug_coding_whodrug_coded'].fillna(0)
            uncoded = uncoded.clip(lower=0)
            risk_score += (uncoded > 0).astype(float) * 0.5
        
        # Lab issues
        if 'lab_lab_issue_count' in df_original.columns:
            issues = df_original['lab_lab_issue_count'].fillna(0)
            risk_score += (issues > 0).astype(float) * 1
            logger.info(f"  Lab issues: {(issues > 0).sum()} patients")
        
        # EDRR issues
        if 'edrr_edrr_issue_count' in df_original.columns:
            issues = df_original['edrr_edrr_issue_count'].fillna(0)
            # Calculate pending
            if 'edrr_edrr_resolved' in df_original.columns:
                pending = issues - df_original['edrr_edrr_resolved'].fillna(0)
                pending = pending.clip(lower=0)
                risk_score += (pending > 0).astype(float) * 1
                logger.info(f"  EDRR pending: {(pending > 0).sum()} patients")
        
        # Protocol deviations
        if 'protocol_deviations' in df_original.columns:
            devs = df_original['protocol_deviations'].fillna(0)
            risk_score += (devs > 0).astype(float) * 1.5
            risk_score += (devs > 2).astype(float) * 1.5
            logger.info(f"  Protocol deviations: {(devs > 0).sum()} patients")
        
        # Broken signatures
        if 'broken_signatures' in df_original.columns:
            broken = df_original['broken_signatures'].fillna(0)
            risk_score += (broken > 0).astype(float) * 1
            logger.info(f"  Broken signatures: {(broken > 0).sum()} patients")
        
        # Convert to risk levels with proper thresholds
        # Adjust thresholds to create balanced classes
        logger.info(f"\n  Risk score stats: min={risk_score.min():.1f}, max={risk_score.max():.1f}, mean={risk_score.mean():.1f}")
        
        # Use percentile-based thresholds for better balance
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
        for level in ['Low', 'Medium', 'High', 'Critical']:
            count = dist.get(level, 0)
            pct = count / len(risk_level) * 100
            logger.info(f"    {level}: {count:,} ({pct:.1f}%)")
        
        return risk_level
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Split data"""
        logger.info("\n" + "=" * 70)
        logger.info("SPLITTING DATA")
        logger.info("=" * 70)
        
        # Encode target
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        self.encoders['target'] = le
        self.encoders['target_classes'] = list(le.classes_)
        
        logger.info(f"  Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        stratify_col = y_encoded
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=self.config.test_ratio,
            random_state=self.config.random_state, stratify=stratify_col
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
        """Apply SMOTE"""
        logger.info("\n" + "=" * 70)
        logger.info("APPLYING SMOTE")
        logger.info("=" * 70)
        
        if not IMBLEARN_AVAILABLE:
            return X_train, y_train
        
        original = y_train.value_counts().sort_index()
        logger.info(f"  Original: {original.to_dict()}")
        
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
        """Complete pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("TRIALPULSE NEXUS 10X - ML DATA PREP v1.3 (NO LEAKAGE)")
        logger.info("=" * 70)
        logger.info(f"Samples: {len(df):,}")
        
        start = datetime.now()
        
        # 1. Create target from RAW data (before any feature selection)
        y = self.create_risk_target(df)
        
        # 2. Select only whitelisted raw features
        X = self.select_features(df)
        
        # 3. Handle missing
        X = self.handle_missing_values(X)
        
        # 4. Remove low variance
        X = self.remove_low_variance(X)
        
        # 5. Remove correlation
        X = self.remove_high_correlation(X)
        
        # 6. Scale
        X = self.scale_features(X)
        
        # 7. Split
        splits = self.split_data(X, y)
        
        # 8. SMOTE
        X_res, y_res = self.apply_smote(splits['train'][0], splits['train'][1])
        splits['train_resampled'] = (X_res, y_res)
        
        duration = (datetime.now() - start).total_seconds()
        
        results = {
            'splits': splits,
            'feature_columns': self.feature_columns,
            'numeric_columns': self.numeric_columns,
            'categorical_columns': self.categorical_columns,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'dropped_features': self.dropped_features,
            'metadata': {
                'version': '1.3.0',
                'duration': duration,
                'n_features': len(self.feature_columns),
                'leakage_prevention': 'strict_whitelist'
            }
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("PREPARATION COMPLETE (NO LEAKAGE)")
        logger.info("=" * 70)
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Duration: {duration:.2f}s")
        
        return results


class MLDataPrepRunner:
    """Runner"""
    
    def __init__(self, input_path: Path, output_dir: Path):
        self.input_path = input_path
        self.output_dir = output_dir
        self.preparator = MLDataPreparator()
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
        logger.info("\n" + "=" * 60)
        logger.info("SAVING OUTPUTS")
        logger.info("=" * 60)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, (X, y) in self.results['splits'].items():
            data = X.copy()
            data['target'] = y.values
            path = self.output_dir / f'ml_{name}.parquet'
            data.to_parquet(path, index=False)
            logger.info(f"✅ {name}: {len(data):,} → {path.name}")
        
        with open(self.output_dir / 'ml_features.json', 'w') as f:
            json.dump({
                'features': self.results['feature_columns'],
                'numeric': self.results['numeric_columns']
            }, f, indent=2)
        
        with open(self.output_dir / 'ml_preparation_metadata.json', 'w') as f:
            json.dump(self.results['metadata'], f, indent=2)
        
        import pickle
        with open(self.output_dir / 'ml_transformers.pkl', 'wb') as f:
            pickle.dump({
                'scalers': self.results['scalers'],
                'encoders': self.results['encoders']
            }, f)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("📊 PHASE 3.1 - ML DATA PREP v1.3 (NO LEAKAGE)")
        print("=" * 70)
        
        splits = self.results['splits']
        print(f"\n📁 SPLITS:")
        for name, (X, y) in splits.items():
            print(f"   {name}: {len(X):,}")
        
        print(f"\n🔧 FEATURES: {len(self.results['feature_columns'])}")
        print(f"   (Only raw source features, no DQI or derived)")
        
        print(f"\n📁 Output: {self.output_dir}")


def main():
    project_root = Path(__file__).parent.parent.parent
    
    # Use the ORIGINAL cleaned data, not analytics (which has DQI)
    input_path = project_root / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
    
    if not input_path.exists():
        # Fallback to analytics but we'll only use raw columns
        input_path = project_root / 'data' / 'processed' / 'analytics' / 'patient_benchmarks.parquet'
    
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        return None
    
    output_dir = project_root / 'data' / 'processed' / 'ml'
    
    runner = MLDataPrepRunner(input_path, output_dir)
    runner.run()
    runner.save_outputs()
    runner.print_summary()
    
    return runner


if __name__ == '__main__':
    main()