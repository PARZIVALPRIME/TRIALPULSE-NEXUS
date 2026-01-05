"""
TRIALPULSE NEXUS 10X - ML Data Preparation v3.0 (ENHANCED FEATURES)

Improvements:
1. More prediction features (derived ratios, interactions)
2. Relaxed correlation threshold for feature diversity
3. Better feature engineering for risk prediction
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


# EXPANDED PREDICTION FEATURES - More signals for better prediction
PREDICTION_FEATURES_BASE = [
    # Workload metrics
    'expected_visits_rave_edc_bo4',
    'pages_entered',
    'total_crfs',
    'crfs_require_signature',
    'crfs_require_verification_sdv',
    'forms_verified',
    
    # Progress indicators
    'crfs_frozen',
    'crfs_locked',
    'crfs_signed',
    'crfs_verified_sdv',
    
    # Query workload (all types)
    'dm_queries',
    'clinical_queries',
    'medical_queries',
    'site_queries',
    'queries_answered',
    
    # Coding workload
    'meddra_coding_meddra_total',
    'whodrug_coding_whodrug_total',
    
    # SAE workload
    'sae_dm_sae_dm_total',
    'sae_safety_sae_safety_total',
    
    # Additional workload features
    'edrr_edrr_resolved',
]

# OUTCOME FEATURES - Used ONLY for target definition
OUTCOME_FEATURES = [
    'broken_signatures',
    'crfs_never_signed',
    'crfs_overdue_for_signs_within_45_days_of_data_entry',
    'crfs_overdue_for_signs_between_45_to_90_days_of_data_entry',
    'crfs_overdue_for_signs_beyond_90_days_of_data_entry',
    'protocol_deviations',
    'pages_with_nonconformant_data',
    'safety_queries',
    'visit_missing_visit_count',
    'visit_visits_overdue_count',
    'lab_lab_issue_count',
    'lab_lab_missing_names',
    'lab_lab_missing_ranges',
    'edrr_edrr_issue_count',
    'sae_dm_sae_dm_completed',
    'sae_safety_sae_safety_completed',
    'meddra_coding_meddra_coded',
    'whodrug_coding_whodrug_coded',
    'inactivated_inactivated_form_count',
    'pages_pages_missing_count',
]


@dataclass
class MLDataConfigV3:
    """Enhanced config for v3"""
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    random_state: int = 42
    correlation_threshold: float = 0.95  # Relaxed from 0.90
    variance_threshold: float = 0.001  # More permissive
    smote_k_neighbors: int = 3


class MLDataPreparatorV3:
    """Enhanced data preparation with feature engineering"""
    
    def __init__(self, config: MLDataConfigV3 = None):
        self.config = config or MLDataConfigV3()
        self.feature_columns: List[str] = []
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for better prediction"""
        logger.info("\n" + "=" * 70)
        logger.info("ENGINEERING DERIVED FEATURES")
        logger.info("=" * 70)
        
        df = df.copy()
        n_derived = 0
        
        # Query ratios
        if 'queries_answered' in df.columns:
            total_queries = (
                df.get('dm_queries', 0) + 
                df.get('clinical_queries', 0) + 
                df.get('medical_queries', 0) + 
                df.get('site_queries', 0)
            )
            df['query_answer_rate'] = np.where(
                total_queries > 0,
                df['queries_answered'] / (total_queries + 1),
                1.0
            )
            n_derived += 1
            
            df['total_query_load'] = total_queries
            n_derived += 1
        
        # CRF completion ratios
        if 'total_crfs' in df.columns and 'crfs_signed' in df.columns:
            df['crf_completion_rate'] = np.where(
                df['total_crfs'] > 0,
                df['crfs_signed'] / (df['total_crfs'] + 1),
                1.0
            )
            n_derived += 1
        
        if 'crfs_require_verification_sdv' in df.columns and 'crfs_verified_sdv' in df.columns:
            df['sdv_completion_rate'] = np.where(
                df['crfs_require_verification_sdv'] > 0,
                df['crfs_verified_sdv'] / (df['crfs_require_verification_sdv'] + 1),
                1.0
            )
            n_derived += 1
        
        # Freeze/lock ratio
        if 'total_crfs' in df.columns:
            if 'crfs_frozen' in df.columns:
                df['freeze_rate'] = np.where(
                    df['total_crfs'] > 0,
                    df['crfs_frozen'] / (df['total_crfs'] + 1),
                    0.0
                )
                n_derived += 1
            
            if 'crfs_locked' in df.columns:
                df['lock_rate'] = np.where(
                    df['total_crfs'] > 0,
                    df['crfs_locked'] / (df['total_crfs'] + 1),
                    0.0
                )
                n_derived += 1
        
        # SAE load indicator
        if 'sae_dm_sae_dm_total' in df.columns:
            df['has_sae_dm'] = (df['sae_dm_sae_dm_total'] > 0).astype(float)
            n_derived += 1
        
        if 'sae_safety_sae_safety_total' in df.columns:
            df['has_sae_safety'] = (df['sae_safety_sae_safety_total'] > 0).astype(float)
            n_derived += 1
        
        # Work intensity (pages per CRF)
        if 'pages_entered' in df.columns and 'total_crfs' in df.columns:
            df['pages_per_crf'] = np.where(
                df['total_crfs'] > 0,
                df['pages_entered'] / (df['total_crfs'] + 1),
                0.0
            )
            n_derived += 1
        
        # Query intensity (queries per page)
        if 'pages_entered' in df.columns:
            df['queries_per_page'] = np.where(
                df['pages_entered'] > 0,
                df.get('total_query_load', 0) / (df['pages_entered'] + 1),
                0.0
            )
            n_derived += 1
        
        # Coding load
        coding_total = df.get('meddra_coding_meddra_total', 0) + df.get('whodrug_coding_whodrug_total', 0)
        df['total_coding_load'] = coding_total
        n_derived += 1
        
        # Binary flags for high-volume cases
        if 'total_query_load' in df.columns:
            df['high_query_volume'] = (df['total_query_load'] > 10).astype(float)
            n_derived += 1
        
        if 'total_crfs' in df.columns:
            df['high_crf_volume'] = (df['total_crfs'] > 50).astype(float)
            n_derived += 1
        
        logger.info(f"  Created {n_derived} derived features")
        
        return df
    
    def select_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select prediction features including derived ones"""
        logger.info("\n" + "=" * 70)
        logger.info("SELECTING PREDICTION FEATURES")
        logger.info("=" * 70)
        
        # Base features
        available_base = [c for c in PREDICTION_FEATURES_BASE if c in df.columns]
        
        # Derived features
        derived_features = [
            'query_answer_rate', 'total_query_load', 'crf_completion_rate',
            'sdv_completion_rate', 'freeze_rate', 'lock_rate',
            'has_sae_dm', 'has_sae_safety', 'pages_per_crf',
            'queries_per_page', 'total_coding_load',
            'high_query_volume', 'high_crf_volume'
        ]
        available_derived = [c for c in derived_features if c in df.columns]
        
        all_features = available_base + available_derived
        
        # Validate no leakage
        outcome_set = set(OUTCOME_FEATURES)
        leaks = [f for f in all_features if f in outcome_set]
        if leaks:
            logger.error(f"LEAKAGE DETECTED: {leaks}")
            raise ValueError("Feature leakage detected!")
        
        # Filter to numeric with variance
        self.feature_columns = []
        for col in all_features:
            if col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    if df[col].nunique() > 1 and df[col].std() > self.config.variance_threshold:
                        self.feature_columns.append(col)
        
        logger.info(f"  Base features: {len(available_base)}")
        logger.info(f"  Derived features: {len(available_derived)}")
        logger.info(f"  Total selected: {len(self.feature_columns)}")
        
        return df[self.feature_columns].copy()
    
    def create_risk_target(self, df: pd.DataFrame) -> pd.Series:
        """Create 4-tier risk target from outcome features"""
        logger.info("\n" + "=" * 70)
        logger.info("CREATING RISK TARGET")
        logger.info("=" * 70)
        
        risk_score = pd.Series(0.0, index=df.index)
        
        # CRITICAL tier indicators
        if 'sae_dm_sae_dm_total' in df.columns and 'sae_dm_sae_dm_completed' in df.columns:
            pending = df['sae_dm_sae_dm_total'].fillna(0) - df['sae_dm_sae_dm_completed'].fillna(0)
            pending = pending.clip(lower=0)
            risk_score += (pending > 0).astype(float) * 4.0
        
        if 'sae_safety_sae_safety_total' in df.columns and 'sae_safety_sae_safety_completed' in df.columns:
            pending = df['sae_safety_sae_safety_total'].fillna(0) - df['sae_safety_sae_safety_completed'].fillna(0)
            pending = pending.clip(lower=0)
            risk_score += (pending > 0).astype(float) * 4.0
        
        if 'broken_signatures' in df.columns:
            risk_score += (df['broken_signatures'].fillna(0) > 0).astype(float) * 3.0
        
        # HIGH tier
        if 'crfs_never_signed' in df.columns:
            val = df['crfs_never_signed'].fillna(0)
            risk_score += (val > 5).astype(float) * 2.5
            risk_score += (val > 0).astype(float) * 1.0
        
        if 'crfs_overdue_for_signs_beyond_90_days_of_data_entry' in df.columns:
            risk_score += (df['crfs_overdue_for_signs_beyond_90_days_of_data_entry'].fillna(0) > 0).astype(float) * 2.5
        
        if 'protocol_deviations' in df.columns:
            val = df['protocol_deviations'].fillna(0)
            risk_score += (val > 0).astype(float) * 2.0
            risk_score += (val > 2).astype(float) * 1.5
        
        # MEDIUM tier
        if 'visit_missing_visit_count' in df.columns:
            risk_score += (df['visit_missing_visit_count'].fillna(0) > 0).astype(float) * 1.5
        
        if 'pages_pages_missing_count' in df.columns:
            risk_score += (df['pages_pages_missing_count'].fillna(0) > 0).astype(float) * 1.0
        
        if 'meddra_coding_meddra_total' in df.columns and 'meddra_coding_meddra_coded' in df.columns:
            uncoded = df['meddra_coding_meddra_total'].fillna(0) - df['meddra_coding_meddra_coded'].fillna(0)
            risk_score += (uncoded.clip(lower=0) > 0).astype(float) * 0.5
        
        if 'lab_lab_issue_count' in df.columns:
            risk_score += (df['lab_lab_issue_count'].fillna(0) > 0).astype(float) * 1.0
        
        if 'safety_queries' in df.columns:
            risk_score += (df['safety_queries'].fillna(0) > 0).astype(float) * 2.0
        
        # Convert to tiers
        p50 = risk_score.quantile(0.50)
        p80 = risk_score.quantile(0.80)
        p95 = risk_score.quantile(0.95)
        
        risk_level = pd.cut(
            risk_score,
            bins=[-np.inf, p50, p80, p95, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        dist = risk_level.value_counts()
        logger.info(f"  Distribution:")
        for level in ['Critical', 'High', 'Medium', 'Low']:
            count = dist.get(level, 0)
            logger.info(f"    {level}: {count:,} ({count/len(risk_level)*100:.1f}%)")
        
        return risk_level
    
    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        return df
    
    def remove_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        logger.info("\n  Removing correlated features...")
        
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
        
        if to_drop:
            logger.info(f"    Dropping {len(to_drop)} correlated features")
            for col in to_drop:
                if col in self.feature_columns:
                    self.feature_columns.remove(col)
            df = df.drop(columns=to_drop, errors='ignore')
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features"""
        df = df.copy()
        existing = [c for c in self.feature_columns if c in df.columns]
        if existing:
            scaler = RobustScaler()
            df[existing] = scaler.fit_transform(df[existing])
            self.scalers['numeric'] = scaler
        return df
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Create stratified splits"""
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        self.encoders['target'] = le
        self.encoders['target_classes'] = list(le.classes_)
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=self.config.test_ratio,
            random_state=self.config.random_state, stratify=y_encoded
        )
        
        val_ratio = self.config.val_ratio / (1 - self.config.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio,
            random_state=self.config.random_state, stratify=y_temp
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE"""
        if not IMBLEARN_AVAILABLE:
            return X, y
        
        min_samples = y.value_counts().min()
        k = min(self.config.smote_k_neighbors, min_samples - 1)
        
        if k < 1:
            sampler = RandomOverSampler(random_state=self.config.random_state)
        else:
            sampler = SMOTE(k_neighbors=k, random_state=self.config.random_state)
        
        try:
            X_res, y_res = sampler.fit_resample(X, y)
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        except:
            return X, y
    
    def prepare_for_training(self, df: pd.DataFrame) -> Dict:
        """Complete pipeline"""
        logger.info("\n" + "=" * 70)
        logger.info("ML DATA PREP v3.0 (ENHANCED FEATURES)")
        logger.info("=" * 70)
        logger.info(f"Input: {len(df):,} samples")
        
        # 1. Create target from outcomes
        y = self.create_risk_target(df)
        
        # 2. Engineer derived features
        df = self.engineer_features(df)
        
        # 3. Select prediction features
        X = self.select_prediction_features(df)
        
        # 4. Handle missing
        X = self.handle_missing(X)
        
        # 5. Remove correlation
        X = self.remove_correlation(X)
        
        # 6. Scale
        X = self.scale_features(X)
        
        # 7. Split
        splits = self.split_data(X, y)
        
        # 8. SMOTE
        X_res, y_res = self.apply_smote(splits['train'][0], splits['train'][1])
        splits['train_resampled'] = (X_res, y_res)
        
        logger.info(f"\n  Final features: {len(self.feature_columns)}")
        for f in self.feature_columns:
            logger.info(f"    ✓ {f}")
        
        return {
            'splits': splits,
            'feature_columns': self.feature_columns,
            'scalers': self.scalers,
            'encoders': self.encoders
        }
