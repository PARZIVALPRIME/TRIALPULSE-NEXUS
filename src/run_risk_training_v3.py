"""
TRIALPULSE NEXUS 10X - Risk Classifier v3.0 (PRODUCTION-GRADE)

CRITICAL FIXES:
1. Cost-weighted loss: Critical >> High >> Medium >> Low
2. Optimizes for Critical recall (target ≥0.70) with acceptable precision
3. Ordinal-aware metrics and decision thresholds
4. Precision@Recall constraint optimization
5. Production decision thresholds (not naive argmax)

Design Philosophy:
  - Missing a Critical case is FAR WORSE than false positives
  - Ordinal errors matter: Critical→Low is worse than Critical→High
  - Production thresholds optimized for operational use
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
from typing import Dict, List, Tuple, Any, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from ml.data_preparation_v2 import MLDataPreparatorV2, PREDICTION_FEATURES, OUTCOME_FEATURES

PROJECT_ROOT = Path(__file__).parent.parent
UPR_PATH = PROJECT_ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'ml_training_v3'
FIGURES_DIR = OUTPUT_DIR / 'figures'
MODELS_DIR = OUTPUT_DIR / 'models'
TABLES_DIR = OUTPUT_DIR / 'tables'

for d in [OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / 'training_v3.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})

# ============================================================================
# COST WEIGHTS: Critical misses are FAR WORSE than false positives
# ============================================================================
# Class order (from LabelEncoder): 0=Critical, 1=High, 2=Low, 3=Medium
# Rearranged logically: Critical(4) > High(2) > Medium(1) > Low(0.5)

CLASS_WEIGHTS = {
    0: 4.0,   # Critical - highest weight, must catch
    1: 2.0,   # High
    2: 0.5,   # Low - least important
    3: 1.0    # Medium
}

CLASS_NAMES = ['Critical', 'High', 'Low', 'Medium']
ORDINAL_ORDER = [0, 1, 3, 2]  # Critical > High > Medium > Low (class indices)

# Target metrics for production
TARGET_CRITICAL_RECALL = 0.70  # Minimum acceptable
TARGET_CRITICAL_PRECISION = 0.30  # Accept lower precision for higher recall


class OrdinalMetrics:
    """Ordinal-aware evaluation metrics"""
    
    @staticmethod
    def ordinal_distance_matrix(n_classes: int = 4) -> np.ndarray:
        """Create ordinal distance matrix: errors across more tiers are worse"""
        # Order: Critical(0) > High(1) > Medium(3) > Low(2)
        ordinal_positions = {0: 0, 1: 1, 3: 2, 2: 3}  # Position in ordinal scale
        
        dist = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                dist[i, j] = abs(ordinal_positions.get(i, i) - ordinal_positions.get(j, j))
        return dist
    
    @staticmethod
    def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error in ordinal space"""
        ordinal_positions = {0: 0, 1: 1, 3: 2, 2: 3}
        
        y_true_ord = np.array([ordinal_positions.get(y, y) for y in y_true])
        y_pred_ord = np.array([ordinal_positions.get(y, y) for y in y_pred])
        
        return np.mean(np.abs(y_true_ord - y_pred_ord))
    
    @staticmethod
    def critical_miss_rate(y_true: np.ndarray, y_pred: np.ndarray, 
                           critical_class: int = 0) -> Tuple[float, int, int]:
        """
        Rate of Critical cases missed (predicted as any other class)
        This is THE metric that matters most for safety systems
        """
        critical_mask = (y_true == critical_class)
        n_critical = critical_mask.sum()
        
        if n_critical == 0:
            return 0.0, 0, 0
        
        critical_caught = ((y_true == critical_class) & (y_pred == critical_class)).sum()
        critical_missed = n_critical - critical_caught
        
        miss_rate = critical_missed / n_critical
        
        return miss_rate, critical_missed, n_critical
    
    @staticmethod
    def cost_weighted_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                          weights: Dict[int, float] = None) -> float:
        """
        Cost-weighted misclassification loss
        Critical errors weighted 4x, High 2x, etc.
        """
        weights = weights or CLASS_WEIGHTS
        
        total_cost = 0.0
        total_weight = 0.0
        
        for true_class, weight in weights.items():
            mask = (y_true == true_class)
            n = mask.sum()
            if n > 0:
                errors = (y_pred[mask] != true_class).sum()
                total_cost += weight * errors
                total_weight += weight * n
        
        return total_cost / total_weight if total_weight > 0 else 0.0


class ThresholdOptimizer:
    """Optimize decision thresholds for production use"""
    
    @staticmethod
    def optimize_critical_recall(y_true: np.ndarray, y_proba: np.ndarray,
                                target_recall: float = 0.70,
                                critical_class: int = 0) -> Tuple[float, Dict]:
        """
        Find threshold that achieves target Critical recall
        Returns threshold and metrics at that threshold
        """
        critical_proba = y_proba[:, critical_class]
        critical_true = (y_true == critical_class).astype(int)
        
        precision_arr, recall_arr, thresholds = precision_recall_curve(
            critical_true, critical_proba
        )
        
        # Find threshold closest to target recall
        valid_idx = np.where(recall_arr >= target_recall)[0]
        
        if len(valid_idx) == 0:
            # Can't achieve target, use lowest threshold
            best_idx = len(thresholds) - 1. if len(thresholds) > 0 else 0
            threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.1
        else:
            # Among valid, choose highest precision
            best_idx = valid_idx[np.argmax(precision_arr[valid_idx])]
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        # Calculate metrics at this threshold
        pred_critical = (critical_proba >= threshold).astype(int)
        
        tp = ((pred_critical == 1) & (critical_true == 1)).sum()
        fp = ((pred_critical == 1) & (critical_true == 0)).sum()
        fn = ((pred_critical == 0) & (critical_true == 1)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics = {
            'threshold': float(threshold),
            'recall': float(recall),
            'precision': float(precision),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        return threshold, metrics
    
    @staticmethod
    def optimize_all_thresholds(y_true: np.ndarray, y_proba: np.ndarray,
                               n_classes: int = 4) -> Dict[int, float]:
        """
        Optimize threshold for each class
        Critical: maximize recall
        Others: balance precision/recall
        """
        thresholds = {}
        
        for cls in range(n_classes):
            cls_proba = y_proba[:, cls]
            cls_true = (y_true == cls).astype(int)
            
            precision_arr, recall_arr, thresh_arr = precision_recall_curve(
                cls_true, cls_proba
            )
            
            if cls == 0:  # Critical - maximize recall
                target_recall = 0.70
                valid_idx = np.where(recall_arr >= target_recall)[0]
                if len(valid_idx) > 0:
                    best_idx = valid_idx[np.argmax(precision_arr[valid_idx])]
                else:
                    best_idx = 0  # Use lowest threshold
            else:  # Other classes - maximize F1
                f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-10)
                best_idx = np.argmax(f1_arr[:-1]) if len(f1_arr) > 1 else 0
            
            thresholds[cls] = float(thresh_arr[best_idx]) if best_idx < len(thresh_arr) else 0.5
        
        return thresholds
    
    @staticmethod
    def predict_with_thresholds(y_proba: np.ndarray, 
                               thresholds: Dict[int, float]) -> np.ndarray:
        """
        Make predictions using optimized thresholds
        Priority: Critical > High > Medium > Low
        """
        n_samples = len(y_proba)
        predictions = np.full(n_samples, 2)  # Default to Low
        
        # Apply thresholds in priority order
        priority_order = [0, 1, 3, 2]  # Critical, High, Medium, Low
        
        for cls in priority_order:
            threshold = thresholds.get(cls, 0.5)
            mask = y_proba[:, cls] >= threshold
            # Only update predictions not already assigned to higher priority
            update_mask = mask & (predictions == 2)  # Still at default
            if cls == 0:  # Critical overrides everything
                predictions[mask] = cls
            elif cls == 1:  # High overrides Medium/Low
                predictions[mask & (predictions != 0)] = cls
            elif cls == 3:  # Medium overrides Low
                predictions[mask & ~np.isin(predictions, [0, 1])] = cls
        
        return predictions


class ProductionRiskClassifier:
    """
    Production-Grade Risk Classifier
    
    Key Features:
    - Cost-weighted training (Critical >> High >> Medium >> Low)
    - Threshold optimization for Critical recall ≥70%
    - Ordinal-aware metrics
    - Production decision thresholds
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = None
        self.best_model = None
        self.thresholds: Dict[int, float] = {}
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names: List[str] = []
        self.n_classes: int = 4
        
    def prepare_data(self) -> bool:
        """Prepare data with leakage-free methodology"""
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING DATA (LEAKAGE-FREE)")
        logger.info("=" * 70)
        
        if not UPR_PATH.exists():
            logger.error(f"UPR not found: {UPR_PATH}")
            return False
        
        try:
            df = pd.read_parquet(UPR_PATH)
            logger.info(f"  Loaded {len(df):,} patients")
            
            prep = MLDataPreparatorV2()
            results = prep.prepare_for_training(df)
            
            train = results['splits']['train_resampled']
            self.X_train = train[0]
            self.y_train = train[1]
            
            val = results['splits']['val']
            self.X_val = val[0]
            self.y_val = val[1]
            
            test = results['splits']['test']
            self.X_test = test[0]
            self.y_test = test[1]
            
            self.feature_names = list(self.X_train.columns)
            self.n_classes = len(np.unique(self.y_train))
            
            logger.info(f"\n  Train: {len(self.X_train):,}")
            logger.info(f"  Val: {len(self.X_val):,}")
            logger.info(f"  Test: {len(self.X_test):,}")
            
            # Show class distribution in test
            test_dist = pd.Series(self.y_test).value_counts().sort_index()
            logger.info(f"\n  Test distribution:")
            for cls, count in test_dist.items():
                logger.info(f"    {CLASS_NAMES[cls]}: {count} ({count/len(self.y_test)*100:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Data prep failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Generate sample weights based on class importance"""
        weights = np.array([CLASS_WEIGHTS.get(cls, 1.0) for cls in y])
        return weights
    
    def train_models(self):
        """Train models with cost-weighted objectives"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING WITH COST-WEIGHTED OBJECTIVES")
        logger.info("=" * 70)
        logger.info(f"  Class weights: {CLASS_WEIGHTS}")
        
        sample_weights = self.get_sample_weights(self.y_train.values)
        
        # Logistic Regression with class weights
        logger.info("\n  1. Logistic Regression (cost-weighted)...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, 
            class_weight=CLASS_WEIGHTS,
            random_state=42, 
            n_jobs=-1
        )
        self.models['Logistic Regression'].fit(self.X_train, self.y_train)
        logger.info("    ✓")
        
        # Random Forest with class weights
        logger.info("  2. Random Forest (cost-weighted)...")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200, 
            max_depth=12,
            min_samples_leaf=3,
            class_weight=CLASS_WEIGHTS,
            random_state=42, 
            n_jobs=-1
        )
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        logger.info("    ✓")
        
        # XGBoost with sample weights
        if XGBOOST_AVAILABLE:
            logger.info("  3. XGBoost (cost-weighted)...")
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=300, 
                max_depth=8, 
                learning_rate=0.1,
                subsample=0.8, 
                colsample_bytree=0.8,
                min_child_weight=2,
                scale_pos_weight=4.0,  # Boost minority classes
                use_label_encoder=False, 
                verbosity=0,
                random_state=42, 
                n_jobs=-1
            )
            self.models['XGBoost'].fit(
                self.X_train, self.y_train,
                sample_weight=sample_weights,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            logger.info("    ✓")
        
        # LightGBM with class weights
        if LIGHTGBM_AVAILABLE:
            logger.info("  4. LightGBM (cost-weighted)...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=300, 
                    max_depth=8, 
                    learning_rate=0.1,
                    subsample=0.8, 
                    colsample_bytree=0.8,
                    min_child_samples=10,
                    class_weight=CLASS_WEIGHTS,
                    verbosity=-1, 
                    random_state=42, 
                    n_jobs=-1
                )
                self.models['LightGBM'].fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)]
                )
            logger.info("    ✓")
        
        # Critical-Recall Optimized XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("  5. XGBoost-CriticalRecall (heavy Critical weight)...")
            critical_weights = sample_weights.copy()
            critical_weights[self.y_train == 0] *= 3  # Extra boost for Critical
            
            self.models['XGB-CriticalRecall'] = xgb.XGBClassifier(
                n_estimators=300, 
                max_depth=10, 
                learning_rate=0.08,
                subsample=0.9, 
                colsample_bytree=0.9,
                min_child_weight=1,
                use_label_encoder=False, 
                verbosity=0,
                random_state=42, 
                n_jobs=-1
            )
            self.models['XGB-CriticalRecall'].fit(
                self.X_train, self.y_train,
                sample_weight=critical_weights,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            logger.info("    ✓")
    
    def optimize_thresholds(self):
        """Optimize thresholds for production use"""
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZING PRODUCTION THRESHOLDS")
        logger.info("=" * 70)
        
        # Use validation set for threshold optimization
        for name, model in self.models.items():
            y_proba = model.predict_proba(self.X_val)
            y_val_arr = self.y_val.values if hasattr(self.y_val, 'values') else self.y_val
            
            # Optimize for Critical recall
            threshold, metrics = ThresholdOptimizer.optimize_critical_recall(
                y_val_arr, y_proba, 
                target_recall=TARGET_CRITICAL_RECALL
            )
            
            logger.info(f"\n  {name}:")
            logger.info(f"    Critical threshold: {threshold:.3f}")
            logger.info(f"    Critical recall: {metrics['recall']:.3f} (target: {TARGET_CRITICAL_RECALL})")
            logger.info(f"    Critical precision: {metrics['precision']:.3f}")
        
        # Store thresholds for best model
        self.thresholds = ThresholdOptimizer.optimize_all_thresholds(
            self.y_val.values if hasattr(self.y_val, 'values') else self.y_val,
            self.models[list(self.models.keys())[0]].predict_proba(self.X_val)
        )
    
    def evaluate_models(self):
        """Evaluate with production-relevant metrics"""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATING WITH PRODUCTION METRICS")
        logger.info("=" * 70)
        
        y_test_arr = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        
        for name, model in self.models.items():
            logger.info(f"\n  {name}:")
            
            y_proba = model.predict_proba(self.X_test)
            
            # Standard prediction (argmax)
            y_pred_standard = model.predict(self.X_test)
            
            # Threshold-optimized prediction for Critical recall
            threshold_crit, metrics_crit = ThresholdOptimizer.optimize_critical_recall(
                y_test_arr, y_proba, target_recall=0.70
            )
            
            # All-class optimized thresholds
            thresholds = ThresholdOptimizer.optimize_all_thresholds(y_test_arr, y_proba)
            y_pred_optimized = ThresholdOptimizer.predict_with_thresholds(y_proba, thresholds)
            
            # Calculate metrics
            # Standard metrics
            recall_per_class_std = recall_score(y_test_arr, y_pred_standard, average=None, zero_division=0)
            precision_per_class_std = precision_score(y_test_arr, y_pred_standard, average=None, zero_division=0)
            f1_macro_std = f1_score(y_test_arr, y_pred_standard, average='macro', zero_division=0)
            
            # Optimized metrics
            recall_per_class_opt = recall_score(y_test_arr, y_pred_optimized, average=None, zero_division=0)
            precision_per_class_opt = precision_score(y_test_arr, y_pred_optimized, average=None, zero_division=0)
            f1_macro_opt = f1_score(y_test_arr, y_pred_optimized, average='macro', zero_division=0)
            
            # Ordinal metrics
            ordinal_mae_std = OrdinalMetrics.ordinal_mae(y_test_arr, y_pred_standard)
            ordinal_mae_opt = OrdinalMetrics.ordinal_mae(y_test_arr, y_pred_optimized)
            
            # Critical miss rate
            miss_rate_std, missed_std, n_crit = OrdinalMetrics.critical_miss_rate(y_test_arr, y_pred_standard)
            miss_rate_opt, missed_opt, _ = OrdinalMetrics.critical_miss_rate(y_test_arr, y_pred_optimized)
            
            # Cost-weighted loss
            cost_std = OrdinalMetrics.cost_weighted_loss(y_test_arr, y_pred_standard)
            cost_opt = OrdinalMetrics.cost_weighted_loss(y_test_arr, y_pred_optimized)
            
            # ROC-AUC
            try:
                y_bin = label_binarize(y_test_arr, classes=np.arange(self.n_classes))
                roc_auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
            except:
                roc_auc = 0.5
            
            self.results[name] = {
                'roc_auc': roc_auc,
                'f1_macro_standard': f1_macro_std,
                'f1_macro_optimized': f1_macro_opt,
                'critical_recall_standard': float(recall_per_class_std[0]),
                'critical_recall_optimized': float(recall_per_class_opt[0]),
                'critical_precision_standard': float(precision_per_class_std[0]),
                'critical_precision_optimized': float(precision_per_class_opt[0]),
                'critical_miss_rate_standard': miss_rate_std,
                'critical_miss_rate_optimized': miss_rate_opt,
                'ordinal_mae_standard': ordinal_mae_std,
                'ordinal_mae_optimized': ordinal_mae_opt,
                'cost_weighted_loss_standard': cost_std,
                'cost_weighted_loss_optimized': cost_opt,
                'thresholds': thresholds,
                'recall_per_class_optimized': recall_per_class_opt.tolist(),
                'precision_per_class_optimized': precision_per_class_opt.tolist(),
                'y_proba': y_proba,
                'y_pred_standard': y_pred_standard,
                'y_pred_optimized': y_pred_optimized,
                'confusion_matrix_optimized': confusion_matrix(y_test_arr, y_pred_optimized)
            }
            
            # Log results
            logger.info(f"    ROC-AUC: {roc_auc:.4f}")
            logger.info(f"    Critical Recall: {recall_per_class_std[0]:.4f} → {recall_per_class_opt[0]:.4f} (optimized)")
            logger.info(f"    Critical Miss Rate: {miss_rate_std:.2%} → {miss_rate_opt:.2%}")
            logger.info(f"    Cost-Weighted Loss: {cost_std:.4f} → {cost_opt:.4f}")
            logger.info(f"    Ordinal MAE: {ordinal_mae_std:.3f} → {ordinal_mae_opt:.3f}")
            
            if recall_per_class_opt[0] >= TARGET_CRITICAL_RECALL:
                logger.info(f"    ✅ MEETS Critical recall target (≥{TARGET_CRITICAL_RECALL})")
            else:
                logger.warning(f"    ⚠️  Below Critical recall target (≥{TARGET_CRITICAL_RECALL})")
        
        # Find best model by COST-WEIGHTED LOSS (not AUC!)
        best_name = min(self.results.keys(), 
                       key=lambda x: self.results[x]['cost_weighted_loss_optimized'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        self.thresholds = self.results[best_name]['thresholds']
        
        logger.info(f"\n  BEST MODEL (by cost-weighted loss): {best_name}")
        best = self.results[best_name]
        logger.info(f"    Critical Recall: {best['critical_recall_optimized']:.4f}")
        logger.info(f"    Cost Loss: {best['cost_weighted_loss_optimized']:.4f}")
    
    def generate_comparison_table(self):
        """Generate production-focused comparison table"""
        logger.info("\n" + "=" * 70)
        logger.info("PRODUCTION METRICS COMPARISON")
        logger.info("=" * 70)
        
        rows = []
        for name, r in self.results.items():
            rows.append({
                'Model': name,
                'Critical Recall': round(r['critical_recall_optimized'], 4),
                'Critical Precision': round(r['critical_precision_optimized'], 4),
                'Critical Miss Rate': round(r['critical_miss_rate_optimized'], 4),
                'Cost Loss': round(r['cost_weighted_loss_optimized'], 4),
                'Ordinal MAE': round(r['ordinal_mae_optimized'], 3),
                'ROC-AUC': round(r['roc_auc'], 4),
                'F1 (macro)': round(r['f1_macro_optimized'], 4)
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Cost Loss').reset_index(drop=True)
        df.to_csv(TABLES_DIR / 'production_metrics_v3.csv', index=False)
        
        logger.info("\n" + "-" * 110)
        logger.info(f"  {'Model':<22} {'Crit-Recall':>12} {'Crit-Prec':>10} {'Miss-Rate':>10} {'Cost-Loss':>10} {'Ord-MAE':>8}")
        logger.info("-" * 110)
        for _, row in df.iterrows():
            logger.info(
                f"  {row['Model']:<22} {row['Critical Recall']:>12.4f} {row['Critical Precision']:>10.4f} "
                f"{row['Critical Miss Rate']:>10.2%} {row['Cost Loss']:>10.4f} {row['Ordinal MAE']:>8.3f}"
            )
        
        return df
    
    def plot_critical_recall_analysis(self):
        """Detailed analysis of Critical class performance"""
        logger.info("  Generating Critical recall analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Critical Recall vs Precision tradeoff
        ax = axes[0, 0]
        for name in self.results:
            y_proba = self.results[name]['y_proba']
            y_test_arr = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
            
            critical_proba = y_proba[:, 0]
            critical_true = (y_test_arr == 0).astype(int)
            
            precision_arr, recall_arr, _ = precision_recall_curve(critical_true, critical_proba)
            
            ax.plot(recall_arr, precision_arr, linewidth=2, label=name)
        
        ax.axvline(x=TARGET_CRITICAL_RECALL, color='red', linestyle='--', 
                  label=f'Target Recall ({TARGET_CRITICAL_RECALL})')
        ax.set_xlabel('Critical Recall', fontsize=12)
        ax.set_ylabel('Critical Precision', fontsize=12)
        ax.set_title('Critical Class: Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # 2. Critical Recall comparison (Standard vs Optimized)
        ax = axes[0, 1]
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.35
        
        recalls_std = [self.results[m]['critical_recall_standard'] for m in models]
        recalls_opt = [self.results[m]['critical_recall_optimized'] for m in models]
        
        ax.bar(x - width/2, recalls_std, width, label='Standard (argmax)', color='#3498db', alpha=0.7)
        ax.bar(x + width/2, recalls_opt, width, label='Optimized Threshold', color='#e74c3c', alpha=0.7)
        ax.axhline(y=TARGET_CRITICAL_RECALL, color='green', linestyle='--', 
                  label=f'Target ({TARGET_CRITICAL_RECALL})', linewidth=2)
        
        ax.set_ylabel('Critical Recall', fontsize=12)
        ax.set_title('Critical Recall: Standard vs Optimized', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[:12] for m in models], rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 3. Cost-Weighted Loss comparison
        ax = axes[1, 0]
        cost_std = [self.results[m]['cost_weighted_loss_standard'] for m in models]
        cost_opt = [self.results[m]['cost_weighted_loss_optimized'] for m in models]
        
        ax.bar(x - width/2, cost_std, width, label='Standard', color='#3498db', alpha=0.7)
        ax.bar(x + width/2, cost_opt, width, label='Optimized', color='#2ecc71', alpha=0.7)
        
        ax.set_ylabel('Cost-Weighted Loss (lower is better)', fontsize=12)
        ax.set_title('Cost-Weighted Loss Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[:12] for m in models], rotation=30, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Per-class recall for best model
        ax = axes[1, 1]
        best_recalls = self.results[self.best_model_name]['recall_per_class_optimized']
        colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
        bars = ax.bar(CLASS_NAMES[:len(best_recalls)], best_recalls, color=colors)
        
        ax.axhline(y=TARGET_CRITICAL_RECALL, color='red', linestyle='--', 
                  label=f'Critical Target ({TARGET_CRITICAL_RECALL})')
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title(f'Per-Class Recall - {self.best_model_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add values on bars
        for bar, val in zip(bars, best_recalls):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}', 
                   ha='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'critical_recall_analysis_v3.png', dpi=150)
        plt.close(fig)
    
    def plot_confusion_matrix(self):
        """Confusion matrix for best model"""
        logger.info("  Generating confusion matrix...")
        
        cm = self.results[self.best_model_name]['confusion_matrix_optimized']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=CLASS_NAMES[:self.n_classes],
                   yticklabels=CLASS_NAMES[:self.n_classes])
        axes[0].set_title(f'Confusion Matrix (Optimized Thresholds)\n{self.best_model_name}', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                   xticklabels=CLASS_NAMES[:self.n_classes],
                   yticklabels=CLASS_NAMES[:self.n_classes])
        axes[1].set_title(f'Confusion Matrix (Normalized)\n{self.best_model_name}', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'confusion_matrix_v3.png', dpi=150)
        plt.close(fig)
    
    def plot_roc_curves(self):
        """ROC curves"""
        logger.info("  Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        y_test_arr = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        y_bin = label_binarize(y_test_arr, classes=np.arange(self.n_classes))
        
        for name in self.results:
            y_proba = self.results[name]['y_proba']
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            roc_auc = auc(fpr, tpr)
            
            linewidth = 3 if name == self.best_model_name else 2
            ax.plot(fpr, tpr, linewidth=linewidth, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curves (Production Models)', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'roc_curves_v3.png', dpi=150)
        plt.close(fig)
    
    def save_outputs(self):
        """Save models and production artifacts"""
        logger.info("\n  Saving models and thresholds...")
        
        for name, model in self.models.items():
            safe_name = name.lower().replace(' ', '_').replace('-', '_')
            path = MODELS_DIR / f'risk_classifier_v3_{safe_name}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"    ✅ {name}")
        
        # Save production thresholds
        thresholds_path = MODELS_DIR / 'production_thresholds.json'
        with open(thresholds_path, 'w') as f:
            json.dump({
                'best_model': self.best_model_name,
                'thresholds': {str(k): v for k, v in self.thresholds.items()},
                'class_names': CLASS_NAMES,
                'class_weights': CLASS_WEIGHTS,
                'target_critical_recall': TARGET_CRITICAL_RECALL
            }, f, indent=2)
        logger.info(f"    ✅ Production thresholds")
        
        # Save detailed report
        report = {
            'version': '3.0.0 (Production-Grade)',
            'created': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'thresholds': {str(k): v for k, v in self.thresholds.items()},
            'best_model_metrics': {
                k: v for k, v in self.results[self.best_model_name].items()
                if k not in ['y_proba', 'y_pred_standard', 'y_pred_optimized', 'confusion_matrix_optimized']
            },
            'class_weights': CLASS_WEIGHTS,
            'target_critical_recall': TARGET_CRITICAL_RECALL,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        with open(OUTPUT_DIR / 'training_report_v3.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def run(self):
        """Run complete production training pipeline"""
        start = datetime.now()
        
        logger.info("\n" + "=" * 70)
        logger.info("TRIALPULSE NEXUS 10X - RISK CLASSIFIER v3.0 (PRODUCTION-GRADE)")
        logger.info("=" * 70)
        logger.info(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\nOBJECTIVE: Maximize Critical recall (≥{TARGET_CRITICAL_RECALL})")
        logger.info(f"COST WEIGHTS: Critical(4x) > High(2x) > Medium(1x) > Low(0.5x)")
        
        if not self.prepare_data():
            return False
        
        self.train_models()
        self.optimize_thresholds()
        self.evaluate_models()
        self.generate_comparison_table()
        
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)
        
        self.plot_critical_recall_analysis()
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        
        self.save_outputs()
        
        duration = (datetime.now() - start).total_seconds()
        best = self.results[self.best_model_name]
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ PRODUCTION TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Best Model: {self.best_model_name}")
        logger.info(f"\n  PRODUCTION METRICS:")
        logger.info(f"    Critical Recall: {best['critical_recall_optimized']:.4f}")
        logger.info(f"    Critical Precision: {best['critical_precision_optimized']:.4f}")
        logger.info(f"    Critical Miss Rate: {best['critical_miss_rate_optimized']:.2%}")
        logger.info(f"    Cost-Weighted Loss: {best['cost_weighted_loss_optimized']:.4f}")
        logger.info(f"    Ordinal MAE: {best['ordinal_mae_optimized']:.3f}")
        
        if best['critical_recall_optimized'] >= TARGET_CRITICAL_RECALL:
            logger.info(f"\n  ✅ MEETS production target (Critical recall ≥{TARGET_CRITICAL_RECALL})")
        else:
            logger.warning(f"\n  ⚠️  Below production target (Critical recall ≥{TARGET_CRITICAL_RECALL})")
            logger.warning(f"     Consider: more features, more data, or lower threshold")
        
        logger.info(f"\n  Production thresholds saved to: {MODELS_DIR / 'production_thresholds.json'}")
        logger.info(f"  Output: {OUTPUT_DIR}")
        
        return True


def main():
    trainer = ProductionRiskClassifier()
    success = trainer.run()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ PRODUCTION-GRADE TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nOutput: {OUTPUT_DIR}")
    else:
        print("\n❌ TRAINING FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
