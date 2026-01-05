"""
TRIALPULSE NEXUS 10X - Risk Classifier Training v2.0 (LEAKAGE-FREE)

Critical Improvements:
- Uses leakage-free data prep (prediction features ≠ outcome features)
- Removes misleading accuracy metric
- Adds per-class recall and macro metrics
- Warns if AUC > 0.90 (suggests leakage)
- Expected realistic AUC: 0.65-0.85
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
from typing import Dict, List, Tuple, Any

# ML imports
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss, classification_report
)
from sklearn.preprocessing import label_binarize

# Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Gradient boosting
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

# Import leakage-free data prep
from ml.data_preparation_v2 import MLDataPreparatorV2, PREDICTION_FEATURES, OUTCOME_FEATURES

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'ml_v2'
UPR_PATH = PROJECT_ROOT / 'data' / 'processed' / 'upr' / 'unified_patient_record.parquet'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'ml_training_v2'
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
        logging.FileHandler(OUTPUT_DIR / 'training_v2.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight'
})

MODEL_COLORS = {
    'Logistic Regression': '#3498db',
    'Decision Tree': '#e74c3c',
    'Random Forest': '#2ecc71',
    'XGBoost': '#f39c12',
    'LightGBM': '#1abc9c',
    'XGB+LGB Ensemble': '#e91e63'
}

CLASS_NAMES = ['Critical', 'High', 'Low', 'Medium']


class LeakageFreeRiskTrainer:
    """Risk classifier training with strict leakage prevention"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        self.best_model_name: str = None
        self.best_model = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names: List[str] = []
        self.n_classes: int = 4
        
    def prepare_data(self) -> bool:
        """Run leakage-free data preparation"""
        logger.info("\n" + "=" * 70)
        logger.info("PREPARING LEAKAGE-FREE DATA")
        logger.info("=" * 70)
        
        if not UPR_PATH.exists():
            logger.error(f"UPR not found: {UPR_PATH}")
            return False
        
        try:
            df = pd.read_parquet(UPR_PATH)
            logger.info(f"  Loaded {len(df):,} patients")
            
            prep = MLDataPreparatorV2()
            results = prep.prepare_for_training(df)
            
            # Save for future runs
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for name, (X, y) in results['splits'].items():
                data = X.copy()
                data['target'] = y.values
                data.to_parquet(DATA_DIR / f'ml_v2_{name}.parquet', index=False)
            
            # Load splits
            train_df = results['splits']['train_resampled']
            self.X_train = train_df[0]
            self.y_train = train_df[1]
            
            val_df = results['splits']['val']
            self.X_val = val_df[0]
            self.y_val = val_df[1]
            
            test_df = results['splits']['test']
            self.X_test = test_df[0]
            self.y_test = test_df[1]
            
            self.feature_names = list(self.X_train.columns)
            self.n_classes = len(np.unique(self.y_train))
            
            logger.info(f"\n  Training: {len(self.X_train):,}")
            logger.info(f"  Validation: {len(self.X_val):,}")
            logger.info(f"  Test: {len(self.X_test):,}")
            logger.info(f"  Features: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_models(self):
        """Train all models"""
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING MODELS")
        logger.info("=" * 70)
        
        # Logistic Regression
        logger.info("\n  1. Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.models['Logistic Regression'].fit(self.X_train, self.y_train)
        logger.info("    ✓")
        
        # Decision Tree
        logger.info("  2. Decision Tree...")
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=8, class_weight='balanced', random_state=42
        )
        self.models['Decision Tree'].fit(self.X_train, self.y_train)
        logger.info("    ✓")
        
        # Random Forest
        logger.info("  3. Random Forest...")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        logger.info("    ✓")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("  4. XGBoost...")
            self.models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, verbosity=0,
                random_state=42, n_jobs=-1
            )
            self.models['XGBoost'].fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            logger.info("    ✓")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("  5. LightGBM...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.models['LightGBM'] = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8,
                    verbosity=-1, random_state=42, n_jobs=-1
                )
                self.models['LightGBM'].fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)]
                )
            logger.info("    ✓")
        
        # Ensemble
        if XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE:
            logger.info("  6. XGB+LGB Ensemble...")
            
            class EnsembleClassifier:
                def __init__(self, xgb_model, lgb_model):
                    self.xgb_model = xgb_model
                    self.lgb_model = lgb_model
                    
                def predict_proba(self, X):
                    return (self.xgb_model.predict_proba(X) + 
                            self.lgb_model.predict_proba(X)) / 2
                    
                def predict(self, X):
                    return np.argmax(self.predict_proba(X), axis=1)
            
            self.models['XGB+LGB Ensemble'] = EnsembleClassifier(
                self.models['XGBoost'], self.models['LightGBM']
            )
            logger.info("    ✓")
    
    def evaluate_models(self):
        """Evaluate all models with proper metrics (NO ACCURACY)"""
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATING MODELS")
        logger.info("=" * 70)
        
        for name, model in self.models.items():
            logger.info(f"\n  {name}...")
            
            try:
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)
                
                # Core metrics (NO ACCURACY - it's misleading)
                precision_macro = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
                recall_macro = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
                
                # Per-class recall (critical for imbalanced data)
                recall_per_class = recall_score(self.y_test, y_pred, average=None, zero_division=0)
                
                # ROC-AUC
                try:
                    y_bin = label_binarize(self.y_test, classes=np.arange(self.n_classes))
                    roc_auc_macro = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
                    roc_auc_weighted = roc_auc_score(y_bin, y_proba, average='weighted', multi_class='ovr')
                except:
                    roc_auc_macro = 0.5
                    roc_auc_weighted = 0.5
                
                # Average Precision
                try:
                    avg_prec = average_precision_score(y_bin, y_proba, average='weighted')
                except:
                    avg_prec = 0.0
                
                self.results[name] = {
                    'f1_macro': f1_macro,
                    'f1_weighted': f1_weighted,
                    'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                    'recall_per_class': recall_per_class.tolist(),
                    'roc_auc_macro': roc_auc_macro,
                    'roc_auc_weighted': roc_auc_weighted,
                    'avg_precision': avg_prec,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'confusion_matrix': confusion_matrix(self.y_test, y_pred)
                }
                
                logger.info(f"    ROC-AUC (macro): {roc_auc_macro:.4f}")
                logger.info(f"    F1 (macro): {f1_macro:.4f}")
                logger.info(f"    Recall (macro): {recall_macro:.4f}")
                
                # LEAKAGE WARNING
                if roc_auc_macro > 0.90:
                    logger.warning(f"    ⚠️  HIGH AUC ({roc_auc_macro:.4f}) - CHECK FOR LEAKAGE!")
                
            except Exception as e:
                logger.error(f"    Error: {e}")
        
        # Find best model (by macro F1, not AUC - more reliable for imbalanced)
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_macro'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"\n  Best model: {best_name}")
        logger.info(f"  F1 (macro): {self.results[best_name]['f1_macro']:.4f}")
        logger.info(f"  ROC-AUC (macro): {self.results[best_name]['roc_auc_macro']:.4f}")
    
    def generate_comparison_table(self):
        """Generate proper model comparison table"""
        logger.info("\n" + "=" * 70)
        logger.info("MODEL COMPARISON TABLE")
        logger.info("=" * 70)
        
        rows = []
        for name in self.results:
            r = self.results[name]
            rows.append({
                'Model': name,
                'ROC-AUC (macro)': round(r['roc_auc_macro'], 4),
                'ROC-AUC (weighted)': round(r['roc_auc_weighted'], 4),
                'F1 (macro)': round(r['f1_macro'], 4),
                'F1 (weighted)': round(r['f1_weighted'], 4),
                'Recall (macro)': round(r['recall_macro'], 4),
                'Precision (macro)': round(r['precision_macro'], 4),
                'Recall-Critical': round(r['recall_per_class'][0], 4) if len(r['recall_per_class']) > 0 else 0,
                'Recall-High': round(r['recall_per_class'][1], 4) if len(r['recall_per_class']) > 1 else 0
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('F1 (macro)', ascending=False).reset_index(drop=True)
        
        df.to_csv(TABLES_DIR / 'model_comparison_v2.csv', index=False)
        
        logger.info("\n" + "-" * 100)
        logger.info(f"  {'Model':<22} {'AUC-macro':>10} {'F1-macro':>10} {'Recall-macro':>12} {'R-Critical':>12} {'R-High':>10}")
        logger.info("-" * 100)
        for _, row in df.iterrows():
            logger.info(
                f"  {row['Model']:<22} {row['ROC-AUC (macro)']:>10.4f} {row['F1 (macro)']:>10.4f} "
                f"{row['Recall (macro)']:>12.4f} {row['Recall-Critical']:>12.4f} {row['Recall-High']:>10.4f}"
            )
        
        return df
    
    def plot_roc_curves(self):
        """Generate ROC curves"""
        logger.info("  Generating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        y_bin = label_binarize(self.y_test, classes=np.arange(self.n_classes))
        
        for name in self.results:
            y_proba = self.results[name]['y_proba']
            fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            roc_auc = auc(fpr, tpr)
            
            color = MODEL_COLORS.get(name, '#666666')
            linewidth = 3 if name == self.best_model_name else 2
            ax.plot(fpr, tpr, color=color, linewidth=linewidth,
                   label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.500)')
        
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('ROC Curves - Leakage-Free Model', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add realistic expectation annotation
        ax.annotate('Expected realistic range: 0.65-0.85', 
                   xy=(0.5, 0.3), fontsize=12, style='italic', color='gray')
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'roc_curves_v2.png', dpi=150)
        plt.close(fig)
    
    def plot_confusion_matrix(self):
        """Generate confusion matrix"""
        logger.info("  Generating confusion matrix...")
        
        cm = self.results[self.best_model_name]['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=CLASS_NAMES[:self.n_classes],
                   yticklabels=CLASS_NAMES[:self.n_classes])
        axes[0].set_title(f'Confusion Matrix (Counts)\n{self.best_model_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted', fontsize=12)
        axes[0].set_ylabel('Actual', fontsize=12)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[1],
                   xticklabels=CLASS_NAMES[:self.n_classes],
                   yticklabels=CLASS_NAMES[:self.n_classes])
        axes[1].set_title(f'Confusion Matrix (Normalized)\n{self.best_model_name}', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted', fontsize=12)
        axes[1].set_ylabel('Actual', fontsize=12)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'confusion_matrix_v2.png', dpi=150)
        plt.close(fig)
    
    def plot_per_class_recall(self):
        """Plot per-class recall - critical for imbalanced data"""
        logger.info("  Generating per-class recall plot...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(CLASS_NAMES[:self.n_classes]))
        width = 0.12
        
        for i, name in enumerate(self.results):
            recalls = self.results[name]['recall_per_class'][:self.n_classes]
            color = MODEL_COLORS.get(name, '#666666')
            ax.bar(x + i * width, recalls, width, label=name, color=color, alpha=0.8)
        
        ax.set_xlabel('Risk Class', fontsize=14)
        ax.set_ylabel('Recall', fontsize=14)
        ax.set_title('Per-Class Recall Comparison\n(Higher is better for catching cases)', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * (len(self.results) - 1) / 2)
        ax.set_xticklabels(CLASS_NAMES[:self.n_classes], fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'per_class_recall_v2.png', dpi=150)
        plt.close(fig)
    
    def generate_shap_explanations(self):
        """Generate SHAP explanations"""
        if not SHAP_AVAILABLE or 'XGBoost' not in self.models:
            logger.warning("  SHAP not available, skipping")
            return
        
        logger.info("  Generating SHAP explanations...")
        
        try:
            model = self.models['XGBoost']
            n_samples = min(500, len(self.X_val))
            X_sample = self.X_val.sample(n=n_samples, random_state=42)
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # SHAP bar chart
            if isinstance(shap_values, list):
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
            
            shap_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_shap
            }).sort_values('importance', ascending=False)
            
            shap_df.to_csv(TABLES_DIR / 'shap_importance_v2.csv', index=False)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            top_n = min(15, len(shap_df))
            top_features = shap_df.head(top_n)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
            ax.barh(top_features['feature'][::-1], top_features['importance'][::-1], color=colors)
            
            ax.set_xlabel('Mean |SHAP Value|', fontsize=14)
            ax.set_title('Top Features by SHAP Importance\n(Should show workload/progress features, NOT outcomes)', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / 'shap_importance_v2.png', dpi=150)
            plt.close(fig)
            
            logger.info("    ✓")
            
        except Exception as e:
            logger.error(f"    SHAP failed: {e}")
    
    def plot_precision_recall_curves(self):
        """Generate PR curves"""
        logger.info("  Generating Precision-Recall curves...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        y_bin = label_binarize(self.y_test, classes=np.arange(self.n_classes))
        
        for name in self.results:
            y_proba = self.results[name]['y_proba']
            precision_arr, recall_arr, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
            ap = average_precision_score(y_bin, y_proba, average='micro')
            
            color = MODEL_COLORS.get(name, '#666666')
            linewidth = 3 if name == self.best_model_name else 2
            ax.plot(recall_arr, precision_arr, color=color, linewidth=linewidth,
                   label=f'{name} (AP = {ap:.3f})')
        
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curves - Leakage-Free Model', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'precision_recall_v2.png', dpi=150)
        plt.close(fig)
    
    def save_models(self):
        """Save trained models"""
        logger.info("\n  Saving models...")
        
        for name, model in self.models.items():
            if name == 'XGB+LGB Ensemble':
                continue
            
            try:
                safe_name = name.lower().replace(' ', '_')
                path = MODELS_DIR / f'risk_classifier_v2_{safe_name}.pkl'
                with open(path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"    ✅ {name}")
            except Exception as e:
                logger.warning(f"    Failed: {name} - {e}")
    
    def generate_report(self):
        """Generate training report"""
        best = self.results[self.best_model_name]
        
        report = {
            'version': '2.0.0 (Leakage-Free)',
            'created': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'metrics': {
                'roc_auc_macro': round(best['roc_auc_macro'], 4),
                'f1_macro': round(best['f1_macro'], 4),
                'recall_macro': round(best['recall_macro'], 4),
                'recall_per_class': best['recall_per_class']
            },
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'n_training_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'leakage_prevention': {
                'prediction_features': PREDICTION_FEATURES[:10] + ['...'],
                'outcome_features_excluded': OUTCOME_FEATURES[:10] + ['...']
            }
        }
        
        with open(OUTPUT_DIR / 'training_report_v2.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def run(self):
        """Run complete training pipeline"""
        start = datetime.now()
        
        logger.info("\n" + "=" * 70)
        logger.info("TRIALPULSE NEXUS 10X - RISK CLASSIFIER v2.0 (LEAKAGE-FREE)")
        logger.info("=" * 70)
        logger.info(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Prepare data
        if not self.prepare_data():
            return False
        
        # Train models
        self.train_models()
        
        # Evaluate
        self.evaluate_models()
        self.generate_comparison_table()
        
        # Visualizations
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 70)
        
        self.plot_roc_curves()
        self.plot_confusion_matrix()
        self.plot_per_class_recall()
        self.plot_precision_recall_curves()
        self.generate_shap_explanations()
        
        # Save
        self.save_models()
        self.generate_report()
        
        duration = (datetime.now() - start).total_seconds()
        
        best = self.results[self.best_model_name]
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ TRAINING COMPLETE (LEAKAGE-FREE)")
        logger.info("=" * 70)
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Best Model: {self.best_model_name}")
        logger.info(f"  ROC-AUC (macro): {best['roc_auc_macro']:.4f}")
        logger.info(f"  F1 (macro): {best['f1_macro']:.4f}")
        logger.info(f"  Recall (macro): {best['recall_macro']:.4f}")
        
        if best['roc_auc_macro'] > 0.90:
            logger.warning("")
            logger.warning("  ⚠️  AUC > 0.90 - INVESTIGATE FOR REMAINING LEAKAGE!")
        elif best['roc_auc_macro'] < 0.65:
            logger.info("")
            logger.info("  ℹ️  Low AUC is expected for true prediction tasks")
            logger.info("     Consider adding more predictive features")
        else:
            logger.info("")
            logger.info("  ✅ AUC in realistic range (0.65-0.85) for prediction")
        
        logger.info(f"\n  Output: {OUTPUT_DIR}")
        
        return True


def main():
    trainer = LeakageFreeRiskTrainer()
    success = trainer.run()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ LEAKAGE-FREE TRAINING COMPLETE")
        print("=" * 70)
        print(f"\nOutput: {OUTPUT_DIR}")
    else:
        print("\n❌ TRAINING FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
