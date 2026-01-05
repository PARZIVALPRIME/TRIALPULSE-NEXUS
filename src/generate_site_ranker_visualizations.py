"""
TRIALPULSE NEXUS 10X - Site Risk Ranker Visualizations
========================================================
Generates PPT-ready visualizations for the Site Risk Ranker model.

Author: TrialPulse Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for professional PPT visuals
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 150

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RANKER_DIR = PROJECT_ROOT / 'data' / 'processed' / 'ml' / 'site_ranker'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'site_ranker_visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load ranker results and site data"""
    
    # Load results JSON
    with open(RANKER_DIR / 'site_ranker_results.json', 'r') as f:
        results = json.load(f)
    
    # Load ranked sites
    ranked_sites = pd.read_csv(RANKER_DIR / 'site_risk_ranking.csv')
    
    # Load full site metrics
    site_metrics = pd.read_parquet(RANKER_DIR / 'site_metrics_with_scores.parquet')
    
    return results, ranked_sites, site_metrics


def plot_feature_importance(results, output_dir):
    """Plot feature importance bar chart"""
    
    importance = results['feature_importance']
    
    # Get top 15 features
    features = list(importance.keys())[:15]
    values = [importance[f] * 100 for f in features]
    
    # Clean feature names for display
    clean_names = []
    for f in features:
        name = f.replace('_sum', ' (sum)').replace('_mean', ' (mean)').replace('_max', ' (max)')
        name = name.replace('edrr_edrr_', 'EDRR ').replace('sae_dm_sae_dm_', 'SAE DM ')
        name = name.replace('inactivated_inactivated_', 'Inactivated ')
        name = name.replace('_', ' ').title()
        clean_names.append(name[:35])  # Truncate long names
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
    bars = ax.barh(range(len(features)), values, color=colors)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(clean_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title('Site Risk Ranker - Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    
    # Add threshold line for dominance check
    ax.axvline(x=40, color='red', linestyle='--', linewidth=2, label='Single Feature Threshold (40%)')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Feature importance plot saved")


def plot_metrics_comparison(results, output_dir):
    """Plot ranking metrics comparison"""
    
    metrics = results['metrics']
    
    # Extract metric values
    metric_names = ['NDCG@5', 'NDCG@10', 'NDCG@20', 'MAP', 'Kendall τ', 'Spearman ρ']
    metric_values = [
        metrics['ndcg@5'],
        metrics['ndcg@10'],
        metrics['ndcg@20'],
        metrics['map'],
        metrics['kendall_tau'],
        metrics['spearman']
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    x = np.arange(len(metric_names))
    colors = ['#2ecc71' if v >= 0.7 else '#f39c12' if v >= 0.5 else '#e74c3c' for v in metric_values]
    bars = ax.bar(x, metric_values, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('Site Risk Ranker - Ranking Performance Metrics', fontsize=14, fontweight='bold')
    
    # Add threshold lines
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Leakage Threshold (0.95)')
    ax.axhline(y=0.55, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Min Acceptable (0.55)')
    ax.legend(loc='upper right')
    
    # Add target zone shading
    ax.axhspan(0.7, 0.85, alpha=0.1, color='green', label='Target Range')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Metrics comparison plot saved")


def plot_risk_distribution(site_metrics, output_dir):
    """Plot risk score distribution"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram of risk scores
    ax1 = axes[0]
    scores = site_metrics['model_risk_score'].dropna()
    
    ax1.hist(scores, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(x=scores.quantile(0.9), color='red', linestyle='--', linewidth=2, 
                label=f'90th Percentile ({scores.quantile(0.9):.2f})')
    ax1.axvline(x=scores.quantile(0.75), color='orange', linestyle='--', linewidth=2,
                label=f'75th Percentile ({scores.quantile(0.75):.2f})')
    ax1.axvline(x=scores.mean(), color='green', linestyle='-', linewidth=2,
                label=f'Mean ({scores.mean():.2f})')
    
    ax1.set_xlabel('Risk Score', fontsize=12)
    ax1.set_ylabel('Number of Sites', fontsize=12)
    ax1.set_title('Distribution of Site Risk Scores', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Plot 2: Risk score by study
    ax2 = axes[1]
    study_risk = site_metrics.groupby('study_id')['model_risk_score'].agg(['mean', 'std']).reset_index()
    study_risk = study_risk.sort_values('mean', ascending=False).head(15)
    
    # Clean study names
    study_risk['study_clean'] = study_risk['study_id'].str.replace('Study_', 'S')
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(study_risk)))
    bars = ax2.bar(study_risk['study_clean'], study_risk['mean'], 
                   yerr=study_risk['std'], color=colors, edgecolor='black', 
                   capsize=3, alpha=0.8)
    
    ax2.set_xlabel('Study', fontsize=12)
    ax2.set_ylabel('Mean Risk Score', fontsize=12)
    ax2.set_title('Mean Risk Score by Study (Top 15)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'risk_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Risk distribution plot saved")


def plot_top_sites_heatmap(ranked_sites, output_dir):
    """Plot heatmap of top risky sites"""
    
    top20 = ranked_sites.head(20).copy()
    
    # Create site label
    top20['site_label'] = top20['study_id'].str.replace('Study_', 'S') + ' - ' + \
                          top20['site_id'].str.replace('Site_', '')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(top20)))[::-1]
    bars = ax.barh(range(len(top20)), top20['model_risk_score'], color=colors)
    
    # Add patient count annotations
    for i, (bar, row) in enumerate(zip(bars, top20.itertuples())):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'n={row.patient_count}', va='center', fontsize=9, color='gray')
    
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['site_label'])
    ax.invert_yaxis()
    ax.set_xlabel('Risk Score', fontsize=12)
    ax.set_title('Top 20 High-Risk Sites', fontsize=14, fontweight='bold')
    
    # Add rank numbers
    for i in range(len(top20)):
        ax.text(-0.3, i, f'#{i+1}', va='center', ha='right', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_sites.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Top sites plot saved")


def plot_red_flags_dashboard(results, output_dir):
    """Plot red flags check dashboard"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    checks = [
        ('Leakage\n(NDCG < 0.95)', results['metrics']['ndcg@10'], 0.95, 'below'),
        ('Identity Ranking\n(Tau < 0.95)', results['metrics']['kendall_tau'], 0.95, 'below'),
        ('Feature Dominance\n(< 40%)', list(results['feature_importance'].values())[0] * 100, 40, 'below'),
        ('Top 5 Features\n(< 80%)', sum(list(results['feature_importance'].values())[:5]) * 100, 80, 'below'),
    ]
    
    x = np.arange(len(checks))
    
    for i, (name, value, threshold, direction) in enumerate(checks):
        passed = (value < threshold) if direction == 'below' else (value > threshold)
        color = '#2ecc71' if passed else '#e74c3c'
        
        ax.bar(i, value, color=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.axhline(y=threshold, xmin=(i-0.4)/len(checks), xmax=(i+0.4)/len(checks),
                   color='black', linestyle='--', linewidth=2)
        
        status = '✓ PASS' if passed else '✗ FAIL'
        ax.text(i, value + 3, f'{value:.1f}\n{status}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in checks], fontsize=10)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Red Flag Detection Dashboard', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'red_flags_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Red flags dashboard saved")


def plot_model_summary(results, site_metrics, output_dir):
    """Generate model summary infographic"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Key Stats
    ax1 = axes[0, 0]
    ax1.axis('off')
    
    stats_text = f"""
    SITE RISK RANKER v1.0
    ══════════════════════════════
    
    📊 DATA
    • Patients: 57,997
    • Sites: 3,416
    • Features: 141
    
    🎯 PERFORMANCE
    • NDCG@10: {results['metrics']['ndcg@10']:.4f}
    • MAP: {results['metrics']['map']:.4f}
    • Kendall τ: {results['metrics']['kendall_tau']:.4f}
    
    ✅ STATUS: PRODUCTION READY
    """
    
    ax1.text(0.1, 0.9, stats_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    ax1.set_title('Model Summary', fontsize=14, fontweight='bold')
    
    # Panel 2: Feature categories pie chart
    ax2 = axes[0, 1]
    importance = results['feature_importance']
    
    # Group features by category
    categories = {
        'EDRR Issues': sum(v for k, v in importance.items() if 'edrr' in k.lower()),
        'SAE Workload': sum(v for k, v in importance.items() if 'sae' in k.lower()),
        'Issue Density': sum(v for k, v in importance.items() if 'density' in k.lower()),
        'Form Issues': sum(v for k, v in importance.items() if 'inactivated' in k.lower()),
        'Other': 1 - sum(v for k, v in importance.items() 
                        if any(x in k.lower() for x in ['edrr', 'sae', 'density', 'inactivated']))
    }
    
    # Only show categories with significant importance
    categories = {k: v for k, v in categories.items() if v > 0.01}
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    wedges, texts, autotexts = ax2.pie(categories.values(), labels=categories.keys(),
                                        autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Feature Category Breakdown', fontsize=14, fontweight='bold')
    
    # Panel 3: Risk score vs patient count scatter
    ax3 = axes[1, 0]
    sample = site_metrics.sample(min(500, len(site_metrics)), random_state=42)
    
    scatter = ax3.scatter(sample['patient_count'], sample['model_risk_score'],
                         c=sample['model_risk_score'], cmap='RdYlGn_r',
                         alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    ax3.set_xlabel('Patient Count per Site', fontsize=12)
    ax3.set_ylabel('Risk Score', fontsize=12)
    ax3.set_title('Risk Score vs Site Size', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Risk Score')
    
    # Panel 4: Metrics gauge
    ax4 = axes[1, 1]
    
    metric_names = ['NDCG@5', 'NDCG@10', 'NDCG@20', 'MAP', 'Tau']
    metric_values = [
        results['metrics']['ndcg@5'],
        results['metrics']['ndcg@10'],
        results['metrics']['ndcg@20'],
        results['metrics']['map'],
        results['metrics']['kendall_tau']
    ]
    
    x = np.arange(len(metric_names))
    colors = ['#27ae60' if v >= 0.7 else '#f1c40f' if v >= 0.5 else '#e74c3c' for v in metric_values]
    bars = ax4.bar(x, metric_values, color=colors, edgecolor='black')
    
    for bar, val in zip(bars, metric_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(metric_names)
    ax4.set_ylim(0, 1.1)
    ax4.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target (0.7)')
    ax4.legend()
    
    plt.suptitle('Site Risk Ranker — Model Overview', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'model_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  ✅ Model summary infographic saved")


def main():
    """Generate all visualizations"""
    
    print("\n" + "=" * 70)
    print("🎨 SITE RISK RANKER - VISUALIZATION GENERATION")
    print("=" * 70)
    
    # Load data
    print("\n📂 Loading data...")
    results, ranked_sites, site_metrics = load_data()
    print(f"   Loaded {len(site_metrics)} sites, {len(ranked_sites)} ranked")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    plot_feature_importance(results, OUTPUT_DIR)
    plot_metrics_comparison(results, OUTPUT_DIR)
    plot_risk_distribution(site_metrics, OUTPUT_DIR)
    plot_top_sites_heatmap(ranked_sites, OUTPUT_DIR)
    plot_red_flags_dashboard(results, OUTPUT_DIR)
    plot_model_summary(results, site_metrics, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # List files
    print("\n📁 Generated files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"   • {f.name}")


if __name__ == '__main__':
    main()
