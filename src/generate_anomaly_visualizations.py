"""
TRIALPULSE NEXUS 10X - Anomaly Detection Visualizations
========================================================
Generates PPT-ready visualizations for the Anomaly Detector model.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
plt.style.use('seaborn-v0_8-whitegrid')
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'analytics' / 'v2'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'outputs' / 'anomaly_visualizations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLORS = {
    'Critical': '#dc3545',
    'High': '#fd7e14', 
    'Medium': '#ffc107',
    'Normal': '#28a745',
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
}


def load_data():
    """Load anomaly detection results."""
    print("Loading data...")
    patients = pd.read_parquet(DATA_DIR / 'patient_anomalies_v2.parquet')
    sites = pd.read_csv(DATA_DIR / 'site_anomalies_v2.csv')
    with open(DATA_DIR / 'summary_v2.json') as f:
        summary = json.load(f)
    return patients, sites, summary


def plot_score_distribution(patients, output_dir):
    """Plot anomaly score distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    ax.hist(patients['anomaly_score'], bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    
    # Add threshold lines
    thresholds = patients['anomaly_score'].quantile([0.90, 0.95, 0.99])
    colors = [COLORS['Medium'], COLORS['High'], COLORS['Critical']]
    labels = ['Medium (P90)', 'High (P95)', 'Critical (P99)']
    
    for thresh, color, label in zip(thresholds, colors, labels):
        ax.axvline(x=thresh, color=color, linestyle='--', linewidth=2, label=f'{label}: {thresh:.3f}')
    
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Patient Count', fontsize=12)
    ax.set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / '1_score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Score distribution plot saved")


def plot_severity_pie(patients, output_dir):
    """Plot severity distribution pie chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    severity_counts = patients['severity'].value_counts()
    colors = [COLORS.get(s, '#999') for s in severity_counts.index]
    
    wedges, texts, autotexts = ax1.pie(
        severity_counts.values,
        labels=severity_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05 if s == 'Critical' else 0 for s in severity_counts.index]
    )
    ax1.set_title('Severity Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(severity_counts.index, severity_counts.values, color=colors, edgecolor='white')
    ax2.set_xlabel('Severity', fontsize=12)
    ax2.set_ylabel('Patient Count', fontsize=12)
    ax2.set_title('Severity Counts', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, severity_counts.values):
        ax2.annotate(f'{count:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '2_severity_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Severity distribution saved")


def plot_method_comparison(patients, output_dir):
    """Plot comparison of different scoring methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = [
        ('score_if', 'Isolation Forest', COLORS['primary']),
        ('score_recon', 'PCA Reconstruction', COLORS['secondary']),
        ('score_zscore', 'Z-Score Max', '#10b981'),
    ]
    
    for ax, (col, name, color) in zip(axes, methods):
        ax.hist(patients[col], bins=50, color=color, alpha=0.7, edgecolor='white')
        ax.set_xlabel('Score', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        
        # Add stats
        mean_val = patients[col].mean()
        p95 = patients[col].quantile(0.95)
        ax.axvline(mean_val, color='black', linestyle='-', linewidth=1.5, label=f'Mean: {mean_val:.3f}')
        ax.axvline(p95, color='red', linestyle='--', linewidth=1.5, label=f'P95: {p95:.3f}')
        ax.legend(fontsize=9)
    
    plt.suptitle('Anomaly Detection Methods Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '3_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Method comparison saved")


def plot_top_anomalies(patients, output_dir):
    """Plot top 20 anomalous patients."""
    top20 = patients.nlargest(20, 'anomaly_score')[['patient_key', 'study_id', 'site_id', 'anomaly_score', 'severity']]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = [COLORS.get(s, '#999') for s in top20['severity']]
    bars = ax.barh(range(len(top20)), top20['anomaly_score'], color=colors, edgecolor='white')
    
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"{row['study_id']}/{row['site_id']}" for _, row in top20.iterrows()], fontsize=9)
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_title('Top 20 Most Anomalous Patients', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS[s], label=s) for s in ['Critical', 'High', 'Medium']]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_top_anomalies.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Top anomalies saved")


def plot_site_heatmap(sites, output_dir):
    """Plot site anomaly heatmap."""
    # Get top 30 sites by anomaly rate
    top_sites = sites.nlargest(30, 'anomaly_rate')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Prepare data for heatmap
    pivot_data = top_sites.set_index('site_id')[['avg_score', 'max_score', 'anomaly_rate', 'patients', 'anomaly_count']]
    pivot_data.columns = ['Avg Score', 'Max Score', 'Anomaly Rate', 'Total Patients', 'Anomalies']
    
    # Normalize for color mapping
    pivot_norm = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min() + 1e-10)
    
    sns.heatmap(pivot_norm, annot=pivot_data.round(2), fmt='', cmap='YlOrRd', 
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalized Value'})
    
    ax.set_title('Top 30 Sites by Anomaly Rate - Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Site ID', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / '5_site_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Site heatmap saved")


def plot_study_comparison(patients, output_dir):
    """Plot anomaly rates by study."""
    study_stats = patients.groupby('study_id').agg({
        'anomaly_score': 'mean',
        'is_anomaly': ['sum', 'mean'],
        'patient_key': 'count'
    }).reset_index()
    study_stats.columns = ['study_id', 'avg_score', 'anomaly_count', 'anomaly_rate', 'patient_count']
    study_stats = study_stats.sort_values('anomaly_rate', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart - anomaly rate
    colors = plt.cm.RdYlGn_r(study_stats['anomaly_rate'] / study_stats['anomaly_rate'].max())
    ax1.barh(study_stats['study_id'], study_stats['anomaly_rate'], color=colors)
    ax1.set_xlabel('Anomaly Rate', fontsize=12)
    ax1.set_ylabel('Study', fontsize=12)
    ax1.set_title('Anomaly Rate by Study', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    
    # Scatter - count vs rate
    scatter = ax2.scatter(study_stats['patient_count'], study_stats['anomaly_rate'], 
                         c=study_stats['avg_score'], cmap='RdYlGn_r', s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Patient Count', fontsize=12)
    ax2.set_ylabel('Anomaly Rate', fontsize=12)
    ax2.set_title('Patient Count vs Anomaly Rate', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Avg Score')
    
    plt.tight_layout()
    plt.savefig(output_dir / '6_study_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Study comparison saved")


def plot_score_correlation(patients, output_dir):
    """Plot correlation between different scores."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pairs = [
        ('score_if', 'score_recon', 'IF vs PCA'),
        ('score_if', 'score_zscore', 'IF vs Z-Score'),
        ('score_recon', 'score_zscore', 'PCA vs Z-Score'),
    ]
    
    # Sample for faster plotting
    sample = patients.sample(min(5000, len(patients)), random_state=42)
    
    for ax, (x, y, title) in zip(axes, pairs):
        colors = [COLORS.get(s, '#999') for s in sample['severity']]
        ax.scatter(sample[x], sample[y], c=colors, alpha=0.5, s=10)
        ax.set_xlabel(x.replace('score_', '').upper(), fontsize=11)
        ax.set_ylabel(y.replace('score_', '').upper(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add correlation
        corr = sample[[x, y]].corr().iloc[0, 1]
        ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                   fontsize=11, fontweight='bold', va='top')
    
    plt.suptitle('Score Correlation Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '7_score_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Score correlation saved")


def plot_summary_infographic(summary, patients, sites, output_dir):
    """Create summary infographic."""
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('ANOMALY DETECTION v2.0 - RESULTS SUMMARY', fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Key metrics boxes
    metrics = [
        ('Total Patients', f"{summary['total_patients']:,}", COLORS['primary']),
        ('Anomalies Detected', f"{summary['anomalies']:,}", COLORS['High']),
        ('Anomaly Rate', f"{summary['anomaly_rate']:.1%}", COLORS['Critical']),
        ('Duration', f"{summary['duration']:.1f}s", COLORS['Normal']),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(color)
        ax.text(0.5, 0.6, value, ha='center', va='center', fontsize=24, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=12, color='white', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Severity distribution
    ax1 = fig.add_subplot(gs[1, :2])
    sev_order = ['Critical', 'High', 'Medium', 'Normal']
    sev_data = [summary['severity'].get(s, 0) for s in sev_order]
    colors = [COLORS[s] for s in sev_order]
    bars = ax1.bar(sev_order, sev_data, color=colors, edgecolor='white', linewidth=2)
    ax1.set_title('Severity Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    for bar, val in zip(bars, sev_data):
        ax1.annotate(f'{val:,}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Site severity
    ax2 = fig.add_subplot(gs[1, 2:])
    site_sev = sites['severity'].value_counts()
    site_colors = [COLORS.get(s, '#999') for s in site_sev.index]
    ax2.pie(site_sev.values, labels=site_sev.index, colors=site_colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Site Severity Distribution', fontsize=14, fontweight='bold')
    
    # Score histogram
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.hist(patients['anomaly_score'], bins=40, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax3.set_xlabel('Anomaly Score')
    ax3.set_ylabel('Count')
    ax3.set_title('Score Distribution', fontsize=14, fontweight='bold')
    
    # Model info
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.axis('off')
    model_text = f"""
    MODEL: Anomaly Detector v2.0 Lite
    
    METHODS:
    • Isolation Forest (50%)
    • PCA Reconstruction (30%)
    • Z-Score Analysis (20%)
    
    FEATURES: {len(patients.columns)} metrics
    
    OUTPUT:
    • {summary['sites']} sites analyzed
    • {summary['critical_sites']} critical sites
    """
    ax4.text(0.1, 0.9, model_text, fontsize=12, family='monospace', va='top', transform=ax4.transAxes)
    ax4.set_title('Model Information', fontsize=14, fontweight='bold')
    
    plt.savefig(output_dir / '8_summary_infographic.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Summary infographic saved")


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("ANOMALY DETECTION VISUALIZATIONS")
    print("=" * 60)
    
    patients, sites, summary = load_data()
    print(f"Loaded {len(patients):,} patients, {len(sites):,} sites\n")
    
    print("Generating visualizations...")
    plot_score_distribution(patients, OUTPUT_DIR)
    plot_severity_pie(patients, OUTPUT_DIR)
    plot_method_comparison(patients, OUTPUT_DIR)
    plot_top_anomalies(patients, OUTPUT_DIR)
    plot_site_heatmap(sites, OUTPUT_DIR)
    plot_study_comparison(patients, OUTPUT_DIR)
    plot_score_correlation(patients, OUTPUT_DIR)
    plot_summary_infographic(summary, patients, sites, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print(f"✅ All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
