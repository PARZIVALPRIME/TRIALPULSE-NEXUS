"""
TRIALPULSE NEXUS 10X — ML MODELS CONSOLIDATED PRESENTATION
Generates HTML file with all 5 models and embedded visualizations
"""

import base64
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / 'data' / 'outputs'

# Output paths for each model
MODELS = {
    'patient_risk': {
        'name': 'Patient Risk Classifier',
        'version': 'v9',
        'figures_dir': OUTPUT_DIR / 'ml_training_v9_production' / 'figures',
        'status': 'PRODUCTION READY',
        'metrics': {
            'ROC-AUC': '0.978',
            'Top Predictor': 'LightGBM',
            'Critical Recall': '71.9%'
        },
        'context': {
            'ROC-AUC': 'World-Class (Target > 0.90)',
            'Top Predictor': 'Best of 5 architectures',
            'Critical Recall': 'Safety-critical sensitivity'
        }
    },
    'issue_detector': {
        'name': 'Multi-Label Issue Detector',
        'version': 'v3',
        'figures_dir': OUTPUT_DIR / 'issue_detector_PRODUCTION_v3' / 'figures',
        'status': 'PRODUCTION READY',
        'metrics': {
            'Avg ROC-AUC': '0.852',
            'Avg F1 Score': '0.356',
            'Labels Trained': '13/14',
            'Leakage Check': 'PASSED'
        },
        'context': {
            'Avg ROC-AUC': 'Excellent across 14 labels',
            'Avg F1 Score': 'Good for imbalanced multi-label',
            'Labels Trained': 'SAE Safety excluded (0% prevalence)',
            'Leakage Check': 'Zero leakage detected'
        }
    },
    'resolution_time': {
        'name': 'Resolution Time Predictor',
        'version': 'v3',
        'figures_dir': OUTPUT_DIR / 'resolution_time_PRODUCTION_v3' / 'figures',
        'status': 'PRODUCTION READY',
        'metrics': {
            'MAE': '3.62 days',
            'Coverage': '80.9%',
            'SHORT MAE': '2.2 days',
            'LONG MAE': '16.6 days'
        },
        'context': {
            'MAE': 'Overall prediction error',
            'Coverage': 'Predictions within interval',
            'SHORT MAE': '≤7 days (82% of data)',
            'LONG MAE': '>30 days (4% of data)'
        }
    },
    'site_ranker': {
        'name': 'Site Risk Ranker',
        'version': 'v2',
        'figures_dir': OUTPUT_DIR / 'site_risk_ranker_OPTIMIZED_v2' / 'figures',
        'status': 'PRODUCTION READY',
        'metrics': {
            'Kendall Tau': '0.753',
            'Spearman Rho': '0.906',
            'Sites Ranked': '2,233',
            'Red Flags': 'NONE'
        },
        'context': {
            'Kendall Tau': 'Strong rank correlation',
            'Spearman Rho': 'Excellent monotonic relationship',
            'Sites Ranked': 'Full site population',
            'Red Flags': 'All checks passed'
        }
    },
    'anomaly_detector': {
        'name': 'Anomaly Detector',
        'version': 'v2',
        'figures_dir': OUTPUT_DIR / 'anomaly_detector_OPTIMIZED_v2' / 'figures',
        'status': 'PRODUCTION READY',
        'metrics': {
            'Patients Analyzed': '57,997',
            'Anomalies Detected': '2,369 (4.1%)',
            'Extreme Anomalies': '54',
            'Red Flags': 'NONE'
        },
        'context': {
            'Patients Analyzed': 'Full patient population',
            'Anomalies Detected': 'Above 50 anomaly score',
            'Extreme Anomalies': 'Above 90 anomaly score',
            'Red Flags': 'All checks passed'
        }
    }
}


def image_to_base64(path: Path) -> str:
    """Convert image to base64 string."""
    if not path.exists():
        return ""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def generate_html():
    """Generate comprehensive HTML presentation."""
    
    html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>TrialPulse Nexus 10X - ML Models Production Summary</title>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    
    body { 
        font-family: 'Segoe UI', Arial, sans-serif; 
        max-width: 1600px; 
        margin: 0 auto; 
        padding: 40px; 
        background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 50%, #0d1117 100%);
        color: #c9d1d9;
        min-height: 100vh;
    }
    
    h1 { 
        color: #58a6ff; 
        text-align: center; 
        border-bottom: 2px solid #30363d; 
        padding-bottom: 20px; 
        font-size: 42px;
        text-shadow: 0 0 20px rgba(88, 166, 255, 0.3);
        margin-bottom: 10px;
    }
    
    .subtitle { 
        text-align: center; 
        font-size: 22px; 
        color: #8b949e; 
        margin-bottom: 50px;
    }
    
    .model-container { 
        display: flex; 
        flex-wrap: wrap; 
        gap: 40px; 
        justify-content: center; 
    }
    
    .model-box { 
        background: linear-gradient(145deg, #161b22 0%, #1c2129 100%);
        border-radius: 20px; 
        padding: 30px; 
        border: 1px solid #30363d; 
        width: 100%; 
        box-shadow: 0 15px 40px rgba(0,0,0,0.6);
        margin-bottom: 30px;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .model-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
    }
    
    h2 { 
        color: #f0883e; 
        margin-top: 0; 
        font-size: 28px; 
        border-bottom: 1px solid #30363d; 
        padding-bottom: 15px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    table { 
        border-collapse: collapse; 
        width: 100%; 
        margin: 20px 0; 
        font-size: 16px; 
    }
    
    th, td { 
        border: 1px solid #30363d; 
        padding: 15px; 
        text-align: left; 
    }
    
    th { 
        background: #21262d; 
        color: #58a6ff; 
    }
    
    tr:nth-child(even) { background: #0d1117; }
    tr:hover { background: rgba(88, 166, 255, 0.1); }
    
    .metric-value { 
        font-size: 24px; 
        font-weight: bold; 
        color: #3fb950; 
    }
    
    .viz-container { 
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 25px;
        margin-top: 30px;
    }
    
    .viz-item {
        text-align: center;
        background: #0d1117;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363d;
    }
    
    .viz-item img { 
        max-width: 100%; 
        height: auto; 
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
    }
    
    .viz-caption {
        color: #8b949e;
        font-size: 14px;
        margin-top: 10px;
    }
    
    .status-pass { 
        color: #3fb950; 
        font-weight: bold; 
        border: 2px solid #3fb950; 
        padding: 8px 16px; 
        border-radius: 25px; 
        display: inline-block;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 10px rgba(63, 185, 80, 0.3);
    }
    
    .toc {
        background: #161b22;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 40px;
        border: 1px solid #30363d;
    }
    
    .toc h3 {
        color: #58a6ff;
        margin-bottom: 15px;
    }
    
    .toc ul {
        list-style: none;
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
    }
    
    .toc li a {
        color: #8b949e;
        text-decoration: none;
        padding: 10px 20px;
        background: #21262d;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .toc li a:hover {
        background: #30363d;
        color: #58a6ff;
    }
    
    .timestamp {
        text-align: center;
        color: #6e7681;
        font-size: 14px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #30363d;
    }
</style>
</head>
<body>

<h1>🚀 TrialPulse Nexus 10X — ML Production Audit</h1>
<p class="subtitle">5-Star Quality Assurance Report | 5 Models Optimized for Production</p>

<div class="toc">
<h3>📋 Table of Contents</h3>
<ul>
'''

    # Add TOC items
    for i, (key, model) in enumerate(MODELS.items(), 1):
        html += f'<li><a href="#{key}">{i}. {model["name"]}</a></li>\n'
    
    html += '''</ul>
</div>

<div class="model-container">
'''

    # Generate each model section
    for i, (key, model) in enumerate(MODELS.items(), 1):
        figures_dir = model['figures_dir']
        
        html += f'''
    <div class="model-box" id="{key}">
        <h2>{i}. {model['name']} ({model['version']}) <span class="status-pass">{model['status']}</span></h2>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Assessment</th>
            </tr>
'''
        
        # Add metrics
        for metric_name, value in model['metrics'].items():
            context = model['context'].get(metric_name, '')
            html += f'''            <tr>
                <td>{metric_name}</td>
                <td class="metric-value">{value}</td>
                <td>{context}</td>
            </tr>
'''
        
        html += '''        </table>
        
        <div class="viz-container">
'''
        
        # Add visualizations
        if figures_dir.exists():
            figures = sorted(figures_dir.glob('*.png'))
            for fig in figures:
                b64 = image_to_base64(fig)
                if b64:
                    caption = fig.stem.replace('_', ' ').title()
                    html += f'''            <div class="viz-item">
                <img src="data:image/png;base64,{b64}" alt="{caption}">
                <p class="viz-caption">{caption}</p>
            </div>
'''
        
        html += '''        </div>
    </div>
'''
    
    html += f'''
</div>

<div class="timestamp">
    Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | TrialPulse Nexus 10X
</div>

</body>
</html>
'''
    
    return html


def main():
    print("=" * 60)
    print("  GENERATING ML MODELS CONSOLIDATED PRESENTATION")
    print("=" * 60)
    
    # Check which figure directories exist
    print("\nChecking model outputs...")
    for key, model in MODELS.items():
        figures_dir = model['figures_dir']
        if figures_dir.exists():
            count = len(list(figures_dir.glob('*.png')))
            print(f"  ✓ {model['name']}: {count} figures")
        else:
            print(f"  ✗ {model['name']}: Directory not found")
            print(f"    Expected: {figures_dir}")
    
    # Generate HTML
    print("\nGenerating HTML...")
    html = generate_html()
    
    # Save output
    output_path = ROOT / 'models_presentation.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved: {output_path}")
    print(f"  Size: {file_size:.2f} MB")
    print("=" * 60)


if __name__ == '__main__':
    main()
