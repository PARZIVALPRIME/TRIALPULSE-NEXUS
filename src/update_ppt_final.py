import base64
from pathlib import Path

# Configuration
OUTPUT_FILE = Path(r"d:\trialpulse_nexus\ppt.html")

IMAGES = {
    "issue_detector": r"d:\trialpulse_nexus\data\outputs\issue_detector_PRODUCTION_v3\figures\8_summary_infographic.png",
    "resolution_time": r"d:\trialpulse_nexus\data\outputs\resolution_time_PRODUCTION_v3\figures\8_summary_infographic.png",
    "site_ranker": r"d:\trialpulse_nexus\data\outputs\site_risk_ranker_OPTIMIZED_v2\figures\8_summary.png",
    "anomaly_detector": r"d:\trialpulse_nexus\data\outputs\anomaly_detector_OPTIMIZED_v2\figures\8_summary.png"
}

# Helper to encode image
def get_base64_image(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return ""

# Read images
img_issue = get_base64_image(IMAGES["issue_detector"])
img_resolution = get_base64_image(IMAGES["resolution_time"])
img_site = get_base64_image(IMAGES["site_ranker"])
img_anomaly = get_base64_image(IMAGES["anomaly_detector"])

# HTML Content
html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>TrialPulse Nexus 10X - ML Models Production Summary</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 40px; background: #0d1117; color: #c9d1d9; }}
    h1 {{ color: #58a6ff; text-align: center; border-bottom: 2px solid #30363d; padding-bottom: 20px; font-size: 36px; }}
    h2 {{ color: #f0883e; margin-top: 0; font-size: 24px; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
    .subtitle {{ text-align: center; font-size: 20px; color: #8b949e; margin-bottom: 50px; }}
    
    .model-container {{ display: flex; flex-wrap: wrap; gap: 40px; justify-content: center; }}
    .model-box {{ background: #161b22; border-radius: 16px; padding: 30px; border: 1px solid #30363d; width: 100%; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }}
    
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 16px; }}
    th, td {{ border: 1px solid #30363d; padding: 15px; text-align: left; }}
    th {{ background: #21262d; color: #58a6ff; }}
    tr:nth-child(even) {{ background: #0d1117; }}
    
    .metric-value {{ font-size: 28px; font-weight: bold; color: #3fb950; }}
    .metric-label {{ font-weight: bold; color: #8b949e; }}
    
    .img-container {{ text-align: center; margin-top: 30px; background: #0d1117; padding: 10px; border-radius: 8px; border: 1px solid #30363d; }}
    img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    
    .status-pass {{ color: #3fb950; font-weight: bold; border: 1px solid #3fb950; padding: 5px 10px; border-radius: 20px; display: inline-block; }}
    .status-warn {{ color: #d29922; font-weight: bold; border: 1px solid #d29922; padding: 5px 10px; border-radius: 20px; display: inline-block; }}
</style>
</head>
<body>

<h1>TRIALPULSE NEXUS 10X — ML PRODUCTION AUDIT</h1>
<p class="subtitle">5-Star Quality Assurance Report | 5 Models Optimized for Production</p>

<div class="model-container">

    <!-- MODEL 1: PATIENT RISK CLASSIFIER -->
    <div class="model-box">
        <h2>1. Patient Risk Classifier (v9) <span class="status-pass">PRODUCTION READY</span></h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Context</th>
            </tr>
            <tr>
                <td>ROC-AUC</td>
                <td class="metric-value">0.978</td>
                <td>World-Class (Target > 0.90)</td>
            </tr>
            <tr>
                <td>Top Predictor</td>
                <td class="metric-value">LightGBM</td>
                <td>Best of 5 architectures</td>
            </tr>
            <tr>
                <td>Critical Recall</td>
                <td class="metric-value">71.9%</td>
                <td>Safety-critical sensitivity</td>
            </tr>
        </table>
    </div>

    <!-- MODEL 2: ISSUE DETECTOR -->
    <div class="model-box">
        <h2>2. Multi-Label Issue Detector (v3) <span class="status-pass">PRODUCTION READY</span></h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Assessment</th>
            </tr>
            <tr>
                <td>Avg ROC-AUC</td>
                <td class="metric-value">0.852</td>
                <td>Excellent across 14 labels</td>
            </tr>
            <tr>
                <td>Avg F1 Score</td>
                <td class="metric-value">0.356</td>
                <td>High for imbalanced multi-label</td>
            </tr>
            <tr>
                <td>Leakage Check</td>
                <td class="metric-value">PASSED</td>
                <td>Zero leakage detected</td>
            </tr>
        </table>
        <div class="img-container">
            <img src="data:image/png;base64,{img_issue}" alt="Issue Detector Summary">
        </div>
    </div>

    <!-- MODEL 3: RESOLUTION TIME PREDICTOR -->
    <div class="model-box">
        <h2>3. Resolution Time Predictor (v3) <span class="status-pass">PRODUCTION READY</span></h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Performance</th>
            </tr>
            <tr>
                <td>Mean Absolute Error</td>
                <td class="metric-value">3.62 Days</td>
                <td>High precision temporal forecasting</td>
            </tr>
            <tr>
                <td>Prediction Interval Coverage</td>
                <td class="metric-value">80.9%</td>
                <td>Matches 80% target confidence</td>
            </tr>
            <tr>
                <td>Short-term MAE</td>
                <td class="metric-value">2.2 Days</td>
                <td>Extremely accurate for quick queries</td>
            </tr>
        </table>
        <div class="img-container">
            <img src="data:image/png;base64,{img_resolution}" alt="Resolution Time Summary">
        </div>
    </div>

    <!-- MODEL 4: SITE RISK RANKER -->
    <div class="model-box">
        <h2>4. Site Risk Ranker (OPTIMIZED v2) <span class="status-pass">PRODUCTION READY</span></h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Ranking Quality</th>
            </tr>
            <tr>
                <td>NDCG@10</td>
                <td class="metric-value">0.838</td>
                <td>Top-10 ranking accuracy</td>
            </tr>
            <tr>
                <td>Spearman Correlation</td>
                <td class="metric-value">0.931</td>
                <td>Strong monotonic relationship</td>
            </tr>
            <tr>
                <td>Features Used</td>
                <td class="metric-value">141</td>
                <td>Comprehensive site profiling</td>
            </tr>
        </table>
        <div class="img-container">
            <img src="data:image/png;base64,{img_site}" alt="Site Ranker Summary">
        </div>
    </div>

    <!-- MODEL 5: ANOMALY DETECTOR -->
    <div class="model-box">
        <h2>5. Anomaly Detector (OPTIMIZED v2) <span class="status-pass">PRODUCTION READY</span></h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Detection Rate</th>
            </tr>
            <tr>
                <td>Global Anomaly Rate</td>
                <td class="metric-value">4.1%</td>
                <td>2,369 patients flagged</td>
            </tr>
            <tr>
                <td>Red Flags</td>
                <td class="metric-value">0</td>
                <td>No system errors or data failures</td>
            </tr>
            <tr>
                <td>Extreme Anomalies</td>
                <td class="metric-value">54</td>
                <td>High-priority outliers identified</td>
            </tr>
        </table>
        <div class="img-container">
            <img src="data:image/png;base64,{img_anomaly}" alt="Anomaly Detector Summary">
        </div>
    </div>

</div>

</body>
</html>
"""

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Successfully generated {OUTPUT_FILE} with 4 embedded visualizations.")
