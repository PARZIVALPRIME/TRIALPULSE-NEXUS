"""
Runner script for Anomaly Detection Engine.

Usage:
    python src/run_anomaly_detection.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import DATA_PROCESSED
from src.ml.anomaly_detector import run_anomaly_detection, AnomalyConfig

if __name__ == '__main__':
    # Input path
    upr_path = DATA_PROCESSED / 'upr' / 'unified_patient_record.parquet'
    
    # Output directory
    output_dir = DATA_PROCESSED / 'analytics'
    
    # Optional: Custom configuration
    config = AnomalyConfig(
        contamination=0.05,  # 5% expected anomalies
        if_n_estimators=200,
        weight_isolation_forest=0.4,
        weight_lof=0.3,
        weight_reconstruction=0.3,
    )
    
    # Run anomaly detection
    summary = run_anomaly_detection(
        upr_path=upr_path,
        output_dir=output_dir,
        config=config
    )
    
    print("\n✅ Anomaly detection complete!")
    print(f"   Anomalies found: {summary['anomaly_detection']['total_anomalies']:,}")
    print(f"   Critical sites: {summary['site_level']['critical_sites']:,}")