"""Run Anomaly Detection Lite v2.0"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import DATA_PROCESSED
from src.ml.anomaly_detector_v2 import run_lite

if __name__ == '__main__':
    run_lite(
        DATA_PROCESSED / 'upr' / 'unified_patient_record.parquet',
        DATA_PROCESSED / 'analytics'
    )
