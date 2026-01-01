# src/run_issue_detector.py
"""
Runner script for Phase 3.3: Issue Detector
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.issue_detector import run_issue_detection

if __name__ == "__main__":
    detector, df_results, report = run_issue_detection()