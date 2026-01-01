# src/run_calibration_test.py

"""
TRIALPULSE NEXUS - Confidence Calibration Test Runner
Phase 10.3
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.governance.confidence_calibration import test_confidence_calibration

if __name__ == "__main__":
    test_confidence_calibration()