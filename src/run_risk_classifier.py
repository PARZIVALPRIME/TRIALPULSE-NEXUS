"""
TRIALPULSE NEXUS 10X - Phase 3.2 Runner
Risk Classifier Training
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from ml.risk_classifier import main

if __name__ == '__main__':
    main()