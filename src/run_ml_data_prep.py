"""
TRIALPULSE NEXUS 10X - Phase 3.1 Runner
ML Data Preparation
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from ml.data_preparation import main

if __name__ == '__main__':
    main()