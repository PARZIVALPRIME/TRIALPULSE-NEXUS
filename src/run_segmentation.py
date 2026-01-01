"""
TRIALPULSE NEXUS 10X - Phase 1.4 Runner
Population Segmentation
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from data.segmentation import main

if __name__ == '__main__':
    main()