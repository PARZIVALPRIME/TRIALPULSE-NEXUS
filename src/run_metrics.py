"""
TRIALPULSE NEXUS 10X - Phase 1.5 Runner
Metrics Engine
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from data.metrics_engine import main

if __name__ == '__main__':
    main()