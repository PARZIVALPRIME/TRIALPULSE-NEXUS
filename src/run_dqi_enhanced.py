"""
TRIALPULSE NEXUS 10X - Phase 2.1 Runner
Enhanced DQI Calculator
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from analytics.dqi_enhanced import main

if __name__ == '__main__':
    main()