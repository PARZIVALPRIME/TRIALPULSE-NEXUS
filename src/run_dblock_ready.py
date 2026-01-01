"""
TRIALPULSE NEXUS 10X - Phase 2.3 Runner
DB Lock Ready Engine
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from analytics.dblock_ready import main

if __name__ == '__main__':
    main()