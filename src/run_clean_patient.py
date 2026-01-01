"""
TRIALPULSE NEXUS 10X - Phase 2.2 Runner
Enhanced Clean Patient Calculator
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from analytics.clean_patient_enhanced import main

if __name__ == '__main__':
    main()