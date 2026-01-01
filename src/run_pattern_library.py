"""
TRIALPULSE NEXUS 10X - Phase 3.6 Runner
Pattern Library Execution Script
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from ml.pattern_library import main

if __name__ == "__main__":
    library, summary = main()