"""
TRIALPULSE NEXUS 10X - Cascade Explorer Test Runner
Tests the cascade explorer dashboard functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard.pages.cascade_explorer import test_cascade_explorer

if __name__ == "__main__":
    success = test_cascade_explorer()
    sys.exit(0 if success else 1)