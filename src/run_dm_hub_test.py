# src/run_dm_hub_test.py
"""
Test runner for Data Manager Hub
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import and run test
from dashboard.pages.dm_hub import test_dm_hub

if __name__ == "__main__":
    test_dm_hub()