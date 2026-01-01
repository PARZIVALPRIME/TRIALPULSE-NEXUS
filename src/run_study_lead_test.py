"""
Test runner for Study Lead Command Dashboard
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard.pages.study_lead import test_study_lead_view

if __name__ == "__main__":
    success = test_study_lead_view()
    sys.exit(0 if success else 1)