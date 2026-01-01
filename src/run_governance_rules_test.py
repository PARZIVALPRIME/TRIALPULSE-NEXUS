# src/run_governance_rules_test.py

"""
TRIALPULSE NEXUS - Governance Rules Engine Test Runner
Phase 10.2
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.governance.rules_engine import test_governance_rules

if __name__ == "__main__":
    test_governance_rules()