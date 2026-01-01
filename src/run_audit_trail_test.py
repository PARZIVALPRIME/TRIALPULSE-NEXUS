# src/run_audit_trail_test.py

"""
TRIALPULSE NEXUS - Audit Trail System Test Runner
Phase 10.1
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.governance.audit_trail import test_audit_trail

if __name__ == "__main__":
    test_audit_trail()