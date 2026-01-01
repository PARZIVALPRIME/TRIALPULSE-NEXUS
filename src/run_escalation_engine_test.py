# src/run_escalation_engine_test.py
"""
TRIALPULSE NEXUS 10X - Escalation Engine Test Runner
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.collaboration.escalation_engine import test_escalation_engine

if __name__ == "__main__":
    success = test_escalation_engine()
    sys.exit(0 if success else 1)