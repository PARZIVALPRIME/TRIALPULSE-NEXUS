"""
TRIALPULSE NEXUS 10X - Phase 9.1 Trial State Model Test Runner
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.trial_state_model import test_trial_state_model

if __name__ == "__main__":
    success = test_trial_state_model()
    sys.exit(0 if success else 1)