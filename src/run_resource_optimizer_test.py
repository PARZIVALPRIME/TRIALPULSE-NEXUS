"""
TRIALPULSE NEXUS 10X - Resource Optimizer Test Runner

Phase 9.3: Resource Optimizer
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.resource_optimizer import test_resource_optimizer

if __name__ == "__main__":
    passed, failed = test_resource_optimizer()
    sys.exit(0 if failed == 0 else 1)