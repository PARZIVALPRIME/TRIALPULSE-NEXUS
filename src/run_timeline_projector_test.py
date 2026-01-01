"""
TRIALPULSE NEXUS 10X - Timeline Projector Test Runner

Phase 9.4: Timeline Projector
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.timeline_projector import test_timeline_projector

if __name__ == "__main__":
    passed, failed = test_timeline_projector()
    sys.exit(0 if failed == 0 else 1)