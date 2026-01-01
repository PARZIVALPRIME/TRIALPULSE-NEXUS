#!/usr/bin/env python3
"""
Runner for Enhanced SUPERVISOR Agent v2.1
Phase 5.3: Task decomposition, intelligent routing, conflict resolution
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Run the test
from src.agents.supervisor_enhanced import test_supervisor

if __name__ == "__main__":
    test_supervisor()