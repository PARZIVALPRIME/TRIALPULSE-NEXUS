# src/run_scenario_simulator_test.py

"""
TRIALPULSE NEXUS 10X - Phase 9.2 Scenario Simulator Test Runner

Tests:
1. Initialize Scenario Simulator
2. Monte Carlo Engine
3. Simulate Close Site
4. Simulate Add Resource
5. Simulate Deadline Check
6. Simulate Process Change
7. Compare Scenarios
8. Uncertainty Band Calculations
9. Scenario Parameter Sampling
10. Convergence Check
11. Scenarios Run History
12. Convenience Functions

Run: python src/run_scenario_simulator_test.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.simulation.scenario_simulator import test_scenario_simulator

if __name__ == "__main__":
    passed, failed = test_scenario_simulator()
    sys.exit(0 if failed == 0 else 1)