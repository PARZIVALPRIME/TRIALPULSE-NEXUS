"""
TRIALPULSE NEXUS 10X - Alert System Test Runner
Phase 8.6
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import directly from the module file, not through __init__.py
import importlib.util

# Load alert_system.py directly
alert_system_path = project_root / "src" / "collaboration" / "alert_system.py"
spec = importlib.util.spec_from_file_location("alert_system", alert_system_path)
alert_system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alert_system)

if __name__ == "__main__":
    success = alert_system.test_alert_system()
    sys.exit(0 if success else 1)