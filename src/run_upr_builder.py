"""
TRIALPULSE NEXUS 10X - UPR Builder Runner
==========================================
Dedicated entry point for UPR building.
"""

import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.upr_builder import UPRBuilder


def run_upr_builder():
    """Run the UPR building pipeline."""
    builder = UPRBuilder()
    upr = builder.build()
    
    if builder.manifest.status == "completed" and not builder.manifest.errors:
        print("\n" + "=" * 70)
        print("✅ UPR BUILD COMPLETED SUCCESSFULLY!")
        print(f"   Patients: {len(upr):,}")
        print(f"   Columns: {len(upr.columns)}")
        print("=" * 70)
        return 0
    else:
        print(f"\n⚠️ UPR BUILD COMPLETED WITH ISSUES")
        return 1


if __name__ == "__main__":
    sys.exit(run_upr_builder())