# src/run_investigation_rooms_test.py
"""
TRIALPULSE NEXUS 10X - Investigation Rooms Test Runner
Phase 8.2
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from collaboration.investigation_rooms import test_investigation_rooms
    test_investigation_rooms()