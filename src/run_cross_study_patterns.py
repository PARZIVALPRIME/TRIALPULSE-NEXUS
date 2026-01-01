"""
Runner script for Phase 4.5: Cross-Study Pattern Matcher
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.knowledge.cross_study_pattern_matcher import main

if __name__ == "__main__":
    main()