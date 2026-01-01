# src/run_vector_store.py
"""
TRIALPULSE NEXUS 10X - Vector Store Runner
Phase 4.2: ChromaDB Vector Store
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge.vector_store import main

if __name__ == "__main__":
    main()