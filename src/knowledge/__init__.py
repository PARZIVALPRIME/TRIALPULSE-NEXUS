# File: src/knowledge/__init__.py
"""
TRIALPULSE NEXUS 10X - Knowledge Module

Components:
- Embedding Pipeline (4.1)
- Vector Store - ChromaDB (4.2)
- RAG Knowledge Base (4.3)
- Causal Hypothesis Engine (4.4)
- Cross-Study Pattern Matcher (4.5)
"""

from pathlib import Path

# Module version
__version__ = "1.0.0"

# Core components
from .embedding_pipeline import EmbeddingPipeline
from .vector_store import VectorStore
from .rag_knowledge_base import RAGKnowledgeBase
from .causal_hypothesis_engine import CausalHypothesisEngine
from .cross_study_pattern_matcher import CrossStudyPatternMatcher

# Public API (explicit exports)
__all__ = [
    'EmbeddingPipeline',
    'VectorStore',
    'RAGKnowledgeBase',
    'CausalHypothesisEngine',
    'CrossStudyPatternMatcher'
]
