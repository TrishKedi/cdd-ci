"""Core services for code duplication detection.

This package provides the main service classes for:
- Code analysis and processing
- Embedding generation and indexing
- Similarity lookup and matching
- AI-powered re-ranking
- Match processing and formatting
"""

from .embedding_service import EmbeddingIndex
from .similarity_lookup_service import SimilarityLookup
from .code_analysis_service import CodeBaseProcessor
from .reranking_service import LLMReranker
from .match_processor_service import MatchProcessor
from .openai_service import OpenAIService

__all__ = [
    "EmbeddingIndex", 
    "SimilarityLookup", 
    "CodeBaseProcessor",
    "LLMReranker",
    "MatchProcessor",
    "OpenAIService"
]