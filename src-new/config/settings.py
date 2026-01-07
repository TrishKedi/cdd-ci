"""Configuration settings for the code duplication detector.

This module defines all configuration constants including file paths,
directories, and default values used throughout the application.
"""

import os
from pathlib import Path

# Base directories
ROOT = os.path.dirname(os.path.abspath(__file__))
data_dir = Path(__file__).parent.parent.parent / "data"

# Data subdirectories
cache_dir = data_dir / "cache"
export_dir = data_dir / "exports"

# String paths for compatibility with legacy code
tmp_dir = str(data_dir / "tmp")
index_dir = str(data_dir / "faiss")
cache_file = str(cache_dir / ".code_block_cache.json")

# Export file paths
json_export_file = str(export_dir / "matches.json")
jsonl_export_file = str(export_dir / "matches.jsonl")

# Legacy file constants (consider deprecating)
INDEX_FILE = 'code-ui.faiss'
EMBEDDIGS_FILE = 'codeBlocks-w-embeddings.json'  # Note: Typo in original

# Default configuration values
default_similarity_threshold: float = 0.85





