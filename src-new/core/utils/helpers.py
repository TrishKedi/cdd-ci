"""Utility functions for the code duplication detection system.

This module provides essential helper functions for file path manipulation,
hashing, indexing, and data processing operations. These utilities support
the core functionality of the similarity search and code analysis pipeline.

Key Functionality:
    - Path manipulation and normalization
    - File system operations with proper error handling
    - Cryptographic hashing for content identification
    - Index file management and naming conventions
    - Shard key generation for data partitioning
    - JSON data loading with validation

The functions in this module are designed to be:
    - Pure functions where possible (no side effects)
    - Cross-platform compatible using pathlib
    - Type-safe with comprehensive annotations
    - Well-documented with usage examples
    - Defensive against edge cases and invalid inputs


"""

import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Union

from config.settings import index_dir, INDEX_FILE, EMBEDDIGS_FILE

# Set up logging for utility operations
logger = logging.getLogger(__name__)


def get_index_path() -> str:
    """Get the path to the main FAISS index file, creating directory if needed.
    
    Ensures the index directory exists and returns the full path to the
    primary FAISS index file. This function handles directory creation
    with proper error handling and logging.
    
    Returns:
        str: Full path to the main FAISS index file
    
    Raises:
        OSError: If directory creation fails due to permissions or disk space
        ValueError: If index_dir or INDEX_FILE configuration is invalid
    """
    try:
        # Create index directory if it doesn't exist (with parents)
        # Using pathlib for better cross-platform compatibility
        index_path_obj = Path(index_dir)
        index_path_obj.mkdir(parents=True, exist_ok=True)
        
        # Build full path to the index file
        full_path = index_path_obj / INDEX_FILE
        
        logger.debug(f"Index path resolved to: {full_path}")
        return str(full_path)
        
    except (OSError, ValueError) as e:
        logger.error(f"Failed to create or access index path: {e}")
        raise OSError(f"Cannot access index directory '{index_dir}': {e}") from e


def sha256_hex(s: str) -> str:
    """Generate SHA-256 hash of a string in hexadecimal format.
    
    Creates a cryptographically secure hash of the input string using
    the SHA-256 algorithm. This is commonly used for content deduplication,
    integrity verification, and generating unique identifiers for code chunks.
    
    Args:
        s (str): Input string to hash. Can be code content, file paths,
            or any text that needs a deterministic identifier.
    
    Returns:
        str: 64-character hexadecimal string representing the SHA-256 hash
    """
    # Encode string to bytes using UTF-8 for consistent hashing across platforms
    # UTF-8 handles all Unicode characters properly
    encoded_bytes = s.encode("utf-8")
    
    # Generate SHA-256 hash and convert to hexadecimal
    # SHA-256 provides 256 bits of security, excellent for collision resistance
    hash_object = hashlib.sha256(encoded_bytes)
    
    return hash_object.hexdigest()

def path_prefix_of(path: str) -> str:
    """Extract the directory prefix from a file path with trailing slash.
    
    Extracts the directory portion of a file path and ensures it ends with
    a forward slash for consistent path concatenation. Returns empty string
    if the path has no directory component (i.e., it's just a filename).
    
    Args:
        path (str): File path to extract directory prefix from.
            Can be absolute, relative, or just a filename.
    
    Returns:
        str: Directory prefix with trailing slash, or empty string if no directory.
            Examples:
            - "src/api/auth.js" -> "src/api/"
            - "utils.py" -> ""
            - "/home/user/project/main.py" -> "/home/user/project/"
    """
    # Use PurePosixPath for consistent forward-slash behavior across platforms
    # This normalizes Windows backslashes to forward slashes
    path_obj = PurePosixPath(path)
    
    # Get parent directory; "." means current directory (no parent)
    parent_str = str(path_obj.parent)
    
    # Return directory with trailing slash, or empty string if no directory
    # The trailing slash ensures safe path concatenation operations
    if parent_str != ".":
        return parent_str + "/"
    else:
        return ""

def shard_key_for(repo_name: str, language: str) -> str:
    """Generate a shard key for data partitioning based on repository and language.
    
    Creates a composite key used for horizontal data partitioning (sharding)
    in the database and FAISS indexes. This enables efficient data distribution
    and parallel processing across multiple shards or nodes.
    
    Args:
        repo_name (str): Name of the repository (typically the directory name)
        language (str): Programming language code (e.g., 'js', 'py', 'java')
    
    Returns:
        str: Formatted shard key in the format "repo:{name}|lang:{language}"
    """
    # Use consistent format with clear component separation
    # The pipe character (|) is safe for most contexts and easy to parse
    return f"repo:{repo_name}|lang:{language}"

def filename_from_path(path: str) -> str:
    """Extract the filename (including extension) from a file path.
    
    Extracts just the filename portion from a complete file path,
    handling both absolute and relative paths consistently across platforms.
    
    Args:
        path (str): File path to extract filename from. Can be absolute,
            relative, or just a filename. Handles both Unix and Windows formats.
    
    Returns:
        str: Filename with extension, or empty string if path ends with slash
    
    Raises:
        ValueError: If path is empty or contains only whitespace
    """
    # Validate input to prevent subtle bugs
    if not path or not path.strip():
        raise ValueError("Path cannot be empty or contain only whitespace")
    
    # Use PurePosixPath for consistent cross-platform path handling
    # This normalizes Windows paths to use forward slashes
    path_obj = PurePosixPath(path.strip())
    
    # Extract the filename (last component of the path)
    # Returns empty string if path ends with separator (directory path)
    return path_obj.name

def index_file_name(path: str) -> str:
    """Generate FAISS index filename from repository path.
    
    Creates a standardized filename for FAISS index files based on the
    repository name. This ensures consistent naming conventions across
    the system and prevents filename conflicts.
    
    Args:
        path (str): Repository path (file system path or URL)
    
    Returns:
        str: FAISS index filename in format "{repo_name}.faiss"
    
    Raises:
        ValueError: If path is empty or results in empty filename
    """
    # Extract repository name from path
    repo_name = filename_from_path(path)
    
    # Validate that we got a meaningful filename
    if not repo_name:
        raise ValueError(f"Cannot generate index filename from path: '{path}'")
    
    # Generate standardized FAISS index filename
    # The .faiss extension is the standard for FAISS index files
    return f"{repo_name}.faiss"

def get_index_path_from_repo(repo_path: str) -> str:
    """Get the full path to a repository's FAISS index file.
    
    Combines the configured index directory with the repository-specific
    index filename to create a complete file path. Creates the index
    directory if it doesn't exist.
    
    Args:
        repo_path (str): Repository path used to derive the index filename
    
    Returns:
        str: Complete file path to the repository's FAISS index
    
    Raises:
        OSError: If directory creation fails due to permissions or disk space
        ValueError: If repo_path is invalid or results in invalid filename
    """
    try:
        # Generate standardized index filename from repository path
        index_filename = index_file_name(repo_path)
        
        # Ensure index directory exists (create with parents if needed)
        index_dir_path = Path(index_dir)
        index_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Construct full path to the index file
        full_index_path = index_dir_path / index_filename
        
        logger.debug(f"Index path for '{repo_path}': {full_index_path}")
        return str(full_index_path)
        
    except (OSError, ValueError) as e:
        logger.error(f"Failed to create index path for '{repo_path}': {e}")
        raise OSError(
            f"Cannot create index path for repository '{repo_path}': {e}"
        ) from e


# Module Constants
# ===============

# File extensions for different file types
FAISS_EXTENSION: str = ".faiss"
JSON_EXTENSION: str = ".json"
CODE_BLOCKS_FILENAME: str = "codeBlocks.json"

# Hash algorithm configuration
DEFAULT_HASH_ALGORITHM: str = "sha256"
HASH_ENCODING: str = "utf-8"

# Path separator for shard keys
SHARD_KEY_SEPARATOR: str = "|"
SHARD_KEY_REPO_PREFIX: str = "repo:"
SHARD_KEY_LANG_PREFIX: str = "lang:"


# Additional Utility Functions
# ===========================

def normalize_repo_name(repo_name: str) -> str:
    """Normalize repository name for consistent file naming.
    
    Sanitizes repository names to ensure they're safe for use as filenames
    across different file systems. Removes or replaces problematic characters.
    
    Args:
        repo_name (str): Raw repository name that may contain special characters
    
    Returns:
        str: Normalized repository name safe for file system usage
    """
    import re
    
    # Replace problematic characters with hyphens
    # Keep alphanumeric, hyphens, underscores, and dots
    normalized = re.sub(r'[^a-zA-Z0-9._-]', '-', repo_name)
    
    # Remove consecutive hyphens and trim
    normalized = re.sub(r'-+', '-', normalized).strip('-')
    
    # Ensure we have a meaningful name
    if not normalized:
        normalized = "unnamed-repo"
    
    return normalized


def validate_file_path(file_path: str) -> bool:
    """Validate that a file path is safe and accessible.
    
    Performs basic validation to ensure a file path is safe for use
    and doesn't contain path traversal attacks or invalid characters.
    
    Args:
        file_path (str): File path to validate
    
    Returns:
        bool: True if path appears safe and valid, False otherwise
    """
    try:
        # Convert to Path object for validation
        path_obj = Path(file_path)
        
        # Check for path traversal attempts
        if '..' in path_obj.parts:
            return False
        
        # Check for null bytes or other problematic characters
        if '\x00' in file_path:
            return False
        
        return True
        
    except (ValueError, OSError):
        return False


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes with error handling.
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        float: File size in megabytes, or 0.0 if file doesn't exist
    """
    try:
        file_size_bytes = Path(file_path).stat().st_size
        return file_size_bytes / (1024 * 1024)  # Convert to MB
    except OSError:
        return 0.0


# Public API
# ==========

__all__ = [
    # Path manipulation functions
    'get_index_path',
    'get_index_path_from_repo',
    'path_prefix_of',
    'filename_from_path',
    'index_file_name',
    
    # Hashing and identification
    'sha256_hex',
    'shard_key_for',
    
    # Utility functions
    'normalize_repo_name',
    'validate_file_path',
    'get_file_size_mb',
    
    # Constants
    'FAISS_EXTENSION',
    'JSON_EXTENSION',
    'CODE_BLOCKS_FILENAME',
    'DEFAULT_HASH_ALGORITHM',
    'HASH_ENCODING',
    'SHARD_KEY_SEPARATOR',
    'SHARD_KEY_REPO_PREFIX',
    'SHARD_KEY_LANG_PREFIX'
]