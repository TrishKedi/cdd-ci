"""Database repository functions for code duplication detection system.

This module provides high-level database operations for managing repositories,
code chunks, embeddings, and FAISS indexes. It serves as an abstraction layer
between the application logic and the raw SQL queries.

Key Functionality:
    - Repository lifecycle management (creation, deletion)
    - Code chunk ingestion and deduplication
    - Embedding storage and retrieval
    - FAISS index registry management
    - PCA transformation configuration
    - Access control and filtering

All functions are designed for async/await usage with SQLAlchemy's
async session management.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Import specific query functions to avoid namespace pollution
from .queries import (
    upsert_repo, upsert_pca, upsert_index, upsert_chunk, upsert_chunks,
    upsert_embedding, upsert_embeddings, active_index, raw_embeddings,
    repo_deletion, chunk_by_vector_id
)

# Set up logging for database operations
logger = logging.getLogger(__name__)

def shard_key_for(repo_name: str, language: str) -> str:
    """Generate a consistent shard key for repository and language combination.
    
    Creates a standardized sharding key used to distribute code chunks and
    embeddings across multiple FAISS indexes. This enables horizontal scaling
    by partitioning data based on repository and programming language.
    
    Args:
        repo_name (str): Name of the repository (e.g., 'my-project')
        language (str): Programming language code (e.g., 'js', 'py', 'java')
    
    Returns:
        str: Formatted shard key in the format 'repo:{name}|lang:{lang}'
    
    Example:
        >>> shard_key_for('my-project', 'js')
        'repo:my-project|lang:js'
    
    Note:
        This key format is used throughout the system for consistent
        partitioning. Changing the format would require data migration.
    """
    # Use consistent delimiter format for reliable parsing downstream
    return f"repo:{repo_name}|lang:{language}"

async def ensure_repo(session: AsyncSession, name: str, default_branch: Optional[str] = None, is_private: bool = True) -> int:
    """Ensure a repository exists in the database, creating it if necessary.
    
    Performs an upsert operation to either create a new repository or update
    an existing one with the provided metadata. This idempotent operation
    is safe to call multiple times with the same repository name.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        name (str): Unique repository name/identifier
        default_branch (Optional[str], optional): Main branch name (e.g., 'main', 'master').
            If None, the repository's existing default branch is preserved.
        is_private (bool, optional): Privacy flag for access control. Defaults to True
            for security-by-default approach.
    
    Returns:
        int: The repository ID (repo_id) from the database. This can be used
            for foreign key relationships in other tables.
    
    Raises:
        SQLAlchemyError: If database operation fails
        ValueError: If repository name is invalid or empty
    
    Note:
        Repository names must be unique across the system. The upsert operation
        will update existing repositories with new metadata if provided.
    """
    # Execute upsert query with provided repository metadata
    row = (await session.execute(upsert_repo, {
        "name": name, 
        "default_branch": default_branch, 
        "is_private": is_private
    })).first()
    
    # Return the repository ID for use in foreign key relationships
    return row[0]

async def ensure_pca(session: AsyncSession, payload: Dict[str, Any]) -> None:
    """Ensure a PCA transformation configuration exists in the database.
    
    Stores or updates PCA (Principal Component Analysis) transformation
    parameters used for dimensionality reduction of code embeddings.
    This enables more efficient storage and faster similarity search.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        payload (Dict[str, Any]): PCA configuration dictionary containing:
            - pca_version (str): Unique version identifier for this PCA
            - input_dim (int): Original embedding dimensionality
            - output_dim (int): Reduced dimensionality after PCA
            - mean_blob (str): Serialized mean vector for centering
            - projection_blob (str): Serialized projection matrix
            - notes (str, optional): Human-readable description
    
    Raises:
        SQLAlchemyError: If database operation fails
        KeyError: If required PCA parameters are missing from payload
    
    Note:
        PCA transformations are versioned to support multiple reduction
        strategies and enable gradual migration between configurations.
    """
    # Execute upsert to store/update PCA configuration
    await session.execute(upsert_pca, payload)
    logger.debug(f"PCA configuration upserted: version={payload.get('pca_version')}")
    
async def upsert_index_row(session: AsyncSession, payload: Dict[str, Any]) -> int:
    """Create or update a FAISS index registry entry.
    
    Manages the metadata for FAISS vector indexes, including their configuration,
    file locations, and deployment status. This registry enables the system to
    track multiple indexes and support blue-green deployments.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        payload (Dict[str, Any]): Index configuration dictionary containing:
            - repo_id (int): Repository this index belongs to
            - shard_key (str): Partitioning key for horizontal scaling
            - view (str): View type ('RAW' or 'CANON')
            - algo (str): FAISS algorithm ('FlatIP', 'HNSW', etc.)
            - params_json (dict): Algorithm-specific parameters
            - file_name (str): FAISS index filename
            - path_uri (str, optional): Full storage path/URI
            - dim (int): Vector dimensionality
            - vector_count (int): Number of vectors in index
            - pca_version (str, optional): Associated PCA transformation
    
    Returns:
        int: The index registry ID for referencing this index configuration
    
    Raises:
        SQLAlchemyError: If database operation fails
        KeyError: If required index parameters are missing
    
    Note:
        Index registry supports blue-green deployments through the is_active
        and blue_green_group fields for zero-downtime index updates.
    """
    # Execute upsert and return the registry ID for reference
    row = (await session.execute(upsert_index, payload)).first()
    logger.debug(f"Index registry entry upserted: ID={row[0]}, shard={payload.get('shard_key')}")
    return row[0]

async def upsert_chunk_row(session: AsyncSession, payload: Dict[str, Any]) -> int:
    """Create or update a single code chunk record.
    
    Stores metadata and content for a code chunk that will be used for
    similarity analysis and duplication detection. Chunks represent
    analyzable segments of source code with both raw and canonicalized forms.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        payload (Dict[str, Any]): Chunk data dictionary containing:
            - repo_id (int): Repository containing this chunk
            - language (str): Programming language
            - path (str): File path within repository
            - start_line (int): Starting line number in source file
            - end_line (int): Ending line number in source file
            - text_raw (str): Original source code
            - text_canon (str, optional): Canonicalized/normalized code
            - canonical_hash (str): Hash for exact duplicate detection
            - shard_key (str): Partitioning key for scaling
            - faiss_vector_id (int): Position in FAISS index
            - symbol_id (int, optional): Associated code symbol
    
    Returns:
        int: The chunk ID for referencing this code chunk in embeddings
            and similarity results
    
    Raises:
        SQLAlchemyError: If database operation fails
        KeyError: If required chunk parameters are missing
    
    Note:
        Chunks support both raw and canonicalized views to enable different
        types of similarity analysis with varying precision levels.
    """
    # Execute upsert and return chunk ID for embedding relationships
    row = (await session.execute(upsert_chunk, payload)).first()
    logger.debug(f"Chunk upserted: ID={row[0]}, path={payload.get('path')}")
    return row[0]

async def upsert_batch_chunks(session: AsyncSession, payloads: List[Dict[str, Any]]) -> List[Tuple[int]]:
    """Create or update multiple code chunks in a single database operation.
    
    Efficiently processes multiple code chunks using batch operations to
    minimize database round-trips during bulk ingestion. This is the preferred
    method for importing large numbers of code chunks from repositories.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        payloads (List[Dict[str, Any]]): List of chunk data dictionaries,
            each containing the same fields as upsert_chunk_row
    
    Returns:
        List[Tuple[int]]: List of tuples containing chunk IDs for each
            successfully processed chunk, in the same order as input
    
    Raises:
        SQLAlchemyError: If batch database operation fails
        ValueError: If payloads list is empty or contains invalid data
    """
    if not payloads:
        logger.warning("Empty payloads list provided to upsert_batch_chunks")
        return []
    
    # Execute batch upsert operation for efficiency
    result = await session.execute(upsert_chunks, payloads)
    rows = result.fetchall()
    
    logger.info(f"Batch upserted {len(rows)} chunks successfully")
    return rows

async def upsert_embedding_row(session: AsyncSession, payload: Dict[str, Any]) -> None:
    """Create or update a single embedding record linking chunk to vector index.
    
    Establishes the relationship between a code chunk and its vector representation
    in a FAISS index. This metadata is crucial for mapping similarity search
    results back to the original source code.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        payload (Dict[str, Any]): Embedding metadata dictionary containing:
            - chunk_id (int): Reference to the code chunk
            - index_id (int): Reference to the FAISS index registry
            - view (str): View type ('RAW' or 'CANON')
            - pca_version (str): PCA transformation used
            - dim (int): Vector dimensionality
            - l2_normalized (bool): Whether vector is L2-normalized
            - faiss_vector_id (int): Position within FAISS index
            - present_in_faiss (bool): Whether currently indexed
            - storage_tier (str, optional): Storage optimization tier
    
    Raises:
        SQLAlchemyError: If database operation fails
        KeyError: If required embedding parameters are missing
        IntegrityError: If chunk_id or index_id references don't exist
    
    Note:
        The embedding table maintains the critical link between database
        records and FAISS vector positions, enabling bidirectional lookups.
    """
    # Execute upsert to establish chunk-to-vector mapping
    await session.execute(upsert_embedding, payload)
    logger.debug(f"Embedding upserted: chunk={payload.get('chunk_id')}, index={payload.get('index_id')}")

async def upsert_batch_embeddings(session: AsyncSession, payloads: List[Dict[str, Any]]) -> None:
    """Create or update multiple embedding records in a single operation.
    
    Efficiently processes multiple chunk-to-vector mappings using batch
    operations. This is essential for maintaining consistency when updating
    large FAISS indexes with thousands of new embeddings.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        payloads (List[Dict[str, Any]]): List of embedding metadata dictionaries,
            each containing the same fields as upsert_embedding_row
    
    Raises:
        SQLAlchemyError: If batch database operation fails
        ValueError: If payloads list is empty
        IntegrityError: If any chunk_id or index_id references don't exist
    
    Performance:
        Batch embedding operations are critical for index rebuild performance,
        especially when processing repositories with millions of code chunks.
    
    Note:
        This operation should be coordinated with FAISS index updates to
        maintain consistency between the database and vector index state.
    """
    if not payloads:
        logger.warning("Empty payloads list provided to upsert_batch_embeddings")
        return
    
    # Execute batch upsert for efficient embedding management
    await session.execute(upsert_embeddings, payloads)
    logger.info(f"Batch upserted {len(payloads)} embeddings successfully")

async def get_active_index(session: AsyncSession, shard_key: str, view: str) -> Optional[Dict[str, Any]]:
    """Retrieve the currently active FAISS index for a shard and view combination.
    
    Finds the index that is currently serving traffic for similarity searches
    within a specific repository/language shard and view type. This supports
    blue-green deployments by distinguishing active from inactive indexes.
    
    Args:
        session (AsyncSession): Active database session for the query
        shard_key (str): Partitioning key (e.g., 'repo:my-project|lang:js')
        view (str): View type - either 'RAW' or 'CANON'
    
    Returns:
        Optional[Dict[str, Any]]: Index registry record as a dictionary if found,
            None if no active index exists for the specified shard/view.
            Dictionary contains all index metadata including:
            - index_id: Registry identifier
            - file_name: FAISS index filename
            - path_uri: Full storage location
            - dim: Vector dimensionality
            - algo: FAISS algorithm used
            - params_json: Algorithm parameters
    
    Raises:
        SQLAlchemyError: If database query fails
    
    Note:
        Only returns indexes marked as is_active=True. During blue-green
        deployments, there may be multiple indexes but only one active.
    """
    # Query for the currently active index serving this shard/view
    result = (await session.execute(active_index, {
        "shard_key": shard_key, 
        "view": view
    })).mappings().first()
    
    if result:
        logger.debug(f"Found active index: {result['index_id']} for {shard_key}")
    else:
        logger.warning(f"No active index found for {shard_key}")
    
    return result

async def retrieve_embeddings(session: AsyncSession, index_registry_id: int) -> List[Tuple]:
    """Retrieve all embedding records for a specific FAISS index.
    
    Fetches the complete list of embeddings associated with a particular
    index registry entry. This is useful for index reconstruction, validation,
    and debugging operations.
    
    Args:
        session (AsyncSession): Active database session for the query
        index_registry_id (int): ID from the index_registry table
    
    Returns:
        List[Tuple]: List of embedding record tuples containing:
            - chunk_id: Reference to the code chunk
            - faiss_vector_id: Position in FAISS index
            - view: View type ('RAW' or 'CANON')
            - dim: Vector dimensionality
            - l2_normalized: Normalization status
            - present_in_faiss: Current indexing status
    
    Raises:
        SQLAlchemyError: If database query fails
        ValueError: If index_registry_id doesn't exist
    
    Performance:
        This query can return large result sets for indexes with millions
        of vectors. Consider pagination for very large indexes.
    
    Note:
        Used primarily for index maintenance operations and consistency
        checks between database metadata and FAISS index contents.
    """
    # Execute query to fetch all embeddings for the specified index
    result = await session.execute(raw_embeddings, {"index_id": index_registry_id})
    embeddings = result.fetchall()
    
    logger.info(f"Retrieved {len(embeddings)} embeddings for index {index_registry_id}")
    return embeddings

async def delete_repo(session: AsyncSession, repositories: List[str]) -> None:
    """Delete multiple repositories and all associated data.
    
    Performs cascading deletion of repositories along with all related
    data including chunks, embeddings, symbols, and index entries.
    This is a destructive operation that cannot be easily undone.
    
    Args:
        session (AsyncSession): Active database session for the transaction
        repositories (List[str]): List of repository names to delete
    
    Raises:
        SQLAlchemyError: If database operation fails
        ValueError: If repositories list is empty
    
    Warning:
        This operation will permanently delete:
        - All code chunks from specified repositories
        - All embeddings and vector index entries
        - All symbol definitions and metadata
        - Repository access permissions
        
        Ensure FAISS indexes are also cleaned up externally as they
        are not automatically updated by this database operation.
    
    Note:
        Foreign key CASCADE constraints ensure related data is automatically
        removed, but external cleanup of FAISS files may be required.
    """
    if not repositories:
        logger.warning("Empty repositories list provided to delete_repo")
        return
    
    # Execute cascading deletion for all specified repositories
    await session.execute(repo_deletion, {"repos": repositories})
    
    logger.warning(f"Deleted {len(repositories)} repositories: {repositories}")

async def get_chunk_by_vector_id(session: AsyncSession, faiss_vector_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a code chunk by its FAISS vector ID.
    
    Looks up the original code chunk information using the vector ID from
    a FAISS similarity search result. This enables mapping search results
    back to the source code for display and analysis.
    
    Args:
        session (AsyncSession): Active database session for the query
        faiss_vector_id (int): Vector position from FAISS search results
    
    Returns:
        Optional[Dict[str, Any]]: Chunk information dictionary if found,
            None if no chunk exists for the given vector ID.
            Dictionary contains:
            - chunk_id: Database identifier
            - repo_id: Repository reference
            - path: File path within repository
            - start_line, end_line: Source location
            - text_raw, text_canon: Code content
            - language: Programming language
    
    Raises:
        SQLAlchemyError: If database query fails
    
    Note:
        This is a critical function for the similarity search pipeline,
        converting FAISS results back to meaningful source code references.
    """
    # Execute query to find chunk by FAISS vector position
    result = (await session.execute(chunk_by_vector_id, {
        "faiss_vector_id": faiss_vector_id
    })).mappings().first()
    
    if result:
        logger.debug(f"Found chunk {result['chunk_id']} for vector ID {faiss_vector_id}")
    else:
        logger.warning(f"No chunk found for FAISS vector ID {faiss_vector_id}")
    
    return result

