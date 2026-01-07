"""Database ingestion functions for code duplication detection.

Provides functions for ingesting code repositories into the database
and FAISS vector indexes with batch operations and caching.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

from cachetools import TTLCache

from .db import AsyncSessionLocal
from .repos import (
    ensure_repo, ensure_pca, upsert_chunk_row, upsert_batch_chunks,
    upsert_embedding_row, upsert_batch_embeddings, upsert_index_row,
    get_active_index, retrieve_embeddings, delete_repo, get_chunk_by_vector_id
)
from core.utils.helpers import filename_from_path, shard_key_for, path_prefix_of, sha256_hex
from config.settings import index_dir

# Set up logging for ingestion operations
logger = logging.getLogger(__name__)

# Cache Configuration

CACHE_TTL: int = 3600  # 1 hour
CACHE_MAXSIZE: int = 100  # Maximum cache entries

# Embedding cache with LRU eviction and TTL expiration
_embedding_cache: TTLCache[str, List[Tuple]] = TTLCache(
    maxsize=CACHE_MAXSIZE,
    ttl=CACHE_TTL
)

async def ingest_chunk() -> int:
    """Ingest a sample code chunk for testing.
    
    Returns:
        int: The database chunk ID
    """
    async with AsyncSessionLocal.begin() as s:
        repo_id = await ensure_repo(s, "webapp", "main", True)
        
        payload = {
            "repo_id": repo_id, 
            "language": "js", 
            "path": "src/api/auth.js",
            "path_prefix": "src/api/", 
            "symbol_id": None,
            "start_line": 180, 
            "end_line": 230, 
            "wstart": 256, 
            "wend": 512,
            "text_raw": "...",
            "text_canon": "...",
            "view_mask": 3,
            "sz_raw": 480, 
            "sz_canon": 420,
            "canon_hash": "sha256hex",
            "is_rep": True,
            "shard_key": shard_key_for("webapp", "js"),
        }
        
        chunk_id = await upsert_chunk_row(s, payload)
        logger.debug(f"Sample chunk ingested with ID: {chunk_id}")
        return chunk_id

async def ingest_embedding(chunk_id: int, faiss_vector_id: int) -> None:
    """Ingest a vector embedding linking a chunk to its FAISS position.
    
    Args:
        chunk_id (int): Database chunk identifier
        faiss_vector_id (int): FAISS vector position
    """
    async with AsyncSessionLocal.begin() as s:
        await ensure_pca(s, {
            "pca_version": "pca_v1",
            "input_dim": 1536,
            "output_dim": 512,
            "mean_blob": None,
            "projection_blob": None,
            "notes": "placeholder: PCA 1536→512; see artifact store"
        })

        payload = {
            "chunk_id": chunk_id,
            "view": "RAW",
            "pca_version": "pca_v1",
            "dim": 512,
            "l2_normalized": True,
            "storage_tier": "HOT",
            "faiss_vector_id": faiss_vector_id,
            "present_in_faiss": True
        }
        
        await upsert_embedding_row(s, payload)
        logger.debug(f"Embedding created for chunk {chunk_id} at FAISS position {faiss_vector_id}")

def fragments_to_payloads(
    fragments: List[Dict[str, Any]],
    repo_id: int,
    repo_name: str,
    language: str,
) -> List[Dict[str, Any]]:
    """Convert code fragments to database chunk payloads.
    
    Args:
        fragments: Code fragment dictionaries with id, path, code, processedCode
        repo_id: Database repository identifier
        repo_name: Repository name for shard key
        language: Programming language code
    
    Returns:
        List of chunk payloads ready for database insertion
    """
    shard_key = shard_key_for(repo_name, language)
    payloads = []
    
    for frag in fragments:
        faiss_vector_id = frag.get('id')
        path = frag["path"]
        raw = frag["code"] or ""
        canon = frag["processedCode"] or ""
        start = frag.get('start')
        end = frag.get('end')
        
        payloads.append({
            "repo_id": repo_id,
            "language": language,
            "path": path,
            "path_prefix": path_prefix_of(path),
            "symbol_id": None,
            "start_line": start,
            "end_line": end,
            "wstart": 0,
            "wend": 0,
            "text_raw": raw,
            "text_canon": canon,
            "view_mask": 3,
            "sz_raw": len(raw.split()),
            "sz_canon": len(canon.split()),
            "canon_hash": sha256_hex(canon),
            "is_rep": True,
            "shard_key": shard_key,
            "faiss_vector_id": faiss_vector_id
        })
    return payloads

async def ingest_chunks(code_blocks: List[Dict[str, Any]], repo_path: str) -> List[Tuple]:
    """Ingest multiple code chunks in batch operation.
    
    Args:
        code_blocks: List of code fragment dictionaries
        repo_path: Repository path
    
    Returns:
        List of chunk metadata tuples
    """
    async with AsyncSessionLocal.begin() as s:
        repo_name = filename_from_path(repo_path)
        repo_id = await ensure_repo(s, repo_name, "main", True)
        payloads = fragments_to_payloads(code_blocks, repo_id, repo_name, 'js')
        
        logger.info(f"Ingesting {len(payloads)} chunks for repository '{repo_name}'")
        chunks = await upsert_batch_chunks(s, payloads)
        logger.info(f"Successfully ingested {len(chunks)} chunks for repository '{repo_name}'")
        return chunks
        
async def ingest_embeddings(chunks: List[Tuple], index_registry_id: int) -> None:
    """Ingest vector embeddings for multiple code chunks in batch operation.
    
    Creates embedding records that link database chunks to their vector
    representations in a FAISS index. This establishes the critical mapping
    needed for similarity search operations.
    
    Args:
        chunks (List[Tuple]): List of chunk tuples from ingest_chunks, where each
            tuple contains (chunk_id, faiss_vector_id, ...) at minimum
        index_registry_id (int): ID of the FAISS index registry entry that will
            contain these embeddings
    
    Raises:
        SQLAlchemyError: If database batch operation fails
        IndexError: If chunk tuples don't have expected structure
        IntegrityError: If foreign key constraints fail
    """
    async with AsyncSessionLocal.begin() as s:
        await ensure_pca(s, {
            "pca_version": "pca_v1",
            "input_dim": 1536,
            "output_dim": 512,
            "mean_blob": None,
            "projection_blob": None,
            "notes": "placeholder: PCA 1536→512; see artifact store"
        })

        payloads = [
            {
                "chunk_id": chunk[0],
                "index_id": index_registry_id,
                "view": "RAW",
                "pca_version": "pca_v1",
                "dim": 512,
                "l2_normalized": True,
                "storage_tier": "HOT",
                "faiss_vector_id": chunk[1],
                "present_in_faiss": True
            }
            for chunk in chunks
        ]
        
        logger.info(f"Ingesting {len(payloads)} embeddings for index {index_registry_id}")
        await upsert_batch_embeddings(s, payloads)
        logger.info(f"Successfully ingested {len(payloads)} embeddings")

async def ingest_index(repo_path: str) -> Tuple[int, str]:
    """Register a new FAISS index in the database registry.
    
    Creates a registry entry for a FAISS vector index, marking it as active
    and ready to serve similarity search requests. This is typically called
    after creating the physical FAISS index file.
    
    Args:
        repo_path (str): File system path to the repository
    
    Returns:
        Tuple[int, str]: Tuple containing (index_registry_id, faiss_filename)
            - index_registry_id: Database ID for the index registry entry
            - faiss_filename: Generated filename for the FAISS index file
    
    Raises:
        SQLAlchemyError: If database operation fails
        ValueError: If repo_path is invalid
    """
    async with AsyncSessionLocal.begin() as s:
        repo_name = filename_from_path(repo_path)
        file_name = f"{repo_name}.faiss"
        path_uri = index_dir
        shard_key = shard_key_for(repo_name, 'js')
        repo_id = await ensure_repo(s, repo_name, "main", True)
   
        payload = {
            "repo_id": repo_id,
            "shard_key": shard_key,
            "view": "RAW",
            "algo": "FlatIP",
            "file_name": file_name,
            "path_uri": path_uri,
            "is_active": True
        }

        logger.info(f"Registering FAISS index for repository '{repo_name}'")
        index_id = await upsert_index_row(s, payload)
        logger.info(f"FAISS index registered with ID {index_id}, filename: {file_name}")
        return index_id, file_name


async def get_embeddings(index_registry_id: int) -> List[Tuple]:
    """Retrieve embeddings for a FAISS index with intelligent caching.
    
    Implements a sophisticated caching strategy to optimize database access patterns:
    - LRU (Least Recently Used) eviction policy
    - TTL (Time To Live) expiration for data freshness
    - Automatic size management to prevent memory bloat
    
    Cache Benefits:
        - Sub-millisecond response for cached data
        - Reduced database load for frequently accessed indexes
        - Automatic memory management with bounded growth
        - Data freshness guaranteed through TTL expiration
    
    Args:
        index_registry_id (int): Database ID of the FAISS index registry entry
        
    Returns:
        List[Tuple]: List of embedding tuples containing chunk metadata and
            vector positions required for similarity search operations
        
    Cache Behavior:
        - Cache Hit: Returns data immediately from memory
        - Cache Miss: Fetches from database, caches result, returns data
        - TTL Expiry: Entries older than CACHE_TTL seconds are automatically removed
        - Size Limit: LRU eviction when cache exceeds CACHE_MAXSIZE entries
    
    """
    cache_key = f"embeddings_{index_registry_id}"
    
    # Check cache first - TTLCache handles expiration automatically
    if cache_key in _embedding_cache:
        logger.debug(f"Cache hit for index {index_registry_id}")
        return _embedding_cache[cache_key]

    # Cache miss: fetch from database and store result
    logger.debug(f"Cache miss for index {index_registry_id} - querying database")
    async with AsyncSessionLocal.begin() as s:
        embeddings = await retrieve_embeddings(s, index_registry_id)
        
        # Cache the result - TTLCache handles size limits and expiration
        _embedding_cache[cache_key] = embeddings
        logger.info(f"Cached {len(embeddings)} embeddings for index {index_registry_id}")
        return embeddings

async def get_index(repo_path: str) -> Optional[Tuple]:
    """Retrieve the active FAISS index for a specific repository.
    
    Finds the currently active index registry entry for a repository,
    which is needed to determine which FAISS index file to load for
    similarity search operations.
    
    Args:
        repo_path (str): File system path to the repository
    
    Returns:
        Optional[Tuple]: Active index registry tuple containing:
            - index_registry_id: Database identifier
            - repo_id: Repository identifier
            - shard_key: Partitioning key
            - file_name: FAISS index filename
            - path_uri: Storage directory
            - Additional metadata fields
            Returns None if no active index exists for the repository.
    
    Raises:
        SQLAlchemyError: If database query fails
    """
    async with AsyncSessionLocal.begin() as s:
        repo_name = filename_from_path(repo_path)
        shard_key = shard_key_for(repo_name, 'js')
        
        index = await get_active_index(s, shard_key, 'RAW')
        return index


async def finalize_index_build(
    raw_code_blocks: List[Dict[str, Any]], 
    repo_path: str, 
    index_registry_id: int
) -> None:
    """Complete the index building process by ingesting chunks and embeddings.
    
    This function orchestrates the final steps of FAISS index creation:
    1. Ingests code chunks into the database
    2. Creates embedding records linking chunks to FAISS vectors
    3. Establishes the complete mapping for similarity search
    
    Args:
        raw_code_blocks (List[Dict[str, Any]]): List of processed code fragments
            containing code content, metadata, and FAISS vector positions
        repo_path (str): File system path to the repository
        index_registry_id (int): Database ID of the index registry entry
    
    Raises:
        SQLAlchemyError: If database operations fail
        ValueError: If raw_code_blocks is empty or malformed
    """
    chunks = await ingest_chunks(raw_code_blocks, repo_path)
    await ingest_embeddings(chunks, index_registry_id)


async def delete_single_repo(repo_urls: List[str]) -> List[str]:
    """Delete multiple repositories and return their associated FAISS filenames.
    
    Removes repositories from the database and returns the list of FAISS
    index filenames that should be cleaned up from the file system.
    
    Args:
        repo_urls (List[str]): List of repository paths or URLs to delete
    
    Returns:
        List[str]: List of FAISS index filenames that should be deleted
            from the file system. Empty list if no repositories provided.
    
    Raises:
        SQLAlchemyError: If database deletion operations fail
    """
    if not repo_urls:
        return []
    
    async with AsyncSessionLocal.begin() as s:
        repo_names = [filename_from_path(repo_url) for repo_url in repo_urls]
        file_names = [f"{filename_from_path(repo_url)}.faiss" for repo_url in repo_urls]
        
        await delete_repo(s, repo_names)
        return file_names

async def get_single_chunk(faiss_vector_id: int) -> Optional[Tuple]:
    """Retrieve a single code chunk by its FAISS vector position.
    
    Performs a reverse lookup from a FAISS vector ID to find the associated
    code chunk information, which is essential for displaying similarity
    search results with proper source code context.
    
    Args:
        faiss_vector_id (int): Position of the vector in the FAISS index
    
    Returns:
        Optional[Tuple]: Chunk information tuple containing:
            - chunk_id: Database identifier
            - repo_id: Repository identifier
            - path: File path within repository
            - start_line, end_line: Source location
            - text_raw, text_canon: Code content
            - Additional metadata fields
            Returns None if the vector ID is not found.
    
    Raises:
        SQLAlchemyError: If database query fails
    """
    async with AsyncSessionLocal.begin() as s:
        chunk = await get_chunk_by_vector_id(s, faiss_vector_id)
        return chunk

