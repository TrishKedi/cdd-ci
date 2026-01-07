"""SQL query definitions for code duplication detection.

Pre-compiled SQLAlchemy queries for database operations.
Designed for PostgreSQL with upsert and conflict resolution.
"""

from typing import Final, Dict
from sqlalchemy import select, func, text, delete, bindparam
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import Select, Insert, Delete

# Import specific table definitions to avoid namespace pollution
from .tables import (
    repo, chunk, embedding, index_registry, pca_transform
)


# Repository Management

upsert_repo: Final[Insert] = insert(repo).values(
    name=func.trim(text(":name")),
    default_branch=text(":default_branch"),
    is_private=text(":is_private")
).on_conflict_do_update(
    index_elements=[repo.c.name],
    set_={"updated_at": func.now()}
).returning(repo.c.repo_id)

upsert_pca: Final[Insert] = insert(pca_transform).values(
    pca_version=text(":pca_version"),
    input_dim=text(":input_dim"),
    output_dim=text(":output_dim"),
    mean_blob=text(":mean_blob"),
    projection_blob=text(":projection_blob"),
    notes=text(":notes")
).on_conflict_do_nothing()

# Vector Index Management

upsert_index: Final[Insert] = insert(index_registry).values(
    repo_id=text(":repo_id"),
    shard_key=text(":shard_key"),
    view=text(":view"),
    algo=text(":algo"),
    file_name=text(":file_name"),
    path_uri=text(":path_uri"),
    is_active=text(":is_active")
).returning(index_registry.c.index_id)

# Code Chunk Management

upsert_chunk: Final[Insert] = insert(chunk).values(
    repo_id=text(":repo_id"),
    language=text(":language"),
    path=text(":path"),
    path_prefix=text(":path_prefix"),
    symbol_id=text(":symbol_id"),
    start_line=text(":start_line"),
    end_line=text(":end_line"),
    window_start_token=text(":wstart"),
    window_end_token=text(":wend"),
    text_raw=text(":text_raw"),
    text_canon=text(":text_canon"),
    view_mask=text(":view_mask"),
    size_tokens_raw=text(":sz_raw"),
    size_tokens_canon=text(":sz_canon"),
    canonical_hash=text(":canon_hash"),
    is_representative=text(":is_rep"),
    shard_key=text(":shard_key"),
).returning(chunk.c.chunk_id)

# Batch chunk upsert
upsert_chunks: Final[Insert] = (
    insert(chunk).values(
        repo_id=text(":repo_id"),
        language=text(":language"),
        path=text(":path"),
        path_prefix=text(":path_prefix"),
        symbol_id=text(":symbol_id"),
        start_line=text(":start_line"),
        end_line=text(":end_line"),
        window_start_token=text(":wstart"),
        window_end_token=text(":wend"),
        text_raw=text(":text_raw"),
        text_canon=text(":text_canon"),
        view_mask=text(":view_mask"),
        size_tokens_raw=text(":sz_raw"),
        size_tokens_canon=text(":sz_canon"),
    canonical_hash=text(":canon_hash")
)
).returning(
    chunk.c.chunk_id,
    chunk.c.faiss_vector_id,
    chunk.c.repo_id,
    chunk.c.path,
    chunk.c.start_line,
    chunk.c.end_line,
    chunk.c.window_start_token,
    chunk.c.window_end_token,
)
# Vector Embedding Management

upsert_embedding: Final[Insert] = insert(embedding).values(
    chunk_id=text(":chunk_id"),
    view=text(":view"),
    pca_version=text(":pca_version"),
    dim=text(":dim"),
    l2_normalized=text(":l2_normalized"),
    storage_tier=text(":storage_tier"),
    faiss_vector_id=text(":faiss_vector_id"),
    present_in_faiss=text(":present_in_faiss")
)
upsert_embeddings: Final[Insert] = upsert_embedding

# Query and Retrieval Operations

active_index: Final[Select] = select(
    index_registry.c.index_id,
    index_registry.c.path_uri,
    index_registry.c.file_name
).where(
    index_registry.c.shard_key == text(":shard_key"),
    index_registry.c.view == text(":view"),
    index_registry.c.is_active.is_(True)
)
raw_embeddings: Final[Select] = (
    select(
        embedding.c.chunk_id,
        embedding.c.faiss_vector_id,
        chunk.c.path,
        chunk.c.text_raw,
        chunk.c.start_line,
        chunk.c.end_line,
        embedding.c.index_id
    )
    .join(chunk, chunk.c.chunk_id == embedding.c.chunk_id)
    .where(embedding.c.index_id == text(":index_id"))
)


chunk_by_vector_id: Final[Select] = select(
    chunk.c.path,
    chunk.c.text_raw,
    chunk.c.start_line,
    chunk.c.end_line
).where(
    chunk.c.faiss_vector_id == text(':faiss_vector_id')
)

# Data Management

repo_deletion: Final[Delete]
repo_deletion: Final[Delete] = delete(repo).where(
    repo.c.name.in_(bindparam('repos', expanding=True))
)



# Query Collections

# All upsert queries
UPSERT_QUERIES: Final[Dict[str, Insert]] = {
    'repo': upsert_repo,
    'pca': upsert_pca, 
    'index': upsert_index,
    'chunk': upsert_chunk,
    'chunks': upsert_chunks,
    'embedding': upsert_embedding,
    'embeddings': upsert_embeddings,
}

# All select queries
SELECT_QUERIES: Final[Dict[str, Select]] = {
    'active_index': active_index,
    'raw_embeddings': raw_embeddings,
    'chunk_by_vector_id': chunk_by_vector_id
}

# Maintenance queries
MAINTENANCE_QUERIES: Final[Dict[str, Delete]] = {
    'repo_deletion': repo_deletion,
}


