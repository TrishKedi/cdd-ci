"""Database table definitions for code duplication detection system.

Tables: repo, symbol, chunk, embedding, pca_transform, index_registry
Schema: code_search
"""

from typing import Final, List
from sqlalchemy import (
    MetaData, Table, Column, BigInteger, Integer, Text, Boolean, TIMESTAMP,
    Enum, ForeignKey, JSON, SmallInteger
)

# Database metadata configuration for code search schema
metadata: Final[MetaData] = MetaData(schema="code_search")

# Supported programming languages
Lang: Final[Enum] = Enum(
    'js', 'ts', 'py', 'java', 'go', 'rb', 'cs', 'cpp', 'rs', 'php', 'other',
    name='lang', 
    schema='code_search'
)

# Code view types: RAW (original) or CANON (normalized)
ViewKind: Final[Enum] = Enum(
    'RAW', 'CANON',
    name='view_kind', 
    schema='code_search'
)

# Deduplication levels
DedupLevel: Final[Enum] = Enum(
    'EXACT', 'NEAR',
    name='dedup_level', 
    schema='code_search'
)

# FAISS algorithms for vector search
FaissAlgo: Final[Enum] = Enum(
    'FlatIP', 'HNSW', 'IVFFlat', 'IVFPQ', 'IVFOPQ',
    name='faiss_algo', 
    schema='code_search'
)

# Storage tiers for lifecycle management
StorageTier: Final[Enum] = Enum(
    'HOT', 'WARM',
    name='storage_tier', 
    schema='code_search'
)

# Repository metadata
repo: Final[Table] = Table("repo", metadata,
    Column("repo_id", BigInteger, primary_key=True),
    Column("name", Text, nullable=False, unique=True),
    Column("default_branch", Text),
    Column("is_private", Boolean, nullable=False, default=True),
    Column("created_at", TIMESTAMP(timezone=True)),
    Column("updated_at", TIMESTAMP(timezone=True)),
)



# Code symbols (functions, classes, methods)
symbol: Final[Table] = Table("symbol", metadata,
    Column("symbol_id", BigInteger, primary_key=True),
    Column("repo_id", BigInteger, ForeignKey(repo.c.repo_id, ondelete="CASCADE"), nullable=False),
    Column("language", Lang, nullable=False),
    Column("path", Text, nullable=False),
    Column("name", Text),
    Column("signature", Text),
    Column("start_line", Integer),
    Column("end_line", Integer),
    Column("created_at", TIMESTAMP(timezone=True)),
    Column("updated_at", TIMESTAMP(timezone=True)),
)

# Code chunks for similarity analysis
chunk: Final[Table] = Table("chunk", metadata,
    Column("chunk_id", BigInteger, primary_key=True),
    Column("repo_id", BigInteger, ForeignKey(repo.c.repo_id, ondelete="CASCADE"), nullable=False),
    Column("language", Lang, nullable=False),
    Column("path", Text, nullable=False),
    Column("path_prefix", Text),
    Column("symbol_id", BigInteger, ForeignKey(symbol.c.symbol_id, ondelete="SET NULL")),
    Column("start_line", Integer),
    Column("end_line", Integer),
    Column("window_start_token", Integer),
    Column("window_end_token", Integer),
    Column("text_raw", Text),
    Column("text_canon", Text),
    Column("view_mask", SmallInteger, nullable=False, default=1),
    Column("size_tokens_raw", Integer),
    Column("size_tokens_canon", Integer),
    Column("canonical_hash", Text),
    Column("is_representative", Boolean, nullable=False, default=True),
    Column("dedup_group_id", BigInteger),
    Column("shard_key", Text, nullable=False),
    Column('faiss_vector_id', Integer(), nullable=False),
    Column("ingested_at", TIMESTAMP(timezone=True)),
    Column("updated_at", TIMESTAMP(timezone=True)),
)

# PCA transformation configurations
pca_transform: Final[Table] = Table("pca_transform", metadata,
    Column("pca_version", Text, primary_key=True),
    Column("input_dim", Integer, nullable=False),
    Column("output_dim", Integer, nullable=False),
    Column("mean_blob", Text),
    Column("projection_blob", Text),
    Column("notes", Text),
    Column("created_at", TIMESTAMP(timezone=True)),
)

# FAISS index metadata
index_registry: Final[Table] = Table("index_registry", metadata,
    Column("index_id", BigInteger, primary_key=True),
    Column("repo_id", BigInteger, ForeignKey(repo.c.repo_id, ondelete="CASCADE"), nullable=False),
    Column("shard_key", Text, nullable=False),
    Column("view", ViewKind, nullable=False),
    Column("storage_tier", StorageTier),
    Column("algo", FaissAlgo, nullable=False),
    Column("params_json", JSON),
    Column("pca_version", Text, ForeignKey(pca_transform.c.pca_version)),
    Column("segment_id", BigInteger),
    Column("file_name", Text, nullable=False),
    Column("path_uri", Text),
    Column("dim", Integer),
    Column("vector_count", Integer, default=0),
    Column("created_at", TIMESTAMP(timezone=True)),
    Column("is_active", Boolean, nullable=False, default=False),
    Column("blue_green_group", Text)
)

# Links chunks to their vector embeddings
embedding: Final[Table] = Table("embedding", metadata,
    Column("chunk_id", BigInteger, ForeignKey(chunk.c.chunk_id, ondelete="CASCADE"), primary_key=True),
    Column("index_id", BigInteger, ForeignKey(index_registry.c.index_id, ondelete="CASCADE"), primary_key=True),
    Column("view", ViewKind, primary_key=True),
    Column("pca_version", Text, ForeignKey(pca_transform.c.pca_version), nullable=False),
    Column("dim", Integer, nullable=False),
    Column("l2_normalized", Boolean, nullable=False, default=True),
    Column("storage_tier", StorageTier),
    Column("faiss_vector_id", Integer),
    Column("present_in_faiss", Boolean, nullable=False, default=False),
    Column("last_indexed_at", TIMESTAMP(timezone=True)),
)

# All tables for programmatic access
ALL_TABLES: Final[List[Table]] = [
    repo, symbol, chunk, pca_transform, index_registry, embedding
]

# View mask constants
VIEW_MASK_RAW: Final[int] = 1
VIEW_MASK_CANON: Final[int] = 2
VIEW_MASK_BOTH: Final[int] = 3


