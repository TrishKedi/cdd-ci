"""Database connection and session management for the code duplication detection system.

This module provides the core database infrastructure using SQLAlchemy's async engine
and session management. It handles connection pooling, transaction management, and
provides utilities for safe database operations.

Key Components:
    - Async Engine: PostgreSQL connection with optimized pooling configuration
    - Session Factory: Creates properly configured database sessions
    - Session Generator: Provides transactional context with safety timeouts

The database layer is designed for high-performance async operations with:
    - Connection pooling for concurrent request handling
    - Statement timeouts to prevent runaway queries
    - Automatic connection health checks (pool_pre_ping)
    - Future-compatible SQLAlchemy 2.0 mode

Usage:
    >>> async with AsyncSessionLocal() as session:
    ...     result = await session.execute(text("SELECT 1"))
    
    >>> async for session in get_session():
    ...     # Session with automatic timeout configuration
    ...     await session.execute(some_query)
"""

import logging
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    async_sessionmaker, 
    AsyncSession,
    AsyncEngine
)

from .settings import settings

# Set up logging for database operations
logger = logging.getLogger(__name__)

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.POOL_SIZE,
    max_overflow=settings.MAX_OVERFLOW,
    pool_pre_ping=True,
    future=True,
    pool_recycle=3600,
    pool_timeout=30,
    echo=settings.DEBUG_SQL if hasattr(settings, 'DEBUG_SQL') else False
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False,
    class_=AsyncSession
)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session with automatic timeout and transaction management.
    
    Provides a properly configured database session with built-in safety measures:
    - Statement timeout to prevent runaway queries
    - Automatic session cleanup on context exit
    - Transaction rollback on exceptions
    - Connection health validation
    
    The session is configured with a statement timeout to prevent long-running
    queries from blocking the application. This is especially important for
    similarity search operations that might involve complex vector calculations.
    
    Yields:
        AsyncSession: Configured database session with timeout protection
    
    Raises:
        SQLAlchemyError: If database connection or timeout configuration fails
        TimeoutError: If statement timeout is exceeded during query execution
    """
    async with AsyncSessionLocal() as session:
        try:
            timeout_query = text(
                f"SET LOCAL statement_timeout = '{settings.STATEMENT_TIMEOUT_MS}ms'"
            )
            await session.execute(timeout_query)
            
            logger.debug(f"Database session created with {settings.STATEMENT_TIMEOUT_MS}ms timeout")
            yield session
            
        except Exception as e:
            logger.error(f"Database session error: {e}")
            raise
            
        finally:
            logger.debug("Database session closed")


async def test_connection() -> bool:
    """Test database connectivity and basic functionality.
    
    Performs a simple connectivity test to verify that the database
    is accessible and responding to queries. Useful for health checks
    and deployment validation.
    
    Returns:
        bool: True if database is accessible and responsive, False otherwise
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(text("SELECT 1 as test"))
            test_value = result.scalar()
            
            success = test_value == 1
            
            if success:
                logger.info("Database connectivity test passed")
            else:
                logger.error(f"Database connectivity test failed: unexpected result {test_value}")
                
            return success
            
    except Exception as e:
        logger.error(f"Database connectivity test failed: {e}")
        return False


async def get_database_info() -> dict[str, str]:
    """Retrieve database version and configuration information.
    
    Gathers diagnostic information about the database server that can
    be useful for debugging, monitoring, and ensuring compatibility.
    
    Returns:
        dict[str, str]: Dictionary containing database information including:
            - version: PostgreSQL version string
            - encoding: Database character encoding
            - timezone: Current database timezone
    
    Raises:
        SQLAlchemyError: If database query fails
    """
    async with AsyncSessionLocal() as session:
        query = text("""
            SELECT 
                version() as version,
                pg_encoding_to_char(encoding) as encoding,
                current_setting('timezone') as timezone
            FROM pg_database 
            WHERE datname = current_database()
        """)
        
        result = await session.execute(query)
        row = result.fetchone()
        
        if row:
            return {
                'version': row.version,
                'encoding': row.encoding,
                'timezone': row.timezone
            }
        else:
            logger.warning("Could not retrieve database information")
            return {}


async def close_engine() -> None:
    """Properly close the database engine and all connections.
    
    Gracefully shuts down the database connection pool and closes all
    active connections. Should be called during application shutdown
    to ensure clean resource cleanup.
    """
    try:
        await engine.dispose()
        logger.info("Database engine closed successfully")
    except Exception as e:
        logger.error(f"Error closing database engine: {e}")
        raise


def get_engine_stats() -> dict[str, int]:
    """Get current database connection pool statistics.
    
    Returns:
        dict[str, int]: Dictionary containing pool statistics:
            - pool_size: Configured base pool size
            - checked_in: Available connections in pool
            - checked_out: Active connections in use
            - overflow: Overflow connections created
            - total: Total connections (pool + overflow)
    """
    pool = engine.pool
    
    return {
        'pool_size': pool.size(),
        'checked_in': pool.checkedin(),
        'checked_out': pool.checkedout(),
        'overflow': pool.overflow(),
        'total': pool.checkedin() + pool.checkedout() + pool.overflow()
    }


DEFAULT_STATEMENT_TIMEOUT: int = 30000
DEFAULT_POOL_SIZE: int = 10
DEFAULT_MAX_OVERFLOW: int = 20

if not hasattr(settings, 'DATABASE_URL') or not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL must be configured in settings")

if not hasattr(settings, 'STATEMENT_TIMEOUT_MS'):
    logger.warning(f"STATEMENT_TIMEOUT_MS not configured, using default {DEFAULT_STATEMENT_TIMEOUT}ms")
    settings.STATEMENT_TIMEOUT_MS = DEFAULT_STATEMENT_TIMEOUT

logger.info(f"Database configured with pool_size={settings.POOL_SIZE}, "
           f"max_overflow={settings.MAX_OVERFLOW}, "
           f"timeout={settings.STATEMENT_TIMEOUT_MS}ms")


__all__ = [
    # Core database objects
    'engine',
    'AsyncSessionLocal', 
    'get_session',
    
    # Utility functions
    'test_connection',
    'get_database_info',
    'close_engine',
    'get_engine_stats',
    
    # Constants
    'DEFAULT_STATEMENT_TIMEOUT',
    'DEFAULT_POOL_SIZE', 
    'DEFAULT_MAX_OVERFLOW'
]
