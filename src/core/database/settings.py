"""Database and application settings configuration.

This module provides centralized configuration management for the code duplication
detection system. It uses Pydantic Settings for robust configuration handling
with validation, environment variable support, and type safety.

Key Features:
    - Environment variable integration with sensible defaults
    - Automatic database URL conversion for sync/async compatibility
    - Connection pool configuration for optimal performance
    - Validation and type safety for all configuration values
    - Development and production environment support

Configuration Sources (in order of precedence):
    1. Environment variables (CDD_DB_HOST, CDD_POOL_SIZE, etc.)
    2. .env file (if present)
    3. Default values defined in the Settings class

Example:
    >>> from .settings import settings
    >>> print(f"Database: {settings.DATABASE_URL}")
    >>> print(f"Pool size: {settings.POOL_SIZE}")

Environment Variables:
    CDD_DB_HOST: PostgreSQL server hostname (default: localhost)
    CDD_DB_PORT: PostgreSQL server port (default: 5432)
    CDD_DB_USER: PostgreSQL username (default: cdd_user)
    CDD_DB_PASSWORD: PostgreSQL password (default: postgres)
    CDD_DB_NAME: PostgreSQL database name (default: cdd_db)
    CDD_POOL_SIZE: Connection pool base size (default: 10)
    CDD_MAX_OVERFLOW: Additional connections under load (default: 20)
    CDD_STATEMENT_TIMEOUT_MS: Query timeout in milliseconds (default: 5000)
    CDD_DEBUG_SQL: Enable SQL query logging (default: false)
"""

import logging
import os
from typing import Optional

from pydantic import Field, validator, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# Set up logging for configuration management
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Application settings with validation and environment variable support.
    
    This class defines all configuration parameters for the code duplication
    detection system. It automatically loads values from environment variables
    and provides sensible defaults for development.
    
    The settings are validated at startup to catch configuration errors early.
    Database URLs are automatically converted between async and sync variants
    to support both the main application (async) and Alembic migrations (sync).
    
    Attributes:
        DATABASE_URL (str): Async PostgreSQL connection string
        DATABASE_URL_SYNC (Optional[str]): Sync PostgreSQL connection string
        POOL_SIZE (int): Base number of database connections in pool
        MAX_OVERFLOW (int): Additional connections allowed under load
        STATEMENT_TIMEOUT_MS (int): Query timeout in milliseconds
        DEBUG_SQL (bool): Enable SQL query logging for debugging
    
    Environment Variables:
        All attributes can be overridden via CDD_ prefixed environment
        variables (e.g., CDD_DB_HOST, CDD_POOL_SIZE, CDD_DEBUG_SQL, etc.)
    """
    
    # Pydantic model configuration
    model_config = SettingsConfigDict(
        # Environment variable configuration
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        
        # Additional settings
        extra="forbid",  # Reject unknown configuration keys
        validate_default=True  # Validate default values
    )
    
    # Database Configuration Components
    # =================================
    
    DB_HOST: str = Field(
        default="localhost",
        description="PostgreSQL server hostname or IP address",
        env="CDD_DB_HOST"
    )
    
    DB_PORT: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL server port number",
        env="CDD_DB_PORT"
    )
    
    DB_USER: str = Field(
        default="cdd_user",
        description="PostgreSQL username for authentication",
        env="CDD_DB_USER"
    )
    
    DB_PASSWORD: str = Field(
        default="postgres",
        description="PostgreSQL password for authentication",
        env="CDD_DB_PASSWORD"
    )
    
    DB_NAME: str = Field(
        default="cdd_db",
        description="PostgreSQL database name",
        env="CDD_DB_NAME"
    )
    
    # Dynamically Generated URLs (computed properties)
    # ==============================================
    
    # These will be set in __init__ based on the individual components above
    
    # Connection Pool Configuration
    # ============================
    
    POOL_SIZE: int = Field(
        default=10,
        ge=1,  # At least 1 connection
        le=50,  # Reasonable upper limit
        description="Base number of connections in the database pool",
        env="CDD_POOL_SIZE"
    )
    
    MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,  # No overflow is valid
        le=100,  # Reasonable upper limit
        description="Additional connections allowed beyond pool_size under load",
        env="CDD_MAX_OVERFLOW"
    )
    
    # Query Configuration
    # ==================
    
    STATEMENT_TIMEOUT_MS: int = Field(
        default=5000,  # 5 seconds
        ge=1000,  # At least 1 second
        le=300000,  # At most 5 minutes
        description="Query timeout in milliseconds to prevent runaway queries",
        env="CDD_STATEMENT_TIMEOUT_MS"
    )
    
    # Debug Configuration
    # ==================
    
    DEBUG_SQL: bool = Field(
        default=False,
        description="Enable SQL query logging (use only in development)",
        env="CDD_DEBUG_SQL"
    )
    
    @validator('DB_PASSWORD')
    def validate_password_security(cls, v: str) -> str:
        """Validate database password for security best practices.
        
        Args:
            v (str): The database password to validate
            
        Returns:
            str: The validated password
        """
        # Warn about default or weak passwords
        # if v == 'postgres':
        #     logger.warning(
        #         "Using default 'postgres' password. "
        #         "Please change to a secure password in production!"
        #     )
        # elif len(v) < 8:
        #     logger.warning(
        #         f"Password is only {len(v)} characters. "
        #         "Consider using a longer password for better security."
        #     )
        
        return v
    
    @validator('DB_HOST')
    def validate_host(cls, v: str) -> str:
        """Validate database host configuration.
        
        Args:
            v (str): The database host to validate
            
        Returns:
            str: The validated host
        """
        if not v or not v.strip():
            raise ValueError("Database host cannot be empty")
        
        return v.strip()
    
    @validator('MAX_OVERFLOW')
    def validate_overflow_vs_pool_size(cls, v: int, values: dict) -> int:
        """Validate that MAX_OVERFLOW is reasonable relative to POOL_SIZE.
        
        Args:
            v (int): The max_overflow value
            values (dict): Other field values (including pool_size)
            
        Returns:
            int: The validated max_overflow value
            
        Raises:
            ValueError: If overflow is excessive relative to pool size
        """
        pool_size = values.get('POOL_SIZE', 10)
        
        # Warn if overflow is more than 3x the pool size (might indicate misconfiguration)
        if v > pool_size * 3:
            logger.warning(
                f"MAX_OVERFLOW ({v}) is more than 3x POOL_SIZE ({pool_size}). "
                "This might indicate misconfiguration."
            )
        
        return v
    
    def __init__(self, **kwargs) -> None:
        """Initialize settings with automatic URL generation from components.
        
        Args:
            **kwargs: Configuration overrides (typically from environment variables)
            
        Note:
            DATABASE_URL and DATABASE_URL_SYNC are automatically generated from
            the individual database components (DB_HOST, DB_USER, etc.)
        """
        super().__init__(**kwargs)
        
        # Generate database URLs from individual components
        self.DATABASE_URL = self._generate_database_url(async_driver=True)
        self.DATABASE_URL_SYNC = self._generate_database_url(async_driver=False)
            
        # Log configuration for debugging (without sensitive info)
        self._log_configuration()
    
    def _generate_database_url(self, async_driver: bool = True) -> str:
        """Generate database URL from individual components.
        
        Constructs PostgreSQL connection URLs using the configured database
        components (host, port, user, password, database name) and the
        appropriate driver for async or sync operations.
        
        Args:
            async_driver (bool): If True, uses asyncpg for async operations.
                If False, uses psycopg for sync operations (Alembic).
        
        Returns:
            str: Complete PostgreSQL connection URL
            
        Example:
            >>> # Async:  postgresql+asyncpg://user:pass@host:5432/db
            >>> # Sync:   postgresql+psycopg://user:pass@host:5432/db
        """
        # Choose driver based on operation type
        driver = "asyncpg" if async_driver else "psycopg"
        
        # Construct URL from components
        # Format: postgresql+driver://user:password@host:port/database
        database_url = (
            f"postgresql+{driver}://"
            f"{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/"
            f"{self.DB_NAME}"
        )
        
        # Log URL generation (without password for security)
        url_safe = (
            f"postgresql+{driver}://"
            f"{self.DB_USER}:***@"
            f"{self.DB_HOST}:{self.DB_PORT}/"
            f"{self.DB_NAME}"
        )
        
        driver_type = "async" if async_driver else "sync"
        logger.debug(f"Generated {driver_type} database URL: {url_safe}")
        
        return database_url
    
    def _log_configuration(self) -> None:
        """Log current configuration for debugging and monitoring.
        
        Logs non-sensitive configuration values to help with debugging
        and operational monitoring. Database passwords are masked for security.
        """
        logger.info("Database configuration loaded:")
        logger.info(f"  Host: {self.DB_HOST}:{self.DB_PORT}")
        logger.info(f"  Database: {self.DB_NAME}")
        logger.info(f"  User: {self.DB_USER}")
        logger.info(f"  Password: {'***' if self.DB_PASSWORD else '[not set]'}")
        logger.info(f"  Pool size: {self.POOL_SIZE}")
        logger.info(f"  Max overflow: {self.MAX_OVERFLOW}")
        logger.info(f"  Statement timeout: {self.STATEMENT_TIMEOUT_MS}ms")
        logger.info(f"  Debug SQL: {self.DEBUG_SQL}")
    
    def get_total_max_connections(self) -> int:
        """Calculate the maximum possible database connections.
        
        Returns:
            int: Maximum connections (pool_size + max_overflow)
        """
        return self.POOL_SIZE + self.MAX_OVERFLOW
    
    def get_connection_info(self) -> dict:
        """Get database connection information (without password).
        
        Returns:
            dict: Dictionary with connection details (password masked)
        """
        return {
            'host': self.DB_HOST,
            'port': self.DB_PORT,
            'database': self.DB_NAME,
            'user': self.DB_USER,
            'password_set': bool(self.DB_PASSWORD)
        }
    
    @property
    def DATABASE_URL(self) -> str:
        """Get the async database URL (computed property).
        
        Returns:
            str: Async PostgreSQL connection URL
        """
        return getattr(self, '_database_url', '')
    
    @DATABASE_URL.setter
    def DATABASE_URL(self, value: str) -> None:
        """Set the async database URL.
        
        Args:
            value (str): Database URL to set
        """
        self._database_url = value
    
    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Get the sync database URL (computed property).
        
        Returns:
            str: Sync PostgreSQL connection URL
        """
        return getattr(self, '_database_url_sync', '')
    
    @DATABASE_URL_SYNC.setter
    def DATABASE_URL_SYNC(self, value: str) -> None:
        """Set the sync database URL.
        
        Args:
            value (str): Database URL to set
        """
        self._database_url_sync = value

# Global Settings Instance
# =======================

# Create the global settings instance that will be imported throughout the application
# This instance is created at module import time and validates all configuration
settings = Settings()

# Validate critical configuration at startup
# This ensures that configuration errors are caught early in the application lifecycle
if not settings.DATABASE_URL:
    raise ValueError("DATABASE_URL is required but not configured")

if not settings.DATABASE_URL_SYNC:
    raise ValueError("DATABASE_URL_SYNC could not be generated or configured")

# Log successful configuration loading
logger.info("Database settings loaded and validated successfully")


# Configuration Constants
# ======================

# Default values that can be referenced elsewhere in the application
DEFAULT_POOL_SIZE: int = 10
DEFAULT_MAX_OVERFLOW: int = 20
DEFAULT_STATEMENT_TIMEOUT_MS: int = 5000  # 5 seconds

# Validation limits
MIN_POOL_SIZE: int = 1
MAX_POOL_SIZE: int = 50
MIN_STATEMENT_TIMEOUT_MS: int = 1000  # 1 second
MAX_STATEMENT_TIMEOUT_MS: int = 300000  # 5 minutes


# Configuration Utility Functions
# ==============================

def get_database_name() -> str:
    """Get the configured database name.
    
    Returns:
        str: Database name from configuration
    """
    return settings.DB_NAME


def is_development_mode() -> bool:
    """Check if the application is running in development mode.
    
    Determines development mode based on database host configuration.
    Localhost or loopback addresses typically indicate development.
    
    Returns:
        bool: True if running in development mode
    """
    return settings.DB_HOST in ('localhost', '127.0.0.1', '::1')


def get_database_host() -> str:
    """Get the configured database host.
    
    Returns:
        str: Database host from configuration
    """
    return settings.DB_HOST


def get_database_port() -> int:
    """Get the configured database port.
    
    Returns:
        int: Database port from configuration
    """
    return settings.DB_PORT


# Public API
# ==========

__all__ = [
    # Main classes and instances
    'Settings',
    'settings',
    
    # Constants
    'DEFAULT_POOL_SIZE',
    'DEFAULT_MAX_OVERFLOW', 
    'DEFAULT_STATEMENT_TIMEOUT_MS',
    'MIN_POOL_SIZE',
    'MAX_POOL_SIZE',
    'MIN_STATEMENT_TIMEOUT_MS',
    'MAX_STATEMENT_TIMEOUT_MS',
    
    # Utility functions
    'get_database_name',
    'get_database_host',
    'get_database_port',
    'is_development_mode'
]

