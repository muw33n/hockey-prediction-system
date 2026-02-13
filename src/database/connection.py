"""
Database Connection Manager - Shared database connection with retry logic.

Extracted from elo_rating_model.py for reuse across the project.
Location: src/database/connection.py
"""

import time
import pandas as pd
from typing import Dict, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, DisconnectionError

from config.logging_config import get_component_logger, LoggingConfig

logger = get_component_logger(__name__, 'database')


class DatabaseConnectionManager:
    """Enhanced database connection with retry logic and error handling."""

    def __init__(self, connection_string: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize database connection manager.

        Args:
            connection_string: SQLAlchemy connection string
            max_retries: Maximum number of connection attempts
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self.connection_string = connection_string
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._engine = None

    @property
    def engine(self):
        """Lazy connection with retry logic."""
        if self._engine is None:
            self._connect_with_retry()
        return self._engine

    def _connect_with_retry(self):
        """Attempt connection with retry logic."""
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Database connection attempt {attempt + 1}/{self.max_retries}")
                self._engine = create_engine(
                    self.connection_string,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    connect_args={"connect_timeout": 30}
                )
                # Test connection
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Database connection established successfully")
                return

            except Exception as e:
                logger.error(f"Database connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 1.5  # Exponential backoff
                else:
                    logger.error("All database connection attempts failed")
                    raise

    def execute_query_safe(self, query, description: str = "Query",
                          params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Safe query execution with retry logic and performance monitoring.

        Args:
            query: SQL query (string or SQLAlchemy text)
            description: Description for logging
            params: Query parameters

        Returns:
            DataFrame with query results
        """
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                logger.debug(f"Executing {description}")

                df = pd.read_sql(query, self.engine, params=params)

                execution_time = time.time() - start_time

                # Smart logging: INFO for important/slow queries, DEBUG for routine
                if execution_time > 0.5 or len(df) > 1000 or 'historical' in description.lower():
                    logger.info(f"{description} completed: {len(df)} rows in {execution_time:.2f}s")
                else:
                    logger.debug(f"{description} completed: {len(df)} rows in {execution_time:.3f}s")

                return df

            except (OperationalError, DisconnectionError) as e:
                logger.warning(f"{description} failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info("Reconnecting to database...")
                    self._engine = None  # Force reconnection
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"{description} failed after {self.max_retries} attempts")
                    raise
            except Exception as e:
                logger.error(f"{description} failed with unexpected error: {e}")
                LoggingConfig.log_exception(logger, e, description)
                raise

    def close(self):
        """Close the database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")
