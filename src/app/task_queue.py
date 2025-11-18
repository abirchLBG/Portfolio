import logging
import os
import time

from redis import Redis, RedisError, ConnectionError as ConnectionError
from rq import Queue

logger = logging.getLogger(__name__)


def get_redis_connection(
    host: str | None = None,
    port: int = 6379,
    db: int = 0,
    max_retries: int = 5,
    retry_delay: float = 1.0,
) -> Redis:
    """
    Establish Redis connection with retry logic.

    Args:
        host: Redis host (defaults to REDIS_HOST env var or 'redis')
        port: Redis port (defaults to REDIS_PORT env var or 6379)
        db: Redis database number
        max_retries: Maximum number of connection attempts
        retry_delay: Delay in seconds between retries

    Returns:
        Redis connection object

    Raises:
        ConnectionError: If connection cannot be established after max_retries
    """
    host = host or os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", port))

    for attempt in range(1, max_retries + 1):
        try:
            redis_conn = Redis(
                host=host, port=port, db=db, socket_timeout=5, socket_connect_timeout=5
            )
            redis_conn.ping()
            logger.info(f"Successfully connected to Redis at {host}:{port}")
            return redis_conn
        except (RedisError, ConnectionError) as e:
            logger.warning(
                f"Redis connection attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to Redis at {host}:{port} after {max_retries} attempts"
                )
                raise ConnectionError(
                    f"Cannot connect to Redis at {host}:{port} after {max_retries} retries"
                ) from e

    # This should never be reached, but satisfies type checker
    raise ConnectionError("Unexpected error in Redis connection")


# Initialize Redis connection and task queue
redis_conn = get_redis_connection()
task_queue = Queue("default", connection=redis_conn)
