"""
cache.py
Centralised caching layer.

Uses Redis when REDIS_URL is set and redis is reachable.
Falls back to SimpleCache (in-process dict) automatically so the app
works perfectly without Redis for local development.

Usage:
    from app.cache import cache

    @cache.cached(timeout=300, key_prefix="upcoming_games")
    def get_data(): ...

    # Or manually:
    cache.set("key", value, timeout=60)
    value = cache.get("key")
    cache.delete("key")
"""

from flask_caching import Cache
from app.logger import get_logger

logger = get_logger(__name__)

cache = Cache()


def init_cache(app):
    """
    Initialise the cache with the Flask app.
    Tries Redis first, falls back to SimpleCache if Redis isn't available.
    Call this inside create_app().
    """
    from app.config import REDIS_URL, CACHE_DEFAULT_TIMEOUT

    # Try Redis
    try:
        import redis as _redis
        r = _redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        config = {
            "CACHE_TYPE":               "RedisCache",
            "CACHE_REDIS_URL":          REDIS_URL,
            "CACHE_DEFAULT_TIMEOUT":    CACHE_DEFAULT_TIMEOUT,
        }
        cache.init_app(app, config=config)
        logger.info(f"Cache: Redis connected at {REDIS_URL}")
    except Exception as e:
        # Redis not available — use in-process SimpleCache
        config = {
            "CACHE_TYPE":            "SimpleCache",
            "CACHE_DEFAULT_TIMEOUT": CACHE_DEFAULT_TIMEOUT,
        }
        cache.init_app(app, config=config)
        logger.info(f"Cache: Redis unavailable ({e}) — using SimpleCache")
