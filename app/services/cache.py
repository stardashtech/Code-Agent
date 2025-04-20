import json
import pickle
from typing import Any, Optional
from redis import Redis
from app.config import settings

class RedisCache:
    def __init__(self):
        self.redis = Redis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            decode_responses=True
        )
        self.binary_redis = Redis.from_url(
            settings.redis_url,
            password=settings.redis_password,
            decode_responses=False
        )
        
    def get(self, key: str) -> Optional[str]:
        """Get a string value from cache."""
        return self.redis.get(key)
        
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a string value in cache with optional TTL."""
        if ttl is None:
            ttl = settings.cache_ttl
        self.redis.set(key, value, ex=ttl)
        
    def get_json(self, key: str) -> Optional[Any]:
        """Get a JSON value from cache."""
        data = self.get(key)
        if data:
            return json.loads(data)
        return None
        
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a JSON value in cache with optional TTL."""
        self.set(key, json.dumps(value), ttl)
        
    def get_binary(self, key: str) -> Optional[bytes]:
        """Get a binary value from cache."""
        return self.binary_redis.get(key)
        
    def set_binary(self, key: str, value: bytes, ttl: Optional[int] = None) -> None:
        """Set a binary value in cache with optional TTL."""
        if ttl is None:
            ttl = settings.cache_ttl
        self.binary_redis.set(key, value, ex=ttl)
        
    def get_pickle(self, key: str) -> Optional[Any]:
        """Get a pickled value from cache."""
        data = self.get_binary(key)
        if data:
            return pickle.loads(data)
        return None
        
    def set_pickle(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a pickled value in cache with optional TTL."""
        self.set_binary(key, pickle.dumps(value), ttl)
        
    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        self.redis.delete(key)
        
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        return self.redis.exists(key) > 0
        
    def clear(self) -> None:
        """Clear all keys from cache."""
        self.redis.flushall()

# Create a global cache instance
cache = RedisCache() 