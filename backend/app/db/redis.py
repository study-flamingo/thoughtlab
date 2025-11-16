import redis.asyncio as redis
from app.core.config import settings
from typing import Optional


class RedisConnection:
    """Redis connection manager"""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Connect to Redis"""
        if self.client is None:
            self.client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Verify connection
            await self.client.ping()
            print("Connected to Redis")
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            self.client = None
            print("Disconnected from Redis")
    
    def get_client(self):
        """Get Redis client"""
        if self.client is None:
            raise RuntimeError("Redis client not initialized. Call connect() first.")
        return self.client


# Global connection instance
redis_conn = RedisConnection()
