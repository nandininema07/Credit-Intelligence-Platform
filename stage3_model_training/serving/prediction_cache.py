"""
Prediction caching system for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import asyncio
import hashlib
from datetime import datetime, timedelta
import redis
import pickle

logger = logging.getLogger(__name__)

class PredictionCache:
    """Redis-based prediction caching system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.cache_ttl = config.get('cache_ttl', 3600)  # 1 hour default
        self.max_cache_size = config.get('max_cache_size', 10000)
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            redis_config = self.config.get('redis', {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            })
            
            self.redis_client = redis.Redis(
                host=redis_config['host'],
                port=redis_config['port'],
                db=redis_config['db'],
                decode_responses=False
            )
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            logger.info("Connected to Redis cache")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, using in-memory cache: {str(e)}")
            self.redis_client = InMemoryCache(self.max_cache_size)
    
    def _generate_cache_key(self, model_id: str, features: Dict[str, Any]) -> str:
        """Generate cache key from model ID and features"""
        
        # Sort features for consistent hashing
        sorted_features = json.dumps(features, sort_keys=True)
        feature_hash = hashlib.md5(sorted_features.encode()).hexdigest()
        
        return f"prediction:{model_id}:{feature_hash}"
    
    async def get_prediction(self, model_id: str, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached prediction"""
        
        cache_key = self._generate_cache_key(model_id, features)
        
        try:
            if isinstance(self.redis_client, InMemoryCache):
                cached_data = self.redis_client.get(cache_key)
            else:
                cached_data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, cache_key
                )
            
            if cached_data:
                self.cache_stats['hits'] += 1
                
                if isinstance(self.redis_client, InMemoryCache):
                    return cached_data
                else:
                    return pickle.loads(cached_data)
            else:
                self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            self.cache_stats['misses'] += 1
            return None
    
    async def cache_prediction(self, model_id: str, features: Dict[str, Any], 
                             prediction_result: Dict[str, Any]):
        """Cache prediction result"""
        
        cache_key = self._generate_cache_key(model_id, features)
        
        try:
            # Add metadata
            cache_data = {
                **prediction_result,
                'cached_at': datetime.now().isoformat(),
                'model_id': model_id
            }
            
            if isinstance(self.redis_client, InMemoryCache):
                self.redis_client.set(cache_key, cache_data, self.cache_ttl)
            else:
                serialized_data = pickle.dumps(cache_data)
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.setex, cache_key, self.cache_ttl, serialized_data
                )
            
            logger.debug(f"Cached prediction for key {cache_key}")
            
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    async def invalidate_model_cache(self, model_id: str):
        """Invalidate all cached predictions for a model"""
        
        try:
            if isinstance(self.redis_client, InMemoryCache):
                self.redis_client.delete_pattern(f"prediction:{model_id}:*")
            else:
                # Get all keys for this model
                pattern = f"prediction:{model_id}:*"
                keys = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.keys, pattern
                )
                
                if keys:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.redis_client.delete, *keys
                    )
            
            logger.info(f"Invalidated cache for model {model_id}")
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        try:
            if isinstance(self.redis_client, InMemoryCache):
                cache_info = self.redis_client.get_stats()
            else:
                info = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.info
                )
                cache_info = {
                    'used_memory': info.get('used_memory', 0),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0)
                }
            
            # Calculate hit rate
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'total_hits': self.cache_stats['hits'],
                'total_misses': self.cache_stats['misses'],
                'total_evictions': self.cache_stats['evictions'],
                'cache_info': cache_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {'error': str(e)}
    
    async def clear_cache(self):
        """Clear all cached predictions"""
        
        try:
            if isinstance(self.redis_client, InMemoryCache):
                self.redis_client.clear()
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.flushdb
                )
            
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
    
    async def cache_batch_predictions(self, model_id: str, 
                                    features_list: List[Dict[str, Any]],
                                    predictions: List[Dict[str, Any]]):
        """Cache batch predictions"""
        
        for features, prediction in zip(features_list, predictions):
            await self.cache_prediction(model_id, features, prediction)


class InMemoryCache:
    """In-memory cache fallback when Redis is not available"""
    
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        
        if key in self.cache:
            # Check TTL
            entry = self.cache[key]
            if datetime.now() < entry['expires_at']:
                self.access_times[key] = datetime.now()
                return entry['data']
            else:
                # Expired
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: int):
        """Set value in cache with TTL"""
        
        # Check if we need to evict
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'data': value,
            'expires_at': expires_at,
            'created_at': datetime.now()
        }
        self.access_times[key] = datetime.now()
    
    def _evict_lru(self):
        """Evict least recently used item"""
        
        if not self.access_times:
            return
        
        # Find LRU key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Remove from cache
        if lru_key in self.cache:
            del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def delete_pattern(self, pattern: str):
        """Delete keys matching pattern"""
        
        # Simple pattern matching (only supports * at end)
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            keys_to_delete = [key for key in self.cache.keys() if key.startswith(prefix)]
        else:
            keys_to_delete = [pattern] if pattern in self.cache else []
        
        for key in keys_to_delete:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        
        # Clean expired entries
        current_time = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time >= entry['expires_at']
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        return {
            'total_keys': len(self.cache),
            'memory_usage_estimate': len(self.cache) * 1024,  # Rough estimate
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size
        }
