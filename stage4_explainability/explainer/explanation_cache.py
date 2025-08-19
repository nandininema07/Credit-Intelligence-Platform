"""
Explanation cache for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)

class ExplanationCache:
    """Cache for storing and retrieving explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.cache_stats = defaultdict(int)
        self.max_cache_size = config.get('max_cache_size', 1000)
        self.cache_ttl_hours = config.get('cache_ttl_hours', 24)
        self.hit_count = 0
        self.miss_count = 0
        
    async def get_explanation(self, request_id: str, explanation_type) -> Optional[Any]:
        """Get cached explanation if available"""
        
        try:
            cache_key = self._generate_cache_key(request_id, explanation_type)
            
            if cache_key in self.cache:
                cached_item = self.cache[cache_key]
                
                # Check if cache item is still valid
                if self._is_cache_valid(cached_item):
                    self.hit_count += 1
                    self.cache_stats['hits'] += 1
                    logger.info(f"Cache hit for {cache_key}")
                    return cached_item['result']
                else:
                    # Remove expired item
                    del self.cache[cache_key]
                    self.cache_stats['expired'] += 1
            
            self.miss_count += 1
            self.cache_stats['misses'] += 1
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached explanation: {e}")
            return None
    
    async def cache_explanation(self, result) -> bool:
        """Cache explanation result"""
        
        try:
            cache_key = self._generate_cache_key(result.request_id, result.explanation_type)
            
            # Check cache size limit
            if len(self.cache) >= self.max_cache_size:
                await self._evict_oldest_entries()
            
            # Store in cache
            cache_item = {
                'result': result,
                'cached_at': datetime.now(),
                'access_count': 0,
                'last_accessed': datetime.now()
            }
            
            self.cache[cache_key] = cache_item
            self.cache_stats['cached'] += 1
            
            logger.info(f"Cached explanation for {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching explanation: {e}")
            return False
    
    def _generate_cache_key(self, request_id: str, explanation_type) -> str:
        """Generate cache key from request parameters"""
        
        try:
            # Create unique key from request ID and explanation type
            key_data = f"{request_id}_{explanation_type}"
            
            if hasattr(explanation_type, 'value'):
                key_data = f"{request_id}_{explanation_type.value}"
            
            # Hash the key for consistent length
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            return cache_key
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"key_{request_id}_{explanation_type}"
    
    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """Check if cached item is still valid"""
        
        try:
            cached_at = cached_item['cached_at']
            ttl = timedelta(hours=self.cache_ttl_hours)
            
            is_valid = datetime.now() - cached_at < ttl
            
            if is_valid:
                # Update access statistics
                cached_item['access_count'] += 1
                cached_item['last_accessed'] = datetime.now()
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error checking cache validity: {e}")
            return False
    
    async def _evict_oldest_entries(self, count: int = None):
        """Evict oldest cache entries"""
        
        try:
            if count is None:
                count = max(1, len(self.cache) // 10)  # Evict 10% of cache
            
            # Sort by last accessed time
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1]['last_accessed']
            )
            
            # Remove oldest entries
            for i in range(min(count, len(sorted_items))):
                cache_key = sorted_items[i][0]
                del self.cache[cache_key]
                self.cache_stats['evicted'] += 1
            
            logger.info(f"Evicted {count} cache entries")
            
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")
    
    async def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries matching pattern"""
        
        try:
            if pattern is None:
                # Clear entire cache
                cleared_count = len(self.cache)
                self.cache.clear()
                self.cache_stats['invalidated'] += cleared_count
                logger.info(f"Cleared entire cache ({cleared_count} entries)")
            else:
                # Clear entries matching pattern
                keys_to_remove = [key for key in self.cache.keys() if pattern in key]
                
                for key in keys_to_remove:
                    del self.cache[key]
                    self.cache_stats['invalidated'] += 1
                
                logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching '{pattern}'")
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
    
    async def get_similar_explanations(self, request_id: str, explanation_type,
                                     similarity_threshold: float = 0.8) -> List[Any]:
        """Get similar cached explanations"""
        
        try:
            similar_explanations = []
            target_key = self._generate_cache_key(request_id, explanation_type)
            
            for cache_key, cached_item in self.cache.items():
                if cache_key == target_key:
                    continue
                
                # Check if cache item is valid
                if not self._is_cache_valid(cached_item):
                    continue
                
                # Simple similarity check based on explanation type
                cached_result = cached_item['result']
                if hasattr(cached_result, 'explanation_type'):
                    if cached_result.explanation_type == explanation_type:
                        similarity_score = self._calculate_similarity(
                            request_id, cached_result.request_id
                        )
                        
                        if similarity_score >= similarity_threshold:
                            similar_explanations.append({
                                'result': cached_result,
                                'similarity_score': similarity_score
                            })
            
            # Sort by similarity score
            similar_explanations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_explanations[:5]  # Return top 5 similar explanations
            
        except Exception as e:
            logger.error(f"Error getting similar explanations: {e}")
            return []
    
    def _calculate_similarity(self, request_id1: str, request_id2: str) -> float:
        """Calculate similarity between two requests (simplified)"""
        
        try:
            # Simple similarity based on string similarity
            # In a real implementation, this would use more sophisticated methods
            
            # Extract user IDs if present
            user_id1 = request_id1.split('_')[0] if '_' in request_id1 else request_id1
            user_id2 = request_id2.split('_')[0] if '_' in request_id2 else request_id2
            
            if user_id1 == user_id2:
                return 0.9  # Same user, high similarity
            
            # Basic string similarity
            common_chars = set(request_id1) & set(request_id2)
            total_chars = set(request_id1) | set(request_id2)
            
            if len(total_chars) == 0:
                return 0.0
            
            similarity = len(common_chars) / len(total_chars)
            return similarity
            
        except Exception:
            return 0.0
    
    async def preload_explanations(self, requests: List[Dict[str, Any]]):
        """Preload explanations for batch processing"""
        
        try:
            preloaded = 0
            
            for request_data in requests:
                request_id = request_data.get('request_id')
                explanation_type = request_data.get('explanation_type')
                
                if request_id and explanation_type:
                    cache_key = self._generate_cache_key(request_id, explanation_type)
                    
                    if cache_key not in self.cache:
                        # This would typically trigger explanation generation
                        # For now, just mark as requested
                        self.cache_stats['preload_requests'] += 1
                        preloaded += 1
            
            logger.info(f"Preloaded {preloaded} explanation requests")
            
        except Exception as e:
            logger.error(f"Error preloading explanations: {e}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        
        try:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
            
            # Calculate cache size distribution
            explanation_types = defaultdict(int)
            access_patterns = {'low': 0, 'medium': 0, 'high': 0}
            
            for cached_item in self.cache.values():
                if hasattr(cached_item['result'], 'explanation_type'):
                    explanation_types[cached_item['result'].explanation_type.value] += 1
                
                access_count = cached_item.get('access_count', 0)
                if access_count <= 1:
                    access_patterns['low'] += 1
                elif access_count <= 5:
                    access_patterns['medium'] += 1
                else:
                    access_patterns['high'] += 1
            
            return {
                'cache_size': len(self.cache),
                'max_cache_size': self.max_cache_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'cache_ttl_hours': self.cache_ttl_hours,
                'explanation_type_distribution': dict(explanation_types),
                'access_patterns': access_patterns,
                'cache_stats': dict(self.cache_stats),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {'error': str(e)}
    
    async def optimize_cache(self):
        """Optimize cache performance"""
        
        try:
            optimization_actions = []
            
            # Remove expired entries
            expired_keys = []
            for cache_key, cached_item in self.cache.items():
                if not self._is_cache_valid(cached_item):
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.cache[key]
                self.cache_stats['expired'] += 1
            
            if expired_keys:
                optimization_actions.append(f"Removed {len(expired_keys)} expired entries")
            
            # Identify frequently accessed items
            frequent_items = []
            for cache_key, cached_item in self.cache.items():
                access_count = cached_item.get('access_count', 0)
                if access_count > 5:  # Frequently accessed
                    frequent_items.append(cache_key)
            
            optimization_actions.append(f"Identified {len(frequent_items)} frequently accessed items")
            
            # Check cache size and evict if necessary
            if len(self.cache) > self.max_cache_size * 0.9:  # 90% full
                await self._evict_oldest_entries(count=int(self.max_cache_size * 0.1))
                optimization_actions.append("Evicted oldest entries due to high cache usage")
            
            logger.info(f"Cache optimization completed: {'; '.join(optimization_actions)}")
            
            return {
                'optimization_actions': optimization_actions,
                'cache_size_after': len(self.cache),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
            return {'error': str(e)}
    
    async def export_cache_data(self, filepath: str):
        """Export cache data to file"""
        
        try:
            cache_export = {
                'cache_data': {},
                'statistics': self.get_cache_statistics(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Export cache items (excluding the actual result objects for size)
            for cache_key, cached_item in self.cache.items():
                cache_export['cache_data'][cache_key] = {
                    'cached_at': cached_item['cached_at'].isoformat(),
                    'access_count': cached_item['access_count'],
                    'last_accessed': cached_item['last_accessed'].isoformat(),
                    'result_type': type(cached_item['result']).__name__
                }
            
            with open(filepath, 'w') as f:
                json.dump(cache_export, f, indent=2)
            
            logger.info(f"Cache data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting cache data: {e}")
    
    async def import_cache_metadata(self, filepath: str):
        """Import cache metadata from file"""
        
        try:
            with open(filepath, 'r') as f:
                cache_import = json.load(f)
            
            # Update statistics
            if 'statistics' in cache_import:
                imported_stats = cache_import['statistics']
                logger.info(f"Imported cache metadata with {imported_stats.get('cache_size', 0)} entries")
            
            logger.info(f"Cache metadata imported from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing cache metadata: {e}")
    
    def configure_cache(self, new_config: Dict[str, Any]):
        """Update cache configuration"""
        
        try:
            old_config = {
                'max_cache_size': self.max_cache_size,
                'cache_ttl_hours': self.cache_ttl_hours
            }
            
            # Update configuration
            if 'max_cache_size' in new_config:
                self.max_cache_size = new_config['max_cache_size']
            
            if 'cache_ttl_hours' in new_config:
                self.cache_ttl_hours = new_config['cache_ttl_hours']
            
            # If cache size was reduced, evict entries
            if self.max_cache_size < len(self.cache):
                entries_to_evict = len(self.cache) - self.max_cache_size
                self._evict_oldest_entries(entries_to_evict)
            
            logger.info(f"Cache configuration updated: {old_config} -> {new_config}")
            
        except Exception as e:
            logger.error(f"Error configuring cache: {e}")
