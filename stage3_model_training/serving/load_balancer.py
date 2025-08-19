"""
Load balancer for Stage 3 model serving.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import random
from datetime import datetime
import time
from collections import defaultdict
import aiohttp

logger = logging.getLogger(__name__)

class LoadBalancer:
    """Load balancer for model serving endpoints"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.endpoints = {}
        self.health_status = {}
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.strategy = config.get('strategy', 'round_robin')
        self.health_check_interval = config.get('health_check_interval', 30)
        self.current_index = 0
        
    def add_endpoint(self, endpoint_id: str, url: str, weight: int = 1):
        """Add model serving endpoint"""
        
        self.endpoints[endpoint_id] = {
            'url': url,
            'weight': weight,
            'added_at': datetime.now().isoformat(),
            'status': 'unknown'
        }
        
        self.health_status[endpoint_id] = {
            'healthy': True,
            'last_check': None,
            'consecutive_failures': 0
        }
        
        logger.info(f"Added endpoint {endpoint_id}: {url}")
    
    def remove_endpoint(self, endpoint_id: str):
        """Remove model serving endpoint"""
        
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
        
        if endpoint_id in self.health_status:
            del self.health_status[endpoint_id]
        
        if endpoint_id in self.request_counts:
            del self.request_counts[endpoint_id]
        
        if endpoint_id in self.response_times:
            del self.response_times[endpoint_id]
        
        logger.info(f"Removed endpoint {endpoint_id}")
    
    def get_healthy_endpoints(self) -> List[str]:
        """Get list of healthy endpoints"""
        
        return [
            endpoint_id for endpoint_id in self.endpoints.keys()
            if self.health_status[endpoint_id]['healthy']
        ]
    
    def select_endpoint(self) -> Optional[str]:
        """Select endpoint based on load balancing strategy"""
        
        healthy_endpoints = self.get_healthy_endpoints()
        
        if not healthy_endpoints:
            logger.warning("No healthy endpoints available")
            return None
        
        if self.strategy == 'round_robin':
            return self._round_robin_selection(healthy_endpoints)
        elif self.strategy == 'weighted_round_robin':
            return self._weighted_round_robin_selection(healthy_endpoints)
        elif self.strategy == 'least_connections':
            return self._least_connections_selection(healthy_endpoints)
        elif self.strategy == 'least_response_time':
            return self._least_response_time_selection(healthy_endpoints)
        elif self.strategy == 'random':
            return self._random_selection(healthy_endpoints)
        else:
            return self._round_robin_selection(healthy_endpoints)
    
    def _round_robin_selection(self, endpoints: List[str]) -> str:
        """Round robin endpoint selection"""
        
        if not endpoints:
            return None
        
        selected = endpoints[self.current_index % len(endpoints)]
        self.current_index += 1
        
        return selected
    
    def _weighted_round_robin_selection(self, endpoints: List[str]) -> str:
        """Weighted round robin selection"""
        
        if not endpoints:
            return None
        
        # Create weighted list
        weighted_endpoints = []
        for endpoint_id in endpoints:
            weight = self.endpoints[endpoint_id]['weight']
            weighted_endpoints.extend([endpoint_id] * weight)
        
        if not weighted_endpoints:
            return endpoints[0]
        
        selected = weighted_endpoints[self.current_index % len(weighted_endpoints)]
        self.current_index += 1
        
        return selected
    
    def _least_connections_selection(self, endpoints: List[str]) -> str:
        """Select endpoint with least active connections"""
        
        if not endpoints:
            return None
        
        # Use request count as proxy for connections
        min_requests = min(self.request_counts[ep] for ep in endpoints)
        candidates = [ep for ep in endpoints if self.request_counts[ep] == min_requests]
        
        return random.choice(candidates)
    
    def _least_response_time_selection(self, endpoints: List[str]) -> str:
        """Select endpoint with lowest average response time"""
        
        if not endpoints:
            return None
        
        avg_response_times = {}
        
        for endpoint_id in endpoints:
            response_times = self.response_times[endpoint_id]
            if response_times:
                avg_response_times[endpoint_id] = sum(response_times) / len(response_times)
            else:
                avg_response_times[endpoint_id] = 0  # New endpoint gets priority
        
        return min(avg_response_times.keys(), key=lambda x: avg_response_times[x])
    
    def _random_selection(self, endpoints: List[str]) -> str:
        """Random endpoint selection"""
        
        if not endpoints:
            return None
        
        return random.choice(endpoints)
    
    async def forward_request(self, path: str, method: str = 'POST', 
                            data: Dict[str, Any] = None, 
                            headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Forward request to selected endpoint"""
        
        endpoint_id = self.select_endpoint()
        
        if not endpoint_id:
            return {
                'success': False,
                'error': 'No healthy endpoints available'
            }
        
        endpoint_url = self.endpoints[endpoint_id]['url']
        full_url = f"{endpoint_url.rstrip('/')}/{path.lstrip('/')}"
        
        start_time = time.time()
        
        try:
            self.request_counts[endpoint_id] += 1
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == 'GET':
                    async with session.get(full_url, headers=headers) as response:
                        result = await response.json()
                        status_code = response.status
                elif method.upper() == 'POST':
                    async with session.post(full_url, json=data, headers=headers) as response:
                        result = await response.json()
                        status_code = response.status
                else:
                    return {
                        'success': False,
                        'error': f'Unsupported method: {method}'
                    }
            
            response_time = time.time() - start_time
            self.response_times[endpoint_id].append(response_time)
            
            # Keep only recent response times
            if len(self.response_times[endpoint_id]) > 100:
                self.response_times[endpoint_id] = self.response_times[endpoint_id][-50:]
            
            # Update health status on successful request
            self.health_status[endpoint_id]['healthy'] = True
            self.health_status[endpoint_id]['consecutive_failures'] = 0
            
            return {
                'success': True,
                'data': result,
                'endpoint_id': endpoint_id,
                'response_time': response_time,
                'status_code': status_code
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Update health status on failure
            self.health_status[endpoint_id]['consecutive_failures'] += 1
            
            if self.health_status[endpoint_id]['consecutive_failures'] >= 3:
                self.health_status[endpoint_id]['healthy'] = False
                logger.warning(f"Marked endpoint {endpoint_id} as unhealthy after 3 failures")
            
            logger.error(f"Request to {endpoint_id} failed: {str(e)}")
            
            return {
                'success': False,
                'error': str(e),
                'endpoint_id': endpoint_id,
                'response_time': response_time
            }
    
    async def health_check(self, endpoint_id: str) -> bool:
        """Perform health check on endpoint"""
        
        if endpoint_id not in self.endpoints:
            return False
        
        endpoint_url = self.endpoints[endpoint_id]['url']
        health_url = f"{endpoint_url.rstrip('/')}/health"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self.health_status[endpoint_id]['healthy'] = True
                        self.health_status[endpoint_id]['consecutive_failures'] = 0
                        self.health_status[endpoint_id]['last_check'] = datetime.now().isoformat()
                        return True
                    else:
                        raise Exception(f"Health check failed with status {response.status}")
        
        except Exception as e:
            self.health_status[endpoint_id]['consecutive_failures'] += 1
            
            if self.health_status[endpoint_id]['consecutive_failures'] >= 3:
                self.health_status[endpoint_id]['healthy'] = False
            
            self.health_status[endpoint_id]['last_check'] = datetime.now().isoformat()
            
            logger.warning(f"Health check failed for {endpoint_id}: {str(e)}")
            return False
    
    async def start_health_monitoring(self):
        """Start periodic health monitoring"""
        
        async def health_monitor():
            while True:
                for endpoint_id in list(self.endpoints.keys()):
                    await self.health_check(endpoint_id)
                
                await asyncio.sleep(self.health_check_interval)
        
        # Start health monitoring task
        asyncio.create_task(health_monitor())
        logger.info("Started health monitoring")
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        
        stats = {
            'strategy': self.strategy,
            'total_endpoints': len(self.endpoints),
            'healthy_endpoints': len(self.get_healthy_endpoints()),
            'endpoints': {}
        }
        
        for endpoint_id, endpoint_info in self.endpoints.items():
            health_info = self.health_status[endpoint_id]
            response_times = self.response_times[endpoint_id]
            
            endpoint_stats = {
                'url': endpoint_info['url'],
                'weight': endpoint_info['weight'],
                'healthy': health_info['healthy'],
                'request_count': self.request_counts[endpoint_id],
                'consecutive_failures': health_info['consecutive_failures'],
                'last_health_check': health_info['last_check'],
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0
            }
            
            stats['endpoints'][endpoint_id] = endpoint_stats
        
        return stats
    
    def update_endpoint_weight(self, endpoint_id: str, weight: int):
        """Update endpoint weight"""
        
        if endpoint_id in self.endpoints:
            self.endpoints[endpoint_id]['weight'] = weight
            logger.info(f"Updated weight for {endpoint_id} to {weight}")
    
    def set_strategy(self, strategy: str):
        """Set load balancing strategy"""
        
        valid_strategies = [
            'round_robin', 'weighted_round_robin', 'least_connections',
            'least_response_time', 'random'
        ]
        
        if strategy in valid_strategies:
            self.strategy = strategy
            logger.info(f"Load balancing strategy set to {strategy}")
        else:
            logger.error(f"Invalid strategy: {strategy}. Valid options: {valid_strategies}")
    
    async def circuit_breaker_check(self, endpoint_id: str) -> bool:
        """Circuit breaker pattern implementation"""
        
        if endpoint_id not in self.health_status:
            return False
        
        health_info = self.health_status[endpoint_id]
        
        # If too many consecutive failures, circuit is open
        if health_info['consecutive_failures'] >= 5:
            # Check if enough time has passed to try again
            if health_info['last_check']:
                last_check = datetime.fromisoformat(health_info['last_check'])
                time_since_check = datetime.now() - last_check
                
                # Try again after 60 seconds
                if time_since_check.total_seconds() > 60:
                    # Attempt health check
                    return await self.health_check(endpoint_id)
                else:
                    return False
            else:
                return False
        
        return health_info['healthy']
