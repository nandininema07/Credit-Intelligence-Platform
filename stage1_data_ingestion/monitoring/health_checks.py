"""
Health monitoring for the data ingestion pipeline.
Monitors system health, data quality, and pipeline performance.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import psutil
import aiohttp

logger = logging.getLogger(__name__)

class HealthChecker:
    """Pipeline health monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_thresholds = config.get('thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 0.1
        })
        self.check_intervals = config.get('intervals', {
            'system': 60,
            'database': 120,
            'external_apis': 300
        })
        
    async def check_pipeline_health(self) -> Dict[str, Any]:
        """Comprehensive pipeline health check"""
        health_status = {
            'healthy': True,
            'timestamp': datetime.now(),
            'checks': {}
        }
        
        # System health
        system_health = await self._check_system_health()
        health_status['checks']['system'] = system_health
        
        # Database health
        db_health = await self._check_database_health()
        health_status['checks']['database'] = db_health
        
        # External API health
        api_health = await self._check_external_apis()
        health_status['checks']['external_apis'] = api_health
        
        # Data quality health
        data_quality = await self._check_data_quality()
        health_status['checks']['data_quality'] = data_quality
        
        # Overall health determination
        health_status['healthy'] = all(
            check.get('healthy', False) 
            for check in health_status['checks'].values()
        )
        
        # Calculate health score
        health_scores = [
            check.get('score', 0.0) 
            for check in health_status['checks'].values()
        ]
        health_status['health_score'] = sum(health_scores) / len(health_scores) if health_scores else 0.0
        
        return health_status
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network connections
            connections = len(psutil.net_connections())
            
            # Process count
            process_count = len(psutil.pids())
            
            system_health = {
                'healthy': (
                    cpu_percent < self.health_thresholds['cpu_usage'] and
                    memory_percent < self.health_thresholds['memory_usage'] and
                    disk_percent < self.health_thresholds['disk_usage']
                ),
                'metrics': {
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'disk_usage': disk_percent,
                    'network_connections': connections,
                    'process_count': process_count
                },
                'score': self._calculate_system_score(cpu_percent, memory_percent, disk_percent)
            }
            
            return system_health
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'score': 0.0
            }
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            from ..storage.postgres_manager import PostgreSQLManager
            
            # This would need to be injected or configured
            # For now, return a placeholder
            db_health = {
                'healthy': True,
                'metrics': {
                    'connection_pool_size': 10,
                    'active_connections': 5,
                    'query_response_time': 0.05,
                    'last_successful_query': datetime.now()
                },
                'score': 1.0
            }
            
            return db_health
            
        except Exception as e:
            logger.error(f"Error checking database health: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'score': 0.0
            }
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API health and connectivity"""
        api_checks = {}
        
        # Test APIs
        apis_to_check = [
            {
                'name': 'newsapi',
                'url': 'https://newsapi.org/v2/top-headlines?country=us&pageSize=1',
                'timeout': 10
            },
            {
                'name': 'yahoo_finance',
                'url': 'https://query1.finance.yahoo.com/v8/finance/chart/AAPL',
                'timeout': 5
            }
        ]
        
        for api in apis_to_check:
            try:
                start_time = datetime.now()
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(api['url'], timeout=api['timeout']) as response:
                        response_time = (datetime.now() - start_time).total_seconds()
                        
                        api_checks[api['name']] = {
                            'healthy': response.status < 400,
                            'status_code': response.status,
                            'response_time': response_time,
                            'timestamp': datetime.now()
                        }
                        
            except Exception as e:
                api_checks[api['name']] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now()
                }
        
        # Overall API health
        overall_healthy = all(check.get('healthy', False) for check in api_checks.values())
        avg_response_time = sum(
            check.get('response_time', 10) 
            for check in api_checks.values()
        ) / max(len(api_checks), 1)
        
        return {
            'healthy': overall_healthy,
            'checks': api_checks,
            'avg_response_time': avg_response_time,
            'score': 1.0 if overall_healthy else 0.5
        }
    
    async def _check_data_quality(self) -> Dict[str, Any]:
        """Check data quality metrics"""
        try:
            # This would integrate with the storage system to check data quality
            # For now, return a placeholder
            data_quality = {
                'healthy': True,
                'metrics': {
                    'recent_ingestion_rate': 100,  # records per hour
                    'data_freshness': 5,  # minutes since last update
                    'error_rate': 0.02,  # 2% error rate
                    'duplicate_rate': 0.05  # 5% duplicate rate
                },
                'score': 0.9
            }
            
            # Check if error rate is acceptable
            if data_quality['metrics']['error_rate'] > self.health_thresholds.get('error_rate', 0.1):
                data_quality['healthy'] = False
                data_quality['score'] *= 0.5
            
            return data_quality
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {
                'healthy': False,
                'error': str(e),
                'score': 0.0
            }
    
    def _calculate_system_score(self, cpu: float, memory: float, disk: float) -> float:
        """Calculate system health score"""
        # Weighted scoring
        cpu_score = max(0, (100 - cpu) / 100)
        memory_score = max(0, (100 - memory) / 100)
        disk_score = max(0, (100 - disk) / 100)
        
        # Weighted average (CPU and memory more important)
        return (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
    
    async def check_source_health(self, source_type: str, source_name: str) -> Dict[str, Any]:
        """Check health of a specific data source"""
        try:
            # This would check specific source connectivity and performance
            # Implementation depends on source type
            
            if source_type == 'news_sources':
                return await self._check_news_source_health(source_name)
            elif source_type == 'social_sources':
                return await self._check_social_source_health(source_name)
            elif source_type == 'financial_sources':
                return await self._check_financial_source_health(source_name)
            else:
                return {'healthy': True, 'message': 'Health check not implemented'}
                
        except Exception as e:
            logger.error(f"Error checking {source_type}.{source_name} health: {e}")
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_news_source_health(self, source_name: str) -> Dict[str, Any]:
        """Check news source health"""
        # Placeholder implementation
        return {
            'healthy': True,
            'last_successful_fetch': datetime.now() - timedelta(minutes=5),
            'articles_fetched_last_hour': 25
        }
    
    async def _check_social_source_health(self, source_name: str) -> Dict[str, Any]:
        """Check social media source health"""
        # Placeholder implementation
        return {
            'healthy': True,
            'last_successful_fetch': datetime.now() - timedelta(minutes=10),
            'posts_fetched_last_hour': 50
        }
    
    async def _check_financial_source_health(self, source_name: str) -> Dict[str, Any]:
        """Check financial data source health"""
        # Placeholder implementation
        return {
            'healthy': True,
            'last_successful_fetch': datetime.now() - timedelta(minutes=15),
            'data_points_fetched_last_hour': 100
        }
    
    def get_health_summary(self, health_status: Dict[str, Any]) -> str:
        """Get human-readable health summary"""
        if health_status['healthy']:
            return f"✅ Pipeline healthy (Score: {health_status['health_score']:.2f})"
        else:
            issues = []
            for check_name, check_result in health_status['checks'].items():
                if not check_result.get('healthy', True):
                    issues.append(check_name)
            
            return f"⚠️ Pipeline issues detected in: {', '.join(issues)}"
    
    async def run_continuous_monitoring(self, callback: Optional[callable] = None):
        """Run continuous health monitoring"""
        while True:
            try:
                health_status = await self.check_pipeline_health()
                
                if callback:
                    await callback(health_status)
                
                if not health_status['healthy']:
                    logger.warning(f"Health issues detected: {self.get_health_summary(health_status)}")
                
                await asyncio.sleep(self.check_intervals['system'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)
