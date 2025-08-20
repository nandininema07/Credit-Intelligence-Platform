"""
Performance metrics collection for the data ingestion pipeline.
Tracks throughput, latency, and operational metrics.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Pipeline metrics collection and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = defaultdict(deque)
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.buffer_size = config.get('buffer_size', 1000)
        self.retention_hours = config.get('retention_hours', 24)
        
    async def collect_pipeline_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive pipeline metrics"""
        metrics = {
            'timestamp': datetime.now(),
            'counters': dict(self.counters),
            'throughput': self._calculate_throughput(),
            'latency': self._calculate_latency_stats(),
            'error_rates': self._calculate_error_rates(),
            'resource_usage': await self._collect_resource_metrics()
        }
        
        # Store metrics in buffer
        self._store_metrics(metrics)
        
        return metrics
    
    def increment_counter(self, metric_name: str, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        key = self._create_metric_key(metric_name, tags)
        self.counters[key] += 1
        
        # Also store timestamped entry
        self.metrics_buffer[key].append({
            'timestamp': datetime.now(),
            'value': 1,
            'type': 'counter'
        })
        
        self._cleanup_old_metrics(key)
    
    def record_timer(self, metric_name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timing metric"""
        key = self._create_metric_key(metric_name, tags)
        
        self.metrics_buffer[key].append({
            'timestamp': datetime.now(),
            'value': duration,
            'type': 'timer'
        })
        
        self._cleanup_old_metrics(key)
    
    def record_gauge(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric"""
        key = self._create_metric_key(metric_name, tags)
        
        self.metrics_buffer[key].append({
            'timestamp': datetime.now(),
            'value': value,
            'type': 'gauge'
        })
        
        self._cleanup_old_metrics(key)
    
    def _create_metric_key(self, metric_name: str, tags: Dict[str, str] = None) -> str:
        """Create a unique key for metric with tags"""
        if not tags:
            return metric_name
        
        tag_string = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}[{tag_string}]"
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in buffer"""
        key = 'pipeline_metrics'
        self.metrics_buffer[key].append(metrics)
        self._cleanup_old_metrics(key)
    
    def _cleanup_old_metrics(self, key: str):
        """Remove old metrics from buffer"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        while (self.metrics_buffer[key] and 
               len(self.metrics_buffer[key]) > self.buffer_size):
            self.metrics_buffer[key].popleft()
        
        # Remove old entries
        while (self.metrics_buffer[key] and 
               self.metrics_buffer[key][0].get('timestamp', datetime.now()) < cutoff_time):
            self.metrics_buffer[key].popleft()
    
    def _calculate_throughput(self) -> Dict[str, float]:
        """Calculate throughput metrics"""
        throughput = {}
        
        for key, metrics in self.metrics_buffer.items():
            if 'ingestion' in key and metrics:
                # Count entries in last hour
                one_hour_ago = datetime.now() - timedelta(hours=1)
                recent_count = sum(
                    1 for metric in metrics 
                    if metric.get('timestamp', datetime.min) > one_hour_ago
                )
                throughput[key] = recent_count
        
        return throughput
    
    def _calculate_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Calculate latency statistics"""
        latency_stats = {}
        
        for key, metrics in self.metrics_buffer.items():
            if any(m.get('type') == 'timer' for m in metrics):
                timer_values = [
                    m['value'] for m in metrics 
                    if m.get('type') == 'timer'
                ]
                
                if timer_values:
                    latency_stats[key] = {
                        'avg': sum(timer_values) / len(timer_values),
                        'min': min(timer_values),
                        'max': max(timer_values),
                        'p95': self._percentile(timer_values, 95),
                        'p99': self._percentile(timer_values, 99)
                    }
        
        return latency_stats
    
    def _calculate_error_rates(self) -> Dict[str, float]:
        """Calculate error rates"""
        error_rates = {}
        
        # Group metrics by base name (without error/success suffix)
        metric_groups = defaultdict(lambda: {'total': 0, 'errors': 0})
        
        for key in self.counters:
            base_key = key.split('[')[0]  # Remove tags
            if 'error' in base_key:
                base_name = base_key.replace('_errors', '').replace('_error', '')
                metric_groups[base_name]['errors'] += self.counters[key]
            else:
                base_name = base_key.replace('_completed', '').replace('_success', '')
                metric_groups[base_name]['total'] += self.counters[key]
        
        for base_name, counts in metric_groups.items():
            if counts['total'] > 0:
                error_rates[base_name] = counts['errors'] / (counts['total'] + counts['errors'])
        
        return error_rates
    
    async def _collect_resource_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
                'network_connections': len(psutil.net_connections()),
                'timestamp': datetime.now()
            }
        except ImportError:
            return {'error': 'psutil not available'}
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            return {'error': str(e)}
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            'counters': dict(self.counters),
            'recent_throughput': self._get_recent_throughput(),
            'current_latency': self._get_current_latency(),
            'buffer_sizes': {key: len(buffer) for key, buffer in self.metrics_buffer.items()},
            'timestamp': datetime.now()
        }
    
    def _get_recent_throughput(self) -> Dict[str, float]:
        """Get throughput for last 15 minutes"""
        throughput = {}
        fifteen_min_ago = datetime.now() - timedelta(minutes=15)
        
        for key, metrics in self.metrics_buffer.items():
            if 'ingestion' in key:
                recent_count = sum(
                    1 for metric in metrics 
                    if metric.get('timestamp', datetime.min) > fifteen_min_ago
                )
                throughput[key] = recent_count / 15.0  # per minute
        
        return throughput
    
    def _get_current_latency(self) -> Dict[str, float]:
        """Get current average latency"""
        latency = {}
        five_min_ago = datetime.now() - timedelta(minutes=5)
        
        for key, metrics in self.metrics_buffer.items():
            recent_timers = [
                m['value'] for m in metrics 
                if (m.get('type') == 'timer' and 
                    m.get('timestamp', datetime.min) > five_min_ago)
            ]
            
            if recent_timers:
                latency[key] = sum(recent_timers) / len(recent_timers)
        
        return latency
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        metrics_data = {
            'export_timestamp': datetime.now(),
            'counters': dict(self.counters),
            'buffer_data': {
                key: list(buffer) 
                for key, buffer in self.metrics_buffer.items()
            }
        }
        
        if format == 'json':
            return json.dumps(metrics_data, default=str, indent=2)
        else:
            return str(metrics_data)
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.counters.clear()
        self.metrics_buffer.clear()
        self.timers.clear()
        logger.info("All metrics reset")
    
    async def start_metrics_collection(self, interval: int = 60):
        """Start automatic metrics collection"""
        while True:
            try:
                await self.collect_pipeline_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(interval)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return {
            'counters': dict(self.counters),
            'timers': {k: self._calculate_stats(v) for k, v in self.timers.items()},
            'buffer_sizes': {k: len(v) for k, v in self.metrics_buffer.items()},
            'last_updated': datetime.now().isoformat()
        }
    
    async def record_ingestion_cycle(self, data_points_count: int, stats: Dict[str, Any]) -> None:
        """Record metrics for an ingestion cycle - compatibility method for pipeline"""
        try:
            # Record data points count
            self.increment_counter('ingestion_cycle_data_points', {'total': str(data_points_count)})
            
            # Record by source
            for source, count in stats.get('by_source', {}).items():
                self.increment_counter('ingestion_by_source', {'source': source, 'count': str(count)})
            
            # Record by content type
            for content_type, count in stats.get('by_content_type', {}).items():
                self.increment_counter('ingestion_by_content_type', {'type': content_type, 'count': str(count)})
            
            # Record by company
            for company, count in stats.get('by_company', {}).items():
                self.increment_counter('ingestion_by_company', {'company': company, 'count': str(count)})
            
            # Record languages
            for language in stats.get('languages', []):
                self.increment_counter('ingestion_by_language', {'language': language})
            
            logger.debug(f"Recorded ingestion cycle metrics: {data_points_count} data points")
            
        except Exception as e:
            logger.error(f"Error recording ingestion cycle metrics: {e}")
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values"""
        if not values:
            return {'count': 0, 'min': 0, 'max': 0, 'avg': 0}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values)
        }
