"""
Metrics collector for Stage 5 alerting workflows dashboard.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import statistics

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'tags': self.tags
        }

@dataclass
class Metric:
    """Metric definition and data"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    data_points: List[MetricPoint]
    
    def __post_init__(self):
        if self.data_points is None:
            self.data_points = []

class MetricsCollector:
    """Metrics collection and aggregation for dashboard analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retention_hours = config.get('retention_hours', 168)  # 7 days
        self.max_points_per_metric = config.get('max_points_per_metric', 10000)
        self.metrics = {}  # metric_name -> Metric
        self.statistics = {
            'total_metrics': 0,
            'total_data_points': 0,
            'metrics_by_type': {},
            'collection_errors': 0
        }
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("MetricsCollector initialized successfully")
        return True
    
    async def register_metric(self, name: str, metric_type: MetricType,
                            description: str, unit: str = "") -> bool:
        """Register a new metric"""
        
        try:
            if name in self.metrics:
                logger.warning(f"Metric already exists: {name}")
                return True
            
            metric = Metric(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                data_points=[]
            )
            
            self.metrics[name] = metric
            self.statistics['total_metrics'] += 1
            
            # Update type statistics
            type_count = self.statistics['metrics_by_type'].get(metric_type.value, 0)
            self.statistics['metrics_by_type'][metric_type.value] = type_count + 1
            
            logger.info(f"Registered metric: {name} ({metric_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Error registering metric: {e}")
            self.statistics['collection_errors'] += 1
            return False
    
    async def record_value(self, metric_name: str, value: float,
                         tags: Dict[str, str] = None, timestamp: datetime = None) -> bool:
        """Record a metric value"""
        
        try:
            if metric_name not in self.metrics:
                logger.error(f"Metric not registered: {metric_name}")
                return False
            
            if timestamp is None:
                timestamp = datetime.now()
            
            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                tags=tags or {}
            )
            
            metric = self.metrics[metric_name]
            metric.data_points.append(point)
            
            # Maintain max points limit
            if len(metric.data_points) > self.max_points_per_metric:
                metric.data_points = metric.data_points[-self.max_points_per_metric:]
            
            self.statistics['total_data_points'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording metric value: {e}")
            self.statistics['collection_errors'] += 1
            return False
    
    async def increment_counter(self, metric_name: str, increment: float = 1.0,
                              tags: Dict[str, str] = None) -> bool:
        """Increment a counter metric"""
        
        try:
            if metric_name not in self.metrics:
                await self.register_metric(metric_name, MetricType.COUNTER, f"Counter: {metric_name}")
            
            metric = self.metrics[metric_name]
            if metric.metric_type != MetricType.COUNTER:
                logger.error(f"Metric {metric_name} is not a counter")
                return False
            
            # Get current value
            current_value = 0
            if metric.data_points:
                current_value = metric.data_points[-1].value
            
            new_value = current_value + increment
            return await self.record_value(metric_name, new_value, tags)
            
        except Exception as e:
            logger.error(f"Error incrementing counter: {e}")
            return False
    
    async def set_gauge(self, metric_name: str, value: float,
                       tags: Dict[str, str] = None) -> bool:
        """Set a gauge metric value"""
        
        try:
            if metric_name not in self.metrics:
                await self.register_metric(metric_name, MetricType.GAUGE, f"Gauge: {metric_name}")
            
            metric = self.metrics[metric_name]
            if metric.metric_type != MetricType.GAUGE:
                logger.error(f"Metric {metric_name} is not a gauge")
                return False
            
            return await self.record_value(metric_name, value, tags)
            
        except Exception as e:
            logger.error(f"Error setting gauge: {e}")
            return False
    
    async def record_timer(self, metric_name: str, duration_seconds: float,
                         tags: Dict[str, str] = None) -> bool:
        """Record a timer metric"""
        
        try:
            if metric_name not in self.metrics:
                await self.register_metric(metric_name, MetricType.TIMER, f"Timer: {metric_name}", "seconds")
            
            metric = self.metrics[metric_name]
            if metric.metric_type != MetricType.TIMER:
                logger.error(f"Metric {metric_name} is not a timer")
                return False
            
            return await self.record_value(metric_name, duration_seconds, tags)
            
        except Exception as e:
            logger.error(f"Error recording timer: {e}")
            return False
    
    async def get_metric_data(self, metric_name: str, 
                            hours_back: int = 24,
                            tags_filter: Dict[str, str] = None) -> Optional[Dict[str, Any]]:
        """Get metric data with optional filtering"""
        
        try:
            if metric_name not in self.metrics:
                return None
            
            metric = self.metrics[metric_name]
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter by time
            filtered_points = [p for p in metric.data_points if p.timestamp >= cutoff_time]
            
            # Filter by tags
            if tags_filter:
                filtered_points = [
                    p for p in filtered_points
                    if all(p.tags.get(k) == v for k, v in tags_filter.items())
                ]
            
            if not filtered_points:
                return {
                    'name': metric_name,
                    'type': metric.metric_type.value,
                    'description': metric.description,
                    'unit': metric.unit,
                    'data_points': [],
                    'summary': {}
                }
            
            # Calculate summary statistics
            values = [p.value for p in filtered_points]
            summary = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'latest': values[-1] if values else 0,
                'first': values[0] if values else 0
            }
            
            if len(values) > 1:
                summary['median'] = statistics.median(values)
                summary['std_dev'] = statistics.stdev(values)
            
            return {
                'name': metric_name,
                'type': metric.metric_type.value,
                'description': metric.description,
                'unit': metric.unit,
                'data_points': [p.to_dict() for p in filtered_points],
                'summary': summary,
                'time_range_hours': hours_back
            }
            
        except Exception as e:
            logger.error(f"Error getting metric data: {e}")
            return None
    
    async def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get summary of all registered metrics"""
        
        try:
            metrics_summary = []
            
            for metric_name, metric in self.metrics.items():
                summary = {
                    'name': metric_name,
                    'type': metric.metric_type.value,
                    'description': metric.description,
                    'unit': metric.unit,
                    'data_points_count': len(metric.data_points),
                    'latest_value': metric.data_points[-1].value if metric.data_points else None,
                    'latest_timestamp': metric.data_points[-1].timestamp.isoformat() if metric.data_points else None
                }
                metrics_summary.append(summary)
            
            return metrics_summary
            
        except Exception as e:
            logger.error(f"Error getting all metrics: {e}")
            return []
    
    async def aggregate_metrics(self, metric_names: List[str],
                              aggregation: str = 'sum',
                              hours_back: int = 24) -> Optional[Dict[str, Any]]:
        """Aggregate multiple metrics"""
        
        try:
            if aggregation not in ['sum', 'avg', 'min', 'max', 'count']:
                logger.error(f"Unsupported aggregation: {aggregation}")
                return None
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            all_values = []
            
            for metric_name in metric_names:
                if metric_name not in self.metrics:
                    continue
                
                metric = self.metrics[metric_name]
                recent_points = [p for p in metric.data_points if p.timestamp >= cutoff_time]
                values = [p.value for p in recent_points]
                all_values.extend(values)
            
            if not all_values:
                return {
                    'aggregation': aggregation,
                    'metrics': metric_names,
                    'result': 0,
                    'count': 0
                }
            
            if aggregation == 'sum':
                result = sum(all_values)
            elif aggregation == 'avg':
                result = statistics.mean(all_values)
            elif aggregation == 'min':
                result = min(all_values)
            elif aggregation == 'max':
                result = max(all_values)
            elif aggregation == 'count':
                result = len(all_values)
            
            return {
                'aggregation': aggregation,
                'metrics': metric_names,
                'result': result,
                'count': len(all_values),
                'time_range_hours': hours_back
            }
            
        except Exception as e:
            logger.error(f"Error aggregating metrics: {e}")
            return None
    
    async def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get key metrics for dashboard display"""
        
        try:
            # Define key dashboard metrics
            dashboard_data = {}
            
            # System metrics
            system_metrics = [
                'alerts_created_total',
                'alerts_resolved_total', 
                'notifications_sent_total',
                'workflows_executed_total',
                'api_requests_total',
                'processing_time_avg'
            ]
            
            for metric_name in system_metrics:
                if metric_name in self.metrics:
                    data = await self.get_metric_data(metric_name, hours_back=24)
                    if data and data['data_points']:
                        dashboard_data[metric_name] = {
                            'current': data['summary']['latest'],
                            'trend': self._calculate_trend(data['data_points']),
                            'unit': data['unit']
                        }
            
            # Performance metrics
            performance_metrics = await self._get_performance_metrics()
            dashboard_data.update(performance_metrics)
            
            # Alert metrics
            alert_metrics = await self._get_alert_metrics()
            dashboard_data.update(alert_metrics)
            
            return {
                'metrics': dashboard_data,
                'last_updated': datetime.now().isoformat(),
                'collection_period_hours': 24
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            return {}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance-related metrics"""
        
        try:
            performance = {}
            
            # Processing times
            if 'processing_time_avg' in self.metrics:
                data = await self.get_metric_data('processing_time_avg', hours_back=24)
                if data and data['summary']:
                    performance['avg_processing_time'] = {
                        'value': data['summary']['mean'],
                        'unit': 'seconds',
                        'trend': self._calculate_trend(data['data_points'])
                    }
            
            # Throughput
            if 'requests_per_second' in self.metrics:
                data = await self.get_metric_data('requests_per_second', hours_back=1)
                if data and data['summary']:
                    performance['current_throughput'] = {
                        'value': data['summary']['latest'],
                        'unit': 'req/sec'
                    }
            
            # Error rates
            error_metrics = ['errors_total', 'requests_total']
            if all(m in self.metrics for m in error_metrics):
                errors_data = await self.get_metric_data('errors_total', hours_back=24)
                requests_data = await self.get_metric_data('requests_total', hours_back=24)
                
                if errors_data and requests_data:
                    total_errors = errors_data['summary']['latest']
                    total_requests = requests_data['summary']['latest']
                    error_rate = (total_errors / max(total_requests, 1)) * 100
                    
                    performance['error_rate'] = {
                        'value': round(error_rate, 2),
                        'unit': 'percent'
                    }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def _get_alert_metrics(self) -> Dict[str, Any]:
        """Get alert-related metrics"""
        
        try:
            alert_metrics = {}
            
            # Alert volume
            if 'alerts_created_total' in self.metrics:
                data = await self.get_metric_data('alerts_created_total', hours_back=24)
                if data and data['summary']:
                    alert_metrics['alerts_24h'] = {
                        'value': data['summary']['latest'],
                        'trend': self._calculate_trend(data['data_points'])
                    }
            
            # Resolution rate
            creation_metric = 'alerts_created_total'
            resolution_metric = 'alerts_resolved_total'
            
            if creation_metric in self.metrics and resolution_metric in self.metrics:
                created_data = await self.get_metric_data(creation_metric, hours_back=24)
                resolved_data = await self.get_metric_data(resolution_metric, hours_back=24)
                
                if created_data and resolved_data:
                    created = created_data['summary']['latest']
                    resolved = resolved_data['summary']['latest']
                    resolution_rate = (resolved / max(created, 1)) * 100
                    
                    alert_metrics['resolution_rate'] = {
                        'value': round(resolution_rate, 2),
                        'unit': 'percent'
                    }
            
            # Average resolution time
            if 'alert_resolution_time_avg' in self.metrics:
                data = await self.get_metric_data('alert_resolution_time_avg', hours_back=24)
                if data and data['summary']:
                    alert_metrics['avg_resolution_time'] = {
                        'value': round(data['summary']['mean'] / 3600, 2),  # Convert to hours
                        'unit': 'hours'
                    }
            
            return alert_metrics
            
        except Exception as e:
            logger.error(f"Error getting alert metrics: {e}")
            return {}
    
    def _calculate_trend(self, data_points: List[Dict[str, Any]]) -> str:
        """Calculate trend direction from data points"""
        
        try:
            if len(data_points) < 2:
                return 'stable'
            
            # Compare first half with second half
            mid_point = len(data_points) // 2
            first_half_avg = statistics.mean([p['value'] for p in data_points[:mid_point]])
            second_half_avg = statistics.mean([p['value'] for p in data_points[mid_point:]])
            
            change_percent = ((second_half_avg - first_half_avg) / max(first_half_avg, 0.001)) * 100
            
            if change_percent > 5:
                return 'increasing'
            elif change_percent < -5:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 'unknown'
    
    async def cleanup_old_data(self):
        """Remove data points older than retention period"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            total_removed = 0
            
            for metric in self.metrics.values():
                initial_count = len(metric.data_points)
                metric.data_points = [p for p in metric.data_points if p.timestamp >= cutoff_time]
                removed = initial_count - len(metric.data_points)
                total_removed += removed
            
            if total_removed > 0:
                logger.info(f"Cleaned up {total_removed} old metric data points")
            
            return total_removed
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    async def export_metrics(self, metric_names: List[str] = None,
                           hours_back: int = 24,
                           format: str = 'json') -> Optional[str]:
        """Export metrics data"""
        
        try:
            if metric_names is None:
                metric_names = list(self.metrics.keys())
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'time_range_hours': hours_back,
                'metrics': {}
            }
            
            for metric_name in metric_names:
                if metric_name in self.metrics:
                    data = await self.get_metric_data(metric_name, hours_back)
                    if data:
                        export_data['metrics'][metric_name] = data
            
            if format.lower() == 'json':
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"metrics_export_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.info(f"Exported metrics to {filename}")
                return filename
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get metrics collector statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Add current state
            stats.update({
                'registered_metrics': len(self.metrics),
                'retention_hours': self.retention_hours,
                'max_points_per_metric': self.max_points_per_metric,
                'oldest_data_point': None,
                'newest_data_point': None
            })
            
            # Find oldest and newest data points
            all_timestamps = []
            for metric in self.metrics.values():
                for point in metric.data_points:
                    all_timestamps.append(point.timestamp)
            
            if all_timestamps:
                stats['oldest_data_point'] = min(all_timestamps).isoformat()
                stats['newest_data_point'] = max(all_timestamps).isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
