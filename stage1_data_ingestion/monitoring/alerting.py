"""
Pipeline alerting system for monitoring data ingestion health.
Sends alerts when pipeline issues are detected.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    SYSTEM_HEALTH = "system_health"
    DATA_QUALITY = "data_quality"
    API_FAILURE = "api_failure"
    PIPELINE_FAILURE = "pipeline_failure"
    RATE_LIMIT = "rate_limit"

class PipelineAlerting:
    """Pipeline alerting and notification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_handlers = []
        self.alert_history = []
        self.alert_thresholds = config.get('thresholds', {
            'cpu_usage': 85.0,
            'memory_usage': 90.0,
            'error_rate': 0.15,
            'api_response_time': 10.0
        })
        self.cooldown_periods = config.get('cooldowns', {
            'system_health': 300,  # 5 minutes
            'data_quality': 600,   # 10 minutes
            'api_failure': 180     # 3 minutes
        })
        self.last_alerts = {}
        
    def add_alert_handler(self, handler: Callable):
        """Add alert handler function"""
        self.alert_handlers.append(handler)
    
    async def check_and_alert(self, health_status: Dict[str, Any], metrics: Dict[str, Any]):
        """Check conditions and send alerts if needed"""
        alerts_to_send = []
        
        # System health alerts
        if not health_status.get('healthy', True):
            system_alerts = self._check_system_alerts(health_status)
            alerts_to_send.extend(system_alerts)
        
        # Performance alerts
        performance_alerts = self._check_performance_alerts(metrics)
        alerts_to_send.extend(performance_alerts)
        
        # Data quality alerts
        data_quality_alerts = self._check_data_quality_alerts(health_status, metrics)
        alerts_to_send.extend(data_quality_alerts)
        
        # Send alerts
        for alert in alerts_to_send:
            if self._should_send_alert(alert):
                await self._send_alert(alert)
                self._record_alert(alert)
    
    def _check_system_alerts(self, health_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for system health alerts"""
        alerts = []
        system_checks = health_status.get('checks', {}).get('system', {})
        
        if not system_checks.get('healthy', True):
            metrics = system_checks.get('metrics', {})
            
            # CPU alert
            cpu_usage = metrics.get('cpu_usage', 0)
            if cpu_usage > self.alert_thresholds['cpu_usage']:
                alerts.append({
                    'type': AlertType.SYSTEM_HEALTH,
                    'severity': AlertSeverity.HIGH if cpu_usage > 95 else AlertSeverity.MEDIUM,
                    'title': 'High CPU Usage',
                    'message': f'CPU usage is {cpu_usage:.1f}%, exceeding threshold of {self.alert_thresholds["cpu_usage"]}%',
                    'metrics': {'cpu_usage': cpu_usage},
                    'timestamp': datetime.now()
                })
            
            # Memory alert
            memory_usage = metrics.get('memory_usage', 0)
            if memory_usage > self.alert_thresholds['memory_usage']:
                alerts.append({
                    'type': AlertType.SYSTEM_HEALTH,
                    'severity': AlertSeverity.CRITICAL if memory_usage > 95 else AlertSeverity.HIGH,
                    'title': 'High Memory Usage',
                    'message': f'Memory usage is {memory_usage:.1f}%, exceeding threshold of {self.alert_thresholds["memory_usage"]}%',
                    'metrics': {'memory_usage': memory_usage},
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance-related alerts"""
        alerts = []
        
        # Check API response times
        latency_stats = metrics.get('latency', {})
        for api_name, stats in latency_stats.items():
            avg_latency = stats.get('avg', 0)
            if avg_latency > self.alert_thresholds['api_response_time']:
                alerts.append({
                    'type': AlertType.API_FAILURE,
                    'severity': AlertSeverity.MEDIUM,
                    'title': 'Slow API Response',
                    'message': f'{api_name} average response time is {avg_latency:.2f}s, exceeding threshold of {self.alert_thresholds["api_response_time"]}s',
                    'metrics': {'api': api_name, 'avg_latency': avg_latency},
                    'timestamp': datetime.now()
                })
        
        # Check error rates
        error_rates = metrics.get('error_rates', {})
        for source_name, error_rate in error_rates.items():
            if error_rate > self.alert_thresholds['error_rate']:
                alerts.append({
                    'type': AlertType.PIPELINE_FAILURE,
                    'severity': AlertSeverity.HIGH if error_rate > 0.3 else AlertSeverity.MEDIUM,
                    'title': 'High Error Rate',
                    'message': f'{source_name} error rate is {error_rate:.1%}, exceeding threshold of {self.alert_thresholds["error_rate"]:.1%}',
                    'metrics': {'source': source_name, 'error_rate': error_rate},
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def _check_data_quality_alerts(self, health_status: Dict[str, Any], metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for data quality alerts"""
        alerts = []
        
        data_quality = health_status.get('checks', {}).get('data_quality', {})
        if not data_quality.get('healthy', True):
            dq_metrics = data_quality.get('metrics', {})
            
            # Low ingestion rate
            ingestion_rate = dq_metrics.get('recent_ingestion_rate', 0)
            if ingestion_rate < 10:  # Less than 10 records per hour
                alerts.append({
                    'type': AlertType.DATA_QUALITY,
                    'severity': AlertSeverity.MEDIUM,
                    'title': 'Low Data Ingestion Rate',
                    'message': f'Data ingestion rate is {ingestion_rate} records/hour, which is below expected levels',
                    'metrics': {'ingestion_rate': ingestion_rate},
                    'timestamp': datetime.now()
                })
            
            # High duplicate rate
            duplicate_rate = dq_metrics.get('duplicate_rate', 0)
            if duplicate_rate > 0.2:  # More than 20% duplicates
                alerts.append({
                    'type': AlertType.DATA_QUALITY,
                    'severity': AlertSeverity.LOW,
                    'title': 'High Duplicate Rate',
                    'message': f'Duplicate rate is {duplicate_rate:.1%}, indicating potential data quality issues',
                    'metrics': {'duplicate_rate': duplicate_rate},
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def _should_send_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be sent based on cooldown and deduplication"""
        alert_key = f"{alert['type'].value}_{alert['title']}"
        
        # Check cooldown
        if alert_key in self.last_alerts:
            last_sent = self.last_alerts[alert_key]
            cooldown = self.cooldown_periods.get(alert['type'].value, 300)
            
            if datetime.now() - last_sent < timedelta(seconds=cooldown):
                return False
        
        return True
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured handlers"""
        alert_data = {
            'id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'type': alert['type'].value,
            'severity': alert['severity'].value,
            'title': alert['title'],
            'message': alert['message'],
            'metrics': alert.get('metrics', {}),
            'timestamp': alert['timestamp'],
            'source': 'data_ingestion_pipeline'
        }
        
        # Send through all configured handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Error sending alert through handler: {e}")
        
        logger.warning(f"ALERT [{alert['severity'].value.upper()}]: {alert['title']} - {alert['message']}")
    
    def _record_alert(self, alert: Dict[str, Any]):
        """Record alert in history"""
        alert_key = f"{alert['type'].value}_{alert['title']}"
        self.last_alerts[alert_key] = datetime.now()
        
        # Add to history
        self.alert_history.append({
            'type': alert['type'].value,
            'severity': alert['severity'].value,
            'title': alert['title'],
            'message': alert['message'],
            'timestamp': alert['timestamp']
        })
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alert_history = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > cutoff_time
        ]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history 
            if alert['timestamp'] > cutoff_time
        ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        recent_alerts = self.get_alert_history(24)
        
        summary = {
            'total_alerts': len(recent_alerts),
            'by_severity': {},
            'by_type': {},
            'last_alert': None
        }
        
        for alert in recent_alerts:
            # Count by severity
            severity = alert['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # Count by type
            alert_type = alert['type']
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
        
        if recent_alerts:
            summary['last_alert'] = max(recent_alerts, key=lambda x: x['timestamp'])
        
        return summary
