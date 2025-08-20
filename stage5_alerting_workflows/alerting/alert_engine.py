"""
Main alert engine for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class AlertStatus(Enum):
    """Alert status types"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    company_id: str
    factor: str
    current_value: float
    threshold_value: float
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    acknowledged_by: Optional[str]
    resolved_by: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]

class AlertEngine:
    """Main alert engine for processing and managing alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_alerts = {}
        self.alert_history = []
        self.alert_callbacks = []
        self.suppression_rules = {}
        self.statistics = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'suppressed_alerts': 0,
            'alerts_by_severity': {
                'low': 0,
                'medium': 0,
                'high': 0,
                'critical': 0
            }
        }
        self._initialize_engine()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("AlertEngine initialized successfully")
        return True
    
    def _initialize_engine(self):
        """Initialize alert engine"""
        
        self.settings = {
            'max_active_alerts': self.config.get('max_active_alerts', 1000),
            'alert_retention_days': self.config.get('alert_retention_days', 30),
            'auto_resolve_hours': self.config.get('auto_resolve_hours', 24),
            'duplicate_window_minutes': self.config.get('duplicate_window_minutes', 15),
            'enable_suppression': self.config.get('enable_suppression', True),
            'enable_escalation': self.config.get('enable_escalation', True)
        }
    
    async def create_alert(self, title: str, description: str, severity: AlertSeverity,
                          company_id: str, factor: str, current_value: float,
                          threshold_value: float, tags: List[str] = None,
                          metadata: Dict[str, Any] = None) -> Alert:
        """Create a new alert"""
        
        try:
            # Check for duplicates
            if await self._is_duplicate_alert(company_id, factor, title):
                logger.info(f"Duplicate alert suppressed: {title}")
                return None
            
            # Check suppression rules
            if await self._is_suppressed(company_id, factor, severity):
                logger.info(f"Alert suppressed by rules: {title}")
                self.statistics['suppressed_alerts'] += 1
                return None
            
            # Create alert
            alert = Alert(
                id=str(uuid.uuid4()),
                title=title,
                description=description,
                severity=severity,
                status=AlertStatus.ACTIVE,
                company_id=company_id,
                factor=factor,
                current_value=current_value,
                threshold_value=threshold_value,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                acknowledged_at=None,
                resolved_at=None,
                acknowledged_by=None,
                resolved_by=None,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Store alert
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Update statistics
            self.statistics['total_alerts'] += 1
            self.statistics['active_alerts'] += 1
            self.statistics['alerts_by_severity'][severity.value] += 1
            
            # Notify callbacks
            await self._notify_callbacks('alert_created', alert)
            
            logger.info(f"Created alert: {alert.id} - {title}")
            return alert
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return None
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            
            if alert.status != AlertStatus.ACTIVE:
                return False
            
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = datetime.now()
            
            await self._notify_callbacks('alert_acknowledged', alert)
            
            logger.info(f"Acknowledged alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.resolved_by = resolved_by
            alert.updated_at = datetime.now()
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            # Update statistics
            self.statistics['active_alerts'] -= 1
            self.statistics['resolved_alerts'] += 1
            
            await self._notify_callbacks('alert_resolved', alert)
            
            logger.info(f"Resolved alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    async def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert"""
        
        try:
            if alert_id not in self.active_alerts:
                return False
            
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.updated_at = datetime.now()
            
            await self._notify_callbacks('alert_suppressed', alert)
            
            logger.info(f"Suppressed alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error suppressing alert: {e}")
            return False
    
    async def _is_duplicate_alert(self, company_id: str, factor: str, title: str) -> bool:
        """Check if alert is a duplicate within the time window"""
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=self.settings['duplicate_window_minutes'])
            
            for alert in self.active_alerts.values():
                if (alert.company_id == company_id and 
                    alert.factor == factor and 
                    alert.title == title and
                    alert.created_at > cutoff_time):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate alert: {e}")
            return False
    
    async def _is_suppressed(self, company_id: str, factor: str, severity: AlertSeverity) -> bool:
        """Check if alert should be suppressed"""
        
        try:
            if not self.settings['enable_suppression']:
                return False
            
            # Check company-specific suppression
            if company_id in self.suppression_rules:
                rule = self.suppression_rules[company_id]
                if factor in rule.get('factors', []):
                    return True
                if severity.value in rule.get('severities', []):
                    return True
            
            # Check global suppression rules
            if 'global' in self.suppression_rules:
                rule = self.suppression_rules['global']
                if factor in rule.get('factors', []):
                    return True
                if severity.value in rule.get('severities', []):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking suppression: {e}")
            return False
    
    async def _notify_callbacks(self, event_type: str, alert: Alert):
        """Notify registered callbacks"""
        
        try:
            for callback in self.alert_callbacks:
                try:
                    await callback(event_type, alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying callbacks: {e}")
    
    async def get_active_alerts(self, company_id: str = None, 
                              severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        
        try:
            alerts = list(self.active_alerts.values())
            
            if company_id:
                alerts = [a for a in alerts if a.company_id == company_id]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            # Sort by severity and creation time
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.HIGH: 1,
                AlertSeverity.MEDIUM: 2,
                AlertSeverity.LOW: 3
            }
            
            alerts.sort(key=lambda x: (severity_order[x.severity], x.created_at), reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def get_alert_history(self, company_id: str = None, hours: int = 24) -> List[Alert]:
        """Get alert history with optional filtering"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_alerts = []
            
            for alert in self.alert_history:
                if alert.created_at < cutoff_time:
                    continue
                
                if company_id and alert.company_id != company_id:
                    continue
                
                filtered_alerts.append(alert)
            
            # Sort by creation time (most recent first)
            filtered_alerts.sort(key=lambda x: x.created_at, reverse=True)
            
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
    
    async def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        
        try:
            # Check active alerts first
            if alert_id in self.active_alerts:
                return self.active_alerts[alert_id]
            
            # Check history
            for alert in self.alert_history:
                if alert.id == alert_id:
                    return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting alert by ID: {e}")
            return None
    
    async def add_suppression_rule(self, rule_id: str, company_id: str = None,
                                 factors: List[str] = None, severities: List[str] = None,
                                 duration_hours: int = 24):
        """Add alert suppression rule"""
        
        try:
            key = company_id if company_id else 'global'
            
            if key not in self.suppression_rules:
                self.suppression_rules[key] = {}
            
            self.suppression_rules[key][rule_id] = {
                'factors': factors or [],
                'severities': severities or [],
                'duration_hours': duration_hours,
                'created_at': datetime.now()
            }
            
            logger.info(f"Added suppression rule: {rule_id}")
            
        except Exception as e:
            logger.error(f"Error adding suppression rule: {e}")
    
    async def remove_suppression_rule(self, rule_id: str, company_id: str = None):
        """Remove alert suppression rule"""
        
        try:
            key = company_id if company_id else 'global'
            
            if key in self.suppression_rules and rule_id in self.suppression_rules[key]:
                del self.suppression_rules[key][rule_id]
                logger.info(f"Removed suppression rule: {rule_id}")
                
        except Exception as e:
            logger.error(f"Error removing suppression rule: {e}")
    
    def register_callback(self, callback):
        """Register alert callback"""
        
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
    
    def unregister_callback(self, callback):
        """Unregister alert callback"""
        
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    async def cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        
        try:
            cutoff_time = datetime.now() - timedelta(days=self.settings['alert_retention_days'])
            
            # Remove old alerts from history
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.created_at > cutoff_time
            ]
            
            logger.info("Cleaned up old alerts")
            
        except Exception as e:
            logger.error(f"Error cleaning up alerts: {e}")
    
    async def auto_resolve_stale_alerts(self):
        """Auto-resolve stale alerts"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.settings['auto_resolve_hours'])
            
            stale_alerts = []
            
            for alert_id, alert in self.active_alerts.items():
                if alert.created_at < cutoff_time and alert.status == AlertStatus.ACTIVE:
                    stale_alerts.append(alert_id)
            
            for alert_id in stale_alerts:
                await self.resolve_alert(alert_id, 'system_auto_resolve')
            
            if stale_alerts:
                logger.info(f"Auto-resolved {len(stale_alerts)} stale alerts")
                
        except Exception as e:
            logger.error(f"Error auto-resolving stale alerts: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get alert engine statistics"""
        try:
            stats = self.statistics.copy()
            
            # Calculate additional metrics
            total_alerts = stats['total_alerts']
            if total_alerts > 0:
                stats['resolution_rate'] = round((stats['resolved_alerts'] / total_alerts) * 100, 2)
                stats['suppression_rate'] = round((stats['suppressed_alerts'] / total_alerts) * 100, 2)
            else:
                stats['resolution_rate'] = 0
                stats['suppression_rate'] = 0
            
            stats['active_alerts_count'] = len(self.active_alerts)
            stats['last_activity'] = datetime.now().isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    async def evaluate_alert_rules(self, company_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate alert rules for a company - compatibility method for AlertingEngine"""
        # This method is called by AlertingEngine but AlertEngine doesn't have rule evaluation
        # The rule evaluation is handled by RuleEngine in the AlertingEngine
        logger.debug(f"AlertEngine evaluate_alert_rules called for company {company_id}")
        return []
    
    async def export_alerts(self, company_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Export alerts for analysis"""
        
        try:
            alerts = await self.get_alert_history(company_id, hours)
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'company_id': company_id,
                'time_window_hours': hours,
                'total_alerts': len(alerts),
                'alerts': []
            }
            
            for alert in alerts:
                export_data['alerts'].append({
                    'id': alert.id,
                    'title': alert.title,
                    'description': alert.description,
                    'severity': alert.severity.value,
                    'status': alert.status.value,
                    'company_id': alert.company_id,
                    'factor': alert.factor,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'created_at': alert.created_at.isoformat(),
                    'updated_at': alert.updated_at.isoformat(),
                    'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                    'tags': alert.tags,
                    'metadata': alert.metadata
                })
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")
            return {'error': str(e)}
