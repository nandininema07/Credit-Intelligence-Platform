"""
Alert history management for Stage 5 alerting workflows dashboard.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AlertStatus(Enum):
    """Alert status types"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"

@dataclass
class HistoricalAlert:
    """Historical alert record"""
    alert_id: str
    company_id: str
    title: str
    description: str
    severity: str
    factor: str
    current_value: Any
    threshold_value: Any
    status: AlertStatus
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        result['updated_at'] = self.updated_at.isoformat()
        if self.acknowledged_at:
            result['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            result['resolved_at'] = self.resolved_at.isoformat()
        return result

class AlertHistory:
    """Alert history management and analytics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_history_days = config.get('max_history_days', 90)
        self.alerts = {}  # alert_id -> HistoricalAlert
        self.company_index = {}  # company_id -> [alert_ids]
        self.status_index = {}  # status -> [alert_ids]
        self.date_index = {}  # date -> [alert_ids]
        self.statistics = {
            'total_alerts': 0,
            'alerts_by_status': {},
            'alerts_by_severity': {},
            'alerts_by_company': {},
            'resolution_times': []
        }
    
    async def add_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Add alert to history"""
        
        try:
            alert_id = alert_data.get('id') or alert_data.get('alert_id')
            if not alert_id:
                logger.error("Alert ID is required")
                return False
            
            # Create historical alert record
            alert = HistoricalAlert(
                alert_id=alert_id,
                company_id=alert_data.get('company_id', 'Unknown'),
                title=alert_data.get('title', 'Untitled Alert'),
                description=alert_data.get('description', ''),
                severity=alert_data.get('severity', 'medium').lower(),
                factor=alert_data.get('factor', 'Unknown'),
                current_value=alert_data.get('current_value'),
                threshold_value=alert_data.get('threshold_value'),
                status=AlertStatus.ACTIVE,
                created_at=datetime.fromisoformat(alert_data.get('created_at', datetime.now().isoformat())),
                updated_at=datetime.now(),
                tags=alert_data.get('tags', [])
            )
            
            # Store alert
            self.alerts[alert_id] = alert
            
            # Update indexes
            await self._update_indexes(alert)
            
            # Update statistics
            self.statistics['total_alerts'] += 1
            
            status_count = self.statistics['alerts_by_status'].get(alert.status.value, 0)
            self.statistics['alerts_by_status'][alert.status.value] = status_count + 1
            
            severity_count = self.statistics['alerts_by_severity'].get(alert.severity, 0)
            self.statistics['alerts_by_severity'][alert.severity] = severity_count + 1
            
            company_count = self.statistics['alerts_by_company'].get(alert.company_id, 0)
            self.statistics['alerts_by_company'][alert.company_id] = company_count + 1
            
            logger.info(f"Added alert to history: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert to history: {e}")
            return False
    
    async def update_alert_status(self, alert_id: str, new_status: AlertStatus,
                                updated_by: str = None, notes: str = None) -> bool:
        """Update alert status"""
        
        try:
            if alert_id not in self.alerts:
                logger.error(f"Alert not found: {alert_id}")
                return False
            
            alert = self.alerts[alert_id]
            old_status = alert.status
            
            # Update status
            alert.status = new_status
            alert.updated_at = datetime.now()
            
            # Update specific status fields
            if new_status == AlertStatus.ACKNOWLEDGED:
                alert.acknowledged_at = datetime.now()
                alert.acknowledged_by = updated_by
            elif new_status == AlertStatus.RESOLVED:
                alert.resolved_at = datetime.now()
                alert.resolved_by = updated_by
                alert.resolution_notes = notes
                
                # Calculate resolution time
                if alert.created_at:
                    resolution_time = (alert.resolved_at - alert.created_at).total_seconds()
                    self.statistics['resolution_times'].append(resolution_time)
            
            # Update indexes
            await self._update_status_index(alert, old_status)
            
            # Update statistics
            old_status_count = self.statistics['alerts_by_status'].get(old_status.value, 0)
            self.statistics['alerts_by_status'][old_status.value] = max(0, old_status_count - 1)
            
            new_status_count = self.statistics['alerts_by_status'].get(new_status.value, 0)
            self.statistics['alerts_by_status'][new_status.value] = new_status_count + 1
            
            logger.info(f"Updated alert status: {alert_id} -> {new_status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert status: {e}")
            return False
    
    async def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """Get specific alert by ID"""
        
        try:
            if alert_id in self.alerts:
                return self.alerts[alert_id].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting alert: {e}")
            return None
    
    async def get_alerts_by_company(self, company_id: str, 
                                  limit: int = 50,
                                  status_filter: List[AlertStatus] = None) -> List[Dict[str, Any]]:
        """Get alerts for specific company"""
        
        try:
            company_alerts = self.company_index.get(company_id, [])
            
            # Apply status filter
            if status_filter:
                filtered_alerts = []
                for alert_id in company_alerts:
                    alert = self.alerts.get(alert_id)
                    if alert and alert.status in status_filter:
                        filtered_alerts.append(alert_id)
                company_alerts = filtered_alerts
            
            # Limit results and sort by creation time (newest first)
            company_alerts = company_alerts[:limit]
            
            # Convert to dictionaries
            result = []
            for alert_id in company_alerts:
                if alert_id in self.alerts:
                    result.append(self.alerts[alert_id].to_dict())
            
            # Sort by created_at descending
            result.sort(key=lambda x: x['created_at'], reverse=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting alerts by company: {e}")
            return []
    
    async def get_alerts_by_status(self, status: AlertStatus,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts by status"""
        
        try:
            status_alerts = self.status_index.get(status, [])
            status_alerts = status_alerts[:limit]
            
            result = []
            for alert_id in status_alerts:
                if alert_id in self.alerts:
                    result.append(self.alerts[alert_id].to_dict())
            
            # Sort by updated_at descending
            result.sort(key=lambda x: x['updated_at'], reverse=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting alerts by status: {e}")
            return []
    
    async def get_alerts_by_date_range(self, start_date: datetime, end_date: datetime,
                                     limit: int = 200) -> List[Dict[str, Any]]:
        """Get alerts within date range"""
        
        try:
            result = []
            
            for alert in self.alerts.values():
                if start_date <= alert.created_at <= end_date:
                    result.append(alert.to_dict())
            
            # Sort by created_at descending
            result.sort(key=lambda x: x['created_at'], reverse=True)
            
            return result[:limit]
            
        except Exception as e:
            logger.error(f"Error getting alerts by date range: {e}")
            return []
    
    async def search_alerts(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search alerts by title, description, or company ID"""
        
        try:
            query_lower = query.lower()
            matching_alerts = []
            
            for alert in self.alerts.values():
                if (query_lower in alert.title.lower() or
                    query_lower in alert.description.lower() or
                    query_lower in alert.company_id.lower() or
                    query_lower in alert.factor.lower()):
                    matching_alerts.append(alert.to_dict())
            
            # Sort by relevance (exact matches first, then by creation time)
            def relevance_score(alert_dict):
                score = 0
                if query_lower in alert_dict['title'].lower():
                    score += 3
                if query_lower in alert_dict['company_id'].lower():
                    score += 2
                if query_lower in alert_dict['description'].lower():
                    score += 1
                return score
            
            matching_alerts.sort(key=lambda x: (relevance_score(x), x['created_at']), reverse=True)
            
            return matching_alerts[:limit]
            
        except Exception as e:
            logger.error(f"Error searching alerts: {e}")
            return []
    
    async def get_alert_trends(self, days_back: int = 30) -> Dict[str, Any]:
        """Get alert trends for analytics"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Daily alert counts
            daily_counts = {}
            severity_trends = {'critical': {}, 'high': {}, 'medium': {}, 'low': {}}
            company_trends = {}
            
            for alert in self.alerts.values():
                if alert.created_at >= cutoff_date:
                    date_key = alert.created_at.strftime('%Y-%m-%d')
                    
                    # Daily counts
                    daily_counts[date_key] = daily_counts.get(date_key, 0) + 1
                    
                    # Severity trends
                    if alert.severity in severity_trends:
                        severity_trends[alert.severity][date_key] = severity_trends[alert.severity].get(date_key, 0) + 1
                    
                    # Company trends
                    if alert.company_id not in company_trends:
                        company_trends[alert.company_id] = {}
                    company_trends[alert.company_id][date_key] = company_trends[alert.company_id].get(date_key, 0) + 1
            
            # Calculate resolution rate
            resolved_alerts = len([a for a in self.alerts.values() 
                                 if a.created_at >= cutoff_date and a.status == AlertStatus.RESOLVED])
            total_alerts = len([a for a in self.alerts.values() if a.created_at >= cutoff_date])
            resolution_rate = (resolved_alerts / total_alerts * 100) if total_alerts > 0 else 0
            
            # Calculate average resolution time
            recent_resolution_times = [rt for rt in self.statistics['resolution_times'][-100:]]
            avg_resolution_time = sum(recent_resolution_times) / len(recent_resolution_times) if recent_resolution_times else 0
            
            return {
                'daily_counts': daily_counts,
                'severity_trends': severity_trends,
                'top_companies': dict(sorted(
                    [(k, sum(v.values())) for k, v in company_trends.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]),
                'resolution_rate': round(resolution_rate, 2),
                'average_resolution_time_hours': round(avg_resolution_time / 3600, 2),
                'total_alerts_period': total_alerts,
                'period_days': days_back
            }
            
        except Exception as e:
            logger.error(f"Error getting alert trends: {e}")
            return {}
    
    async def get_company_summary(self, company_id: str) -> Dict[str, Any]:
        """Get comprehensive alert summary for company"""
        
        try:
            company_alerts = [self.alerts[aid] for aid in self.company_index.get(company_id, [])
                            if aid in self.alerts]
            
            if not company_alerts:
                return {
                    'company_id': company_id,
                    'total_alerts': 0,
                    'active_alerts': 0,
                    'resolved_alerts': 0
                }
            
            # Calculate statistics
            total_alerts = len(company_alerts)
            active_alerts = len([a for a in company_alerts if a.status == AlertStatus.ACTIVE])
            resolved_alerts = len([a for a in company_alerts if a.status == AlertStatus.RESOLVED])
            
            # Severity breakdown
            severity_counts = {}
            for alert in company_alerts:
                severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            
            # Recent activity (last 30 days)
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_alerts = [a for a in company_alerts if a.created_at >= recent_cutoff]
            
            # Most common factors
            factor_counts = {}
            for alert in company_alerts:
                factor_counts[alert.factor] = factor_counts.get(alert.factor, 0) + 1
            
            top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'company_id': company_id,
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'resolved_alerts': resolved_alerts,
                'acknowledged_alerts': len([a for a in company_alerts if a.status == AlertStatus.ACKNOWLEDGED]),
                'severity_breakdown': severity_counts,
                'recent_alerts_30d': len(recent_alerts),
                'top_risk_factors': [{'factor': f, 'count': c} for f, c in top_factors],
                'first_alert': min([a.created_at for a in company_alerts]).isoformat() if company_alerts else None,
                'last_alert': max([a.created_at for a in company_alerts]).isoformat() if company_alerts else None
            }
            
        except Exception as e:
            logger.error(f"Error getting company summary: {e}")
            return {'error': str(e)}
    
    async def _update_indexes(self, alert: HistoricalAlert):
        """Update all indexes for alert"""
        
        try:
            # Company index
            if alert.company_id not in self.company_index:
                self.company_index[alert.company_id] = []
            if alert.alert_id not in self.company_index[alert.company_id]:
                self.company_index[alert.company_id].append(alert.alert_id)
            
            # Status index
            if alert.status not in self.status_index:
                self.status_index[alert.status] = []
            if alert.alert_id not in self.status_index[alert.status]:
                self.status_index[alert.status].append(alert.alert_id)
            
            # Date index
            date_key = alert.created_at.strftime('%Y-%m-%d')
            if date_key not in self.date_index:
                self.date_index[date_key] = []
            if alert.alert_id not in self.date_index[date_key]:
                self.date_index[date_key].append(alert.alert_id)
                
        except Exception as e:
            logger.error(f"Error updating indexes: {e}")
    
    async def _update_status_index(self, alert: HistoricalAlert, old_status: AlertStatus):
        """Update status index when alert status changes"""
        
        try:
            # Remove from old status index
            if old_status in self.status_index and alert.alert_id in self.status_index[old_status]:
                self.status_index[old_status].remove(alert.alert_id)
            
            # Add to new status index
            if alert.status not in self.status_index:
                self.status_index[alert.status] = []
            if alert.alert_id not in self.status_index[alert.status]:
                self.status_index[alert.status].append(alert.alert_id)
                
        except Exception as e:
            logger.error(f"Error updating status index: {e}")
    
    async def cleanup_old_alerts(self):
        """Remove alerts older than retention period"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
            
            alerts_to_remove = []
            for alert_id, alert in self.alerts.items():
                if alert.created_at < cutoff_date:
                    alerts_to_remove.append(alert_id)
            
            # Remove old alerts
            for alert_id in alerts_to_remove:
                alert = self.alerts[alert_id]
                
                # Remove from indexes
                if alert.company_id in self.company_index:
                    if alert_id in self.company_index[alert.company_id]:
                        self.company_index[alert.company_id].remove(alert_id)
                
                if alert.status in self.status_index:
                    if alert_id in self.status_index[alert.status]:
                        self.status_index[alert.status].remove(alert_id)
                
                date_key = alert.created_at.strftime('%Y-%m-%d')
                if date_key in self.date_index:
                    if alert_id in self.date_index[date_key]:
                        self.date_index[date_key].remove(alert_id)
                
                # Remove from main storage
                del self.alerts[alert_id]
            
            if alerts_to_remove:
                logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")
            
            return len(alerts_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")
            return 0
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get alert history statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Add current counts
            stats.update({
                'current_alerts': len(self.alerts),
                'max_history_days': self.max_history_days,
                'companies_monitored': len(self.company_index),
                'oldest_alert': min([a.created_at for a in self.alerts.values()]).isoformat() if self.alerts else None,
                'newest_alert': max([a.created_at for a in self.alerts.values()]).isoformat() if self.alerts else None
            })
            
            # Calculate average resolution time
            if self.statistics['resolution_times']:
                avg_resolution_hours = sum(self.statistics['resolution_times']) / len(self.statistics['resolution_times']) / 3600
                stats['average_resolution_time_hours'] = round(avg_resolution_hours, 2)
            else:
                stats['average_resolution_time_hours'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
