"""
Live feed for Stage 5 alerting workflows dashboard.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class FeedEventType(Enum):
    """Types of feed events"""
    ALERT_CREATED = "alert_created"
    ALERT_UPDATED = "alert_updated"
    ALERT_RESOLVED = "alert_resolved"
    SCORE_UPDATED = "score_updated"
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    NOTIFICATION_SENT = "notification_sent"

@dataclass
class FeedEvent:
    """Live feed event"""
    event_id: str
    event_type: FeedEventType
    timestamp: datetime
    company_id: str
    title: str
    description: str
    severity: str
    data: Dict[str, Any]
    source: str = "system"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

class LiveFeed:
    """Live feed for real-time alert and system events"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_events = config.get('max_events', 1000)
        self.retention_hours = config.get('retention_hours', 24)
        self.events = []
        self.subscribers = {}
        self.event_filters = {}
        self.statistics = {
            'total_events': 0,
            'events_by_type': {},
            'events_by_severity': {},
            'active_subscribers': 0
        }
    
    async def add_event(self, event_type: FeedEventType, company_id: str,
                       title: str, description: str, severity: str = 'medium',
                       data: Dict[str, Any] = None, source: str = 'system') -> str:
        """Add new event to live feed"""
        
        try:
            import uuid
            event_id = str(uuid.uuid4())
            
            event = FeedEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                company_id=company_id,
                title=title,
                description=description,
                severity=severity.lower(),
                data=data or {},
                source=source
            )
            
            # Add to events list
            self.events.insert(0, event)  # Insert at beginning for chronological order
            
            # Maintain max events limit
            if len(self.events) > self.max_events:
                self.events = self.events[:self.max_events]
            
            # Update statistics
            self.statistics['total_events'] += 1
            
            event_type_count = self.statistics['events_by_type'].get(event_type.value, 0)
            self.statistics['events_by_type'][event_type.value] = event_type_count + 1
            
            severity_count = self.statistics['events_by_severity'].get(severity, 0)
            self.statistics['events_by_severity'][severity] = severity_count + 1
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            logger.info(f"Added live feed event: {title} ({event_id})")
            return event_id
            
        except Exception as e:
            logger.error(f"Error adding live feed event: {e}")
            return None
    
    async def get_recent_events(self, limit: int = 50, 
                              event_types: List[FeedEventType] = None,
                              severity_filter: List[str] = None,
                              company_filter: List[str] = None,
                              hours_back: int = None) -> List[Dict[str, Any]]:
        """Get recent events with optional filters"""
        
        try:
            # Start with all events
            filtered_events = self.events.copy()
            
            # Apply time filter
            if hours_back:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                filtered_events = [e for e in filtered_events if e.timestamp >= cutoff_time]
            
            # Apply event type filter
            if event_types:
                filtered_events = [e for e in filtered_events if e.event_type in event_types]
            
            # Apply severity filter
            if severity_filter:
                severity_lower = [s.lower() for s in severity_filter]
                filtered_events = [e for e in filtered_events if e.severity in severity_lower]
            
            # Apply company filter
            if company_filter:
                filtered_events = [e for e in filtered_events if e.company_id in company_filter]
            
            # Limit results
            filtered_events = filtered_events[:limit]
            
            # Convert to dictionaries
            return [event.to_dict() for event in filtered_events]
            
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def subscribe(self, subscriber_id: str, callback: Callable,
                       event_types: List[FeedEventType] = None,
                       severity_filter: List[str] = None) -> bool:
        """Subscribe to live feed events"""
        
        try:
            subscription = {
                'callback': callback,
                'event_types': event_types,
                'severity_filter': [s.lower() for s in severity_filter] if severity_filter else None,
                'created_at': datetime.now()
            }
            
            self.subscribers[subscriber_id] = subscription
            self.statistics['active_subscribers'] = len(self.subscribers)
            
            logger.info(f"Added live feed subscriber: {subscriber_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to live feed: {e}")
            return False
    
    async def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from live feed events"""
        
        try:
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
                self.statistics['active_subscribers'] = len(self.subscribers)
                logger.info(f"Removed live feed subscriber: {subscriber_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error unsubscribing from live feed: {e}")
            return False
    
    async def _notify_subscribers(self, event: FeedEvent):
        """Notify all subscribers of new event"""
        
        try:
            for subscriber_id, subscription in self.subscribers.items():
                try:
                    # Check event type filter
                    if subscription['event_types'] and event.event_type not in subscription['event_types']:
                        continue
                    
                    # Check severity filter
                    if subscription['severity_filter'] and event.severity not in subscription['severity_filter']:
                        continue
                    
                    # Call subscriber callback
                    callback = subscription['callback']
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event.to_dict())
                    else:
                        callback(event.to_dict())
                        
                except Exception as e:
                    logger.error(f"Error notifying subscriber {subscriber_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying subscribers: {e}")
    
    async def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get specific event by ID"""
        
        try:
            for event in self.events:
                if event.event_id == event_id:
                    return event.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error getting event by ID: {e}")
            return None
    
    async def get_events_by_company(self, company_id: str, 
                                  limit: int = 20) -> List[Dict[str, Any]]:
        """Get events for specific company"""
        
        try:
            company_events = [e for e in self.events if e.company_id == company_id]
            company_events = company_events[:limit]
            return [event.to_dict() for event in company_events]
            
        except Exception as e:
            logger.error(f"Error getting events by company: {e}")
            return []
    
    async def get_event_timeline(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get event timeline for dashboard visualization"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
            
            # Group events by hour
            timeline = {}
            for event in recent_events:
                hour_key = event.timestamp.strftime('%Y-%m-%d %H:00')
                
                if hour_key not in timeline:
                    timeline[hour_key] = {
                        'total': 0,
                        'by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                        'by_type': {}
                    }
                
                timeline[hour_key]['total'] += 1
                
                # Count by severity
                severity = event.severity
                if severity in timeline[hour_key]['by_severity']:
                    timeline[hour_key]['by_severity'][severity] += 1
                
                # Count by type
                event_type = event.event_type.value
                type_count = timeline[hour_key]['by_type'].get(event_type, 0)
                timeline[hour_key]['by_type'][event_type] = type_count + 1
            
            # Sort timeline by hour
            sorted_timeline = dict(sorted(timeline.items()))
            
            return {
                'timeline': sorted_timeline,
                'total_events': len(recent_events),
                'time_range': f"{hours_back} hours",
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting event timeline: {e}")
            return {}
    
    async def cleanup_old_events(self):
        """Remove events older than retention period"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            
            initial_count = len(self.events)
            self.events = [e for e in self.events if e.timestamp >= cutoff_time]
            removed_count = initial_count - len(self.events)
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old events from live feed")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old events: {e}")
            return 0
    
    async def get_feed_statistics(self) -> Dict[str, Any]:
        """Get live feed statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Add current counts
            stats.update({
                'current_events': len(self.events),
                'max_events': self.max_events,
                'retention_hours': self.retention_hours,
                'oldest_event': self.events[-1].timestamp.isoformat() if self.events else None,
                'newest_event': self.events[0].timestamp.isoformat() if self.events else None
            })
            
            # Calculate event rates
            if self.events:
                time_span = (self.events[0].timestamp - self.events[-1].timestamp).total_seconds() / 3600
                if time_span > 0:
                    stats['events_per_hour'] = round(len(self.events) / time_span, 2)
                else:
                    stats['events_per_hour'] = 0
            else:
                stats['events_per_hour'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting feed statistics: {e}")
            return {'error': str(e)}
    
    async def export_events(self, format: str = 'json', 
                          hours_back: int = 24) -> Optional[str]:
        """Export events to file"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            export_events = [e for e in self.events if e.timestamp >= cutoff_time]
            
            if not export_events:
                logger.warning("No events to export")
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format.lower() == 'json':
                filename = f"live_feed_export_{timestamp}.json"
                data = [event.to_dict() for event in export_events]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
                
                logger.info(f"Exported {len(export_events)} events to {filename}")
                return filename
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Error exporting events: {e}")
            return None
