"""
Main alerting engine for Stage 5.
Handles real-time alerting, notifications, and workflow automation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, asdict
from enum import Enum

# Import existing Stage 5 components
from .alerting.alert_engine import AlertEngine
from .notifications.email_notifier import EmailNotifier
from .notifications.slack_integration import SlackIntegration
from .notifications.teams_integration import TeamsIntegration
from .workflows.jira_integration import JiraIntegration
from .workflows.pdf_generator import PDFGenerator
from .dashboard.live_feed import LiveFeed
from .dashboard.alert_history import AlertHistory

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Alert:
    id: str
    company: str
    alert_type: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]
    status: str = "active"

class AlertingEngine:
    """Main alerting and workflow engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.alert_tasks = []
        
        # Initialize components
        self.alert_engine = AlertEngine(config.get('alerting', {}))
        self.email_notifier = EmailNotifier(config.get('email', {}))
        self.slack_integration = SlackIntegration(config.get('slack', {}))
        self.teams_integration = TeamsIntegration(config.get('teams', {}))
        self.jira_integration = JiraIntegration(config.get('jira', {}))
        self.pdf_generator = PDFGenerator(config.get('pdf', {}))
        self.live_feed = LiveFeed(config.get('live_feed', {}))
        self.alert_history = AlertHistory(config.get('alert_history', {}))
        
        # Alert tracking
        self.active_alerts = {}
        self.alert_cooldowns = {}
        
    async def initialize(self):
        """Initialize the alerting engine"""
        try:
            # Initialize all components
            await self.alert_engine.initialize()
            await self.email_notifier.initialize()
            await self.slack_integration.initialize()
            await self.teams_integration.initialize()
            await self.jira_integration.initialize()
            await self.live_feed.initialize()
            await self.alert_history.initialize()
            
            logger.info("Stage 5 Alerting Engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Stage 5: {e}")
            raise
    
    async def start_engine(self):
        """Start the alerting engine"""
        if self.running:
            logger.warning("Alerting engine is already running")
            return
            
        try:
            self.running = True
            logger.info("Starting alerting engine...")
            
            # Start alerting tasks
            self.alert_tasks = [
                asyncio.create_task(self._alert_monitoring_loop()),
                asyncio.create_task(self._notification_processing_loop()),
                asyncio.create_task(self._workflow_processing_loop()),
                asyncio.create_task(self._cleanup_loop())
            ]
            
            await asyncio.gather(*self.alert_tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Alerting engine error: {e}")
            await self.stop_engine()
    
    async def stop_engine(self):
        """Stop the alerting engine"""
        self.running = False
        logger.info("Stopping alerting engine...")
        
        # Cancel all tasks
        for task in self.alert_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.alert_tasks:
            await asyncio.gather(*self.alert_tasks, return_exceptions=True)
            
        self.alert_tasks = []
        logger.info("Alerting engine stopped")
    
    async def _alert_monitoring_loop(self):
        """Main alert monitoring loop"""
        interval = self.config.get('alert_check_interval', 60)
        
        while self.running:
            try:
                # Check for new alerts from all sources
                await self._check_score_alerts()
                await self._check_system_alerts()
                await self._check_threshold_alerts()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _notification_processing_loop(self):
        """Process notification queue"""
        while self.running:
            try:
                # Process pending notifications
                await self._process_notification_queue()
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in notification processing: {e}")
                await asyncio.sleep(30)
    
    async def _workflow_processing_loop(self):
        """Process workflow automation"""
        while self.running:
            try:
                # Process workflow triggers
                await self._process_workflow_triggers()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in workflow processing: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Cleanup old alerts and data"""
        while self.running:
            try:
                await self._cleanup_old_alerts()
                await asyncio.sleep(3600)  # Run every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(1800)
    
    async def _check_score_alerts(self):
        """Check for credit score-based alerts"""
        try:
            # This would typically check recent score changes
            # For now, generate mock alerts based on thresholds
            severity_thresholds = self.config.get('severity_thresholds', {})
            
            # Mock score data
            companies = ['Apple Inc.', 'Microsoft Corporation', 'Tesla Inc.']
            for company in companies:
                # Simulate score check
                current_score = 650 + (hash(company) % 200)  # Mock score 650-850
                
                # Check thresholds
                if current_score < severity_thresholds.get('critical', 400):
                    await self._create_alert(company, 'score_critical', AlertSeverity.CRITICAL,
                                           f"Critical credit score: {current_score}")
                elif current_score < severity_thresholds.get('high', 500):
                    await self._create_alert(company, 'score_high', AlertSeverity.HIGH,
                                           f"High risk credit score: {current_score}")
                
        except Exception as e:
            logger.error(f"Error checking score alerts: {e}")
    
    async def _check_system_alerts(self):
        """Check for system health alerts"""
        try:
            # Check system metrics and health
            # For now, simulate occasional system alerts
            import random
            if random.random() < 0.01:  # 1% chance of system alert
                await self._create_alert("System", 'system_health', AlertSeverity.MEDIUM,
                                       "High memory usage detected in data processing pipeline")
                
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")
    
    async def _check_threshold_alerts(self):
        """Check for custom threshold alerts"""
        try:
            # Check custom business rules and thresholds
            # This would integrate with the alert engine's rule evaluation
            # For now, we'll skip this since we don't have company data to evaluate
            logger.debug("Threshold alerts check skipped - no company data available")
            
        except Exception as e:
            logger.error(f"Error checking threshold alerts: {e}")
    
    async def _create_alert(self, company: str, alert_type: str, 
                          severity: AlertSeverity, message: str, 
                          metadata: Dict[str, Any] = None):
        """Create and process a new alert"""
        try:
            # Check cooldown
            cooldown_key = f"{company}_{alert_type}"
            cooldown_minutes = self.config.get('cooldown_minutes', 60)
            
            if cooldown_key in self.alert_cooldowns:
                last_alert = self.alert_cooldowns[cooldown_key]
                if (datetime.now() - last_alert).total_seconds() < cooldown_minutes * 60:
                    return  # Skip due to cooldown
            
            # Create alert
            alert_id = f"{company}_{alert_type}_{int(datetime.now().timestamp())}"
            alert = Alert(
                id=alert_id,
                company=company,
                alert_type=alert_type,
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            # Store alert
            self.active_alerts[alert_id] = alert
            self.alert_cooldowns[cooldown_key] = datetime.now()
            
            # Add to history
            await self.alert_history.add_alert(asdict(alert))
            
            # Add to live feed
            await self.live_feed.add_event({
                'type': 'alert_created',
                'alert': asdict(alert),
                'timestamp': datetime.now().isoformat()
            })
            
            # Queue for notifications
            await self._queue_alert_notifications(alert)
            
            # Trigger workflows if needed
            await self._trigger_alert_workflows(alert)
            
            logger.info(f"Created alert: {alert_id} - {severity.value} - {message}")
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
    
    async def _queue_alert_notifications(self, alert: Alert):
        """Queue alert for notifications"""
        try:
            notification_channels = self.config.get('notification_channels', ['email'])
            
            for channel in notification_channels:
                if channel == 'email':
                    await self.email_notifier.send_alert_notification(alert)
                elif channel == 'slack':
                    await self.slack_integration.send_alert(alert)
                elif channel == 'teams':
                    await self.teams_integration.send_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error queuing notifications for alert {alert.id}: {e}")
    
    async def _trigger_alert_workflows(self, alert: Alert):
        """Trigger automated workflows for alert"""
        try:
            # Create Jira ticket for critical alerts
            if alert.severity == AlertSeverity.CRITICAL:
                await self.jira_integration.create_ticket_from_alert(alert)
            
            # Generate PDF report for high/critical alerts
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                await self.pdf_generator.generate_alert_report(alert)
                
        except Exception as e:
            logger.error(f"Error triggering workflows for alert {alert.id}: {e}")
    
    async def _process_notification_queue(self):
        """Process pending notifications"""
        try:
            # Process email queue
            await self.email_notifier.process_queue()
            
            # Process Slack queue
            await self.slack_integration.process_queue()
            
            # Process Teams queue
            await self.teams_integration.process_queue()
            
        except Exception as e:
            logger.error(f"Error processing notification queue: {e}")
    
    async def _process_workflow_triggers(self):
        """Process workflow automation triggers"""
        try:
            # Process Jira workflow queue
            await self.jira_integration.process_workflow_queue()
            
            # Process PDF generation queue
            await self.pdf_generator.process_generation_queue()
            
        except Exception as e:
            logger.error(f"Error processing workflow triggers: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old alerts and data"""
        try:
            # Remove old active alerts
            cutoff_time = datetime.now() - timedelta(hours=24)
            expired_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.timestamp < cutoff_time
            ]
            
            for alert_id in expired_alerts:
                del self.active_alerts[alert_id]
            
            # Clean up cooldowns
            cutoff_cooldown = datetime.now() - timedelta(hours=2)
            expired_cooldowns = [
                key for key, timestamp in self.alert_cooldowns.items()
                if timestamp < cutoff_cooldown
            ]
            
            for key in expired_cooldowns:
                del self.alert_cooldowns[key]
            
            # Clean up history
            await self.alert_history.cleanup_old_alerts()
            
            if expired_alerts or expired_cooldowns:
                logger.info(f"Cleaned up {len(expired_alerts)} alerts and {len(expired_cooldowns)} cooldowns")
                
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
    
    async def check_company_alerts(self, company_name: str, 
                                 score_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alerts specific to a company"""
        try:
            alerts_generated = []
            score = score_result.get('score', 0)
            
            # Check score-based alerts
            severity_thresholds = self.config.get('severity_thresholds', {})
            
            if score < severity_thresholds.get('critical', 400):
                alert = await self._create_alert(
                    company_name, 'score_critical', AlertSeverity.CRITICAL,
                    f"Critical credit score alert: {score}"
                )
                if alert:
                    alerts_generated.append(asdict(alert))
                    
            elif score < severity_thresholds.get('high', 500):
                alert = await self._create_alert(
                    company_name, 'score_high', AlertSeverity.HIGH,
                    f"High risk credit score alert: {score}"
                )
                if alert:
                    alerts_generated.append(asdict(alert))
            
            return alerts_generated
            
        except Exception as e:
            logger.error(f"Error checking company alerts for {company_name}: {e}")
            return []
    
    async def get_alerting_status(self) -> Dict[str, Any]:
        """Get current alerting engine status"""
        components_status = {}
        
        # Check component status safely
        for component_name, component in [
            ('email_notifier', self.email_notifier),
            ('slack_integration', self.slack_integration),
            ('teams_integration', self.teams_integration),
            ('jira_integration', self.jira_integration),
            ('live_feed', self.live_feed)
        ]:
            if hasattr(component, 'get_status'):
                try:
                    status = component.get_status()
                    if hasattr(status, '__await__'):
                        components_status[component_name] = await status
                    else:
                        components_status[component_name] = status
                except Exception as e:
                    components_status[component_name] = {'healthy': False, 'error': str(e)}
            else:
                components_status[component_name] = {'healthy': True, 'initialized': component is not None}
        
        return {
            'healthy': True,
            'running': self.running,
            'active_alerts': len(self.active_alerts),
            'cooldowns_active': len(self.alert_cooldowns),
            'notification_channels': self.config.get('notification_channels', []),
            'components_status': components_status
        }
    
    async def get_alert_feed(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts for the live feed"""
        try:
            return await self.live_feed.get_recent_events(limit)
        except Exception as e:
            logger.error(f"Error getting alert feed: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = 'acknowledged'
                alert.metadata['acknowledged_by'] = user
                alert.metadata['acknowledged_at'] = datetime.now().isoformat()
                
                await self.alert_history.update_alert_status(alert_id, 'acknowledged', user)
                logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, user: str, resolution: str = None) -> bool:
        """Resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = 'resolved'
                alert.metadata['resolved_by'] = user
                alert.metadata['resolved_at'] = datetime.now().isoformat()
                if resolution:
                    alert.metadata['resolution'] = resolution
                
                await self.alert_history.update_alert_status(alert_id, 'resolved', user)
                
                # Send resolution notifications
                await self._send_resolution_notifications(alert)
                
                logger.info(f"Alert {alert_id} resolved by {user}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def _send_resolution_notifications(self, alert: Alert):
        """Send notifications when alert is resolved"""
        try:
            notification_channels = self.config.get('notification_channels', [])
            
            for channel in notification_channels:
                if channel == 'slack':
                    await self.slack_integration.send_resolution_notification(alert)
                elif channel == 'teams':
                    await self.teams_integration.send_resolution_notification(alert)
                    
        except Exception as e:
            logger.error(f"Error sending resolution notifications: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.stop_engine()
            
            # Cleanup components
            if hasattr(self.email_notifier, 'cleanup'):
                await self.email_notifier.cleanup()
            if hasattr(self.slack_integration, 'cleanup'):
                await self.slack_integration.cleanup()
            if hasattr(self.live_feed, 'cleanup'):
                await self.live_feed.cleanup()
            if hasattr(self.alert_history, 'cleanup'):
                await self.alert_history.cleanup()
                
            self.active_alerts.clear()
            self.alert_cooldowns.clear()
            
            logger.info("Stage 5 cleanup completed")
        except Exception as e:
            logger.error(f"Error during Stage 5 cleanup: {e}")
