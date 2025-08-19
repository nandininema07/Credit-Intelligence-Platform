"""
Alert management service integrating Stage 5 alerting workflows
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta

from app.models.alert import (
    Alert, AlertCreate, AlertUpdate, AlertResponse, AlertRule,
    AlertSummary, AlertFeed, AlertMetrics, PaginatedResponse
)
from app.models.common import AlertSeverity, AlertStatus

# Import Stage 5 components
from stage5_alerting_workflows.monitoring.score_monitor import ScoreMonitor
from stage5_alerting_workflows.alerting.alert_engine import AlertEngine
from stage5_alerting_workflows.notifications.email_notifier import EmailNotifier
from stage5_alerting_workflows.notifications.slack_integration import SlackIntegration
from stage5_alerting_workflows.workflows.workflow_engine import WorkflowEngine
from stage5_alerting_workflows.workflows.jira_integration import JiraIntegration
from stage5_alerting_workflows.workflows.pdf_generator import PDFGenerator

logger = logging.getLogger(__name__)

class AlertService:
    """Service for alert management integrating Stage 5 workflows"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.score_monitor = ScoreMonitor()
        self.alert_engine = AlertEngine()
        self.email_notifier = EmailNotifier()
        self.slack_integration = SlackIntegration()
        self.workflow_engine = WorkflowEngine()
        self.jira_integration = JiraIntegration()
        self.pdf_generator = PDFGenerator()
    
    async def get_alerts(
        self,
        page: int = 1,
        size: int = 20,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        company_id: Optional[int] = None,
        alert_type: Optional[str] = None
    ) -> PaginatedResponse[AlertResponse]:
        """Get paginated alerts with filtering"""
        try:
            # Build query
            query = select(Alert)
            
            if status:
                query = query.where(Alert.status == status)
            if severity:
                query = query.where(Alert.severity == severity)
            if company_id:
                query = query.where(Alert.company_id == company_id)
            if alert_type:
                query = query.where(Alert.alert_type == alert_type)
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await self.db.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            offset = (page - 1) * size
            query = query.order_by(desc(Alert.created_at)).offset(offset).limit(size)
            
            result = await self.db.execute(query)
            alerts = result.scalars().all()
            
            # Enrich with additional data
            alert_responses = []
            for alert in alerts:
                response = await self._enrich_alert_data(alert)
                alert_responses.append(response)
            
            pages = (total + size - 1) // size
            
            return PaginatedResponse(
                items=alert_responses,
                total=total,
                page=page,
                size=size,
                pages=pages,
                has_next=page < pages,
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error fetching alerts: {str(e)}")
            raise
    
    async def get_alert_feed(
        self,
        limit: int = 50,
        since: Optional[str] = None
    ) -> AlertFeed:
        """Get real-time alert feed"""
        try:
            query = select(Alert).where(Alert.status == AlertStatus.ACTIVE)
            
            if since:
                since_datetime = datetime.fromisoformat(since.replace('Z', '+00:00'))
                query = query.where(Alert.created_at > since_datetime)
            
            query = query.order_by(desc(Alert.created_at)).limit(limit)
            
            result = await self.db.execute(query)
            alerts = result.scalars().all()
            
            # Enrich alerts
            enriched_alerts = []
            for alert in alerts:
                enriched = await self._enrich_alert_data(alert)
                enriched_alerts.append(enriched)
            
            # Check for new alerts since last check
            has_new_alerts = len(alerts) > 0
            unread_count = len([a for a in alerts if not a.acknowledged_at])
            
            return AlertFeed(
                alerts=enriched_alerts,
                last_updated=datetime.utcnow(),
                has_new_alerts=has_new_alerts,
                unread_count=unread_count,
                filters_applied={"status": "active", "limit": limit}
            )
            
        except Exception as e:
            logger.error(f"Error fetching alert feed: {str(e)}")
            raise
    
    async def get_alert_summary(self, days: int = 7) -> AlertSummary:
        """Get alert summary statistics"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get counts by status
            status_counts = {}
            for status in AlertStatus:
                count_query = select(func.count(Alert.id)).where(
                    and_(Alert.status == status, Alert.created_at >= cutoff_date)
                )
                result = await self.db.execute(count_query)
                status_counts[status.value] = result.scalar()
            
            # Get counts by severity
            severity_counts = {}
            for severity in AlertSeverity:
                count_query = select(func.count(Alert.id)).where(
                    and_(Alert.severity == severity, Alert.created_at >= cutoff_date)
                )
                result = await self.db.execute(count_query)
                severity_counts[severity] = result.scalar()
            
            # Get counts by type (mock data)
            type_counts = {
                'score_change': 45,
                'threshold_breach': 23,
                'anomaly_detection': 18,
                'regulatory_change': 12,
                'market_event': 8
            }
            
            # Get counts by company (mock data)
            company_counts = {
                'Tesla Inc.': 12,
                'Meta Platforms': 8,
                'Netflix Inc.': 6,
                'Apple Inc.': 5,
                'Microsoft Corp.': 4
            }
            
            # Get recent alerts
            recent_query = select(Alert).where(
                Alert.created_at >= cutoff_date
            ).order_by(desc(Alert.created_at)).limit(10)
            
            recent_result = await self.db.execute(recent_query)
            recent_alerts_raw = recent_result.scalars().all()
            
            recent_alerts = []
            for alert in recent_alerts_raw:
                enriched = await self._enrich_alert_data(alert)
                recent_alerts.append(enriched)
            
            total_alerts = sum(status_counts.values())
            
            return AlertSummary(
                total_alerts=total_alerts,
                active_alerts=status_counts.get('active', 0),
                acknowledged_alerts=status_counts.get('acknowledged', 0),
                resolved_alerts=status_counts.get('resolved', 0),
                by_severity=severity_counts,
                by_type=type_counts,
                by_company=company_counts,
                recent_alerts=recent_alerts
            )
            
        except Exception as e:
            logger.error(f"Error fetching alert summary: {str(e)}")
            raise
    
    async def get_alert_metrics(self) -> AlertMetrics:
        """Get alert system performance metrics"""
        try:
            today = datetime.utcnow().date()
            today_start = datetime.combine(today, datetime.min.time())
            
            # Count today's alerts
            today_count_query = select(func.count(Alert.id)).where(
                Alert.created_at >= today_start
            )
            today_result = await self.db.execute(today_count_query)
            total_alerts_today = today_result.scalar()
            
            # Mock metrics - would calculate from actual data
            return AlertMetrics(
                total_alerts_today=total_alerts_today,
                avg_response_time_minutes=12.5,
                resolution_rate_percent=87.3,
                false_positive_rate=0.08,
                top_alert_types={
                    'score_change': 45,
                    'threshold_breach': 23,
                    'anomaly_detection': 18
                },
                busiest_companies={
                    'Tesla Inc.': 12,
                    'Meta Platforms': 8,
                    'Netflix Inc.': 6
                },
                performance_by_severity={
                    AlertSeverity.CRITICAL: {'avg_response_min': 5.2, 'resolution_rate': 0.95},
                    AlertSeverity.HIGH: {'avg_response_min': 8.7, 'resolution_rate': 0.89},
                    AlertSeverity.MEDIUM: {'avg_response_min': 15.3, 'resolution_rate': 0.82},
                    AlertSeverity.LOW: {'avg_response_min': 45.1, 'resolution_rate': 0.76}
                }
            )
            
        except Exception as e:
            logger.error(f"Error fetching alert metrics: {str(e)}")
            raise
    
    async def get_alert_by_id(self, alert_id: int) -> Optional[AlertResponse]:
        """Get alert by ID"""
        try:
            query = select(Alert).where(Alert.id == alert_id)
            result = await self.db.execute(query)
            alert = result.scalar_one_or_none()
            
            if not alert:
                return None
            
            return await self._enrich_alert_data(alert)
            
        except Exception as e:
            logger.error(f"Error fetching alert {alert_id}: {str(e)}")
            raise
    
    async def create_alert(self, alert_data: AlertCreate) -> AlertResponse:
        """Create a new alert using Stage 5 alert engine"""
        try:
            # Use Stage 5 alert engine to create alert
            alert_dict = alert_data.dict()
            stage5_alert = await self.alert_engine.create_alert(
                company_id=alert_dict['company_id'],
                alert_type=alert_dict['alert_type'],
                severity=alert_dict['severity'],
                title=alert_dict['title'],
                message=alert_dict['message'],
                metadata=alert_dict.get('metadata', {}),
                trigger_data=alert_dict.get('trigger_data', {})
            )
            
            # Convert to database model
            alert = Alert(**alert_data.dict())
            alert.created_at = datetime.utcnow()
            alert.hash_key = stage5_alert.get('hash_key')
            
            self.db.add(alert)
            await self.db.commit()
            await self.db.refresh(alert)
            
            # Trigger notifications
            await self._send_alert_notifications(alert)
            
            return await self._enrich_alert_data(alert)
            
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            await self.db.rollback()
            raise
    
    async def update_alert(
        self,
        alert_id: int,
        alert_update: AlertUpdate
    ) -> Optional[AlertResponse]:
        """Update alert status or details"""
        try:
            query = select(Alert).where(Alert.id == alert_id)
            result = await self.db.execute(query)
            alert = result.scalar_one_or_none()
            
            if not alert:
                return None
            
            # Update fields
            update_data = alert_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(alert, field, value)
            
            alert.updated_at = datetime.utcnow()
            
            await self.db.commit()
            await self.db.refresh(alert)
            
            return await self._enrich_alert_data(alert)
            
        except Exception as e:
            logger.error(f"Error updating alert {alert_id}: {str(e)}")
            await self.db.rollback()
            raise
    
    async def acknowledge_alert(
        self,
        alert_id: int,
        acknowledged_by: str
    ) -> bool:
        """Acknowledge an alert"""
        try:
            query = select(Alert).where(Alert.id == alert_id)
            result = await self.db.execute(query)
            alert = result.scalar_one_or_none()
            
            if not alert:
                return False
            
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            await self.db.commit()
            
            # Send acknowledgment notification
            await self._send_acknowledgment_notification(alert, acknowledged_by)
            
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
            await self.db.rollback()
            raise
    
    async def resolve_alert(
        self,
        alert_id: int,
        resolved_by: str,
        resolution_notes: Optional[str] = None
    ) -> bool:
        """Resolve an alert"""
        try:
            query = select(Alert).where(Alert.id == alert_id)
            result = await self.db.execute(query)
            alert = result.scalar_one_or_none()
            
            if not alert:
                return False
            
            alert.status = AlertStatus.RESOLVED
            alert.resolved_by = resolved_by
            alert.resolved_at = datetime.utcnow()
            alert.resolution_notes = resolution_notes
            alert.updated_at = datetime.utcnow()
            
            await self.db.commit()
            
            # Send resolution notification
            await self._send_resolution_notification(alert, resolved_by)
            
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {str(e)}")
            await self.db.rollback()
            raise
    
    async def mute_alert(self, alert_id: int, duration_hours: int) -> bool:
        """Mute an alert for specified duration"""
        try:
            query = select(Alert).where(Alert.id == alert_id)
            result = await self.db.execute(query)
            alert = result.scalar_one_or_none()
            
            if not alert:
                return False
            
            alert.status = AlertStatus.MUTED
            alert.updated_at = datetime.utcnow()
            
            # Set unmute time in metadata
            unmute_time = datetime.utcnow() + timedelta(hours=duration_hours)
            if not alert.metadata:
                alert.metadata = {}
            alert.metadata['muted_until'] = unmute_time.isoformat()
            
            await self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error muting alert {alert_id}: {str(e)}")
            await self.db.rollback()
            raise
    
    async def share_alert(
        self,
        alert_id: int,
        format: str = "pdf",
        recipients: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Share/export alert in specified format"""
        try:
            alert = await self.get_alert_by_id(alert_id)
            if not alert:
                raise ValueError("Alert not found")
            
            if format == "pdf":
                # Generate PDF using Stage 5 PDF generator
                pdf_data = await self.pdf_generator.generate_alert_report(
                    alert_data=alert.dict(),
                    include_charts=True
                )
                
                if recipients:
                    # Send via email
                    await self.email_notifier.send_alert_pdf(
                        recipients=recipients,
                        alert_data=alert.dict(),
                        pdf_data=pdf_data
                    )
                
                return {
                    "format": "pdf",
                    "file_size": len(pdf_data),
                    "recipients_sent": len(recipients) if recipients else 0,
                    "download_url": f"/api/v1/alerts/{alert_id}/download/pdf"
                }
            
            elif format == "json":
                return {
                    "format": "json",
                    "data": alert.dict(),
                    "exported_at": datetime.utcnow().isoformat()
                }
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error sharing alert {alert_id}: {str(e)}")
            raise
    
    async def create_task_from_alert(
        self,
        alert_id: int,
        task_type: str = "jira",
        assignee: Optional[str] = None,
        creator: str = "system"
    ) -> Dict[str, Any]:
        """Create a task/ticket from an alert"""
        try:
            alert = await self.get_alert_by_id(alert_id)
            if not alert:
                raise ValueError("Alert not found")
            
            if task_type == "jira":
                # Create Jira ticket using Stage 5 integration
                ticket_result = await self.jira_integration.create_ticket_from_alert(
                    alert_data=alert.dict(),
                    assignee=assignee,
                    creator=creator
                )
                
                return {
                    "task_type": "jira",
                    "ticket_id": ticket_result.get('ticket_id'),
                    "ticket_url": ticket_result.get('ticket_url'),
                    "assignee": assignee,
                    "created_at": datetime.utcnow().isoformat()
                }
            
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
                
        except Exception as e:
            logger.error(f"Error creating task from alert {alert_id}: {str(e)}")
            raise
    
    async def get_company_alerts(
        self,
        company_id: int,
        status: Optional[AlertStatus] = None,
        limit: int = 20
    ) -> List[AlertResponse]:
        """Get alerts for a specific company"""
        try:
            query = select(Alert).where(Alert.company_id == company_id)
            
            if status:
                query = query.where(Alert.status == status)
            
            query = query.order_by(desc(Alert.created_at)).limit(limit)
            
            result = await self.db.execute(query)
            alerts = result.scalars().all()
            
            # Enrich alerts
            enriched_alerts = []
            for alert in alerts:
                enriched = await self._enrich_alert_data(alert)
                enriched_alerts.append(enriched)
            
            return enriched_alerts
            
        except Exception as e:
            logger.error(f"Error fetching alerts for company {company_id}: {str(e)}")
            raise
    
    async def get_alert_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        try:
            # Mock implementation - would query alert_rules table
            return [
                AlertRule(
                    id=1,
                    name="High Score Drop",
                    description="Alert when score drops more than 10 points",
                    is_active=True,
                    condition_type="change",
                    conditions={"threshold": -10, "timeframe": "1d"},
                    severity=AlertSeverity.HIGH,
                    alert_type="score_change",
                    message_template="Score dropped by {change}% for {company_name}",
                    notification_channels=["email", "slack"],
                    cooldown_minutes=60,
                    max_alerts_per_day=5
                )
            ]
            
        except Exception as e:
            logger.error(f"Error fetching alert rules: {str(e)}")
            raise
    
    async def create_alert_rule(self, rule: AlertRule) -> AlertRule:
        """Create a new alert rule"""
        try:
            # Mock implementation - would insert into alert_rules table
            rule.id = 1  # Mock ID
            logger.info(f"Created alert rule: {rule.name}")
            return rule
            
        except Exception as e:
            logger.error(f"Error creating alert rule: {str(e)}")
            raise
    
    async def _enrich_alert_data(self, alert: Alert) -> AlertResponse:
        """Enrich alert data with additional computed fields"""
        try:
            # Mock company data
            company_names = {
                1: 'Apple Inc.',
                2: 'Microsoft Corp.',
                3: 'Tesla Inc.',
                4: 'Amazon.com Inc.',
                5: 'Meta Platforms'
            }
            
            company_tickers = {
                1: 'AAPL',
                2: 'MSFT',
                3: 'TSLA', 
                4: 'AMZN',
                5: 'META'
            }
            
            # Calculate time since created
            time_diff = datetime.utcnow() - alert.created_at
            if time_diff.days > 0:
                time_since = f"{time_diff.days} days ago"
            elif time_diff.seconds > 3600:
                hours = time_diff.seconds // 3600
                time_since = f"{hours} hours ago"
            else:
                minutes = time_diff.seconds // 60
                time_since = f"{minutes} minutes ago"
            
            # Check if overdue (mock logic)
            is_overdue = (
                alert.status == AlertStatus.ACTIVE and
                alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH] and
                time_diff.total_seconds() > 3600  # 1 hour
            )
            
            response_data = alert.__dict__.copy()
            response_data.update({
                'company_name': company_names.get(alert.company_id, f'Company {alert.company_id}'),
                'company_ticker': company_tickers.get(alert.company_id),
                'time_since_created': time_since,
                'is_overdue': is_overdue,
                'related_alerts': []  # Would query for related alerts
            })
            
            return AlertResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Error enriching alert data: {str(e)}")
            raise
    
    async def _send_alert_notifications(self, alert: Alert):
        """Send notifications for new alert"""
        try:
            alert_data = alert.__dict__.copy()
            
            # Send email notification
            await self.email_notifier.send_alert_notification(alert_data)
            
            # Send Slack notification
            await self.slack_integration.send_alert_notification(alert_data)
            
            # Mark notification as sent
            alert.notification_sent = True
            alert.notification_channels = ["email", "slack"]
            
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Error sending alert notifications: {str(e)}")
    
    async def _send_acknowledgment_notification(self, alert: Alert, acknowledged_by: str):
        """Send acknowledgment notification"""
        try:
            await self.slack_integration.send_acknowledgment_notification(
                alert_data=alert.__dict__.copy(),
                acknowledged_by=acknowledged_by
            )
            
        except Exception as e:
            logger.error(f"Error sending acknowledgment notification: {str(e)}")
    
    async def _send_resolution_notification(self, alert: Alert, resolved_by: str):
        """Send resolution notification"""
        try:
            await self.slack_integration.send_resolution_notification(
                alert_data=alert.__dict__.copy(),
                resolved_by=resolved_by
            )
            
        except Exception as e:
            logger.error(f"Error sending resolution notification: {str(e)}")
