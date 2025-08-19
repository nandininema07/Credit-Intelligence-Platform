"""
Alert-related Pydantic models
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field

from .common import BaseModel, TimestampMixin, AlertSeverity, AlertStatus, TrendDirection

class AlertBase(BaseModel):
    """Base alert model"""
    company_id: int
    title: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=1, max_length=1000)
    severity: AlertSeverity
    alert_type: str = Field(..., max_length=50)  # "score_change", "threshold_breach", "anomaly"
    
class AlertCreate(AlertBase):
    """Model for creating an alert"""
    metadata: Optional[Dict[str, Any]] = None
    trigger_data: Optional[Dict[str, Any]] = None

class AlertUpdate(BaseModel):
    """Model for updating an alert"""
    status: Optional[AlertStatus] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

class Alert(AlertBase, TimestampMixin):
    """Complete alert model"""
    id: int
    status: AlertStatus = AlertStatus.ACTIVE
    
    # Score-related fields
    current_score: Optional[float] = None
    previous_score: Optional[float] = None
    score_change: Optional[float] = None
    threshold_breached: Optional[float] = None
    
    # Acknowledgment and resolution
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    # Metadata
    trigger_data: Optional[Dict[str, Any]] = None
    notification_sent: bool = False
    notification_channels: List[str] = Field(default_factory=list)
    
    # Deduplication
    hash_key: Optional[str] = None
    duplicate_count: int = 0
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None

class AlertResponse(Alert):
    """Alert response with additional computed fields"""
    company_name: str
    company_ticker: Optional[str] = None
    time_since_created: str  # "2 hours ago"
    is_overdue: bool = False
    related_alerts: List[int] = Field(default_factory=list)
    
class AlertRule(BaseModel):
    """Alert rule configuration"""
    id: Optional[int] = None
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    is_active: bool = True
    
    # Rule conditions
    condition_type: str  # "threshold", "change", "anomaly", "custom"
    conditions: Dict[str, Any]  # rule-specific conditions
    
    # Targeting
    company_ids: Optional[List[int]] = None
    industries: Optional[List[str]] = None
    sectors: Optional[List[str]] = None
    
    # Alert properties
    severity: AlertSeverity
    alert_type: str
    message_template: str
    
    # Notification settings
    notification_channels: List[str] = Field(default_factory=list)
    cooldown_minutes: int = Field(default=60, ge=0)
    max_alerts_per_day: int = Field(default=10, ge=1)

class AlertSummary(BaseModel):
    """Alert summary statistics"""
    total_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    by_severity: Dict[AlertSeverity, int]
    by_type: Dict[str, int]
    by_company: Dict[str, int]
    recent_alerts: List[AlertResponse]

class AlertFeed(BaseModel):
    """Real-time alert feed"""
    alerts: List[AlertResponse]
    last_updated: datetime
    has_new_alerts: bool
    unread_count: int
    filters_applied: Dict[str, Any]

class AlertNotification(BaseModel):
    """Alert notification model"""
    alert_id: int
    channel: str  # "email", "slack", "webhook", "sms"
    recipient: str
    status: str  # "pending", "sent", "failed", "delivered"
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

class AlertWorkflow(BaseModel):
    """Alert workflow configuration"""
    id: Optional[int] = None
    name: str
    trigger_conditions: Dict[str, Any]
    workflow_steps: List[Dict[str, Any]]
    is_active: bool = True
    
class AlertEscalation(BaseModel):
    """Alert escalation rules"""
    alert_id: int
    escalation_level: int
    escalated_to: str
    escalated_at: datetime
    reason: str
    auto_escalated: bool = False

class AlertMetrics(BaseModel):
    """Alert system metrics"""
    total_alerts_today: int
    avg_response_time_minutes: float
    resolution_rate_percent: float
    false_positive_rate: float
    top_alert_types: Dict[str, int]
    busiest_companies: Dict[str, int]
    performance_by_severity: Dict[AlertSeverity, Dict[str, float]]
