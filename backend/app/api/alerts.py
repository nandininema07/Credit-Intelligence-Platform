"""
Alert management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from app.database import get_db
from app.models.alert import (
    Alert, AlertCreate, AlertUpdate, AlertResponse, AlertRule,
    AlertSummary, AlertFeed, AlertMetrics, PaginatedResponse
)
from app.models.common import AlertSeverity, AlertStatus
from app.services.alert_service import AlertService
from app.utils.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=PaginatedResponse[AlertResponse])
async def get_alerts(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    status: Optional[AlertStatus] = Query(None),
    severity: Optional[AlertSeverity] = Query(None),
    company_id: Optional[int] = Query(None),
    alert_type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get paginated list of alerts with filtering"""
    try:
        alert_service = AlertService(db)
        result = await alert_service.get_alerts(
            page=page,
            size=size,
            status=status,
            severity=severity,
            company_id=company_id,
            alert_type=alert_type
        )
        return result
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/feed", response_model=AlertFeed)
async def get_alert_feed(
    limit: int = Query(50, ge=1, le=100),
    since: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get real-time alert feed for dashboard"""
    try:
        alert_service = AlertService(db)
        feed = await alert_service.get_alert_feed(limit=limit, since=since)
        return feed
        
    except Exception as e:
        logger.error(f"Error fetching alert feed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/summary", response_model=AlertSummary)
async def get_alert_summary(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get alert summary statistics"""
    try:
        alert_service = AlertService(db)
        summary = await alert_service.get_alert_summary(days=days)
        return summary
        
    except Exception as e:
        logger.error(f"Error fetching alert summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics", response_model=AlertMetrics)
async def get_alert_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get alert system performance metrics"""
    try:
        alert_service = AlertService(db)
        metrics = await alert_service.get_alert_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching alert metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(
    alert_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get alert by ID"""
    try:
        alert_service = AlertService(db)
        alert = await alert_service.get_alert_by_id(alert_id)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        return alert
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/", response_model=AlertResponse)
async def create_alert(
    alert: AlertCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new alert"""
    try:
        alert_service = AlertService(db)
        new_alert = await alert_service.create_alert(alert)
        return new_alert
        
    except Exception as e:
        logger.error(f"Error creating alert: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{alert_id}", response_model=AlertResponse)
async def update_alert(
    alert_id: int = Path(..., ge=1),
    alert_update: AlertUpdate = ...,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update alert status or details"""
    try:
        alert_service = AlertService(db)
        updated_alert = await alert_service.update_alert(alert_id, alert_update)
        
        if not updated_alert:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        return updated_alert
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Acknowledge an alert"""
    try:
        alert_service = AlertService(db)
        success = await alert_service.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=current_user.get("username", "system")
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        return {"message": "Alert acknowledged successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{alert_id}/resolve")
async def resolve_alert(
    alert_id: int = Path(..., ge=1),
    resolution_notes: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Resolve an alert"""
    try:
        alert_service = AlertService(db)
        success = await alert_service.resolve_alert(
            alert_id=alert_id,
            resolved_by=current_user.get("username", "system"),
            resolution_notes=resolution_notes
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        return {"message": "Alert resolved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{alert_id}/mute")
async def mute_alert(
    alert_id: int = Path(..., ge=1),
    duration_hours: int = Query(24, ge=1, le=168),  # Max 1 week
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Mute an alert for specified duration"""
    try:
        alert_service = AlertService(db)
        success = await alert_service.mute_alert(
            alert_id=alert_id,
            duration_hours=duration_hours
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
            
        return {"message": f"Alert muted for {duration_hours} hours"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error muting alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{alert_id}/share")
async def share_alert(
    alert_id: int = Path(..., ge=1),
    format: str = Query("pdf", regex="^(pdf|json|email)$"),
    recipients: Optional[List[str]] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Share/export alert in specified format"""
    try:
        alert_service = AlertService(db)
        result = await alert_service.share_alert(
            alert_id=alert_id,
            format=format,
            recipients=recipients
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error sharing alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{alert_id}/create-task")
async def create_task_from_alert(
    alert_id: int = Path(..., ge=1),
    task_type: str = Query("jira", regex="^(jira|asana|trello)$"),
    assignee: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a task/ticket from an alert"""
    try:
        alert_service = AlertService(db)
        task_result = await alert_service.create_task_from_alert(
            alert_id=alert_id,
            task_type=task_type,
            assignee=assignee,
            creator=current_user.get("username", "system")
        )
        
        return task_result
        
    except Exception as e:
        logger.error(f"Error creating task from alert {alert_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/company/{company_id}/alerts", response_model=List[AlertResponse])
async def get_company_alerts(
    company_id: int = Path(..., ge=1),
    status: Optional[AlertStatus] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get alerts for a specific company"""
    try:
        alert_service = AlertService(db)
        alerts = await alert_service.get_company_alerts(
            company_id=company_id,
            status=status,
            limit=limit
        )
        return alerts
        
    except Exception as e:
        logger.error(f"Error fetching alerts for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/rules", response_model=List[AlertRule])
async def get_alert_rules(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all alert rules"""
    try:
        alert_service = AlertService(db)
        rules = await alert_service.get_alert_rules()
        return rules
        
    except Exception as e:
        logger.error(f"Error fetching alert rules: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/rules", response_model=AlertRule)
async def create_alert_rule(
    rule: AlertRule,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new alert rule"""
    try:
        alert_service = AlertService(db)
        new_rule = await alert_service.create_alert_rule(rule)
        return new_rule
        
    except Exception as e:
        logger.error(f"Error creating alert rule: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
