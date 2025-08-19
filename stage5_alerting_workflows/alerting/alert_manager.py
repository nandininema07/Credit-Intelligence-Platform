"""
Alert manager for real-time credit risk alerts and notifications.
Monitors credit scores, risk changes, and triggers appropriate alerts.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Alert types"""
    CREDIT_SCORE_DROP = "credit_score_drop"
    RISK_CATEGORY_CHANGE = "risk_category_change"
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY_DETECTED = "anomaly_detected"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    MODEL_DRIFT = "model_drift"

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    company_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    acknowledged: bool = False
    
@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    alert_type: AlertType
    conditions: Dict[str, Any]
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 60

class AlertManager:
    """Real-time alert management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = []
        self.notification_callbacks = []
        self.cooldown_tracker = {}
        
        # Load default rules
        self._load_default_rules()
        
    def _load_default_rules(self):
        """Load default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="credit_score_drop_major",
                name="Major Credit Score Drop",
                alert_type=AlertType.CREDIT_SCORE_DROP,
                conditions={"score_drop_threshold": 50, "time_window_hours": 24},
                severity=AlertSeverity.HIGH
            ),
            AlertRule(
                rule_id="credit_score_drop_minor",
                name="Minor Credit Score Drop",
                alert_type=AlertType.CREDIT_SCORE_DROP,
                conditions={"score_drop_threshold": 20, "time_window_hours": 24},
                severity=AlertSeverity.MEDIUM
            ),
            AlertRule(
                rule_id="risk_category_upgrade",
                name="Risk Category Downgrade",
                alert_type=AlertType.RISK_CATEGORY_CHANGE,
                conditions={"direction": "downgrade"},
                severity=AlertSeverity.HIGH
            ),
            AlertRule(
                rule_id="critical_threshold_breach",
                name="Critical Score Threshold",
                alert_type=AlertType.THRESHOLD_BREACH,
                conditions={"threshold": 500, "direction": "below"},
                severity=AlertSeverity.CRITICAL
            ),
            AlertRule(
                rule_id="high_volatility",
                name="High Score Volatility",
                alert_type=AlertType.ANOMALY_DETECTED,
                conditions={"volatility_threshold": 0.15, "lookback_days": 7},
                severity=AlertSeverity.MEDIUM
            )
        ]
        
        self.alert_rules.extend(default_rules)
        logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        self.alert_rules = [r for r in self.alert_rules if r.rule_id != rule_id]
        logger.info(f"Removed alert rule: {rule_id}")
    
    def add_notification_callback(self, callback: Callable):
        """Add notification callback function"""
        self.notification_callbacks.append(callback)
    
    async def check_alerts(self, company_id: str, current_data: Dict[str, Any], 
                          historical_data: List[Dict[str, Any]] = None):
        """Check for alert conditions"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            # Check cooldown
            cooldown_key = f"{company_id}_{rule.rule_id}"
            if self._is_in_cooldown(cooldown_key, rule.cooldown_minutes):
                continue
            
            # Check rule conditions
            if await self._evaluate_rule(rule, company_id, current_data, historical_data):
                alert = await self._create_alert(rule, company_id, current_data)
                triggered_alerts.append(alert)
                
                # Set cooldown
                self.cooldown_tracker[cooldown_key] = datetime.now()
        
        # Process triggered alerts
        for alert in triggered_alerts:
            await self._process_alert(alert)
        
        return triggered_alerts
    
    def _is_in_cooldown(self, cooldown_key: str, cooldown_minutes: int) -> bool:
        """Check if rule is in cooldown period"""
        if cooldown_key not in self.cooldown_tracker:
            return False
        
        last_triggered = self.cooldown_tracker[cooldown_key]
        cooldown_end = last_triggered + timedelta(minutes=cooldown_minutes)
        
        return datetime.now() < cooldown_end
    
    async def _evaluate_rule(self, rule: AlertRule, company_id: str, 
                           current_data: Dict[str, Any], 
                           historical_data: List[Dict[str, Any]]) -> bool:
        """Evaluate if alert rule conditions are met"""
        
        try:
            if rule.alert_type == AlertType.CREDIT_SCORE_DROP:
                return await self._check_score_drop(rule, current_data, historical_data)
            
            elif rule.alert_type == AlertType.RISK_CATEGORY_CHANGE:
                return await self._check_risk_category_change(rule, current_data, historical_data)
            
            elif rule.alert_type == AlertType.THRESHOLD_BREACH:
                return await self._check_threshold_breach(rule, current_data)
            
            elif rule.alert_type == AlertType.ANOMALY_DETECTED:
                return await self._check_anomaly(rule, current_data, historical_data)
            
            elif rule.alert_type == AlertType.DATA_QUALITY_ISSUE:
                return await self._check_data_quality(rule, current_data)
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {str(e)}")
            return False
    
    async def _check_score_drop(self, rule: AlertRule, current_data: Dict[str, Any], 
                              historical_data: List[Dict[str, Any]]) -> bool:
        """Check for credit score drop"""
        if not historical_data:
            return False
        
        current_score = current_data.get('credit_score', 0)
        threshold = rule.conditions.get('score_drop_threshold', 50)
        time_window = rule.conditions.get('time_window_hours', 24)
        
        # Find recent historical score
        cutoff_time = datetime.now() - timedelta(hours=time_window)
        
        recent_scores = []
        for data in historical_data:
            data_time = datetime.fromisoformat(data.get('timestamp', ''))
            if data_time >= cutoff_time:
                recent_scores.append(data.get('credit_score', 0))
        
        if not recent_scores:
            return False
        
        max_recent_score = max(recent_scores)
        score_drop = max_recent_score - current_score
        
        return score_drop >= threshold
    
    async def _check_risk_category_change(self, rule: AlertRule, current_data: Dict[str, Any], 
                                        historical_data: List[Dict[str, Any]]) -> bool:
        """Check for risk category change"""
        if not historical_data:
            return False
        
        current_category = current_data.get('risk_category', '')
        direction = rule.conditions.get('direction', 'any')
        
        # Get most recent historical category
        if historical_data:
            previous_category = historical_data[-1].get('risk_category', '')
        else:
            return False
        
        if current_category == previous_category:
            return False
        
        # Define risk category hierarchy
        risk_hierarchy = {
            'Low Risk': 1,
            'Medium Risk': 2,
            'High Risk': 3,
            'Very High Risk': 4
        }
        
        current_level = risk_hierarchy.get(current_category, 0)
        previous_level = risk_hierarchy.get(previous_category, 0)
        
        if direction == 'downgrade':
            return current_level > previous_level
        elif direction == 'upgrade':
            return current_level < previous_level
        else:
            return current_level != previous_level
    
    async def _check_threshold_breach(self, rule: AlertRule, current_data: Dict[str, Any]) -> bool:
        """Check for threshold breach"""
        current_score = current_data.get('credit_score', 0)
        threshold = rule.conditions.get('threshold', 500)
        direction = rule.conditions.get('direction', 'below')
        
        if direction == 'below':
            return current_score < threshold
        elif direction == 'above':
            return current_score > threshold
        
        return False
    
    async def _check_anomaly(self, rule: AlertRule, current_data: Dict[str, Any], 
                           historical_data: List[Dict[str, Any]]) -> bool:
        """Check for anomalies in credit scores"""
        if not historical_data or len(historical_data) < 5:
            return False
        
        volatility_threshold = rule.conditions.get('volatility_threshold', 0.15)
        lookback_days = rule.conditions.get('lookback_days', 7)
        
        # Get recent scores
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        recent_scores = []
        
        for data in historical_data:
            data_time = datetime.fromisoformat(data.get('timestamp', ''))
            if data_time >= cutoff_time:
                recent_scores.append(data.get('credit_score', 0))
        
        if len(recent_scores) < 3:
            return False
        
        # Calculate volatility (coefficient of variation)
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)
        
        if mean_score == 0:
            return False
        
        volatility = std_score / mean_score
        return volatility > volatility_threshold
    
    async def _check_data_quality(self, rule: AlertRule, current_data: Dict[str, Any]) -> bool:
        """Check for data quality issues"""
        required_fields = rule.conditions.get('required_fields', [])
        missing_threshold = rule.conditions.get('missing_threshold', 0.2)
        
        if not required_fields:
            required_fields = ['credit_score', 'risk_category', 'features']
        
        missing_count = 0
        for field in required_fields:
            if field not in current_data or current_data[field] is None:
                missing_count += 1
        
        missing_ratio = missing_count / len(required_fields)
        return missing_ratio > missing_threshold
    
    async def _create_alert(self, rule: AlertRule, company_id: str, 
                          current_data: Dict[str, Any]) -> Alert:
        """Create alert from rule and data"""
        alert_id = f"{company_id}_{rule.rule_id}_{datetime.now().isoformat()}"
        
        # Generate alert message
        message = await self._generate_alert_message(rule, current_data)
        
        # Prepare alert details
        details = {
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'conditions': rule.conditions,
            'current_data': current_data
        }
        
        alert = Alert(
            alert_id=alert_id,
            company_id=company_id,
            alert_type=rule.alert_type,
            severity=rule.severity,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
        
        return alert
    
    async def _generate_alert_message(self, rule: AlertRule, current_data: Dict[str, Any]) -> str:
        """Generate human-readable alert message"""
        company_id = current_data.get('company_id', 'Unknown Company')
        current_score = current_data.get('credit_score', 0)
        risk_category = current_data.get('risk_category', 'Unknown')
        
        if rule.alert_type == AlertType.CREDIT_SCORE_DROP:
            threshold = rule.conditions.get('score_drop_threshold', 0)
            return f"Credit score for {company_id} has dropped by more than {threshold} points. Current score: {current_score:.0f}"
        
        elif rule.alert_type == AlertType.RISK_CATEGORY_CHANGE:
            return f"Risk category for {company_id} has changed to {risk_category}. Current score: {current_score:.0f}"
        
        elif rule.alert_type == AlertType.THRESHOLD_BREACH:
            threshold = rule.conditions.get('threshold', 0)
            return f"Credit score for {company_id} has breached threshold of {threshold}. Current score: {current_score:.0f}"
        
        elif rule.alert_type == AlertType.ANOMALY_DETECTED:
            return f"Unusual volatility detected in credit score for {company_id}. Current score: {current_score:.0f}"
        
        elif rule.alert_type == AlertType.DATA_QUALITY_ISSUE:
            return f"Data quality issues detected for {company_id}. Some required fields are missing."
        
        return f"Alert triggered for {company_id}: {rule.name}"
    
    async def _process_alert(self, alert: Alert):
        """Process and store alert"""
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        # Send notifications
        for callback in self.notification_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in notification callback: {str(e)}")
    
    def acknowledge_alert(self, alert_id: str, user_id: str = None) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {user_id or 'system'}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, user_id: str = None) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"Alert {alert_id} resolved by {user_id or 'system'}")
            return True
        return False
    
    def get_active_alerts(self, company_id: str = None, 
                         severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        alerts = list(self.active_alerts.values())
        
        # Filter by company
        if company_id:
            alerts = [a for a in alerts if a.company_id == company_id]
        
        # Filter by severity
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Filter unresolved
        alerts = [a for a in alerts if not a.resolved]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total_alerts = len(self.alert_history)
        active_count = len([a for a in self.active_alerts.values() if not a.resolved])
        
        # Count by severity
        severity_counts = {}
        for severity in AlertSeverity:
            count = len([a for a in self.alert_history if a.severity == severity])
            severity_counts[severity.value] = count
        
        # Count by type
        type_counts = {}
        for alert_type in AlertType:
            count = len([a for a in self.alert_history if a.alert_type == alert_type])
            type_counts[alert_type.value] = count
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_count,
            'severity_breakdown': severity_counts,
            'type_breakdown': type_counts,
            'rules_count': len(self.alert_rules)
        }
