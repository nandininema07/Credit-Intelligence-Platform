"""
Priority management for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class Priority(Enum):
    """Alert priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

@dataclass
class PriorityRule:
    """Priority calculation rule"""
    id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    priority_boost: int
    enabled: bool
    weight: float

class PriorityManager:
    """Manage alert priorities and escalation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.priority_rules = {}
        self.escalation_rules = {}
        self.statistics = {
            'priorities_calculated': 0,
            'escalations_triggered': 0,
            'priority_changes': 0
        }
        self._initialize_manager()
    
    def _initialize_manager(self):
        """Initialize priority manager"""
        
        # Default priority rules
        default_rules = [
            {
                'id': 'critical_score_drop',
                'name': 'Critical Score Drop',
                'description': 'Boost priority for critical score drops',
                'conditions': {'severity': 'critical', 'factor': 'credit_score'},
                'priority_boost': 2,
                'weight': 1.0
            },
            {
                'id': 'multiple_factors',
                'name': 'Multiple Factors Affected',
                'description': 'Boost priority when multiple factors are affected',
                'conditions': {'factor_count': {'gt': 2}},
                'priority_boost': 1,
                'weight': 0.8
            },
            {
                'id': 'high_value_company',
                'name': 'High Value Company',
                'description': 'Boost priority for high-value companies',
                'conditions': {'company_tier': 'premium'},
                'priority_boost': 1,
                'weight': 0.6
            }
        ]
        
        for rule_config in default_rules:
            self.priority_rules[rule_config['id']] = PriorityRule(
                id=rule_config['id'],
                name=rule_config['name'],
                description=rule_config['description'],
                conditions=rule_config['conditions'],
                priority_boost=rule_config['priority_boost'],
                enabled=True,
                weight=rule_config['weight']
            )
    
    async def calculate_priority(self, alert_data: Dict[str, Any]) -> Priority:
        """Calculate alert priority based on rules"""
        
        try:
            base_priority = self._get_base_priority(alert_data.get('severity', 'medium'))
            priority_score = base_priority.value
            
            # Apply priority rules
            for rule_id, rule in self.priority_rules.items():
                if not rule.enabled:
                    continue
                
                if await self._evaluate_priority_rule(rule, alert_data):
                    boost = rule.priority_boost * rule.weight
                    priority_score += boost
                    logger.debug(f"Applied priority rule {rule_id}: +{boost}")
            
            # Cap priority score
            priority_score = min(Priority.URGENT.value, max(Priority.LOW.value, priority_score))
            
            # Convert back to Priority enum
            final_priority = Priority(int(priority_score))
            
            self.statistics['priorities_calculated'] += 1
            
            return final_priority
            
        except Exception as e:
            logger.error(f"Error calculating priority: {e}")
            return Priority.MEDIUM
    
    def _get_base_priority(self, severity: str) -> Priority:
        """Get base priority from severity"""
        
        severity_map = {
            'low': Priority.LOW,
            'medium': Priority.MEDIUM,
            'high': Priority.HIGH,
            'critical': Priority.CRITICAL
        }
        
        return severity_map.get(severity.lower(), Priority.MEDIUM)
    
    async def _evaluate_priority_rule(self, rule: PriorityRule, alert_data: Dict[str, Any]) -> bool:
        """Evaluate if a priority rule applies"""
        
        try:
            for condition_key, condition_value in rule.conditions.items():
                if condition_key not in alert_data:
                    return False
                
                actual_value = alert_data[condition_key]
                
                # Handle different condition types
                if isinstance(condition_value, dict):
                    # Complex condition (e.g., {'gt': 2})
                    if not self._evaluate_complex_condition(actual_value, condition_value):
                        return False
                else:
                    # Simple equality condition
                    if actual_value != condition_value:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating priority rule: {e}")
            return False
    
    def _evaluate_complex_condition(self, actual_value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate complex conditions"""
        
        try:
            for operator, expected_value in condition.items():
                if operator == 'gt' and actual_value <= expected_value:
                    return False
                elif operator == 'lt' and actual_value >= expected_value:
                    return False
                elif operator == 'gte' and actual_value < expected_value:
                    return False
                elif operator == 'lte' and actual_value > expected_value:
                    return False
                elif operator == 'eq' and actual_value != expected_value:
                    return False
                elif operator == 'ne' and actual_value == expected_value:
                    return False
                elif operator == 'in' and actual_value not in expected_value:
                    return False
                elif operator == 'contains' and expected_value not in str(actual_value):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating complex condition: {e}")
            return False
    
    async def check_escalation(self, alert_id: str, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if alert should be escalated"""
        
        try:
            current_time = datetime.now()
            created_time = alert_data.get('created_at', current_time)
            
            if isinstance(created_time, str):
                created_time = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
            
            age_minutes = (current_time - created_time).total_seconds() / 60
            
            # Escalation rules based on priority and age
            escalation_thresholds = {
                Priority.CRITICAL: 15,  # 15 minutes
                Priority.HIGH: 30,     # 30 minutes
                Priority.MEDIUM: 60,   # 1 hour
                Priority.LOW: 240      # 4 hours
            }
            
            current_priority = Priority(alert_data.get('priority', Priority.MEDIUM.value))
            threshold = escalation_thresholds.get(current_priority, 60)
            
            if age_minutes > threshold:
                escalation = {
                    'alert_id': alert_id,
                    'reason': 'time_threshold_exceeded',
                    'age_minutes': age_minutes,
                    'threshold_minutes': threshold,
                    'current_priority': current_priority.name,
                    'escalated_at': current_time,
                    'escalation_actions': self._get_escalation_actions(current_priority)
                }
                
                self.statistics['escalations_triggered'] += 1
                
                return escalation
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking escalation: {e}")
            return None
    
    def _get_escalation_actions(self, priority: Priority) -> List[str]:
        """Get escalation actions based on priority"""
        
        actions = {
            Priority.LOW: ['notify_manager'],
            Priority.MEDIUM: ['notify_manager', 'create_ticket'],
            Priority.HIGH: ['notify_manager', 'create_ticket', 'send_sms'],
            Priority.CRITICAL: ['notify_manager', 'create_ticket', 'send_sms', 'call_oncall'],
            Priority.URGENT: ['notify_manager', 'create_ticket', 'send_sms', 'call_oncall', 'page_executive']
        }
        
        return actions.get(priority, ['notify_manager'])
    
    async def update_priority(self, alert_id: str, new_priority: Priority, reason: str) -> bool:
        """Update alert priority"""
        
        try:
            # This would typically update the alert in the database
            # For now, just log the change
            
            logger.info(f"Updated priority for alert {alert_id} to {new_priority.name}: {reason}")
            self.statistics['priority_changes'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating priority: {e}")
            return False
    
    async def add_priority_rule(self, rule_id: str, name: str, description: str,
                              conditions: Dict[str, Any], priority_boost: int,
                              weight: float = 1.0) -> bool:
        """Add new priority rule"""
        
        try:
            rule = PriorityRule(
                id=rule_id,
                name=name,
                description=description,
                conditions=conditions,
                priority_boost=priority_boost,
                enabled=True,
                weight=weight
            )
            
            self.priority_rules[rule_id] = rule
            logger.info(f"Added priority rule: {rule_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding priority rule: {e}")
            return False
    
    async def remove_priority_rule(self, rule_id: str) -> bool:
        """Remove priority rule"""
        
        try:
            if rule_id in self.priority_rules:
                del self.priority_rules[rule_id]
                logger.info(f"Removed priority rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing priority rule: {e}")
            return False
    
    async def get_priority_rules(self) -> List[PriorityRule]:
        """Get all priority rules"""
        
        return list(self.priority_rules.values())
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get priority manager statistics"""
        
        try:
            stats = self.statistics.copy()
            stats.update({
                'total_priority_rules': len(self.priority_rules),
                'enabled_priority_rules': sum(1 for r in self.priority_rules.values() if r.enabled)
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
