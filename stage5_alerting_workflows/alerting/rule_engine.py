"""
Rule engine for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import re

logger = logging.getLogger(__name__)

class RuleType(Enum):
    """Types of alert rules"""
    THRESHOLD = "threshold"
    TREND = "trend"
    ANOMALY = "anomaly"
    COMPOSITE = "composite"
    SCHEDULE = "schedule"

class RuleOperator(Enum):
    """Rule operators"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    BETWEEN = "between"
    CONTAINS = "contains"
    MATCHES = "matches"

@dataclass
class RuleCondition:
    """Rule condition definition"""
    field: str
    operator: RuleOperator
    value: Any
    weight: float = 1.0

@dataclass
class AlertRule:
    """Alert rule definition"""
    id: str
    name: str
    description: str
    rule_type: RuleType
    enabled: bool
    conditions: List[RuleCondition]
    actions: List[str]
    company_ids: List[str]
    factors: List[str]
    severity: str
    cooldown_minutes: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class RuleEngine:
    """Engine for evaluating alert rules"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rules = {}
        self.rule_history = {}
        self.custom_functions = {}
        self.statistics = {
            'rules_evaluated': 0,
            'rules_triggered': 0,
            'rules_created': 0,
            'rules_updated': 0
        }
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize rule engine"""
        
        self.operators = {
            RuleOperator.GREATER_THAN: lambda x, y: x > y,
            RuleOperator.LESS_THAN: lambda x, y: x < y,
            RuleOperator.EQUAL: lambda x, y: x == y,
            RuleOperator.NOT_EQUAL: lambda x, y: x != y,
            RuleOperator.BETWEEN: lambda x, y: y[0] <= x <= y[1] if isinstance(y, (list, tuple)) and len(y) == 2 else False,
            RuleOperator.CONTAINS: lambda x, y: str(y) in str(x),
            RuleOperator.MATCHES: lambda x, y: bool(re.match(str(y), str(x)))
        }
        
        # Create default rules
        asyncio.create_task(self._create_default_rules())
    
    async def _create_default_rules(self):
        """Create default alert rules"""
        
        try:
            default_rules = [
                {
                    'id': 'credit_score_drop_major',
                    'name': 'Major Credit Score Drop',
                    'description': 'Alert when credit score drops significantly',
                    'rule_type': RuleType.THRESHOLD,
                    'conditions': [
                        RuleCondition('credit_score_change', RuleOperator.LESS_THAN, -15.0)
                    ],
                    'actions': ['create_alert', 'send_notification'],
                    'factors': ['credit_score'],
                    'severity': 'high',
                    'cooldown_minutes': 60
                },
                {
                    'id': 'payment_history_decline',
                    'name': 'Payment History Decline',
                    'description': 'Alert when payment history shows decline',
                    'rule_type': RuleType.TREND,
                    'conditions': [
                        RuleCondition('payment_history_trend', RuleOperator.LESS_THAN, -0.1)
                    ],
                    'actions': ['create_alert'],
                    'factors': ['payment_history'],
                    'severity': 'medium',
                    'cooldown_minutes': 120
                },
                {
                    'id': 'high_volatility',
                    'name': 'High Score Volatility',
                    'description': 'Alert when credit score shows high volatility',
                    'rule_type': RuleType.ANOMALY,
                    'conditions': [
                        RuleCondition('volatility_score', RuleOperator.GREATER_THAN, 3.0)
                    ],
                    'actions': ['create_alert'],
                    'factors': ['credit_score'],
                    'severity': 'medium',
                    'cooldown_minutes': 180
                }
            ]
            
            for rule_config in default_rules:
                await self.create_rule(
                    rule_id=rule_config['id'],
                    name=rule_config['name'],
                    description=rule_config['description'],
                    rule_type=rule_config['rule_type'],
                    conditions=rule_config['conditions'],
                    actions=rule_config['actions'],
                    factors=rule_config['factors'],
                    severity=rule_config['severity'],
                    cooldown_minutes=rule_config['cooldown_minutes']
                )
                
        except Exception as e:
            logger.error(f"Error creating default rules: {e}")
    
    async def create_rule(self, rule_id: str, name: str, description: str,
                         rule_type: RuleType, conditions: List[RuleCondition],
                         actions: List[str], factors: List[str], severity: str,
                         cooldown_minutes: int = 60, company_ids: List[str] = None,
                         metadata: Dict[str, Any] = None) -> AlertRule:
        """Create a new alert rule"""
        
        try:
            if rule_id in self.rules:
                raise ValueError(f"Rule {rule_id} already exists")
            
            rule = AlertRule(
                id=rule_id,
                name=name,
                description=description,
                rule_type=rule_type,
                enabled=True,
                conditions=conditions,
                actions=actions,
                company_ids=company_ids or [],
                factors=factors,
                severity=severity,
                cooldown_minutes=cooldown_minutes,
                metadata=metadata or {},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.rules[rule_id] = rule
            self.rule_history[rule_id] = []
            
            self.statistics['rules_created'] += 1
            
            logger.info(f"Created rule: {rule_id} - {name}")
            return rule
            
        except Exception as e:
            logger.error(f"Error creating rule: {e}")
            raise
    
    async def evaluate_rules(self, company_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all applicable rules against data"""
        
        try:
            triggered_rules = []
            
            for rule_id, rule in self.rules.items():
                if not rule.enabled:
                    continue
                
                # Check if rule applies to this company
                if rule.company_ids and company_id not in rule.company_ids:
                    continue
                
                # Check cooldown
                if await self._is_in_cooldown(rule_id, company_id):
                    continue
                
                # Evaluate rule
                result = await self._evaluate_single_rule(rule, company_id, data)
                
                if result['triggered']:
                    triggered_rules.append({
                        'rule_id': rule_id,
                        'rule_name': rule.name,
                        'rule_type': rule.rule_type.value,
                        'severity': rule.severity,
                        'actions': rule.actions,
                        'conditions_met': result['conditions_met'],
                        'score': result['score'],
                        'metadata': result['metadata']
                    })
                    
                    # Record rule trigger
                    await self._record_rule_trigger(rule_id, company_id, result)
                
                self.statistics['rules_evaluated'] += 1
            
            return triggered_rules
            
        except Exception as e:
            logger.error(f"Error evaluating rules: {e}")
            return []
    
    async def _evaluate_single_rule(self, rule: AlertRule, company_id: str, 
                                   data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single rule against data"""
        
        try:
            conditions_met = []
            total_score = 0.0
            total_weight = 0.0
            
            for condition in rule.conditions:
                condition_result = await self._evaluate_condition(condition, data)
                
                if condition_result['met']:
                    conditions_met.append({
                        'field': condition.field,
                        'operator': condition.operator.value,
                        'expected': condition.value,
                        'actual': condition_result['actual_value'],
                        'weight': condition.weight
                    })
                    
                    total_score += condition_result['score'] * condition.weight
                
                total_weight += condition.weight
            
            # Calculate overall score
            overall_score = total_score / total_weight if total_weight > 0 else 0.0
            
            # Determine if rule is triggered
            triggered = len(conditions_met) > 0
            
            # For composite rules, require all conditions
            if rule.rule_type == RuleType.COMPOSITE:
                triggered = len(conditions_met) == len(rule.conditions)
            
            result = {
                'triggered': triggered,
                'conditions_met': conditions_met,
                'score': overall_score,
                'metadata': {
                    'total_conditions': len(rule.conditions),
                    'conditions_met_count': len(conditions_met),
                    'evaluation_time': datetime.now().isoformat(),
                    'company_id': company_id
                }
            }
            
            if triggered:
                self.statistics['rules_triggered'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating single rule: {e}")
            return {'triggered': False, 'conditions_met': [], 'score': 0.0, 'metadata': {}}
    
    async def _evaluate_condition(self, condition: RuleCondition, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single condition"""
        
        try:
            # Get field value from data
            field_value = self._get_field_value(condition.field, data)
            
            if field_value is None:
                return {'met': False, 'score': 0.0, 'actual_value': None}
            
            # Apply operator
            operator_func = self.operators.get(condition.operator)
            if not operator_func:
                logger.error(f"Unknown operator: {condition.operator}")
                return {'met': False, 'score': 0.0, 'actual_value': field_value}
            
            condition_met = operator_func(field_value, condition.value)
            
            # Calculate condition score
            score = 1.0 if condition_met else 0.0
            
            # For numeric comparisons, calculate degree of satisfaction
            if condition_met and isinstance(field_value, (int, float)) and isinstance(condition.value, (int, float)):
                if condition.operator == RuleOperator.GREATER_THAN:
                    score = min(2.0, field_value / condition.value)
                elif condition.operator == RuleOperator.LESS_THAN:
                    score = min(2.0, condition.value / field_value) if field_value != 0 else 2.0
            
            return {
                'met': condition_met,
                'score': score,
                'actual_value': field_value
            }
            
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return {'met': False, 'score': 0.0, 'actual_value': None}
    
    def _get_field_value(self, field: str, data: Dict[str, Any]) -> Any:
        """Get field value from data, supporting nested fields"""
        
        try:
            # Support dot notation for nested fields
            if '.' in field:
                keys = field.split('.')
                value = data
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return None
                return value
            else:
                return data.get(field)
                
        except Exception as e:
            logger.error(f"Error getting field value: {e}")
            return None
    
    async def _is_in_cooldown(self, rule_id: str, company_id: str) -> bool:
        """Check if rule is in cooldown period"""
        
        try:
            if rule_id not in self.rule_history:
                return False
            
            rule = self.rules[rule_id]
            cooldown_period = timedelta(minutes=rule.cooldown_minutes)
            current_time = datetime.now()
            
            # Check recent triggers for this company
            recent_triggers = [
                trigger for trigger in self.rule_history[rule_id]
                if trigger['company_id'] == company_id and 
                current_time - trigger['timestamp'] < cooldown_period
            ]
            
            return len(recent_triggers) > 0
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False
    
    async def _record_rule_trigger(self, rule_id: str, company_id: str, result: Dict[str, Any]):
        """Record a rule trigger"""
        
        try:
            if rule_id not in self.rule_history:
                self.rule_history[rule_id] = []
            
            trigger_record = {
                'company_id': company_id,
                'timestamp': datetime.now(),
                'score': result['score'],
                'conditions_met': result['conditions_met'],
                'metadata': result['metadata']
            }
            
            self.rule_history[rule_id].append(trigger_record)
            
            # Limit history size
            max_history = 1000
            if len(self.rule_history[rule_id]) > max_history:
                self.rule_history[rule_id] = self.rule_history[rule_id][-max_history//2:]
                
        except Exception as e:
            logger.error(f"Error recording rule trigger: {e}")
    
    async def update_rule(self, rule_id: str, **updates) -> AlertRule:
        """Update an existing rule"""
        
        try:
            if rule_id not in self.rules:
                raise ValueError(f"Rule {rule_id} not found")
            
            rule = self.rules[rule_id]
            
            # Update allowed fields
            allowed_updates = [
                'name', 'description', 'enabled', 'conditions', 'actions',
                'company_ids', 'factors', 'severity', 'cooldown_minutes', 'metadata'
            ]
            
            for field, value in updates.items():
                if field in allowed_updates:
                    setattr(rule, field, value)
            
            rule.updated_at = datetime.now()
            self.statistics['rules_updated'] += 1
            
            logger.info(f"Updated rule: {rule_id}")
            return rule
            
        except Exception as e:
            logger.error(f"Error updating rule: {e}")
            raise
    
    async def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule"""
        
        try:
            if rule_id not in self.rules:
                return False
            
            del self.rules[rule_id]
            
            if rule_id in self.rule_history:
                del self.rule_history[rule_id]
            
            logger.info(f"Deleted rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting rule: {e}")
            return False
    
    async def enable_rule(self, rule_id: str) -> bool:
        """Enable a rule"""
        
        try:
            if rule_id not in self.rules:
                return False
            
            self.rules[rule_id].enabled = True
            self.rules[rule_id].updated_at = datetime.now()
            
            logger.info(f"Enabled rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error enabling rule: {e}")
            return False
    
    async def disable_rule(self, rule_id: str) -> bool:
        """Disable a rule"""
        
        try:
            if rule_id not in self.rules:
                return False
            
            self.rules[rule_id].enabled = False
            self.rules[rule_id].updated_at = datetime.now()
            
            logger.info(f"Disabled rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error disabling rule: {e}")
            return False
    
    def register_custom_function(self, name: str, func: Callable):
        """Register custom function for rule evaluation"""
        
        self.custom_functions[name] = func
        logger.info(f"Registered custom function: {name}")
    
    async def get_all_rules(self, enabled_only: bool = False) -> List[AlertRule]:
        """Get all rules"""
        
        try:
            rules = list(self.rules.values())
            
            if enabled_only:
                rules = [r for r in rules if r.enabled]
            
            return rules
            
        except Exception as e:
            logger.error(f"Error getting rules: {e}")
            return []
    
    async def get_rule_by_id(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID"""
        
        return self.rules.get(rule_id)
    
    async def get_rules_for_company(self, company_id: str) -> List[AlertRule]:
        """Get rules applicable to a company"""
        
        try:
            applicable_rules = []
            
            for rule in self.rules.values():
                if not rule.enabled:
                    continue
                
                if not rule.company_ids or company_id in rule.company_ids:
                    applicable_rules.append(rule)
            
            return applicable_rules
            
        except Exception as e:
            logger.error(f"Error getting company rules: {e}")
            return []
    
    async def get_rule_history(self, rule_id: str = None, company_id: str = None,
                              hours: int = 24) -> List[Dict[str, Any]]:
        """Get rule trigger history"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            history = []
            
            rule_ids = [rule_id] if rule_id else self.rule_history.keys()
            
            for rid in rule_ids:
                if rid not in self.rule_history:
                    continue
                
                for trigger in self.rule_history[rid]:
                    if trigger['timestamp'] < cutoff_time:
                        continue
                    
                    if company_id and trigger['company_id'] != company_id:
                        continue
                    
                    history.append({
                        'rule_id': rid,
                        'rule_name': self.rules[rid].name if rid in self.rules else 'Unknown',
                        **trigger
                    })
            
            # Sort by timestamp (most recent first)
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting rule history: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Add current counts
            stats.update({
                'total_rules': len(self.rules),
                'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
                'disabled_rules': sum(1 for r in self.rules.values() if not r.enabled),
                'custom_functions': len(self.custom_functions)
            })
            
            # Add rule type breakdown
            rule_types = {}
            for rule in self.rules.values():
                rule_type = rule.rule_type.value
                rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
            
            stats['rules_by_type'] = rule_types
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    async def export_rules(self) -> Dict[str, Any]:
        """Export rule configurations"""
        
        try:
            export_data = {
                'rules': {},
                'statistics': await self.get_statistics(),
                'exported_at': datetime.now().isoformat()
            }
            
            for rule_id, rule in self.rules.items():
                export_data['rules'][rule_id] = {
                    'id': rule.id,
                    'name': rule.name,
                    'description': rule.description,
                    'rule_type': rule.rule_type.value,
                    'enabled': rule.enabled,
                    'conditions': [
                        {
                            'field': c.field,
                            'operator': c.operator.value,
                            'value': c.value,
                            'weight': c.weight
                        }
                        for c in rule.conditions
                    ],
                    'actions': rule.actions,
                    'company_ids': rule.company_ids,
                    'factors': rule.factors,
                    'severity': rule.severity,
                    'cooldown_minutes': rule.cooldown_minutes,
                    'metadata': rule.metadata,
                    'created_at': rule.created_at.isoformat(),
                    'updated_at': rule.updated_at.isoformat()
                }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting rules: {e}")
            return {'error': str(e)}
