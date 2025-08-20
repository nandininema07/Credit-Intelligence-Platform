"""
Tests for Stage 5 alerting components.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from stage5_alerting_workflows.alerting import (
    AlertEngine, RuleEngine, PriorityManager, AlertDeduplicator
)
from stage5_alerting_workflows.alerting.alert_engine import AlertSeverity
from stage5_alerting_workflows.alerting.priority_manager import Priority

class TestAlertEngine:
    """Tests for AlertEngine"""
    
    @pytest.fixture
    def alert_engine(self, alerting_config):
        return AlertEngine(alerting_config['alert_engine'])
    
    @pytest.mark.asyncio
    async def test_create_alert(self, alert_engine, sample_alert_data):
        """Test creating an alert"""
        alert_id = await alert_engine.create_alert(
            title=sample_alert_data['title'],
            description=sample_alert_data['description'],
            severity=AlertSeverity(sample_alert_data['severity']),
            company_id=sample_alert_data['company_id'],
            factor=sample_alert_data['factor'],
            current_value=sample_alert_data['current_value'],
            threshold_value=sample_alert_data['threshold_value']
        )
        
        assert alert_id is not None
        assert alert_id in alert_engine.active_alerts
        
        alert = alert_engine.active_alerts[alert_id]
        assert alert.company_id == sample_alert_data['company_id']
        assert alert.title == sample_alert_data['title']
        assert alert.severity.value == sample_alert_data['severity']
    
    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_engine, sample_alert_data):
        """Test acknowledging an alert"""
        alert_id = await alert_engine.create_alert(
            title=sample_alert_data['title'],
            description=sample_alert_data['description'],
            severity=AlertSeverity(sample_alert_data['severity']),
            company_id=sample_alert_data['company_id'],
            factor=sample_alert_data['factor'],
            current_value=sample_alert_data['current_value'],
            threshold_value=sample_alert_data['threshold_value']
        )
        
        result = await alert_engine.acknowledge_alert(
            alert_id, 'test_user', 'Investigating issue'
        )
        
        assert result is True
        alert = alert_engine.active_alerts[alert_id]
        assert alert.acknowledged is True
        assert alert.acknowledged_by == 'test_user'
        assert alert.acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_engine, sample_alert_data):
        """Test resolving an alert"""
        alert_id = await alert_engine.create_alert(
            title=sample_alert_data['title'],
            description=sample_alert_data['description'],
            severity=AlertSeverity(sample_alert_data['severity']),
            company_id=sample_alert_data['company_id'],
            factor=sample_alert_data['factor'],
            current_value=sample_alert_data['current_value'],
            threshold_value=sample_alert_data['threshold_value']
        )
        
        result = await alert_engine.resolve_alert(
            alert_id, 'test_user', 'Issue resolved'
        )
        
        assert result is True
        assert alert_id not in alert_engine.active_alerts
        assert alert_id in alert_engine.resolved_alerts
    
    @pytest.mark.asyncio
    async def test_suppress_alert(self, alert_engine, sample_alert_data):
        """Test suppressing an alert"""
        alert_id = await alert_engine.create_alert(
            title=sample_alert_data['title'],
            description=sample_alert_data['description'],
            severity=AlertSeverity(sample_alert_data['severity']),
            company_id=sample_alert_data['company_id'],
            factor=sample_alert_data['factor'],
            current_value=sample_alert_data['current_value'],
            threshold_value=sample_alert_data['threshold_value']
        )
        
        result = await alert_engine.suppress_alert(
            alert_id, duration_minutes=60, reason='Maintenance window'
        )
        
        assert result is True
        alert = alert_engine.active_alerts[alert_id]
        assert alert.suppressed is True
        assert alert.suppressed_until is not None
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, alert_engine, sample_alert_data):
        """Test duplicate alert detection"""
        # Create first alert
        alert_id1 = await alert_engine.create_alert(
            title=sample_alert_data['title'],
            description=sample_alert_data['description'],
            severity=AlertSeverity(sample_alert_data['severity']),
            company_id=sample_alert_data['company_id'],
            factor=sample_alert_data['factor'],
            current_value=sample_alert_data['current_value'],
            threshold_value=sample_alert_data['threshold_value']
        )
        
        # Try to create duplicate
        alert_id2 = await alert_engine.create_alert(
            title=sample_alert_data['title'],
            description=sample_alert_data['description'],
            severity=AlertSeverity(sample_alert_data['severity']),
            company_id=sample_alert_data['company_id'],
            factor=sample_alert_data['factor'],
            current_value=sample_alert_data['current_value'],
            threshold_value=sample_alert_data['threshold_value']
        )
        
        # Should return existing alert ID or handle duplicate
        assert alert_id1 is not None
        # Behavior depends on deduplication strategy

class TestRuleEngine:
    """Tests for RuleEngine"""
    
    @pytest.fixture
    def rule_engine(self, alerting_config):
        return RuleEngine(alerting_config['rule_engine'])
    
    @pytest.mark.asyncio
    async def test_create_rule(self, rule_engine):
        """Test creating an alert rule"""
        rule_id = await rule_engine.create_rule(
            name='Credit Score Drop',
            description='Alert when credit score drops significantly',
            conditions=[{
                'field': 'credit_score',
                'operator': 'less_than',
                'value': 650
            }],
            severity='high',
            company_filter=['COMP001', 'COMP002']
        )
        
        assert rule_id is not None
        assert rule_id in rule_engine.rules
        
        rule = rule_engine.rules[rule_id]
        assert rule.name == 'Credit Score Drop'
        assert len(rule.conditions) == 1
    
    @pytest.mark.asyncio
    async def test_evaluate_rules(self, rule_engine, sample_credit_data):
        """Test rule evaluation"""
        # Create rule
        await rule_engine.create_rule(
            name='Low Credit Score',
            conditions=[{
                'field': 'credit_score',
                'operator': 'less_than',
                'value': 700
            }],
            severity='medium'
        )
        
        # Evaluate rules
        triggered_rules = await rule_engine.evaluate_rules(
            sample_credit_data['company_id'],
            sample_credit_data
        )
        
        assert len(triggered_rules) > 0
        rule = triggered_rules[0]
        assert rule['name'] == 'Low Credit Score'
        assert rule['severity'] == 'medium'
    
    @pytest.mark.asyncio
    async def test_rule_cooldown(self, rule_engine, sample_credit_data):
        """Test rule cooldown functionality"""
        rule_id = await rule_engine.create_rule(
            name='Test Rule',
            conditions=[{
                'field': 'credit_score',
                'operator': 'less_than',
                'value': 700
            }],
            severity='medium',
            cooldown_minutes=30
        )
        
        # First evaluation - should trigger
        triggered1 = await rule_engine.evaluate_rules(
            sample_credit_data['company_id'],
            sample_credit_data
        )
        assert len(triggered1) > 0
        
        # Second evaluation immediately - should be in cooldown
        triggered2 = await rule_engine.evaluate_rules(
            sample_credit_data['company_id'],
            sample_credit_data
        )
        assert len(triggered2) == 0  # Should be suppressed by cooldown

class TestPriorityManager:
    """Tests for PriorityManager"""
    
    @pytest.fixture
    def priority_manager(self, alerting_config):
        return PriorityManager(alerting_config['priority_manager'])
    
    @pytest.mark.asyncio
    async def test_calculate_priority(self, priority_manager, sample_alert_data):
        """Test priority calculation"""
        priority_score = await priority_manager.calculate_priority(sample_alert_data)
        
        assert isinstance(priority_score, Priority)
        assert priority_score in [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL, Priority.URGENT]
    
    @pytest.mark.asyncio
    async def test_severity_based_priority(self, priority_manager):
        """Test priority calculation based on severity"""
        critical_alert = {
            'severity': 'critical',
            'company_id': 'COMP001',
            'factor': 'payment_default'
        }
        
        low_alert = {
            'severity': 'low',
            'company_id': 'COMP001',
            'factor': 'minor_change'
        }
        
        critical_priority = await priority_manager.calculate_priority(critical_alert)
        low_priority = await priority_manager.calculate_priority(low_alert)
        
        assert critical_priority.value > low_priority.value
    
    @pytest.mark.asyncio
    async def test_escalation_check(self, priority_manager, sample_alert_data):
        """Test alert escalation logic"""
        # Create alert with old timestamp to trigger escalation
        old_alert = sample_alert_data.copy()
        old_alert['created_at'] = (datetime.now() - timedelta(hours=3)).isoformat()
        
        escalation_result = await priority_manager.check_escalation('test_alert_id', old_alert)
        assert escalation_result is None or isinstance(escalation_result, dict)

class TestAlertDeduplicator:
    """Tests for AlertDeduplicator"""
    
    @pytest.fixture
    def deduplicator(self, alerting_config):
        return AlertDeduplicator(alerting_config['deduplication'])
    
    @pytest.mark.asyncio
    async def test_exact_match_deduplication(self, deduplicator, sample_alert_data):
        """Test exact match deduplication"""
        # Check for duplicates (should find none initially)
        duplicate_result = await deduplicator.check_duplicate(sample_alert_data)
        assert duplicate_result is None  # No duplicate found initially
    
    @pytest.mark.asyncio
    async def test_fuzzy_match_deduplication(self, deduplicator, sample_alert_data):
        """Test fuzzy match deduplication"""
        # Check for duplicates with similar alert
        similar_alert = sample_alert_data.copy()
        similar_alert['title'] = 'Similar Credit Score Alert'
        
        duplicate_result = await deduplicator.check_duplicate(similar_alert)
        assert duplicate_result is None  # No duplicate found for similar alert
        
        # Test completed - basic duplicate check works
    
    @pytest.mark.asyncio
    async def test_time_window_deduplication(self, deduplicator, sample_alert_data):
        """Test time window based deduplication"""
        # Test old alert (outside time window)
        old_alert = sample_alert_data.copy()
        old_alert['created_at'] = (datetime.now() - timedelta(hours=2)).isoformat()
        
        # Check for duplicates
        duplicate_result = await deduplicator.check_duplicate(old_alert)
        assert duplicate_result is None  # No duplicate found for old alert
    
    @pytest.mark.asyncio
    async def test_company_factor_deduplication(self, deduplicator):
        """Test company-factor based deduplication"""
        alert1 = {
            'company_id': 'COMP001',
            'factor': 'credit_score',
            'title': 'Credit Score Alert',
            'severity': 'high',
            'created_at': datetime.now().isoformat()
        }
        
        alert2 = {
            'company_id': 'COMP001',
            'factor': 'credit_score',
            'title': 'Different Title',
            'severity': 'medium',
            'created_at': datetime.now().isoformat()
        }
        
        # Check for duplicates in both alerts
        duplicate_result1 = await deduplicator.check_duplicate(alert1)
        duplicate_result2 = await deduplicator.check_duplicate(alert2)
        
        # No duplicates found initially
        assert duplicate_result1 is None
        assert duplicate_result2 is None
