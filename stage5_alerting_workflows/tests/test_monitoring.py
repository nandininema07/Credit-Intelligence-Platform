"""
Tests for Stage 5 monitoring components.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from stage5_alerting_workflows.monitoring import (
    ScoreMonitor, ThresholdManager, AnomalyDetector, TrendMonitor
)

class TestScoreMonitor:
    """Tests for ScoreMonitor"""
    
    @pytest.fixture
    def score_monitor(self, monitoring_config):
        return ScoreMonitor(monitoring_config['score_monitor'])
    
    @pytest.mark.asyncio
    async def test_add_score(self, score_monitor, sample_credit_data):
        """Test adding credit score data"""
        result = await score_monitor.add_score(
            sample_credit_data['company_id'],
            sample_credit_data['credit_score'],
            sample_credit_data['factors']
        )
        
        assert result is True
        assert sample_credit_data['company_id'] in score_monitor.company_scores
        
        company_data = score_monitor.company_scores[sample_credit_data['company_id']]
        assert len(company_data['history']) == 1
        assert company_data['current_score'] == sample_credit_data['credit_score']
    
    @pytest.mark.asyncio
    async def test_detect_significant_change(self, score_monitor):
        """Test detection of significant score changes"""
        company_id = 'TEST_COMP'
        
        # Add initial score
        await score_monitor.add_score(company_id, 700, {})
        
        # Add score with significant change
        events = await score_monitor.add_score(company_id, 650, {})
        
        assert len(events) > 0
        event = events[0]
        assert event['event_type'] == 'significant_change'
        assert event['severity'] == 'high'
    
    @pytest.mark.asyncio
    async def test_volatility_detection(self, score_monitor):
        """Test volatility detection"""
        company_id = 'TEST_COMP'
        
        # Add multiple scores with high volatility
        scores = [700, 650, 720, 640, 710, 630]
        for score in scores:
            await score_monitor.add_score(company_id, score, {})
        
        company_data = score_monitor.company_scores[company_id]
        volatility = score_monitor._calculate_volatility(company_data['history'])
        
        assert volatility > 0.05  # Should detect high volatility
    
    @pytest.mark.asyncio
    async def test_get_company_status(self, score_monitor, sample_credit_data):
        """Test getting company status"""
        company_id = sample_credit_data['company_id']
        await score_monitor.add_score(
            company_id,
            sample_credit_data['credit_score'],
            sample_credit_data['factors']
        )
        
        status = await score_monitor.get_company_status(company_id)
        
        assert status is not None
        assert status['company_id'] == company_id
        assert status['current_score'] == sample_credit_data['credit_score']
        assert 'trend' in status
        assert 'volatility' in status

class TestThresholdManager:
    """Tests for ThresholdManager"""
    
    @pytest.fixture
    def threshold_manager(self, monitoring_config):
        return ThresholdManager(monitoring_config['threshold_manager'])
    
    @pytest.mark.asyncio
    async def test_create_threshold(self, threshold_manager):
        """Test creating a threshold"""
        threshold_id = await threshold_manager.create_threshold(
            company_id='TEST_COMP',
            factor='credit_score',
            threshold_type='absolute',
            value=650,
            severity='high',
            direction='below'
        )
        
        assert threshold_id is not None
        assert threshold_id in threshold_manager.thresholds
        
        threshold = threshold_manager.thresholds[threshold_id]
        assert threshold.company_id == 'TEST_COMP'
        assert threshold.factor == 'credit_score'
        assert threshold.value == 650
    
    @pytest.mark.asyncio
    async def test_evaluate_thresholds(self, threshold_manager):
        """Test threshold evaluation"""
        company_id = 'TEST_COMP'
        
        # Create threshold
        await threshold_manager.create_threshold(
            company_id=company_id,
            factor='credit_score',
            threshold_type='absolute',
            value=650,
            severity='high',
            direction='below'
        )
        
        # Test breach
        breaches = await threshold_manager.evaluate_thresholds(
            company_id, {'credit_score': 640}
        )
        
        assert len(breaches) == 1
        breach = breaches[0]
        assert breach['factor'] == 'credit_score'
        assert breach['severity'] == 'high'
        assert breach['breached'] is True
    
    @pytest.mark.asyncio
    async def test_cooldown_period(self, threshold_manager):
        """Test threshold cooldown period"""
        company_id = 'TEST_COMP'
        
        threshold_id = await threshold_manager.create_threshold(
            company_id=company_id,
            factor='credit_score',
            threshold_type='absolute',
            value=650,
            severity='high',
            direction='below'
        )
        
        # First evaluation - should breach
        breaches1 = await threshold_manager.evaluate_thresholds(
            company_id, {'credit_score': 640}
        )
        assert len(breaches1) == 1
        
        # Second evaluation immediately - should be in cooldown
        breaches2 = await threshold_manager.evaluate_thresholds(
            company_id, {'credit_score': 630}
        )
        assert len(breaches2) == 0  # Should be suppressed by cooldown

class TestAnomalyDetector:
    """Tests for AnomalyDetector"""
    
    @pytest.fixture
    def anomaly_detector(self, monitoring_config):
        return AnomalyDetector(monitoring_config['anomaly_detector'])
    
    @pytest.mark.asyncio
    async def test_add_data_point(self, anomaly_detector):
        """Test adding data points"""
        company_id = 'TEST_COMP'
        data = {'credit_score': 700, 'debt_ratio': 0.3}
        
        result = await anomaly_detector.add_data_point(company_id, data)
        assert result is True
        assert company_id in anomaly_detector.company_data
    
    @pytest.mark.asyncio
    async def test_statistical_anomaly_detection(self, anomaly_detector):
        """Test statistical anomaly detection"""
        company_id = 'TEST_COMP'
        
        # Add normal data points
        normal_scores = [700, 705, 695, 710, 690, 702, 698, 708, 692, 706]
        for score in normal_scores:
            await anomaly_detector.add_data_point(
                company_id, {'credit_score': score}
            )
        
        # Add anomalous data point
        anomalies = await anomaly_detector.add_data_point(
            company_id, {'credit_score': 500}  # Significant outlier
        )
        
        assert len(anomalies) > 0
        anomaly = anomalies[0]
        assert anomaly['anomaly_type'] == 'statistical'
        assert anomaly['severity'] in ['medium', 'high', 'critical']
    
    @pytest.mark.asyncio
    async def test_get_company_anomalies(self, anomaly_detector):
        """Test getting company anomalies"""
        company_id = 'TEST_COMP'
        
        # Add some data to trigger anomaly detection
        await anomaly_detector.add_data_point(
            company_id, {'credit_score': 700}
        )
        
        anomalies = await anomaly_detector.get_company_anomalies(company_id)
        assert isinstance(anomalies, list)

class TestTrendMonitor:
    """Tests for TrendMonitor"""
    
    @pytest.fixture
    def trend_monitor(self, monitoring_config):
        return TrendMonitor(monitoring_config['trend_monitor'])
    
    @pytest.mark.asyncio
    async def test_add_data_point(self, trend_monitor):
        """Test adding data points for trend analysis"""
        company_id = 'TEST_COMP'
        
        result = await trend_monitor.add_data_point(
            company_id, 'credit_score', 700
        )
        
        assert result is True
        assert company_id in trend_monitor.company_data
        assert 'credit_score' in trend_monitor.company_data[company_id]
    
    @pytest.mark.asyncio
    async def test_trend_detection(self, trend_monitor):
        """Test trend detection"""
        company_id = 'TEST_COMP'
        factor = 'credit_score'
        
        # Add declining trend data
        declining_scores = [700, 690, 680, 670, 660, 650, 640, 630]
        for score in declining_scores:
            await trend_monitor.add_data_point(company_id, factor, score)
        
        trends = await trend_monitor.analyze_trends(company_id, factor)
        
        assert 'short_term' in trends
        assert 'medium_term' in trends
        assert 'long_term' in trends
        
        # Should detect declining trend
        short_trend = trends['short_term']
        assert short_trend['direction'] == 'declining'
        assert short_trend['strength'] > 0.5
    
    @pytest.mark.asyncio
    async def test_trend_reversal_detection(self, trend_monitor):
        """Test trend reversal detection"""
        company_id = 'TEST_COMP'
        factor = 'credit_score'
        
        # Add data showing trend reversal
        scores = [700, 690, 680, 670, 680, 690, 700, 710]  # Decline then recovery
        for score in scores:
            await trend_monitor.add_data_point(company_id, factor, score)
        
        reversals = await trend_monitor.detect_trend_reversals(company_id, factor)
        
        # Should detect reversal from declining to improving
        assert len(reversals) > 0
        reversal = reversals[0]
        assert reversal['from_trend'] == 'declining'
        assert reversal['to_trend'] == 'improving'
    
    @pytest.mark.asyncio
    async def test_get_company_trends(self, trend_monitor):
        """Test getting comprehensive company trends"""
        company_id = 'TEST_COMP'
        
        # Add data for multiple factors
        await trend_monitor.add_data_point(company_id, 'credit_score', 700)
        await trend_monitor.add_data_point(company_id, 'debt_ratio', 0.3)
        
        trends = await trend_monitor.get_company_trends(company_id)
        
        assert 'company_id' in trends
        assert 'factors' in trends
        assert company_id == trends['company_id']
