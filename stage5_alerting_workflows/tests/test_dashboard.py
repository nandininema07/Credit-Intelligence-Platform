"""
Tests for Stage 5 dashboard components.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from stage5_alerting_workflows.dashboard import (
    LiveFeed, FeedEventType, AlertHistory, AlertStatus, 
    MetricsCollector, MetricType
)

class TestLiveFeed:
    """Tests for LiveFeed"""
    
    @pytest.fixture
    def live_feed(self, dashboard_config):
        return LiveFeed(dashboard_config['live_feed'])
    
    @pytest.mark.asyncio
    async def test_add_event(self, live_feed):
        """Test adding event to live feed"""
        event_id = await live_feed.add_event(
            event_type=FeedEventType.ALERT_CREATED,
            company_id='COMP001',
            title='Test Alert',
            description='Test alert description',
            severity='high'
        )
        
        assert event_id is not None
        assert len(live_feed.events) == 1
        
        event = live_feed.events[0]
        assert event.event_type == FeedEventType.ALERT_CREATED
        assert event.company_id == 'COMP001'
        assert event.severity == 'high'
    
    @pytest.mark.asyncio
    async def test_get_recent_events(self, live_feed):
        """Test getting recent events"""
        # Add multiple events
        for i in range(5):
            await live_feed.add_event(
                event_type=FeedEventType.ALERT_CREATED,
                company_id=f'COMP{i+1:03d}',
                title=f'Alert {i+1}',
                description=f'Description {i+1}',
                severity='medium'
            )
        
        events = await live_feed.get_recent_events(limit=3)
        
        assert len(events) == 3
        assert all(isinstance(event, dict) for event in events)
        assert events[0]['title'] == 'Alert 5'  # Most recent first
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, live_feed):
        """Test event filtering"""
        # Add events with different types and severities
        await live_feed.add_event(
            FeedEventType.ALERT_CREATED, 'COMP001', 'Alert 1', 'Desc 1', 'high'
        )
        await live_feed.add_event(
            FeedEventType.ALERT_RESOLVED, 'COMP002', 'Alert 2', 'Desc 2', 'low'
        )
        await live_feed.add_event(
            FeedEventType.SCORE_UPDATED, 'COMP003', 'Score Update', 'Desc 3', 'medium'
        )
        
        # Filter by event type
        alert_events = await live_feed.get_recent_events(
            event_types=[FeedEventType.ALERT_CREATED]
        )
        assert len(alert_events) == 1
        assert alert_events[0]['title'] == 'Alert 1'
        
        # Filter by severity
        high_events = await live_feed.get_recent_events(
            severity_filter=['high']
        )
        assert len(high_events) == 1
        assert high_events[0]['severity'] == 'high'
    
    @pytest.mark.asyncio
    async def test_subscriber_notifications(self, live_feed):
        """Test subscriber notification system"""
        received_events = []
        
        async def test_callback(event):
            received_events.append(event)
        
        # Subscribe to events
        result = await live_feed.subscribe(
            'test_subscriber',
            test_callback,
            event_types=[FeedEventType.ALERT_CREATED]
        )
        assert result is True
        
        # Add event that should trigger notification
        await live_feed.add_event(
            FeedEventType.ALERT_CREATED,
            'COMP001',
            'Test Alert',
            'Test description',
            'high'
        )
        
        # Allow async notification to complete
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0]['title'] == 'Test Alert'
    
    @pytest.mark.asyncio
    async def test_get_event_timeline(self, live_feed):
        """Test event timeline generation"""
        # Add events over time
        base_time = datetime.now()
        for i in range(10):
            event_time = base_time - timedelta(hours=i)
            await live_feed.add_event(
                FeedEventType.ALERT_CREATED,
                f'COMP{i+1:03d}',
                f'Alert {i+1}',
                f'Description {i+1}',
                ['low', 'medium', 'high'][i % 3]
            )
            # Manually set timestamp for testing
            live_feed.events[0].timestamp = event_time
        
        timeline = await live_feed.get_event_timeline(hours_back=12)
        
        assert 'timeline' in timeline
        assert 'total_events' in timeline
        assert timeline['total_events'] > 0

class TestAlertHistory:
    """Tests for AlertHistory"""
    
    @pytest.fixture
    def alert_history(self, dashboard_config):
        return AlertHistory(dashboard_config['alert_history'])
    
    @pytest.mark.asyncio
    async def test_add_alert(self, alert_history, sample_alert_data):
        """Test adding alert to history"""
        result = await alert_history.add_alert(sample_alert_data)
        
        assert result is True
        assert sample_alert_data['id'] in alert_history.alerts
        
        alert = alert_history.alerts[sample_alert_data['id']]
        assert alert.company_id == sample_alert_data['company_id']
        assert alert.status == AlertStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_update_alert_status(self, alert_history, sample_alert_data):
        """Test updating alert status"""
        # Add alert first
        await alert_history.add_alert(sample_alert_data)
        
        # Update status
        result = await alert_history.update_alert_status(
            sample_alert_data['id'],
            AlertStatus.ACKNOWLEDGED,
            'test_user',
            'Investigating issue'
        )
        
        assert result is True
        alert = alert_history.alerts[sample_alert_data['id']]
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == 'test_user'
        assert alert.acknowledged_at is not None
    
    @pytest.mark.asyncio
    async def test_get_alerts_by_company(self, alert_history):
        """Test getting alerts by company"""
        from conftest import create_test_alerts
        alerts = create_test_alerts(5)
        
        # Add alerts for specific company
        company_alerts = []
        for i, alert in enumerate(alerts):
            if i < 3:
                alert['company_id'] = 'TARGET_COMP'
                company_alerts.append(alert)
            await alert_history.add_alert(alert)
        
        result = await alert_history.get_alerts_by_company('TARGET_COMP')
        
        assert len(result) == 3
        assert all(alert['company_id'] == 'TARGET_COMP' for alert in result)
    
    @pytest.mark.asyncio
    async def test_search_alerts(self, alert_history):
        """Test alert search functionality"""
        from conftest import create_test_alerts
        alerts = create_test_alerts(5)
        
        for alert in alerts:
            await alert_history.add_alert(alert)
        
        # Search by title
        results = await alert_history.search_alerts('Test Alert 1')
        assert len(results) == 1
        assert results[0]['title'] == 'Test Alert 1'
        
        # Search by company
        results = await alert_history.search_alerts('COMP001')
        assert len(results) == 1
        assert results[0]['company_id'] == 'COMP001'
    
    @pytest.mark.asyncio
    async def test_get_alert_trends(self, alert_history):
        """Test alert trend analysis"""
        from conftest import create_test_alerts
        alerts = create_test_alerts(10)
        
        for alert in alerts:
            await alert_history.add_alert(alert)
        
        trends = await alert_history.get_alert_trends(days_back=7)
        
        assert 'daily_counts' in trends
        assert 'severity_trends' in trends
        assert 'resolution_rate' in trends
        assert 'total_alerts_period' in trends

class TestMetricsCollector:
    """Tests for MetricsCollector"""
    
    @pytest.fixture
    def metrics_collector(self, dashboard_config):
        return MetricsCollector(dashboard_config['metrics_collector'])
    
    @pytest.mark.asyncio
    async def test_register_metric(self, metrics_collector):
        """Test registering a metric"""
        result = await metrics_collector.register_metric(
            name='test_counter',
            metric_type=MetricType.COUNTER,
            description='Test counter metric',
            unit='count'
        )
        
        assert result is True
        assert 'test_counter' in metrics_collector.metrics
        
        metric = metrics_collector.metrics['test_counter']
        assert metric.name == 'test_counter'
        assert metric.metric_type == MetricType.COUNTER
    
    @pytest.mark.asyncio
    async def test_record_value(self, metrics_collector):
        """Test recording metric values"""
        await metrics_collector.register_metric(
            'test_gauge', MetricType.GAUGE, 'Test gauge'
        )
        
        result = await metrics_collector.record_value('test_gauge', 42.5)
        
        assert result is True
        metric = metrics_collector.metrics['test_gauge']
        assert len(metric.data_points) == 1
        assert metric.data_points[0].value == 42.5
    
    @pytest.mark.asyncio
    async def test_increment_counter(self, metrics_collector):
        """Test incrementing counter metrics"""
        result = await metrics_collector.increment_counter('test_counter', 5.0)
        
        assert result is True
        assert 'test_counter' in metrics_collector.metrics
        
        metric = metrics_collector.metrics['test_counter']
        assert metric.data_points[-1].value == 5.0
        
        # Increment again
        await metrics_collector.increment_counter('test_counter', 3.0)
        assert metric.data_points[-1].value == 8.0
    
    @pytest.mark.asyncio
    async def test_set_gauge(self, metrics_collector):
        """Test setting gauge values"""
        result = await metrics_collector.set_gauge('cpu_usage', 75.5)
        
        assert result is True
        assert 'cpu_usage' in metrics_collector.metrics
        
        metric = metrics_collector.metrics['cpu_usage']
        assert metric.data_points[-1].value == 75.5
        
        # Set new value
        await metrics_collector.set_gauge('cpu_usage', 82.3)
        assert metric.data_points[-1].value == 82.3
    
    @pytest.mark.asyncio
    async def test_record_timer(self, metrics_collector):
        """Test recording timer metrics"""
        result = await metrics_collector.record_timer('api_response_time', 0.245)
        
        assert result is True
        assert 'api_response_time' in metrics_collector.metrics
        
        metric = metrics_collector.metrics['api_response_time']
        assert metric.metric_type == MetricType.TIMER
        assert metric.data_points[-1].value == 0.245
    
    @pytest.mark.asyncio
    async def test_get_metric_data(self, metrics_collector):
        """Test getting metric data with filtering"""
        await metrics_collector.register_metric(
            'test_metric', MetricType.GAUGE, 'Test metric'
        )
        
        # Add multiple data points
        for i in range(10):
            await metrics_collector.record_value('test_metric', i * 10)
        
        data = await metrics_collector.get_metric_data('test_metric', hours_back=24)
        
        assert data is not None
        assert data['name'] == 'test_metric'
        assert len(data['data_points']) == 10
        assert 'summary' in data
        assert data['summary']['count'] == 10
        assert data['summary']['min'] == 0
        assert data['summary']['max'] == 90
    
    @pytest.mark.asyncio
    async def test_aggregate_metrics(self, metrics_collector):
        """Test metric aggregation"""
        # Create multiple metrics
        for i in range(3):
            metric_name = f'metric_{i}'
            await metrics_collector.register_metric(
                metric_name, MetricType.GAUGE, f'Test metric {i}'
            )
            await metrics_collector.record_value(metric_name, (i + 1) * 10)
        
        # Test sum aggregation
        result = await metrics_collector.aggregate_metrics(
            ['metric_0', 'metric_1', 'metric_2'],
            aggregation='sum'
        )
        
        assert result is not None
        assert result['result'] == 60  # 10 + 20 + 30
        assert result['aggregation'] == 'sum'
    
    @pytest.mark.asyncio
    async def test_get_dashboard_metrics(self, metrics_collector):
        """Test getting dashboard metrics"""
        # Add some system metrics
        await metrics_collector.increment_counter('alerts_created_total', 25)
        await metrics_collector.increment_counter('alerts_resolved_total', 15)
        await metrics_collector.record_timer('processing_time_avg', 2.5)
        
        dashboard_data = await metrics_collector.get_dashboard_metrics()
        
        assert 'metrics' in dashboard_data
        assert 'last_updated' in dashboard_data
        assert isinstance(dashboard_data['metrics'], dict)
