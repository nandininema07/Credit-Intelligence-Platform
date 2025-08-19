"""
Pytest configuration and fixtures for Stage 5 alerting workflows tests.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
import tempfile
import os

# Test data fixtures
@pytest.fixture
def sample_alert_data():
    """Sample alert data for testing"""
    return {
        'id': 'alert_123',
        'company_id': 'COMP001',
        'title': 'Credit Score Decline',
        'description': 'Credit score has declined by 15 points',
        'severity': 'high',
        'factor': 'payment_history',
        'current_value': 650,
        'threshold_value': 700,
        'created_at': datetime.now().isoformat(),
        'status': 'active',
        'tags': ['automated', 'credit-score']
    }

@pytest.fixture
def sample_company_data():
    """Sample company data for testing"""
    return {
        'company_id': 'COMP001',
        'company_name': 'Test Company Inc.',
        'industry': 'Technology',
        'credit_score': 650,
        'risk_category': 'Medium Risk',
        'last_updated': datetime.now().isoformat(),
        'financial_metrics': {
            'debt_to_equity': 0.45,
            'current_ratio': 1.2,
            'revenue_growth': 0.08
        },
        'risk_factors': [
            'Recent payment delays',
            'Declining revenue trend'
        ],
        'recommendations': [
            'Monitor payment patterns',
            'Review credit terms'
        ]
    }

@pytest.fixture
def sample_credit_data():
    """Sample credit data for testing"""
    return {
        'company_id': 'COMP001',
        'timestamp': datetime.now(),
        'credit_score': 650,
        'factors': {
            'payment_history': 0.75,
            'debt_utilization': 0.60,
            'credit_age': 0.80,
            'credit_mix': 0.70,
            'new_credit': 0.65
        },
        'market_data': {
            'industry_avg_score': 680,
            'market_volatility': 0.15
        }
    }

@pytest.fixture
def monitoring_config():
    """Configuration for monitoring components"""
    return {
        'score_monitor': {
            'threshold_change': 10,
            'volatility_window': 30,
            'confidence_threshold': 0.8
        },
        'threshold_manager': {
            'default_cooldown_minutes': 60,
            'max_thresholds_per_company': 10
        },
        'anomaly_detector': {
            'sensitivity': 0.8,
            'min_data_points': 10,
            'lookback_days': 30
        },
        'trend_monitor': {
            'short_window': 7,
            'medium_window': 30,
            'long_window': 90
        }
    }

@pytest.fixture
def alerting_config():
    """Configuration for alerting components"""
    return {
        'alert_engine': {
            'max_active_alerts': 1000,
            'default_ttl_hours': 24
        },
        'rule_engine': {
            'max_rules': 100,
            'default_cooldown_minutes': 30
        },
        'priority_manager': {
            'escalation_time_minutes': 120,
            'max_priority_score': 100
        },
        'deduplication': {
            'time_window_minutes': 60,
            'similarity_threshold': 0.8
        }
    }

@pytest.fixture
def notification_config():
    """Configuration for notification components"""
    return {
        'email': {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'username': 'test@example.com',
            'password': 'test_password',
            'from_address': 'alerts@credtech.com'
        },
        'sms': {
            'provider': 'twilio',
            'account_sid': 'test_sid',
            'auth_token': 'test_token',
            'from_number': '+1234567890'
        },
        'webhook': {
            'default_timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 5
        },
        'slack': {
            'bot_token': 'xoxb-test-token',
            'default_channel': '#alerts'
        },
        'teams': {
            'webhook_urls': {
                'general': 'https://outlook.office.com/webhook/test'
            },
            'default_channel': 'general'
        }
    }

@pytest.fixture
def workflow_config():
    """Configuration for workflow components"""
    return {
        'workflow_engine': {
            'max_concurrent_workflows': 5,
            'default_timeout_minutes': 30
        },
        'jira': {
            'base_url': 'https://test.atlassian.net',
            'username': 'test@example.com',
            'api_token': 'test_token',
            'project_key': 'CRED'
        },
        'pdf_generator': {
            'output_directory': tempfile.mkdtemp(),
            'company_logo': ''
        },
        'export_manager': {
            'output_directory': tempfile.mkdtemp()
        }
    }

@pytest.fixture
def dashboard_config():
    """Configuration for dashboard components"""
    return {
        'live_feed': {
            'max_events': 1000,
            'retention_hours': 24
        },
        'alert_history': {
            'max_history_days': 90
        },
        'metrics_collector': {
            'retention_hours': 168,
            'max_points_per_metric': 10000
        }
    }

@pytest.fixture
def sample_workflow_tasks():
    """Sample workflow tasks for testing"""
    return [
        {
            'task_id': 'task_1',
            'name': 'Data Ingestion',
            'task_type': 'data_ingestion',
            'parameters': {
                'company_id': 'COMP001',
                'data_sources': ['news', 'financial']
            },
            'dependencies': [],
            'timeout_minutes': 10
        },
        {
            'task_id': 'task_2',
            'name': 'Feature Engineering',
            'task_type': 'feature_engineering',
            'parameters': {
                'company_id': 'COMP001'
            },
            'dependencies': ['task_1'],
            'timeout_minutes': 15
        },
        {
            'task_id': 'task_3',
            'name': 'Model Scoring',
            'task_type': 'model_scoring',
            'parameters': {
                'company_id': 'COMP001',
                'model_name': 'xgboost'
            },
            'dependencies': ['task_2'],
            'timeout_minutes': 5
        }
    ]

@pytest.fixture
def sample_summary_data():
    """Sample summary data for testing"""
    return {
        'date': '2024-01-15',
        'total_alerts': 25,
        'critical_alerts': 3,
        'high_alerts': 8,
        'medium_alerts': 10,
        'low_alerts': 4,
        'resolved_alerts': 15,
        'top_companies': [
            {'company_id': 'COMP001', 'alert_count': 5, 'primary_risk_factor': 'payment_history'},
            {'company_id': 'COMP002', 'alert_count': 3, 'primary_risk_factor': 'debt_utilization'},
            {'company_id': 'COMP003', 'alert_count': 2, 'primary_risk_factor': 'market_volatility'}
        ],
        'common_factors': ['payment_history', 'debt_utilization', 'market_volatility'],
        'trend': 5.2
    }

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def temp_directory():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

# Mock classes for external dependencies
class MockSMTPServer:
    """Mock SMTP server for email testing"""
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connected = False
        self.authenticated = False
        self.sent_messages = []
    
    def starttls(self):
        pass
    
    def login(self, username, password):
        self.authenticated = True
    
    def send_message(self, message):
        self.sent_messages.append(message)
        return {}
    
    def quit(self):
        self.connected = False

class MockHTTPResponse:
    """Mock HTTP response for API testing"""
    
    def __init__(self, status=200, json_data=None, text_data=""):
        self.status = status
        self._json_data = json_data or {}
        self._text_data = text_data
    
    async def json(self):
        return self._json_data
    
    async def text(self):
        return self._text_data

@pytest.fixture
def mock_http_session():
    """Mock aiohttp session for testing"""
    
    class MockSession:
        def __init__(self):
            self.requests = []
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def post(self, url, **kwargs):
            self.requests.append(('POST', url, kwargs))
            return MockHTTPResponse(status=200, json_data={'success': True})
        
        async def get(self, url, **kwargs):
            self.requests.append(('GET', url, kwargs))
            return MockHTTPResponse(status=200, json_data={'data': 'test'})
        
        async def put(self, url, **kwargs):
            self.requests.append(('PUT', url, kwargs))
            return MockHTTPResponse(status=204)
    
    return MockSession()

# Test utilities
def create_test_alerts(count: int = 5) -> List[Dict[str, Any]]:
    """Create multiple test alerts"""
    alerts = []
    for i in range(count):
        alert = {
            'id': f'alert_{i+1}',
            'company_id': f'COMP{i+1:03d}',
            'title': f'Test Alert {i+1}',
            'description': f'Test alert description {i+1}',
            'severity': ['low', 'medium', 'high', 'critical'][i % 4],
            'factor': 'test_factor',
            'current_value': 100 - (i * 10),
            'threshold_value': 80,
            'created_at': (datetime.now() - timedelta(hours=i)).isoformat(),
            'status': 'active'
        }
        alerts.append(alert)
    return alerts

def create_test_companies(count: int = 3) -> List[Dict[str, Any]]:
    """Create multiple test companies"""
    companies = []
    for i in range(count):
        company = {
            'company_id': f'COMP{i+1:03d}',
            'company_name': f'Test Company {i+1}',
            'industry': ['Technology', 'Finance', 'Healthcare'][i % 3],
            'credit_score': 700 - (i * 50),
            'risk_category': ['Low Risk', 'Medium Risk', 'High Risk'][i % 3],
            'last_updated': datetime.now().isoformat()
        }
        companies.append(company)
    return companies
