"""
Pytest configuration and shared fixtures for Stage 4 explainability tests.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
import asyncio

@pytest.fixture
def sample_credit_profile():
    """Sample credit profile for testing"""
    return {
        'user_id': 'test_user_123',
        'credit_score': 680,
        'factors': {
            'payment_history': 0.7,
            'credit_utilization': 0.6,
            'credit_length': 0.5,
            'credit_mix': 0.6,
            'new_credit': 0.8
        },
        'demographics': {
            'age': 35,
            'income': 65000,
            'employment_length': 5
        }
    }

@pytest.fixture
def sample_credit_data():
    """Sample credit dataset for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'payment_history': np.random.uniform(0.3, 1.0, 100),
        'credit_utilization': np.random.uniform(0.1, 0.9, 100),
        'credit_length': np.random.uniform(0.2, 0.8, 100),
        'credit_mix': np.random.uniform(0.3, 0.9, 100),
        'new_credit': np.random.uniform(0.4, 1.0, 100)
    })

@pytest.fixture
def mock_credit_model():
    """Mock credit scoring model for testing"""
    model = Mock()
    model.predict.return_value = np.random.uniform(300, 850, 100)
    model.feature_importances_ = np.array([0.35, 0.30, 0.15, 0.10, 0.10])
    return model

@pytest.fixture
def default_config():
    """Default configuration for testing"""
    return {
        'confidence_threshold': 0.7,
        'max_iterations': 100,
        'enable_caching': True,
        'enable_logging': False,
        'random_state': 42
    }

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def poor_credit_profile():
    """Sample poor credit profile for testing improvement scenarios"""
    return {
        'user_id': 'poor_credit_user',
        'credit_score': 550,
        'factors': {
            'payment_history': 0.4,
            'credit_utilization': 0.9,
            'credit_length': 0.2,
            'credit_mix': 0.3,
            'new_credit': 0.5
        }
    }

@pytest.fixture
def excellent_credit_profile():
    """Sample excellent credit profile for testing maintenance scenarios"""
    return {
        'user_id': 'excellent_credit_user',
        'credit_score': 820,
        'factors': {
            'payment_history': 0.95,
            'credit_utilization': 0.15,
            'credit_length': 0.85,
            'credit_mix': 0.8,
            'new_credit': 0.9
        }
    }
