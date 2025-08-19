"""
Tests for Stage 3 serving components.
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from ..serving.model_api import ModelServingAPI, PredictionRequest
from ..serving.prediction_cache import PredictionCache, InMemoryCache
from ..serving.load_balancer import LoadBalancer

class TestModelServingAPI(unittest.TestCase):
    """Test model serving API"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(5)])
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=5, random_state=42)
        self.model.fit(self.X_df, self.y)
        
        self.config = {
            'host': '0.0.0.0',
            'port': 8080
        }
    
    def test_api_initialization(self):
        """Test API initialization"""
        api = ModelServingAPI(self.config)
        self.assertIsInstance(api, ModelServingAPI)
        self.assertIsNotNone(api.app)
    
    def test_prediction_request_model(self):
        """Test prediction request model"""
        features = {f'feature_{i}': float(self.X[0, i]) for i in range(5)}
        
        request = PredictionRequest(
            features=features,
            model_id='test_model',
            return_probabilities=True
        )
        
        self.assertEqual(request.model_id, 'test_model')
        self.assertTrue(request.return_probabilities)
        self.assertEqual(len(request.features), 5)

class TestPredictionCache(unittest.TestCase):
    """Test prediction caching"""
    
    def setUp(self):
        """Set up test cache"""
        self.config = {
            'cache_ttl': 300,
            'max_cache_size': 100,
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0
            }
        }
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = PredictionCache(self.config)
        self.assertIsInstance(cache, PredictionCache)
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        cache = PredictionCache(self.config)
        
        features = {'feature_1': 1.0, 'feature_2': 2.0}
        key1 = cache._generate_cache_key('model_1', features)
        key2 = cache._generate_cache_key('model_1', features)
        key3 = cache._generate_cache_key('model_2', features)
        
        # Same model and features should generate same key
        self.assertEqual(key1, key2)
        
        # Different model should generate different key
        self.assertNotEqual(key1, key3)
    
    def test_in_memory_cache(self):
        """Test in-memory cache functionality"""
        cache = InMemoryCache(max_size=10)
        
        # Test set and get
        cache.set('key1', {'prediction': 0.8}, 300)
        result = cache.get('key1')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['prediction'], 0.8)
        
        # Test non-existent key
        result = cache.get('non_existent')
        self.assertIsNone(result)
    
    def test_cache_eviction(self):
        """Test cache eviction policy"""
        cache = InMemoryCache(max_size=2)
        
        # Fill cache to capacity
        cache.set('key1', {'data': 1}, 300)
        cache.set('key2', {'data': 2}, 300)
        
        # Access key1 to make it more recently used
        cache.get('key1')
        
        # Add another item, should evict key2 (LRU)
        cache.set('key3', {'data': 3}, 300)
        
        self.assertIsNotNone(cache.get('key1'))
        self.assertIsNone(cache.get('key2'))
        self.assertIsNotNone(cache.get('key3'))

class TestLoadBalancer(unittest.TestCase):
    """Test load balancer functionality"""
    
    def setUp(self):
        """Set up test load balancer"""
        self.config = {
            'strategy': 'round_robin',
            'health_check_interval': 30
        }
    
    def test_load_balancer_initialization(self):
        """Test load balancer initialization"""
        lb = LoadBalancer(self.config)
        self.assertIsInstance(lb, LoadBalancer)
        self.assertEqual(lb.strategy, 'round_robin')
    
    def test_endpoint_management(self):
        """Test endpoint addition and removal"""
        lb = LoadBalancer(self.config)
        
        # Add endpoints
        lb.add_endpoint('endpoint1', 'http://localhost:8080', weight=1)
        lb.add_endpoint('endpoint2', 'http://localhost:8081', weight=2)
        
        self.assertEqual(len(lb.endpoints), 2)
        self.assertIn('endpoint1', lb.endpoints)
        self.assertIn('endpoint2', lb.endpoints)
        
        # Remove endpoint
        lb.remove_endpoint('endpoint1')
        self.assertEqual(len(lb.endpoints), 1)
        self.assertNotIn('endpoint1', lb.endpoints)
    
    def test_round_robin_selection(self):
        """Test round robin endpoint selection"""
        lb = LoadBalancer(self.config)
        
        # Add endpoints
        lb.add_endpoint('endpoint1', 'http://localhost:8080')
        lb.add_endpoint('endpoint2', 'http://localhost:8081')
        
        # Test round robin selection
        selections = []
        for _ in range(4):
            selected = lb.select_endpoint()
            selections.append(selected)
        
        # Should alternate between endpoints
        expected = ['endpoint1', 'endpoint2', 'endpoint1', 'endpoint2']
        self.assertEqual(selections, expected)
    
    def test_weighted_round_robin_selection(self):
        """Test weighted round robin selection"""
        config = {**self.config, 'strategy': 'weighted_round_robin'}
        lb = LoadBalancer(config)
        
        # Add endpoints with different weights
        lb.add_endpoint('endpoint1', 'http://localhost:8080', weight=1)
        lb.add_endpoint('endpoint2', 'http://localhost:8081', weight=2)
        
        # Test weighted selection
        selections = []
        for _ in range(6):
            selected = lb.select_endpoint()
            selections.append(selected)
        
        # endpoint2 should appear twice as often as endpoint1
        endpoint1_count = selections.count('endpoint1')
        endpoint2_count = selections.count('endpoint2')
        
        self.assertEqual(endpoint2_count, endpoint1_count * 2)
    
    def test_least_connections_selection(self):
        """Test least connections selection"""
        config = {**self.config, 'strategy': 'least_connections'}
        lb = LoadBalancer(config)
        
        # Add endpoints
        lb.add_endpoint('endpoint1', 'http://localhost:8080')
        lb.add_endpoint('endpoint2', 'http://localhost:8081')
        
        # Simulate different request counts
        lb.request_counts['endpoint1'] = 5
        lb.request_counts['endpoint2'] = 3
        
        # Should select endpoint with fewer connections
        selected = lb.select_endpoint()
        self.assertEqual(selected, 'endpoint2')
    
    def test_health_status_management(self):
        """Test health status management"""
        lb = LoadBalancer(self.config)
        
        # Add endpoints
        lb.add_endpoint('endpoint1', 'http://localhost:8080')
        lb.add_endpoint('endpoint2', 'http://localhost:8081')
        
        # Mark one endpoint as unhealthy
        lb.health_status['endpoint1']['healthy'] = False
        
        # Should only return healthy endpoints
        healthy_endpoints = lb.get_healthy_endpoints()
        self.assertEqual(len(healthy_endpoints), 1)
        self.assertIn('endpoint2', healthy_endpoints)
        self.assertNotIn('endpoint1', healthy_endpoints)
    
    def test_load_balancer_stats(self):
        """Test load balancer statistics"""
        lb = LoadBalancer(self.config)
        
        # Add endpoints
        lb.add_endpoint('endpoint1', 'http://localhost:8080')
        lb.add_endpoint('endpoint2', 'http://localhost:8081')
        
        # Simulate some activity
        lb.request_counts['endpoint1'] = 10
        lb.request_counts['endpoint2'] = 15
        lb.response_times['endpoint1'] = [100, 150, 120]
        lb.response_times['endpoint2'] = [200, 180, 190]
        
        stats = lb.get_load_balancer_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_endpoints'], 2)
        self.assertEqual(stats['healthy_endpoints'], 2)
        self.assertIn('endpoints', stats)
        
        # Check endpoint-specific stats
        endpoint1_stats = stats['endpoints']['endpoint1']
        self.assertEqual(endpoint1_stats['request_count'], 10)
        self.assertAlmostEqual(endpoint1_stats['avg_response_time'], 123.33, places=1)

if __name__ == '__main__':
    unittest.main()
