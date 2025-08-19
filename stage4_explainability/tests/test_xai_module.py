"""
Tests for XAI module components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import asyncio

from stage4_explainability.xai.counterfactual_analysis import CounterfactualAnalyzer
from stage4_explainability.xai.global_explanations import GlobalExplainer

class TestCounterfactualAnalyzer:
    """Test cases for CounterfactualAnalyzer"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'methods': ['simple', 'genetic'],
            'max_iterations': 100,
            'population_size': 50
        }
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'payment_history': [0.8, 0.6, 0.9, 0.7],
            'credit_utilization': [0.3, 0.7, 0.2, 0.5],
            'credit_length': [0.6, 0.4, 0.8, 0.5],
            'credit_mix': [0.7, 0.5, 0.8, 0.6],
            'new_credit': [0.8, 0.6, 0.9, 0.7]
        })
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.predict.return_value = np.array([720, 650, 780, 680])
        return model
    
    def test_analyzer_initialization(self, sample_config):
        analyzer = CounterfactualAnalyzer(sample_config)
        assert analyzer.config == sample_config
        assert hasattr(analyzer, 'methods')
        assert hasattr(analyzer, 'feature_ranges')
    
    @pytest.mark.asyncio
    async def test_generate_counterfactuals(self, sample_config, sample_data, mock_model):
        analyzer = CounterfactualAnalyzer(sample_config)
        
        instance = sample_data.iloc[0].to_dict()
        target_outcome = 750
        
        result = await analyzer.generate_counterfactuals(
            instance, target_outcome, mock_model
        )
        
        assert 'counterfactuals' in result
        assert 'metadata' in result
        assert len(result['counterfactuals']) > 0
    
    @pytest.mark.asyncio
    async def test_simple_counterfactual_generation(self, sample_config, sample_data, mock_model):
        analyzer = CounterfactualAnalyzer(sample_config)
        
        instance = sample_data.iloc[0].to_dict()
        target_outcome = 750
        
        counterfactual = await analyzer._generate_simple_counterfactual(
            instance, target_outcome, mock_model
        )
        
        assert counterfactual is not None
        assert 'changes_made' in counterfactual
        assert 'predicted_outcome' in counterfactual
    
    def test_calculate_distance(self, sample_config):
        analyzer = CounterfactualAnalyzer(sample_config)
        
        original = {'feature1': 0.5, 'feature2': 0.3}
        counterfactual = {'feature1': 0.7, 'feature2': 0.4}
        
        distance = analyzer._calculate_distance(original, counterfactual)
        assert isinstance(distance, float)
        assert distance >= 0
    
    def test_validate_counterfactual(self, sample_config):
        analyzer = CounterfactualAnalyzer(sample_config)
        
        original = {'payment_history': 0.6, 'credit_utilization': 0.8}
        counterfactual = {'payment_history': 0.8, 'credit_utilization': 0.4}
        
        is_valid = analyzer._validate_counterfactual(original, counterfactual)
        assert isinstance(is_valid, bool)

class TestGlobalExplainer:
    """Test cases for GlobalExplainer"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'importance_methods': ['model_based', 'permutation'],
            'n_permutations': 50,
            'random_state': 42
        }
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'payment_history': np.random.uniform(0, 1, 100),
            'credit_utilization': np.random.uniform(0, 1, 100),
            'credit_length': np.random.uniform(0, 1, 100),
            'credit_mix': np.random.uniform(0, 1, 100),
            'new_credit': np.random.uniform(0, 1, 100)
        })
    
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.predict.return_value = np.random.uniform(300, 850, 100)
        return model
    
    def test_explainer_initialization(self, sample_config):
        explainer = GlobalExplainer(sample_config)
        assert explainer.config == sample_config
        assert hasattr(explainer, 'importance_methods')
    
    @pytest.mark.asyncio
    async def test_explain_global(self, sample_config, sample_data, mock_model):
        explainer = GlobalExplainer(sample_config)
        
        result = await explainer.explain_global(sample_data, mock_model)
        
        assert 'importance_methods' in result
        assert 'consensus_ranking' in result
        assert 'feature_interactions' in result
    
    @pytest.mark.asyncio
    async def test_model_based_importance(self, sample_config, sample_data, mock_model):
        explainer = GlobalExplainer(sample_config)
        
        # Mock model with feature_importances_
        mock_model.feature_importances_ = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        importance = await explainer._calculate_model_based_importance(sample_data, mock_model)
        
        assert isinstance(importance, dict)
        assert len(importance) == len(sample_data.columns)
    
    @pytest.mark.asyncio
    async def test_permutation_importance(self, sample_config, sample_data, mock_model):
        explainer = GlobalExplainer(sample_config)
        
        importance = await explainer._calculate_permutation_importance(sample_data, mock_model)
        
        assert isinstance(importance, dict)
        assert len(importance) == len(sample_data.columns)
    
    def test_calculate_consensus_ranking(self, sample_config):
        explainer = GlobalExplainer(sample_config)
        
        importance_methods = {
            'method1': {'feature1': 0.5, 'feature2': 0.3, 'feature3': 0.2},
            'method2': {'feature1': 0.4, 'feature2': 0.4, 'feature3': 0.2},
            'method3': {'feature1': 0.6, 'feature2': 0.2, 'feature3': 0.2}
        }
        
        consensus = explainer._calculate_consensus_ranking(importance_methods)
        
        assert isinstance(consensus, dict)
        assert len(consensus) == 3
        assert 'feature1' in consensus

@pytest.mark.asyncio
async def test_integration_xai_modules():
    """Integration test for XAI modules"""
    
    # Sample data
    data = pd.DataFrame({
        'payment_history': [0.8, 0.6, 0.9],
        'credit_utilization': [0.3, 0.7, 0.2],
        'credit_length': [0.6, 0.4, 0.8]
    })
    
    # Mock model
    model = Mock()
    model.predict.return_value = np.array([720, 650, 780])
    model.feature_importances_ = np.array([0.5, 0.3, 0.2])
    
    # Test counterfactual analyzer
    cf_config = {'methods': ['simple'], 'max_iterations': 10}
    cf_analyzer = CounterfactualAnalyzer(cf_config)
    
    instance = data.iloc[0].to_dict()
    cf_result = await cf_analyzer.generate_counterfactuals(instance, 750, model)
    
    assert 'counterfactuals' in cf_result
    
    # Test global explainer
    global_config = {'importance_methods': ['model_based'], 'n_permutations': 10}
    global_explainer = GlobalExplainer(global_config)
    
    global_result = await global_explainer.explain_global(data, model)
    
    assert 'importance_methods' in global_result
    assert 'consensus_ranking' in global_result

if __name__ == "__main__":
    pytest.main([__file__])
