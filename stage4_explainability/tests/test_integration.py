"""
Integration tests for Stage 4 explainability module.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, AsyncMock

from stage4_explainability.xai.counterfactual_analysis import CounterfactualAnalyzer
from stage4_explainability.xai.global_explanations import GlobalExplainer
from stage4_explainability.chatbot.chat_engine import ChatEngine
from stage4_explainability.explainer.explanation_generator import ExplanationGenerator
from stage4_explainability.natural_language.text_generation import TextGenerator
from stage4_explainability.simulation.what_if_analyzer import WhatIfAnalyzer

@pytest.mark.asyncio
async def test_full_explainability_pipeline():
    """Test complete explainability pipeline integration"""
    
    # Sample credit profile
    profile = {
        'user_id': 'test_user_123',
        'credit_score': 680,
        'factors': {
            'payment_history': 0.7,
            'credit_utilization': 0.8,
            'credit_length': 0.4,
            'credit_mix': 0.6,
            'new_credit': 0.7
        },
        'demographics': {
            'age': 35,
            'income': 65000,
            'employment_length': 5
        }
    }
    
    # Mock model
    mock_model = Mock()
    mock_model.predict.return_value = np.array([680])
    mock_model.feature_importances_ = np.array([0.35, 0.30, 0.15, 0.10, 0.10])
    
    # Sample data for global explanations
    sample_data = pd.DataFrame({
        'payment_history': np.random.uniform(0.3, 1.0, 100),
        'credit_utilization': np.random.uniform(0.1, 0.9, 100),
        'credit_length': np.random.uniform(0.2, 0.8, 100),
        'credit_mix': np.random.uniform(0.3, 0.9, 100),
        'new_credit': np.random.uniform(0.4, 1.0, 100)
    })
    
    config = {
        'confidence_threshold': 0.7,
        'max_scenarios': 5,
        'enable_caching': True
    }
    
    # Initialize components
    cf_analyzer = CounterfactualAnalyzer(config)
    global_explainer = GlobalExplainer(config)
    explanation_generator = ExplanationGenerator(config)
    text_generator = TextGenerator(config)
    what_if_analyzer = WhatIfAnalyzer(config)
    
    # Step 1: Generate local explanations (counterfactual)
    instance = profile['factors']
    target_score = 750
    
    cf_result = await cf_analyzer.generate_counterfactuals(
        instance, target_score, mock_model
    )
    
    assert 'counterfactuals' in cf_result
    assert len(cf_result['counterfactuals']) > 0
    
    # Step 2: Generate global explanations
    global_result = await global_explainer.explain_global(sample_data, mock_model)
    
    assert 'importance_methods' in global_result
    assert 'consensus_ranking' in global_result
    
    # Step 3: Generate comprehensive explanation
    explanation_data = {
        'user_id': profile['user_id'],
        'credit_score': profile['credit_score'],
        'factors': profile['factors'],
        'counterfactuals': cf_result['counterfactuals'],
        'global_importance': global_result,
        'explanation_type': 'comprehensive'
    }
    
    comprehensive_explanation = await explanation_generator.generate_explanation(
        explanation_data, explanation_type='comprehensive'
    )
    
    assert 'local_explanation' in comprehensive_explanation
    assert 'global_explanation' in comprehensive_explanation
    assert 'counterfactual_explanation' in comprehensive_explanation
    
    # Step 4: Generate natural language explanation
    natural_language_text = await text_generator.generate_explanation_text(
        explanation_data, style=text_generator.TextStyle.CONVERSATIONAL
    )
    
    assert isinstance(natural_language_text, str)
    assert len(natural_language_text) > 50
    assert 'credit' in natural_language_text.lower()
    
    # Step 5: Generate what-if scenarios
    scenarios = what_if_analyzer.get_predefined_scenarios()[:3]
    scenario_results = await what_if_analyzer.analyze_multiple_scenarios(scenarios, profile)
    
    assert len(scenario_results) == 3
    assert all(hasattr(r, 'predicted_score') for r in scenario_results)
    
    # Step 6: Verify integration consistency
    # The counterfactual should suggest changes that align with what-if scenarios
    if cf_result['counterfactuals']:
        cf_changes = cf_result['counterfactuals'][0].get('changes_made', {})
        scenario_changes = scenario_results[0].scenario.changes
        
        # At least some factors should overlap
        common_factors = set(cf_changes.keys()) & set(scenario_changes.keys())
        assert len(common_factors) > 0

@pytest.mark.asyncio
async def test_chatbot_explainer_integration():
    """Test chatbot integration with explanation components"""
    
    config = {
        'session_timeout': 3600,
        'confidence_threshold': 0.7,
        'enable_explanations': True
    }
    
    # Initialize chatbot
    chat_engine = ChatEngine(config)
    
    # Mock explanation generator
    mock_explainer = AsyncMock()
    mock_explainer.generate_explanation.return_value = {
        'local_explanation': {'importance': {'payment_history': 0.4, 'credit_utilization': 0.3}},
        'narrative': 'Your payment history is the most important factor affecting your credit score.'
    }
    
    chat_engine.explanation_generator = mock_explainer
    
    # Test explanation request
    result = await chat_engine.process_message(
        user_id="test_user",
        message="Why is my credit score 680?",
        session_id="test_session"
    )
    
    assert 'response' in result
    assert 'intent' in result
    # Should recognize this as an explanation request
    assert result['intent']['intent'] in ['explanation_request', 'credit_score_inquiry']

@pytest.mark.asyncio
async def test_end_to_end_credit_analysis():
    """End-to-end test of credit analysis workflow"""
    
    # User profile with credit issues
    user_profile = {
        'user_id': 'user_456',
        'credit_score': 580,  # Poor score
        'factors': {
            'payment_history': 0.5,  # Poor payment history
            'credit_utilization': 0.9,  # High utilization
            'credit_length': 0.3,  # Short credit history
            'credit_mix': 0.4,  # Limited credit mix
            'new_credit': 0.6  # Recent inquiries
        }
    }
    
    config = {'enable_all_features': True}
    
    # Initialize all components
    components = {
        'cf_analyzer': CounterfactualAnalyzer(config),
        'global_explainer': GlobalExplainer(config),
        'explanation_generator': ExplanationGenerator(config),
        'text_generator': TextGenerator(config),
        'what_if_analyzer': WhatIfAnalyzer(config),
        'chat_engine': ChatEngine(config)
    }
    
    # Mock model for predictions
    mock_model = Mock()
    mock_model.predict.return_value = np.array([580])
    mock_model.feature_importances_ = np.array([0.35, 0.30, 0.15, 0.10, 0.10])
    
    # Step 1: User asks about their credit score
    chat_result = await components['chat_engine'].process_message(
        user_id=user_profile['user_id'],
        message="My credit score is 580. What's wrong with it?",
        session_id="analysis_session"
    )
    
    assert 'response' in chat_result
    
    # Step 2: Generate detailed explanation
    explanation_request = {
        'user_id': user_profile['user_id'],
        'credit_score': user_profile['credit_score'],
        'factors': user_profile['factors'],
        'explanation_type': 'local'
    }
    
    local_explanation = await components['explanation_generator'].generate_explanation(
        explanation_request, explanation_type='local'
    )
    
    assert 'combined_importance' in local_explanation
    
    # Step 3: Generate improvement scenarios
    improvement_scenarios = await components['what_if_analyzer'].analyze_multiple_scenarios(
        components['what_if_analyzer'].get_predefined_scenarios()[:3],
        user_profile
    )
    
    # Should show significant improvement potential
    best_scenario = improvement_scenarios[0]
    assert best_scenario.score_change > 20  # Should predict meaningful improvement
    
    # Step 4: Generate counterfactual analysis
    target_score = 650  # Realistic improvement target
    cf_analysis = await components['cf_analyzer'].generate_counterfactuals(
        user_profile['factors'], target_score, mock_model
    )
    
    assert 'counterfactuals' in cf_analysis
    
    # Step 5: Create natural language summary
    summary_data = {
        'user_id': user_profile['user_id'],
        'current_score': user_profile['credit_score'],
        'target_score': target_score,
        'local_explanation': local_explanation,
        'best_scenario': best_scenario,
        'counterfactual': cf_analysis['counterfactuals'][0] if cf_analysis['counterfactuals'] else None,
        'explanation_type': 'improvement_plan'
    }
    
    improvement_plan = await components['text_generator'].generate_explanation_text(
        summary_data, style=components['text_generator'].TextStyle.CONVERSATIONAL
    )
    
    assert isinstance(improvement_plan, str)
    assert len(improvement_plan) > 100
    # Should mention key improvement areas
    assert any(factor.replace('_', ' ') in improvement_plan.lower() 
              for factor in ['payment history', 'credit utilization'])

@pytest.mark.asyncio
async def test_performance_and_scalability():
    """Test performance with multiple concurrent requests"""
    
    config = {'enable_caching': True, 'max_concurrent': 5}
    
    # Create multiple user profiles
    profiles = []
    for i in range(10):
        profiles.append({
            'user_id': f'user_{i}',
            'credit_score': np.random.randint(500, 800),
            'factors': {
                'payment_history': np.random.uniform(0.3, 1.0),
                'credit_utilization': np.random.uniform(0.1, 0.9),
                'credit_length': np.random.uniform(0.2, 0.8),
                'credit_mix': np.random.uniform(0.3, 0.9),
                'new_credit': np.random.uniform(0.4, 1.0)
            }
        })
    
    # Initialize components
    explanation_generator = ExplanationGenerator(config)
    text_generator = TextGenerator(config)
    
    # Process multiple requests concurrently
    async def process_profile(profile):
        explanation = await explanation_generator.generate_explanation(
            profile, explanation_type='local'
        )
        
        text = await text_generator.generate_explanation_text(
            explanation, style=text_generator.TextStyle.SIMPLE
        )
        
        return {'profile': profile, 'explanation': explanation, 'text': text}
    
    # Measure performance
    import time
    start_time = time.time()
    
    tasks = [process_profile(profile) for profile in profiles[:5]]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Should complete within reasonable time (adjust threshold as needed)
    assert processing_time < 30  # 30 seconds for 5 profiles
    assert len(results) == 5
    assert all('explanation' in r and 'text' in r for r in results)

@pytest.mark.asyncio
async def test_error_handling_and_resilience():
    """Test error handling across components"""
    
    config = {'enable_error_recovery': True}
    
    # Initialize components
    explanation_generator = ExplanationGenerator(config)
    text_generator = TextGenerator(config)
    
    # Test with invalid data
    invalid_profile = {
        'user_id': None,  # Invalid user ID
        'credit_score': -100,  # Invalid score
        'factors': {
            'invalid_factor': 2.0,  # Invalid factor value
            'payment_history': 'invalid'  # Invalid data type
        }
    }
    
    # Should handle errors gracefully
    try:
        explanation = await explanation_generator.generate_explanation(
            invalid_profile, explanation_type='local'
        )
        # Should return error information rather than crash
        assert 'error' in explanation or 'combined_importance' in explanation
    except Exception as e:
        # If it raises an exception, it should be handled gracefully
        assert isinstance(e, (ValueError, TypeError))
    
    # Test text generation with missing data
    incomplete_data = {
        'explanation_type': 'local'
        # Missing required fields
    }
    
    text_result = await text_generator.generate_explanation_text(incomplete_data)
    
    # Should return some default text rather than crash
    assert isinstance(text_result, str)
    assert len(text_result) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
