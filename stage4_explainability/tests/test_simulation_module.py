"""
Tests for simulation module components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import asyncio

from stage4_explainability.simulation.what_if_analyzer import WhatIfAnalyzer, WhatIfScenario, ScenarioType
from stage4_explainability.simulation.sensitivity_analysis import SensitivityAnalyzer, SensitivityMethod
from stage4_explainability.simulation.scenario_generator import ScenarioGenerator, ScenarioCategory, ScenarioComplexity

class TestWhatIfAnalyzer:
    """Test cases for WhatIfAnalyzer"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'max_scenarios': 10,
            'confidence_threshold': 0.7
        }
    
    @pytest.fixture
    def sample_profile(self):
        return {
            'credit_score': 680,
            'factors': {
                'payment_history': 0.7,
                'credit_utilization': 0.6,
                'credit_length': 0.5,
                'credit_mix': 0.6,
                'new_credit': 0.8
            }
        }
    
    @pytest.fixture
    def sample_scenario(self):
        return WhatIfScenario(
            name="Test Scenario",
            description="Test scenario for unit testing",
            scenario_type=ScenarioType.SINGLE_FACTOR,
            changes={'credit_utilization': -0.2},
            timeline="3-6 months"
        )
    
    def test_analyzer_initialization(self, sample_config):
        analyzer = WhatIfAnalyzer(sample_config)
        assert analyzer.config == sample_config
        assert hasattr(analyzer, 'factor_impacts')
        assert hasattr(analyzer, 'interaction_effects')
    
    @pytest.mark.asyncio
    async def test_analyze_scenario(self, sample_config, sample_profile, sample_scenario):
        analyzer = WhatIfAnalyzer(sample_config)
        
        result = await analyzer.analyze_scenario(sample_scenario, sample_profile)
        
        assert result.scenario == sample_scenario
        assert hasattr(result, 'predicted_score')
        assert hasattr(result, 'score_change')
        assert hasattr(result, 'confidence')
        assert result.predicted_score != sample_profile['credit_score']
    
    @pytest.mark.asyncio
    async def test_calculate_score_change(self, sample_config, sample_profile):
        analyzer = WhatIfAnalyzer(sample_config)
        
        changes = {'credit_utilization': -0.2, 'payment_history': 0.1}
        current_factors = sample_profile['factors']
        
        score_change = await analyzer._calculate_score_change(changes, current_factors)
        
        assert isinstance(score_change, float)
        assert score_change != 0  # Should have some impact
    
    @pytest.mark.asyncio
    async def test_generate_custom_scenario(self, sample_config, sample_profile):
        analyzer = WhatIfAnalyzer(sample_config)
        
        target_score = 750
        scenario = await analyzer.generate_custom_scenario(target_score, sample_profile)
        
        assert isinstance(scenario, WhatIfScenario)
        assert scenario.scenario_type == ScenarioType.OPTIMIZATION
        assert len(scenario.changes) > 0
    
    def test_get_predefined_scenarios(self, sample_config):
        analyzer = WhatIfAnalyzer(sample_config)
        
        scenarios = analyzer.get_predefined_scenarios()
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert all(isinstance(s, WhatIfScenario) for s in scenarios)

class TestSensitivityAnalyzer:
    """Test cases for SensitivityAnalyzer"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'perturbation_steps': 10,
            'confidence_threshold': 0.8
        }
    
    @pytest.fixture
    def sample_profile(self):
        return {
            'credit_score': 680,
            'factors': {
                'payment_history': 0.7,
                'credit_utilization': 0.6,
                'credit_length': 0.5,
                'credit_mix': 0.6,
                'new_credit': 0.8
            }
        }
    
    def test_analyzer_initialization(self, sample_config):
        analyzer = SensitivityAnalyzer(sample_config)
        assert analyzer.config == sample_config
        assert hasattr(analyzer, 'perturbation_ranges')
        assert hasattr(analyzer, 'factor_constraints')
    
    @pytest.mark.asyncio
    async def test_analyze_factor_sensitivity(self, sample_config, sample_profile):
        analyzer = SensitivityAnalyzer(sample_config)
        
        result = await analyzer.analyze_factor_sensitivity(
            'credit_utilization', sample_profile, SensitivityMethod.ONE_AT_A_TIME
        )
        
        assert result.factor == 'credit_utilization'
        assert hasattr(result, 'sensitivity_score')
        assert hasattr(result, 'impact_range')
        assert hasattr(result, 'confidence')
        assert len(result.perturbation_results) > 0
    
    @pytest.mark.asyncio
    async def test_generate_perturbations(self, sample_config):
        analyzer = SensitivityAnalyzer(sample_config)
        
        factor = 'payment_history'
        base_value = 0.7
        perturbation_config = analyzer.perturbation_ranges[factor]
        
        perturbations = await analyzer._generate_perturbations(
            factor, base_value, perturbation_config, SensitivityMethod.ONE_AT_A_TIME
        )
        
        assert isinstance(perturbations, list)
        assert len(perturbations) > 0
        assert all(0 <= p <= 1 for p in perturbations)
    
    @pytest.mark.asyncio
    async def test_analyze_all_factors(self, sample_config, sample_profile):
        analyzer = SensitivityAnalyzer(sample_config)
        
        results = await analyzer.analyze_all_factors(sample_profile)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(hasattr(r, 'factor') and hasattr(r, 'sensitivity_score') for r in results)
        # Results should be sorted by sensitivity score
        assert results[0].sensitivity_score >= results[-1].sensitivity_score
    
    @pytest.mark.asyncio
    async def test_identify_critical_factors(self, sample_config, sample_profile):
        analyzer = SensitivityAnalyzer(sample_config)
        
        critical_factors = await analyzer.identify_critical_factors(sample_profile, threshold=0.3)
        
        assert isinstance(critical_factors, list)
        assert all(isinstance(f, str) for f in critical_factors)
    
    @pytest.mark.asyncio
    async def test_generate_sensitivity_report(self, sample_config, sample_profile):
        analyzer = SensitivityAnalyzer(sample_config)
        
        report = await analyzer.generate_sensitivity_report(sample_profile)
        
        assert 'profile_summary' in report
        assert 'sensitivity_analysis' in report
        assert 'critical_factors' in report
        assert 'recommendations' in report

class TestScenarioGenerator:
    """Test cases for ScenarioGenerator"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'max_scenarios': 15,
            'complexity_levels': ['simple', 'moderate', 'complex']
        }
    
    @pytest.fixture
    def sample_profile(self):
        return {
            'credit_score': 680,
            'factors': {
                'payment_history': 0.7,
                'credit_utilization': 0.6,
                'credit_length': 0.5,
                'credit_mix': 0.6,
                'new_credit': 0.8
            }
        }
    
    def test_generator_initialization(self, sample_config):
        generator = ScenarioGenerator(sample_config)
        assert generator.config == sample_config
        assert hasattr(generator, 'factor_relationships')
        assert hasattr(generator, 'scenario_templates')
    
    @pytest.mark.asyncio
    async def test_generate_scenarios(self, sample_config, sample_profile):
        generator = ScenarioGenerator(sample_config)
        
        scenarios = await generator.generate_scenarios(sample_profile, max_scenarios=5)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) <= 5
        assert all(hasattr(s, 'name') and hasattr(s, 'changes') for s in scenarios)
    
    @pytest.mark.asyncio
    async def test_generate_improvement_scenarios(self, sample_config, sample_profile):
        generator = ScenarioGenerator(sample_config)
        
        scenarios = await generator._generate_improvement_scenarios(sample_profile)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert all(s.category == ScenarioCategory.IMPROVEMENT for s in scenarios)
    
    @pytest.mark.asyncio
    async def test_create_single_factor_scenario(self, sample_config, sample_profile):
        generator = ScenarioGenerator(sample_config)
        
        scenario = await generator._create_single_factor_scenario('credit_utilization', sample_profile)
        
        assert scenario is not None
        assert scenario.complexity == ScenarioComplexity.SIMPLE
        assert 'credit_utilization' in scenario.changes
        assert len(scenario.changes) == 1
    
    @pytest.mark.asyncio
    async def test_create_multi_factor_scenario(self, sample_config, sample_profile):
        generator = ScenarioGenerator(sample_config)
        
        scenario = await generator._create_multi_factor_scenario(sample_profile)
        
        assert scenario is not None
        assert scenario.complexity == ScenarioComplexity.COMPLEX
        assert len(scenario.changes) > 1
    
    @pytest.mark.asyncio
    async def test_generate_stress_scenarios(self, sample_config, sample_profile):
        generator = ScenarioGenerator(sample_config)
        
        scenarios = await generator._generate_stress_scenarios(sample_profile)
        
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert all(s.category == ScenarioCategory.STRESS_TEST for s in scenarios)
    
    @pytest.mark.asyncio
    async def test_predict_outcome(self, sample_config, sample_profile):
        generator = ScenarioGenerator(sample_config)
        
        changes = {'credit_utilization': -0.2, 'payment_history': 0.1}
        outcome = await generator._predict_outcome(changes, sample_profile)
        
        assert 'predicted_score' in outcome
        assert 'score_change' in outcome
        assert 'factors_improved' in outcome
        assert outcome['predicted_score'] != sample_profile['credit_score']
    
    @pytest.mark.asyncio
    async def test_generate_custom_scenario(self, sample_config, sample_profile):
        generator = ScenarioGenerator(sample_config)
        
        target_changes = {'payment_history': 0.15, 'credit_utilization': -0.25}
        scenario = await generator.generate_custom_scenario(target_changes, sample_profile)
        
        assert scenario.name == "Custom Credit Strategy"
        assert scenario.changes == target_changes
        assert scenario.category == ScenarioCategory.OPTIMIZATION

@pytest.mark.asyncio
async def test_simulation_integration():
    """Integration test for simulation components"""
    
    config = {
        'max_scenarios': 5,
        'confidence_threshold': 0.7,
        'perturbation_steps': 5
    }
    
    profile = {
        'credit_score': 650,
        'factors': {
            'payment_history': 0.6,
            'credit_utilization': 0.8,
            'credit_length': 0.4,
            'credit_mix': 0.5,
            'new_credit': 0.7
        }
    }
    
    # Test what-if analyzer
    what_if_analyzer = WhatIfAnalyzer(config)
    scenarios = what_if_analyzer.get_predefined_scenarios()[:2]
    
    what_if_results = await what_if_analyzer.analyze_multiple_scenarios(scenarios, profile)
    assert len(what_if_results) == 2
    assert all(hasattr(r, 'predicted_score') for r in what_if_results)
    
    # Test sensitivity analyzer
    sensitivity_analyzer = SensitivityAnalyzer(config)
    sensitivity_results = await sensitivity_analyzer.analyze_all_factors(profile)
    assert len(sensitivity_results) > 0
    
    # Test scenario generator
    scenario_generator = ScenarioGenerator(config)
    generated_scenarios = await scenario_generator.generate_scenarios(profile, max_scenarios=3)
    assert len(generated_scenarios) <= 3
    assert all(hasattr(s, 'expected_outcome') for s in generated_scenarios)
    
    # Test integration - use sensitivity results to inform scenario generation
    critical_factors = await sensitivity_analyzer.identify_critical_factors(profile, threshold=0.3)
    
    if critical_factors:
        # Create custom scenario targeting most sensitive factor
        target_changes = {critical_factors[0]: 0.2}
        custom_scenario = await scenario_generator.generate_custom_scenario(target_changes, profile)
        
        # Analyze the custom scenario
        what_if_result = await what_if_analyzer.analyze_scenario(
            WhatIfScenario(
                name=custom_scenario.name,
                description=custom_scenario.description,
                scenario_type=ScenarioType.OPTIMIZATION,
                changes=custom_scenario.changes
            ),
            profile
        )
        
        assert what_if_result.score_change != 0

if __name__ == "__main__":
    pytest.main([__file__])
