"""
Scenario generator for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class ScenarioCategory(Enum):
    """Categories of scenarios"""
    IMPROVEMENT = "improvement"
    MAINTENANCE = "maintenance"
    RISK_MITIGATION = "risk_mitigation"
    OPTIMIZATION = "optimization"
    STRESS_TEST = "stress_test"

class ScenarioComplexity(Enum):
    """Scenario complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

@dataclass
class GeneratedScenario:
    """Generated scenario with metadata"""
    name: str
    description: str
    category: ScenarioCategory
    complexity: ScenarioComplexity
    changes: Dict[str, float]
    expected_outcome: Dict[str, Any]
    implementation_difficulty: str
    timeline: str
    success_probability: float
    risk_level: str
    prerequisites: List[str]
    alternatives: List[str]

class ScenarioGenerator:
    """Generator for credit improvement scenarios"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.factor_relationships = {}
        self.scenario_templates = {}
        self._initialize_generator()
        
    def _initialize_generator(self):
        """Initialize scenario generator"""
        
        self.factor_relationships = {
            'payment_history': {
                'improvement_actions': ['setup_autopay', 'payment_reminders', 'debt_counseling'],
                'time_to_impact': 3,
                'max_improvement': 0.25,
                'difficulty': 'low'
            },
            'credit_utilization': {
                'improvement_actions': ['debt_paydown', 'credit_limit_increase', 'balance_transfer'],
                'time_to_impact': 1,
                'max_improvement': 0.40,
                'difficulty': 'moderate'
            },
            'credit_length': {
                'improvement_actions': ['keep_old_accounts', 'authorized_user_addition'],
                'time_to_impact': 12,
                'max_improvement': 0.15,
                'difficulty': 'low'
            },
            'credit_mix': {
                'improvement_actions': ['installment_loan', 'secured_credit_card', 'retail_account'],
                'time_to_impact': 6,
                'max_improvement': 0.20,
                'difficulty': 'moderate'
            },
            'new_credit': {
                'improvement_actions': ['avoid_applications', 'strategic_timing', 'pre_qualification'],
                'time_to_impact': 12,
                'max_improvement': 0.15,
                'difficulty': 'low'
            }
        }
        
        self.scenario_templates = {
            ScenarioCategory.IMPROVEMENT: {
                'aggressive': {'change_multiplier': 1.2, 'timeline_multiplier': 0.8, 'risk_level': 'medium'},
                'conservative': {'change_multiplier': 0.8, 'timeline_multiplier': 1.5, 'risk_level': 'low'},
                'balanced': {'change_multiplier': 1.0, 'timeline_multiplier': 1.0, 'risk_level': 'low'}
            },
            ScenarioCategory.MAINTENANCE: {
                'status_quo': {'change_multiplier': 0.1, 'timeline_multiplier': 1.0, 'risk_level': 'low'}
            },
            ScenarioCategory.OPTIMIZATION: {
                'maximize': {'change_multiplier': 1.5, 'timeline_multiplier': 1.2, 'risk_level': 'medium'}
            }
        }
    
    async def generate_scenarios(self, current_profile: Dict[str, Any],
                               target_score: Optional[float] = None,
                               max_scenarios: int = 10) -> List[GeneratedScenario]:
        """Generate multiple scenarios based on profile and goals"""
        
        try:
            scenarios = []
            current_factors = current_profile.get('factors', {})
            
            # Generate improvement scenarios
            improvement_scenarios = await self._generate_improvement_scenarios(current_profile)
            scenarios.extend(improvement_scenarios)
            
            # Generate optimization scenarios
            optimization_scenarios = await self._generate_optimization_scenarios(current_profile)
            scenarios.extend(optimization_scenarios)
            
            # Generate stress test scenarios
            stress_scenarios = await self._generate_stress_scenarios(current_profile)
            scenarios.extend(stress_scenarios)
            
            # Rank scenarios
            ranked_scenarios = await self._rank_scenarios(scenarios, current_profile, target_score)
            
            return ranked_scenarios[:max_scenarios]
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {e}")
            return []
    
    async def _generate_improvement_scenarios(self, current_profile: Dict[str, Any]) -> List[GeneratedScenario]:
        """Generate improvement scenarios"""
        
        try:
            scenarios = []
            current_factors = current_profile.get('factors', {})
            
            # Single factor improvements
            for factor in self.factor_relationships.keys():
                current_value = current_factors.get(factor, 0.5)
                if current_value < 0.8:  # Room for improvement
                    scenario = await self._create_single_factor_scenario(factor, current_profile)
                    if scenario:
                        scenarios.append(scenario)
            
            # Multi-factor improvement
            multi_scenario = await self._create_multi_factor_scenario(current_profile)
            if multi_scenario:
                scenarios.append(multi_scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating improvement scenarios: {e}")
            return []
    
    async def _create_single_factor_scenario(self, factor: str, current_profile: Dict[str, Any]) -> Optional[GeneratedScenario]:
        """Create scenario focusing on single factor"""
        
        try:
            current_factors = current_profile.get('factors', {})
            current_value = current_factors.get(factor, 0.5)
            
            # Calculate improvement
            max_improvement = self.factor_relationships[factor]['max_improvement']
            target_improvement = min(max_improvement, (1.0 - current_value) * 0.7)
            
            changes = {factor: target_improvement}
            
            # Predict outcome
            expected_outcome = await self._predict_outcome(changes, current_profile)
            
            # Generate timeline
            timeline = await self._estimate_timeline(changes)
            
            scenario = GeneratedScenario(
                name=f"Improve {factor.replace('_', ' ').title()}",
                description=f"Focus on improving {factor.replace('_', ' ')} through targeted actions",
                category=ScenarioCategory.IMPROVEMENT,
                complexity=ScenarioComplexity.SIMPLE,
                changes=changes,
                expected_outcome=expected_outcome,
                implementation_difficulty=self.factor_relationships[factor]['difficulty'].title(),
                timeline=timeline,
                success_probability=0.8,
                risk_level='low',
                prerequisites=await self._get_prerequisites(factor),
                alternatives=self.factor_relationships[factor]['improvement_actions'][:2]
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error creating single factor scenario: {e}")
            return None
    
    async def _create_multi_factor_scenario(self, current_profile: Dict[str, Any]) -> Optional[GeneratedScenario]:
        """Create multi-factor improvement scenario"""
        
        try:
            current_factors = current_profile.get('factors', {})
            
            # Select top 3 factors with improvement potential
            factor_scores = []
            for factor, config in self.factor_relationships.items():
                current_value = current_factors.get(factor, 0.5)
                improvement_potential = (1.0 - current_value) * config['max_improvement']
                factor_scores.append((factor, improvement_potential))
            
            factor_scores.sort(key=lambda x: x[1], reverse=True)
            selected_factors = [f[0] for f in factor_scores[:3]]
            
            # Generate changes
            changes = {}
            for factor in selected_factors:
                current_value = current_factors.get(factor, 0.5)
                max_improvement = self.factor_relationships[factor]['max_improvement']
                target_improvement = min(max_improvement * 0.6, (1.0 - current_value) * 0.5)
                changes[factor] = target_improvement
            
            expected_outcome = await self._predict_outcome(changes, current_profile)
            timeline = await self._estimate_timeline(changes)
            
            scenario = GeneratedScenario(
                name="Comprehensive Credit Improvement",
                description="Improve multiple credit factors simultaneously",
                category=ScenarioCategory.IMPROVEMENT,
                complexity=ScenarioComplexity.COMPLEX,
                changes=changes,
                expected_outcome=expected_outcome,
                implementation_difficulty="Moderate",
                timeline=timeline,
                success_probability=0.7,
                risk_level='medium',
                prerequisites=["Stable income", "Organized financial plan"],
                alternatives=["Sequential improvement", "Targeted single-factor approach"]
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error creating multi-factor scenario: {e}")
            return None
    
    async def _generate_optimization_scenarios(self, current_profile: Dict[str, Any]) -> List[GeneratedScenario]:
        """Generate optimization scenarios"""
        
        try:
            scenarios = []
            
            # Score maximization scenario
            max_scenario = await self._create_maximization_scenario(current_profile)
            if max_scenario:
                scenarios.append(max_scenario)
            
            # Efficiency optimization
            efficiency_scenario = await self._create_efficiency_scenario(current_profile)
            if efficiency_scenario:
                scenarios.append(efficiency_scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating optimization scenarios: {e}")
            return []
    
    async def _create_maximization_scenario(self, current_profile: Dict[str, Any]) -> Optional[GeneratedScenario]:
        """Create score maximization scenario"""
        
        try:
            current_factors = current_profile.get('factors', {})
            
            # Maximize all improvable factors
            changes = {}
            for factor, config in self.factor_relationships.items():
                current_value = current_factors.get(factor, 0.5)
                if current_value < 0.9:
                    max_improvement = config['max_improvement']
                    target_improvement = min(max_improvement, (1.0 - current_value) * 0.8)
                    changes[factor] = target_improvement
            
            expected_outcome = await self._predict_outcome(changes, current_profile)
            
            scenario = GeneratedScenario(
                name="Maximum Score Optimization",
                description="Aggressive strategy to maximize credit score",
                category=ScenarioCategory.OPTIMIZATION,
                complexity=ScenarioComplexity.COMPLEX,
                changes=changes,
                expected_outcome=expected_outcome,
                implementation_difficulty="Challenging",
                timeline="12-24 months",
                success_probability=0.6,
                risk_level='medium',
                prerequisites=["Significant financial resources", "Long-term commitment"],
                alternatives=["Gradual improvement", "Targeted optimization"]
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error creating maximization scenario: {e}")
            return None
    
    async def _create_efficiency_scenario(self, current_profile: Dict[str, Any]) -> Optional[GeneratedScenario]:
        """Create efficiency optimization scenario"""
        
        try:
            current_factors = current_profile.get('factors', {})
            
            # Focus on high-impact, low-effort improvements
            efficiency_factors = ['credit_utilization', 'payment_history']
            changes = {}
            
            for factor in efficiency_factors:
                if factor in current_factors:
                    current_value = current_factors[factor]
                    if current_value < 0.8:
                        target_improvement = min(0.15, (1.0 - current_value) * 0.6)
                        changes[factor] = target_improvement
            
            expected_outcome = await self._predict_outcome(changes, current_profile)
            
            scenario = GeneratedScenario(
                name="Efficient Credit Optimization",
                description="Focus on high-impact, low-effort improvements",
                category=ScenarioCategory.OPTIMIZATION,
                complexity=ScenarioComplexity.MODERATE,
                changes=changes,
                expected_outcome=expected_outcome,
                implementation_difficulty="Easy",
                timeline="3-6 months",
                success_probability=0.85,
                risk_level='low',
                prerequisites=["Basic financial discipline"],
                alternatives=["Comprehensive approach", "Single-factor focus"]
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error creating efficiency scenario: {e}")
            return None
    
    async def _generate_stress_scenarios(self, current_profile: Dict[str, Any]) -> List[GeneratedScenario]:
        """Generate stress test scenarios"""
        
        try:
            scenarios = []
            
            # Economic downturn scenario
            economic_changes = {
                'payment_history': -0.10,
                'credit_utilization': 0.15,
                'new_credit': 0.05
            }
            
            economic_scenario = GeneratedScenario(
                name="Economic Downturn Stress Test",
                description="Simulate impact of economic recession",
                category=ScenarioCategory.STRESS_TEST,
                complexity=ScenarioComplexity.MODERATE,
                changes=economic_changes,
                expected_outcome=await self._predict_outcome(economic_changes, current_profile),
                implementation_difficulty="N/A",
                timeline="6-12 months",
                success_probability=0.3,
                risk_level='high',
                prerequisites=["Emergency fund", "Stable employment"],
                alternatives=["Risk mitigation strategies", "Insurance coverage"]
            )
            
            scenarios.append(economic_scenario)
            
            # Job loss scenario
            job_loss_changes = {
                'payment_history': -0.15,
                'credit_utilization': 0.20
            }
            
            job_loss_scenario = GeneratedScenario(
                name="Job Loss Impact Analysis",
                description="Simulate impact of unemployment",
                category=ScenarioCategory.STRESS_TEST,
                complexity=ScenarioComplexity.SIMPLE,
                changes=job_loss_changes,
                expected_outcome=await self._predict_outcome(job_loss_changes, current_profile),
                implementation_difficulty="N/A",
                timeline="3-6 months",
                success_probability=0.2,
                risk_level='high',
                prerequisites=["Emergency savings", "Job search plan"],
                alternatives=["Income diversification", "Expense reduction"]
            )
            
            scenarios.append(job_loss_scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating stress scenarios: {e}")
            return []
    
    async def _predict_outcome(self, changes: Dict[str, float], current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Predict scenario outcomes"""
        
        try:
            current_score = current_profile.get('credit_score', 650)
            
            # Simple score prediction model
            score_weights = {
                'payment_history': 175,
                'credit_utilization': 150,
                'credit_length': 75,
                'credit_mix': 50,
                'new_credit': 50
            }
            
            score_change = 0
            for factor, change in changes.items():
                if factor in score_weights:
                    factor_impact = change * score_weights[factor]
                    score_change += factor_impact
            
            predicted_score = max(300, min(850, current_score + score_change))
            
            return {
                'predicted_score': predicted_score,
                'score_change': score_change,
                'percentage_improvement': (score_change / current_score) * 100 if current_score > 0 else 0,
                'factors_improved': list(changes.keys()),
                'total_factors_changed': len(changes)
            }
            
        except Exception as e:
            logger.error(f"Error predicting outcome: {e}")
            return {'error': str(e)}
    
    async def _estimate_timeline(self, changes: Dict[str, float]) -> str:
        """Estimate implementation timeline"""
        
        try:
            max_timeline = 0
            
            for factor in changes.keys():
                if factor in self.factor_relationships:
                    factor_timeline = self.factor_relationships[factor]['time_to_impact']
                    max_timeline = max(max_timeline, factor_timeline)
            
            if max_timeline <= 3:
                return "1-3 months"
            elif max_timeline <= 6:
                return "3-6 months"
            elif max_timeline <= 12:
                return "6-12 months"
            else:
                return "12+ months"
                
        except Exception as e:
            logger.error(f"Error estimating timeline: {e}")
            return "6-12 months"
    
    async def _get_prerequisites(self, factor: str) -> List[str]:
        """Get prerequisites for factor improvement"""
        
        prerequisites_map = {
            'payment_history': ["Stable income", "Payment tracking system"],
            'credit_utilization': ["Available funds", "Debt paydown plan"],
            'credit_length': ["Existing credit accounts", "Account management discipline"],
            'credit_mix': ["Good credit standing", "Stable employment"],
            'new_credit': ["Credit discipline", "Strategic planning"]
        }
        
        return prerequisites_map.get(factor, ["Basic financial planning"])
    
    async def _rank_scenarios(self, scenarios: List[GeneratedScenario],
                            current_profile: Dict[str, Any],
                            target_score: Optional[float]) -> List[GeneratedScenario]:
        """Rank scenarios by effectiveness and feasibility"""
        
        try:
            if not scenarios:
                return scenarios
            
            # Calculate ranking scores
            for scenario in scenarios:
                ranking_score = 0.0
                
                # Score improvement potential (40% weight)
                outcome = scenario.expected_outcome
                if isinstance(outcome, dict) and 'score_change' in outcome:
                    score_change = outcome['score_change']
                    ranking_score += (score_change / 100) * 0.4
                
                # Success probability (30% weight)
                ranking_score += scenario.success_probability * 0.3
                
                # Implementation difficulty (20% weight)
                difficulty_scores = {'Easy': 1.0, 'Moderate': 0.7, 'Challenging': 0.4, 'Very Difficult': 0.1}
                difficulty_score = difficulty_scores.get(scenario.implementation_difficulty, 0.5)
                ranking_score += difficulty_score * 0.2
                
                # Risk level (10% weight)
                risk_scores = {'low': 1.0, 'medium': 0.6, 'high': 0.2}
                risk_score = risk_scores.get(scenario.risk_level, 0.5)
                ranking_score += risk_score * 0.1
                
                scenario.ranking_score = ranking_score
            
            # Sort by ranking score
            scenarios.sort(key=lambda x: getattr(x, 'ranking_score', 0), reverse=True)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error ranking scenarios: {e}")
            return scenarios
    
    async def generate_custom_scenario(self, target_changes: Dict[str, float],
                                     current_profile: Dict[str, Any]) -> GeneratedScenario:
        """Generate custom scenario with specific changes"""
        
        try:
            expected_outcome = await self._predict_outcome(target_changes, current_profile)
            timeline = await self._estimate_timeline(target_changes)
            
            # Assess complexity
            complexity = ScenarioComplexity.SIMPLE
            if len(target_changes) > 2:
                complexity = ScenarioComplexity.COMPLEX
            elif len(target_changes) > 1:
                complexity = ScenarioComplexity.MODERATE
            
            # Calculate success probability
            success_prob = 0.8 - (len(target_changes) - 1) * 0.1
            success_prob = max(0.3, min(0.9, success_prob))
            
            scenario = GeneratedScenario(
                name="Custom Credit Strategy",
                description="User-defined credit improvement strategy",
                category=ScenarioCategory.OPTIMIZATION,
                complexity=complexity,
                changes=target_changes,
                expected_outcome=expected_outcome,
                implementation_difficulty="Moderate",
                timeline=timeline,
                success_probability=success_prob,
                risk_level='medium',
                prerequisites=["Financial planning", "Commitment to changes"],
                alternatives=["Predefined scenarios", "Professional consultation"]
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error generating custom scenario: {e}")
            return GeneratedScenario(
                name="Error Scenario",
                description="Could not generate custom scenario",
                category=ScenarioCategory.OPTIMIZATION,
                complexity=ScenarioComplexity.SIMPLE,
                changes={},
                expected_outcome={'error': str(e)},
                implementation_difficulty="Unknown",
                timeline="Unknown",
                success_probability=0.0,
                risk_level='high',
                prerequisites=[],
                alternatives=[]
            )
    
    def get_supported_factors(self) -> List[str]:
        """Get list of supported factors"""
        return list(self.factor_relationships.keys())
    
    def get_scenario_categories(self) -> List[ScenarioCategory]:
        """Get list of scenario categories"""
        return list(ScenarioCategory)
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """Get generator statistics"""
        
        return {
            'supported_factors': len(self.factor_relationships),
            'scenario_categories': len(ScenarioCategory),
            'complexity_levels': len(ScenarioComplexity),
            'template_categories': len(self.scenario_templates),
            'timestamp': datetime.now().isoformat()
        }
