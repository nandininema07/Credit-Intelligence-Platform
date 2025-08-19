"""
What-if analyzer for Stage 4 explainability.
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

class ScenarioType(Enum):
    """Types of what-if scenarios"""
    SINGLE_FACTOR = "single_factor"
    MULTIPLE_FACTORS = "multiple_factors"
    TEMPORAL = "temporal"
    STRESS_TEST = "stress_test"
    OPTIMIZATION = "optimization"

@dataclass
class WhatIfScenario:
    """What-if scenario definition"""
    name: str
    description: str
    scenario_type: ScenarioType
    changes: Dict[str, Any]
    timeline: Optional[str] = None
    probability: Optional[float] = None
    effort_level: Optional[str] = None

@dataclass
class ScenarioResult:
    """Result of what-if analysis"""
    scenario: WhatIfScenario
    predicted_score: float
    score_change: float
    confidence: float
    risk_assessment: Dict[str, Any]
    timeline_estimate: str
    implementation_steps: List[str]
    side_effects: List[str]

class WhatIfAnalyzer:
    """Analyzer for what-if scenarios in credit profiles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scenario_templates = {}
        self.factor_impacts = {}
        self.interaction_effects = {}
        self.temporal_models = {}
        self._initialize_analyzer()
        
    def _initialize_analyzer(self):
        """Initialize what-if analyzer"""
        
        # Factor impact estimates (simplified model)
        self.factor_impacts = {
            'payment_history': {
                'weight': 0.35,
                'max_impact': 100,
                'improvement_rate': 0.8,
                'degradation_rate': 1.2
            },
            'credit_utilization': {
                'weight': 0.30,
                'max_impact': 90,
                'improvement_rate': 0.9,
                'degradation_rate': 1.1
            },
            'credit_length': {
                'weight': 0.15,
                'max_impact': 40,
                'improvement_rate': 0.3,
                'degradation_rate': 0.5
            },
            'credit_mix': {
                'weight': 0.10,
                'max_impact': 30,
                'improvement_rate': 0.6,
                'degradation_rate': 0.7
            },
            'new_credit': {
                'weight': 0.10,
                'max_impact': 25,
                'improvement_rate': 0.7,
                'degradation_rate': 0.9
            }
        }
        
        # Interaction effects between factors
        self.interaction_effects = {
            ('payment_history', 'credit_utilization'): 0.15,
            ('credit_utilization', 'credit_length'): 0.10,
            ('payment_history', 'credit_length'): 0.12,
            ('credit_mix', 'credit_length'): 0.08,
            ('new_credit', 'credit_utilization'): -0.05
        }
        
        # Initialize scenario templates
        self._initialize_scenario_templates()
        
    def _initialize_scenario_templates(self):
        """Initialize predefined scenario templates"""
        
        self.scenario_templates = {
            'pay_down_debt': WhatIfScenario(
                name="Pay Down Credit Card Debt",
                description="Reduce credit card balances to lower utilization",
                scenario_type=ScenarioType.SINGLE_FACTOR,
                changes={'credit_utilization': -0.20},
                timeline="3-6 months",
                effort_level="moderate"
            ),
            'perfect_payments': WhatIfScenario(
                name="Perfect Payment History",
                description="Make all payments on time for extended period",
                scenario_type=ScenarioType.TEMPORAL,
                changes={'payment_history': 0.95},
                timeline="12-24 months",
                effort_level="low"
            ),
            'debt_consolidation': WhatIfScenario(
                name="Debt Consolidation",
                description="Consolidate multiple debts into single loan",
                scenario_type=ScenarioType.MULTIPLE_FACTORS,
                changes={
                    'credit_utilization': -0.15,
                    'credit_mix': 0.10,
                    'new_credit': -0.05
                },
                timeline="6-12 months",
                effort_level="high"
            ),
            'credit_limit_increase': WhatIfScenario(
                name="Credit Limit Increase",
                description="Request higher credit limits on existing cards",
                scenario_type=ScenarioType.SINGLE_FACTOR,
                changes={'credit_utilization': -0.10},
                timeline="1-3 months",
                effort_level="low"
            ),
            'close_old_accounts': WhatIfScenario(
                name="Close Old Credit Accounts",
                description="Close unused credit accounts",
                scenario_type=ScenarioType.MULTIPLE_FACTORS,
                changes={
                    'credit_length': -0.08,
                    'credit_utilization': 0.05,
                    'credit_mix': -0.03
                },
                timeline="immediate",
                effort_level="low"
            ),
            'new_credit_card': WhatIfScenario(
                name="Open New Credit Card",
                description="Apply for and open new credit card",
                scenario_type=ScenarioType.MULTIPLE_FACTORS,
                changes={
                    'new_credit': -0.10,
                    'credit_utilization': -0.05,
                    'credit_mix': 0.05
                },
                timeline="3-6 months",
                effort_level="moderate"
            ),
            'mortgage_application': WhatIfScenario(
                name="Apply for Mortgage",
                description="Apply for home mortgage loan",
                scenario_type=ScenarioType.MULTIPLE_FACTORS,
                changes={
                    'new_credit': -0.15,
                    'credit_mix': 0.08,
                    'credit_utilization': 0.02
                },
                timeline="6-12 months",
                effort_level="high"
            )
        }
    
    async def analyze_scenario(self, scenario: WhatIfScenario, 
                             current_profile: Dict[str, Any]) -> ScenarioResult:
        """Analyze a single what-if scenario"""
        
        try:
            current_score = current_profile.get('credit_score', 650)
            current_factors = current_profile.get('factors', {})
            
            # Calculate predicted score change
            predicted_change = await self._calculate_score_change(
                scenario.changes, current_factors
            )
            
            predicted_score = max(300, min(850, current_score + predicted_change))
            
            # Calculate confidence based on scenario complexity
            confidence = await self._calculate_confidence(scenario, current_factors)
            
            # Assess risks
            risk_assessment = await self._assess_scenario_risks(scenario, current_factors)
            
            # Generate implementation steps
            implementation_steps = await self._generate_implementation_steps(scenario)
            
            # Identify potential side effects
            side_effects = await self._identify_side_effects(scenario, current_factors)
            
            # Estimate timeline
            timeline_estimate = scenario.timeline or await self._estimate_timeline(scenario)
            
            return ScenarioResult(
                scenario=scenario,
                predicted_score=predicted_score,
                score_change=predicted_change,
                confidence=confidence,
                risk_assessment=risk_assessment,
                timeline_estimate=timeline_estimate,
                implementation_steps=implementation_steps,
                side_effects=side_effects
            )
            
        except Exception as e:
            logger.error(f"Error analyzing scenario {scenario.name}: {e}")
            return ScenarioResult(
                scenario=scenario,
                predicted_score=current_profile.get('credit_score', 650),
                score_change=0,
                confidence=0.0,
                risk_assessment={'error': str(e)},
                timeline_estimate="unknown",
                implementation_steps=[],
                side_effects=[]
            )
    
    async def _calculate_score_change(self, changes: Dict[str, float], 
                                    current_factors: Dict[str, float]) -> float:
        """Calculate predicted score change from factor changes"""
        
        try:
            total_change = 0.0
            
            # Direct factor impacts
            for factor, change in changes.items():
                if factor in self.factor_impacts:
                    impact_config = self.factor_impacts[factor]
                    weight = impact_config['weight']
                    max_impact = impact_config['max_impact']
                    
                    # Apply improvement/degradation rates
                    if change > 0:
                        rate = impact_config['improvement_rate']
                    else:
                        rate = impact_config['degradation_rate']
                    
                    # Calculate impact
                    factor_impact = change * weight * max_impact * rate
                    total_change += factor_impact
            
            # Interaction effects
            interaction_bonus = await self._calculate_interaction_effects(changes)
            total_change += interaction_bonus
            
            # Apply diminishing returns for large changes
            if abs(total_change) > 50:
                total_change = np.sign(total_change) * (50 + (abs(total_change) - 50) * 0.5)
            
            return total_change
            
        except Exception as e:
            logger.error(f"Error calculating score change: {e}")
            return 0.0
    
    async def _calculate_interaction_effects(self, changes: Dict[str, float]) -> float:
        """Calculate interaction effects between changed factors"""
        
        try:
            interaction_bonus = 0.0
            changed_factors = list(changes.keys())
            
            for i, factor1 in enumerate(changed_factors):
                for factor2 in changed_factors[i+1:]:
                    # Check for interaction effect
                    interaction_key = (factor1, factor2)
                    reverse_key = (factor2, factor1)
                    
                    if interaction_key in self.interaction_effects:
                        effect = self.interaction_effects[interaction_key]
                    elif reverse_key in self.interaction_effects:
                        effect = self.interaction_effects[reverse_key]
                    else:
                        continue
                    
                    # Calculate interaction bonus
                    change1 = changes[factor1]
                    change2 = changes[factor2]
                    
                    # Positive interaction if both changes are in same direction
                    if (change1 > 0 and change2 > 0) or (change1 < 0 and change2 < 0):
                        interaction_bonus += abs(effect) * 20
                    else:
                        interaction_bonus -= abs(effect) * 10
            
            return interaction_bonus
            
        except Exception as e:
            logger.error(f"Error calculating interaction effects: {e}")
            return 0.0
    
    async def _calculate_confidence(self, scenario: WhatIfScenario, 
                                  current_factors: Dict[str, float]) -> float:
        """Calculate confidence in scenario prediction"""
        
        try:
            base_confidence = 0.7
            
            # Adjust based on scenario type
            if scenario.scenario_type == ScenarioType.SINGLE_FACTOR:
                base_confidence += 0.1
            elif scenario.scenario_type == ScenarioType.MULTIPLE_FACTORS:
                base_confidence -= 0.1
            elif scenario.scenario_type == ScenarioType.STRESS_TEST:
                base_confidence -= 0.2
            
            # Adjust based on number of changes
            num_changes = len(scenario.changes)
            if num_changes > 3:
                base_confidence -= (num_changes - 3) * 0.05
            
            # Adjust based on magnitude of changes
            total_magnitude = sum(abs(change) for change in scenario.changes.values())
            if total_magnitude > 0.5:
                base_confidence -= (total_magnitude - 0.5) * 0.2
            
            # Adjust based on current factor values
            for factor, change in scenario.changes.items():
                current_value = current_factors.get(factor, 0.5)
                
                # Lower confidence if pushing factors to extremes
                new_value = current_value + change
                if new_value < 0.1 or new_value > 0.9:
                    base_confidence -= 0.1
            
            return max(0.1, min(0.95, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _assess_scenario_risks(self, scenario: WhatIfScenario, 
                                   current_factors: Dict[str, float]) -> Dict[str, Any]:
        """Assess risks associated with scenario"""
        
        try:
            risks = {
                'overall_risk': 'low',
                'financial_risk': 'low',
                'timeline_risk': 'low',
                'implementation_risk': 'low',
                'specific_risks': []
            }
            
            # Assess based on scenario type
            if scenario.scenario_type == ScenarioType.STRESS_TEST:
                risks['overall_risk'] = 'high'
                risks['specific_risks'].append("Stress test scenario may not reflect realistic conditions")
            
            # Assess financial risks
            if 'new_credit' in scenario.changes and scenario.changes['new_credit'] < -0.1:
                risks['financial_risk'] = 'medium'
                risks['specific_risks'].append("New credit applications may temporarily lower score")
            
            # Assess timeline risks
            if scenario.timeline and ('12' in scenario.timeline or '24' in scenario.timeline):
                risks['timeline_risk'] = 'medium'
                risks['specific_risks'].append("Long timeline increases uncertainty")
            
            # Assess implementation risks
            if scenario.effort_level == 'high':
                risks['implementation_risk'] = 'medium'
                risks['specific_risks'].append("High effort requirement may affect completion")
            
            # Check for extreme changes
            for factor, change in scenario.changes.items():
                if abs(change) > 0.3:
                    risks['overall_risk'] = 'medium'
                    risks['specific_risks'].append(f"Large change in {factor} may be difficult to achieve")
            
            return risks
            
        except Exception as e:
            logger.error(f"Error assessing scenario risks: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}
    
    async def _generate_implementation_steps(self, scenario: WhatIfScenario) -> List[str]:
        """Generate implementation steps for scenario"""
        
        try:
            steps = []
            
            # Generate steps based on scenario changes
            for factor, change in scenario.changes.items():
                if factor == 'credit_utilization' and change < 0:
                    steps.extend([
                        "Calculate total credit card debt",
                        "Create debt paydown plan",
                        "Make extra payments to reduce balances",
                        "Monitor utilization ratio monthly"
                    ])
                elif factor == 'payment_history' and change > 0:
                    steps.extend([
                        "Set up automatic payments for all accounts",
                        "Create payment calendar with due dates",
                        "Monitor accounts for on-time payments",
                        "Review payment history quarterly"
                    ])
                elif factor == 'credit_mix' and change > 0:
                    steps.extend([
                        "Research appropriate credit products",
                        "Compare terms and conditions",
                        "Apply for new credit type",
                        "Manage new account responsibly"
                    ])
                elif factor == 'new_credit' and change < 0:
                    steps.extend([
                        "Avoid new credit applications",
                        "Wait for hard inquiries to age",
                        "Focus on existing account management"
                    ])
                elif factor == 'credit_length':
                    if change > 0:
                        steps.extend([
                            "Keep oldest accounts open and active",
                            "Make small purchases on old cards",
                            "Pay off balances monthly"
                        ])
                    else:
                        steps.extend([
                            "Review accounts to close",
                            "Transfer balances if needed",
                            "Close accounts strategically"
                        ])
            
            # Add general monitoring steps
            steps.extend([
                "Monitor credit score monthly",
                "Review credit reports quarterly",
                "Track progress toward goals"
            ])
            
            # Remove duplicates while preserving order
            unique_steps = []
            for step in steps:
                if step not in unique_steps:
                    unique_steps.append(step)
            
            return unique_steps[:8]  # Limit to 8 steps
            
        except Exception as e:
            logger.error(f"Error generating implementation steps: {e}")
            return ["Monitor progress and adjust strategy as needed"]
    
    async def _identify_side_effects(self, scenario: WhatIfScenario, 
                                   current_factors: Dict[str, float]) -> List[str]:
        """Identify potential side effects of scenario"""
        
        try:
            side_effects = []
            
            # Check for common side effects
            if 'new_credit' in scenario.changes and scenario.changes['new_credit'] < 0:
                side_effects.append("Hard credit inquiries may temporarily lower score")
            
            if 'credit_utilization' in scenario.changes and scenario.changes['credit_utilization'] < -0.15:
                side_effects.append("Rapid debt paydown may strain cash flow")
            
            if 'credit_length' in scenario.changes and scenario.changes['credit_length'] < 0:
                side_effects.append("Closing accounts reduces available credit")
            
            # Check for interaction side effects
            changes = scenario.changes
            if 'credit_utilization' in changes and 'new_credit' in changes:
                if changes['credit_utilization'] < 0 and changes['new_credit'] < 0:
                    side_effects.append("Combining debt paydown with new credit may send mixed signals")
            
            # Check for timeline-related side effects
            if scenario.timeline and 'immediate' in scenario.timeline:
                side_effects.append("Immediate changes may not reflect in credit score right away")
            
            return side_effects
            
        except Exception as e:
            logger.error(f"Error identifying side effects: {e}")
            return []
    
    async def _estimate_timeline(self, scenario: WhatIfScenario) -> str:
        """Estimate timeline for scenario implementation"""
        
        try:
            # Base timeline on scenario type and complexity
            num_changes = len(scenario.changes)
            max_change = max(abs(change) for change in scenario.changes.values())
            
            if scenario.scenario_type == ScenarioType.SINGLE_FACTOR:
                if max_change < 0.1:
                    return "1-3 months"
                elif max_change < 0.2:
                    return "3-6 months"
                else:
                    return "6-12 months"
            
            elif scenario.scenario_type == ScenarioType.MULTIPLE_FACTORS:
                if num_changes <= 2:
                    return "3-6 months"
                elif num_changes <= 4:
                    return "6-12 months"
                else:
                    return "12+ months"
            
            elif scenario.scenario_type == ScenarioType.TEMPORAL:
                return "12-24 months"
            
            else:
                return "6-12 months"
            
        except Exception as e:
            logger.error(f"Error estimating timeline: {e}")
            return "unknown"
    
    async def analyze_multiple_scenarios(self, scenarios: List[WhatIfScenario],
                                       current_profile: Dict[str, Any]) -> List[ScenarioResult]:
        """Analyze multiple what-if scenarios"""
        
        try:
            results = []
            
            # Analyze scenarios in parallel
            tasks = [
                self.analyze_scenario(scenario, current_profile)
                for scenario in scenarios
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Sort by predicted score improvement
            results.sort(key=lambda x: x.score_change, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing multiple scenarios: {e}")
            return []
    
    async def generate_custom_scenario(self, target_score: float, 
                                     current_profile: Dict[str, Any],
                                     constraints: Dict[str, Any] = None) -> WhatIfScenario:
        """Generate custom scenario to reach target score"""
        
        try:
            current_score = current_profile.get('credit_score', 650)
            current_factors = current_profile.get('factors', {})
            
            target_change = target_score - current_score
            
            if abs(target_change) < 5:
                return WhatIfScenario(
                    name="Maintain Current Profile",
                    description="Continue current credit management practices",
                    scenario_type=ScenarioType.SINGLE_FACTOR,
                    changes={},
                    timeline="ongoing"
                )
            
            # Generate changes to reach target
            changes = {}
            remaining_change = target_change
            
            # Prioritize factors by impact and actionability
            factor_priority = [
                'credit_utilization',
                'payment_history', 
                'new_credit',
                'credit_mix',
                'credit_length'
            ]
            
            for factor in factor_priority:
                if abs(remaining_change) < 5:
                    break
                
                if factor not in self.factor_impacts:
                    continue
                
                impact_config = self.factor_impacts[factor]
                max_factor_change = remaining_change / (impact_config['weight'] * impact_config['max_impact'])
                
                # Limit factor changes to reasonable ranges
                max_factor_change = max(-0.3, min(0.3, max_factor_change))
                
                if abs(max_factor_change) > 0.05:  # Only include meaningful changes
                    changes[factor] = max_factor_change
                    
                    # Estimate impact
                    estimated_impact = max_factor_change * impact_config['weight'] * impact_config['max_impact']
                    remaining_change -= estimated_impact
            
            scenario_name = f"Reach {int(target_score)} Credit Score"
            scenario_description = f"Custom plan to achieve target score of {int(target_score)}"
            
            return WhatIfScenario(
                name=scenario_name,
                description=scenario_description,
                scenario_type=ScenarioType.OPTIMIZATION,
                changes=changes,
                timeline=await self._estimate_timeline_for_changes(changes)
            )
            
        except Exception as e:
            logger.error(f"Error generating custom scenario: {e}")
            return WhatIfScenario(
                name="Error Scenario",
                description="Could not generate custom scenario",
                scenario_type=ScenarioType.SINGLE_FACTOR,
                changes={}
            )
    
    async def _estimate_timeline_for_changes(self, changes: Dict[str, float]) -> str:
        """Estimate timeline based on specific changes"""
        
        try:
            max_timeline_months = 0
            
            for factor, change in changes.items():
                if factor == 'payment_history':
                    # Payment history takes time to build
                    months = max(6, abs(change) * 24)
                elif factor == 'credit_utilization':
                    # Can change quickly with debt paydown
                    months = max(1, abs(change) * 6)
                elif factor == 'credit_length':
                    # Takes time to build, immediate to reduce
                    months = max(1, abs(change) * 12) if change > 0 else 1
                elif factor == 'new_credit':
                    # Impact fades over time
                    months = max(3, abs(change) * 12)
                elif factor == 'credit_mix':
                    # Moderate timeline
                    months = max(3, abs(change) * 9)
                else:
                    months = 6
                
                max_timeline_months = max(max_timeline_months, months)
            
            if max_timeline_months <= 3:
                return "1-3 months"
            elif max_timeline_months <= 6:
                return "3-6 months"
            elif max_timeline_months <= 12:
                return "6-12 months"
            elif max_timeline_months <= 24:
                return "12-24 months"
            else:
                return "24+ months"
            
        except Exception as e:
            logger.error(f"Error estimating timeline for changes: {e}")
            return "6-12 months"
    
    def get_predefined_scenarios(self) -> List[WhatIfScenario]:
        """Get list of predefined scenarios"""
        return list(self.scenario_templates.values())
    
    def get_scenario_by_name(self, name: str) -> Optional[WhatIfScenario]:
        """Get predefined scenario by name"""
        return self.scenario_templates.get(name)
    
    async def compare_scenarios(self, results: List[ScenarioResult]) -> Dict[str, Any]:
        """Compare multiple scenario results"""
        
        try:
            if not results:
                return {'error': 'No scenarios to compare'}
            
            comparison = {
                'best_score_improvement': None,
                'lowest_risk': None,
                'fastest_timeline': None,
                'easiest_implementation': None,
                'summary_stats': {}
            }
            
            # Find best in each category
            best_improvement = max(results, key=lambda x: x.score_change)
            comparison['best_score_improvement'] = {
                'scenario': best_improvement.scenario.name,
                'score_change': best_improvement.score_change,
                'confidence': best_improvement.confidence
            }
            
            # Find lowest risk (simplified)
            lowest_risk = min(results, key=lambda x: len(x.risk_assessment.get('specific_risks', [])))
            comparison['lowest_risk'] = {
                'scenario': lowest_risk.scenario.name,
                'risk_level': lowest_risk.risk_assessment.get('overall_risk', 'unknown'),
                'confidence': lowest_risk.confidence
            }
            
            # Summary statistics
            score_changes = [r.score_change for r in results]
            confidences = [r.confidence for r in results]
            
            comparison['summary_stats'] = {
                'avg_score_change': np.mean(score_changes),
                'max_score_change': np.max(score_changes),
                'min_score_change': np.min(score_changes),
                'avg_confidence': np.mean(confidences),
                'total_scenarios': len(results)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing scenarios: {e}")
            return {'error': str(e)}
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """Get what-if analyzer statistics"""
        
        return {
            'predefined_scenarios': len(self.scenario_templates),
            'supported_factors': list(self.factor_impacts.keys()),
            'interaction_effects': len(self.interaction_effects),
            'scenario_types': [t.value for t in ScenarioType],
            'timestamp': datetime.now().isoformat()
        }
