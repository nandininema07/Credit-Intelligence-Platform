"""
Sensitivity analysis for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)

class SensitivityMethod(Enum):
    """Sensitivity analysis methods"""
    ONE_AT_A_TIME = "one_at_a_time"
    GLOBAL_SENSITIVITY = "global_sensitivity"
    VARIANCE_BASED = "variance_based"
    MORRIS = "morris"
    SOBOL = "sobol"

@dataclass
class SensitivityResult:
    """Result of sensitivity analysis"""
    factor: str
    base_value: float
    sensitivity_score: float
    impact_range: Tuple[float, float]
    confidence: float
    method: SensitivityMethod
    perturbation_results: List[Dict[str, Any]]

class SensitivityAnalyzer:
    """Analyzer for factor sensitivity in credit profiles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perturbation_ranges = {}
        self.factor_constraints = {}
        self.interaction_matrix = {}
        self._initialize_analyzer()
        
    def _initialize_analyzer(self):
        """Initialize sensitivity analyzer"""
        
        # Define perturbation ranges for each factor
        self.perturbation_ranges = {
            'payment_history': {
                'min_change': -0.20,
                'max_change': 0.20,
                'step_size': 0.05,
                'realistic_range': (0.0, 1.0)
            },
            'credit_utilization': {
                'min_change': -0.30,
                'max_change': 0.30,
                'step_size': 0.05,
                'realistic_range': (0.0, 1.0)
            },
            'credit_length': {
                'min_change': -0.15,
                'max_change': 0.15,
                'step_size': 0.03,
                'realistic_range': (0.0, 1.0)
            },
            'credit_mix': {
                'min_change': -0.20,
                'max_change': 0.20,
                'step_size': 0.04,
                'realistic_range': (0.0, 1.0)
            },
            'new_credit': {
                'min_change': -0.25,
                'max_change': 0.25,
                'step_size': 0.05,
                'realistic_range': (0.0, 1.0)
            },
            'debt_to_income': {
                'min_change': -0.20,
                'max_change': 0.20,
                'step_size': 0.04,
                'realistic_range': (0.0, 1.0)
            },
            'account_age': {
                'min_change': -0.10,
                'max_change': 0.10,
                'step_size': 0.02,
                'realistic_range': (0.0, 1.0)
            }
        }
        
        # Factor constraints and dependencies
        self.factor_constraints = {
            'credit_utilization': {
                'depends_on': ['total_debt', 'total_credit_limit'],
                'constraint_type': 'ratio',
                'max_realistic': 0.9
            },
            'debt_to_income': {
                'depends_on': ['total_debt', 'monthly_income'],
                'constraint_type': 'ratio',
                'max_realistic': 0.8
            },
            'payment_history': {
                'constraint_type': 'percentage',
                'min_realistic': 0.0,
                'max_realistic': 1.0
            }
        }
        
        # Initialize interaction matrix for factor dependencies
        self._initialize_interaction_matrix()
    
    def _initialize_interaction_matrix(self):
        """Initialize factor interaction matrix"""
        
        self.interaction_matrix = {
            'payment_history': {
                'credit_utilization': 0.15,
                'credit_length': 0.10,
                'new_credit': 0.08
            },
            'credit_utilization': {
                'payment_history': 0.15,
                'debt_to_income': 0.25,
                'new_credit': -0.05
            },
            'credit_length': {
                'payment_history': 0.10,
                'credit_mix': 0.12,
                'account_age': 0.30
            },
            'credit_mix': {
                'credit_length': 0.12,
                'new_credit': 0.08
            },
            'new_credit': {
                'credit_utilization': -0.05,
                'credit_mix': 0.08,
                'payment_history': 0.08
            }
        }
    
    async def analyze_factor_sensitivity(self, factor: str, current_profile: Dict[str, Any],
                                       method: SensitivityMethod = SensitivityMethod.ONE_AT_A_TIME) -> SensitivityResult:
        """Analyze sensitivity of a single factor"""
        
        try:
            if factor not in self.perturbation_ranges:
                raise ValueError(f"Factor {factor} not supported for sensitivity analysis")
            
            base_value = current_profile.get('factors', {}).get(factor, 0.5)
            base_score = current_profile.get('credit_score', 650)
            
            # Get perturbation configuration
            perturbation_config = self.perturbation_ranges[factor]
            
            # Generate perturbations
            perturbations = await self._generate_perturbations(
                factor, base_value, perturbation_config, method
            )
            
            # Analyze each perturbation
            perturbation_results = []
            
            for perturbation in perturbations:
                result = await self._analyze_perturbation(
                    factor, perturbation, current_profile, base_score
                )
                perturbation_results.append(result)
            
            # Calculate sensitivity metrics
            sensitivity_score = await self._calculate_sensitivity_score(
                perturbation_results, method
            )
            
            # Determine impact range
            score_changes = [r['score_change'] for r in perturbation_results]
            impact_range = (min(score_changes), max(score_changes))
            
            # Calculate confidence
            confidence = await self._calculate_sensitivity_confidence(
                perturbation_results, method
            )
            
            return SensitivityResult(
                factor=factor,
                base_value=base_value,
                sensitivity_score=sensitivity_score,
                impact_range=impact_range,
                confidence=confidence,
                method=method,
                perturbation_results=perturbation_results
            )
            
        except Exception as e:
            logger.error(f"Error analyzing factor sensitivity for {factor}: {e}")
            return SensitivityResult(
                factor=factor,
                base_value=current_profile.get('factors', {}).get(factor, 0.5),
                sensitivity_score=0.0,
                impact_range=(0.0, 0.0),
                confidence=0.0,
                method=method,
                perturbation_results=[]
            )
    
    async def _generate_perturbations(self, factor: str, base_value: float,
                                    perturbation_config: Dict[str, Any],
                                    method: SensitivityMethod) -> List[float]:
        """Generate perturbation values for sensitivity analysis"""
        
        try:
            perturbations = []
            
            min_change = perturbation_config['min_change']
            max_change = perturbation_config['max_change']
            step_size = perturbation_config['step_size']
            realistic_range = perturbation_config['realistic_range']
            
            if method == SensitivityMethod.ONE_AT_A_TIME:
                # Generate systematic perturbations
                changes = np.arange(min_change, max_change + step_size, step_size)
                
                for change in changes:
                    new_value = base_value + change
                    
                    # Ensure within realistic range
                    new_value = max(realistic_range[0], min(realistic_range[1], new_value))
                    
                    if abs(new_value - base_value) > 0.01:  # Only meaningful changes
                        perturbations.append(new_value)
            
            elif method == SensitivityMethod.GLOBAL_SENSITIVITY:
                # Generate random perturbations across full range
                num_samples = 50
                
                for _ in range(num_samples):
                    change = np.random.uniform(min_change, max_change)
                    new_value = base_value + change
                    new_value = max(realistic_range[0], min(realistic_range[1], new_value))
                    perturbations.append(new_value)
            
            elif method == SensitivityMethod.MORRIS:
                # Morris method perturbations
                num_trajectories = 10
                num_levels = 4
                
                for _ in range(num_trajectories):
                    # Generate trajectory of perturbations
                    level_values = np.linspace(realistic_range[0], realistic_range[1], num_levels)
                    selected_values = np.random.choice(level_values, size=3, replace=False)
                    perturbations.extend(selected_values)
            
            else:
                # Default to one-at-a-time
                changes = np.arange(min_change, max_change + step_size, step_size)
                for change in changes:
                    new_value = max(realistic_range[0], min(realistic_range[1], base_value + change))
                    perturbations.append(new_value)
            
            return list(set(perturbations))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error generating perturbations: {e}")
            return [base_value]
    
    async def _analyze_perturbation(self, factor: str, new_value: float,
                                  current_profile: Dict[str, Any], base_score: float) -> Dict[str, Any]:
        """Analyze impact of single perturbation"""
        
        try:
            # Create modified profile
            modified_profile = current_profile.copy()
            modified_factors = modified_profile.get('factors', {}).copy()
            modified_factors[factor] = new_value
            modified_profile['factors'] = modified_factors
            
            # Calculate new score (simplified model)
            new_score = await self._predict_score_with_factors(modified_factors)
            
            # Calculate change
            score_change = new_score - base_score
            factor_change = new_value - current_profile.get('factors', {}).get(factor, 0.5)
            
            # Calculate local sensitivity (derivative approximation)
            local_sensitivity = score_change / factor_change if abs(factor_change) > 0.001 else 0
            
            # Consider interaction effects
            interaction_effect = await self._calculate_interaction_impact(
                factor, new_value, modified_factors
            )
            
            return {
                'factor_value': new_value,
                'factor_change': factor_change,
                'predicted_score': new_score,
                'score_change': score_change,
                'local_sensitivity': local_sensitivity,
                'interaction_effect': interaction_effect,
                'total_effect': score_change + interaction_effect
            }
            
        except Exception as e:
            logger.error(f"Error analyzing perturbation: {e}")
            return {
                'factor_value': new_value,
                'factor_change': 0,
                'predicted_score': base_score,
                'score_change': 0,
                'local_sensitivity': 0,
                'interaction_effect': 0,
                'total_effect': 0
            }
    
    async def _predict_score_with_factors(self, factors: Dict[str, float]) -> float:
        """Predict credit score based on factor values (simplified model)"""
        
        try:
            # Simplified credit score model
            base_score = 300
            
            # Factor weights (simplified FICO-like model)
            weights = {
                'payment_history': 175,  # 35% of 500 point range
                'credit_utilization': 150,  # 30% of 500 point range
                'credit_length': 75,   # 15% of 500 point range
                'credit_mix': 50,      # 10% of 500 point range
                'new_credit': 50,      # 10% of 500 point range
                'debt_to_income': 40,  # Additional factor
                'account_age': 30      # Additional factor
            }
            
            total_score = base_score
            
            for factor, value in factors.items():
                if factor in weights:
                    # Convert factor value (0-1) to score contribution
                    contribution = value * weights[factor]
                    total_score += contribution
            
            # Apply non-linear adjustments
            if factors.get('credit_utilization', 0.5) > 0.7:
                # High utilization penalty
                penalty = (factors['credit_utilization'] - 0.7) * 100
                total_score -= penalty
            
            if factors.get('payment_history', 0.5) < 0.8:
                # Poor payment history penalty
                penalty = (0.8 - factors['payment_history']) * 150
                total_score -= penalty
            
            # Ensure score is within valid range
            total_score = max(300, min(850, total_score))
            
            return total_score
            
        except Exception as e:
            logger.error(f"Error predicting score: {e}")
            return 650  # Default score
    
    async def _calculate_interaction_impact(self, changed_factor: str, new_value: float,
                                          all_factors: Dict[str, float]) -> float:
        """Calculate interaction effects when factor changes"""
        
        try:
            interaction_impact = 0.0
            
            if changed_factor not in self.interaction_matrix:
                return interaction_impact
            
            interactions = self.interaction_matrix[changed_factor]
            
            for other_factor, interaction_strength in interactions.items():
                if other_factor in all_factors:
                    other_value = all_factors[other_factor]
                    
                    # Calculate interaction effect
                    # Positive interactions amplify when both factors are good
                    # Negative interactions create penalties
                    
                    if interaction_strength > 0:
                        # Positive interaction
                        interaction_impact += interaction_strength * new_value * other_value * 20
                    else:
                        # Negative interaction
                        interaction_impact += interaction_strength * new_value * other_value * 20
            
            return interaction_impact
            
        except Exception as e:
            logger.error(f"Error calculating interaction impact: {e}")
            return 0.0
    
    async def _calculate_sensitivity_score(self, perturbation_results: List[Dict[str, Any]],
                                         method: SensitivityMethod) -> float:
        """Calculate overall sensitivity score"""
        
        try:
            if not perturbation_results:
                return 0.0
            
            if method == SensitivityMethod.ONE_AT_A_TIME:
                # Use average absolute local sensitivity
                sensitivities = [abs(r['local_sensitivity']) for r in perturbation_results]
                return np.mean(sensitivities)
            
            elif method == SensitivityMethod.GLOBAL_SENSITIVITY:
                # Use variance of score changes
                score_changes = [r['score_change'] for r in perturbation_results]
                return np.std(score_changes)
            
            elif method == SensitivityMethod.MORRIS:
                # Morris method uses mean and standard deviation
                sensitivities = [abs(r['local_sensitivity']) for r in perturbation_results]
                mean_sensitivity = np.mean(sensitivities)
                std_sensitivity = np.std(sensitivities)
                return mean_sensitivity + 0.5 * std_sensitivity
            
            else:
                # Default calculation
                score_changes = [abs(r['score_change']) for r in perturbation_results]
                return np.mean(score_changes)
            
        except Exception as e:
            logger.error(f"Error calculating sensitivity score: {e}")
            return 0.0
    
    async def _calculate_sensitivity_confidence(self, perturbation_results: List[Dict[str, Any]],
                                              method: SensitivityMethod) -> float:
        """Calculate confidence in sensitivity analysis"""
        
        try:
            if not perturbation_results:
                return 0.0
            
            base_confidence = 0.7
            
            # Adjust based on number of perturbations
            num_perturbations = len(perturbation_results)
            if num_perturbations >= 20:
                base_confidence += 0.2
            elif num_perturbations >= 10:
                base_confidence += 0.1
            elif num_perturbations < 5:
                base_confidence -= 0.2
            
            # Adjust based on consistency of results
            score_changes = [r['score_change'] for r in perturbation_results]
            if len(score_changes) > 1:
                cv = np.std(score_changes) / (abs(np.mean(score_changes)) + 0.001)
                if cv < 0.5:  # Low coefficient of variation = consistent results
                    base_confidence += 0.1
                elif cv > 2.0:  # High variation = less reliable
                    base_confidence -= 0.1
            
            # Adjust based on method
            if method == SensitivityMethod.GLOBAL_SENSITIVITY:
                base_confidence += 0.05
            elif method == SensitivityMethod.MORRIS:
                base_confidence += 0.1
            
            return max(0.1, min(0.95, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating sensitivity confidence: {e}")
            return 0.5
    
    async def analyze_all_factors(self, current_profile: Dict[str, Any],
                                method: SensitivityMethod = SensitivityMethod.ONE_AT_A_TIME) -> List[SensitivityResult]:
        """Analyze sensitivity of all factors"""
        
        try:
            factors = list(self.perturbation_ranges.keys())
            
            # Analyze factors in parallel
            tasks = [
                self.analyze_factor_sensitivity(factor, current_profile, method)
                for factor in factors
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Sort by sensitivity score
            results.sort(key=lambda x: x.sensitivity_score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing all factors: {e}")
            return []
    
    async def identify_critical_factors(self, current_profile: Dict[str, Any],
                                      threshold: float = 0.5) -> List[str]:
        """Identify factors with high sensitivity (critical factors)"""
        
        try:
            sensitivity_results = await self.analyze_all_factors(current_profile)
            
            critical_factors = []
            
            for result in sensitivity_results:
                if result.sensitivity_score >= threshold:
                    critical_factors.append(result.factor)
            
            return critical_factors
            
        except Exception as e:
            logger.error(f"Error identifying critical factors: {e}")
            return []
    
    async def analyze_factor_interactions(self, current_profile: Dict[str, Any],
                                        factor_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Analyze interactions between factor pairs"""
        
        try:
            if factor_pairs is None:
                # Generate common factor pairs
                factors = ['payment_history', 'credit_utilization', 'credit_length', 'new_credit']
                factor_pairs = [(factors[i], factors[j]) for i in range(len(factors)) 
                               for j in range(i+1, len(factors))]
            
            interaction_results = {}
            
            for factor1, factor2 in factor_pairs:
                interaction_strength = await self._analyze_pairwise_interaction(
                    factor1, factor2, current_profile
                )
                
                pair_key = f"{factor1}_{factor2}"
                interaction_results[pair_key] = {
                    'factor1': factor1,
                    'factor2': factor2,
                    'interaction_strength': interaction_strength,
                    'interaction_type': 'synergistic' if interaction_strength > 0 else 'antagonistic'
                }
            
            # Sort by interaction strength
            sorted_interactions = sorted(
                interaction_results.items(),
                key=lambda x: abs(x[1]['interaction_strength']),
                reverse=True
            )
            
            return {
                'interactions': dict(sorted_interactions),
                'strongest_interaction': sorted_interactions[0] if sorted_interactions else None,
                'total_interactions': len(interaction_results)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing factor interactions: {e}")
            return {'error': str(e)}
    
    async def _analyze_pairwise_interaction(self, factor1: str, factor2: str,
                                          current_profile: Dict[str, Any]) -> float:
        """Analyze interaction between two factors"""
        
        try:
            base_factors = current_profile.get('factors', {})
            base_score = current_profile.get('credit_score', 650)
            
            # Get base values
            base_value1 = base_factors.get(factor1, 0.5)
            base_value2 = base_factors.get(factor2, 0.5)
            
            # Test different combinations
            test_change = 0.1
            
            # Individual effects
            factors1_changed = base_factors.copy()
            factors1_changed[factor1] = min(1.0, base_value1 + test_change)
            score1 = await self._predict_score_with_factors(factors1_changed)
            effect1 = score1 - base_score
            
            factors2_changed = base_factors.copy()
            factors2_changed[factor2] = min(1.0, base_value2 + test_change)
            score2 = await self._predict_score_with_factors(factors2_changed)
            effect2 = score2 - base_score
            
            # Combined effect
            factors_both_changed = base_factors.copy()
            factors_both_changed[factor1] = min(1.0, base_value1 + test_change)
            factors_both_changed[factor2] = min(1.0, base_value2 + test_change)
            score_both = await self._predict_score_with_factors(factors_both_changed)
            effect_both = score_both - base_score
            
            # Interaction effect = combined effect - sum of individual effects
            interaction_effect = effect_both - (effect1 + effect2)
            
            return interaction_effect
            
        except Exception as e:
            logger.error(f"Error analyzing pairwise interaction: {e}")
            return 0.0
    
    async def generate_sensitivity_report(self, current_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive sensitivity analysis report"""
        
        try:
            # Analyze all factors
            sensitivity_results = await self.analyze_all_factors(current_profile)
            
            # Identify critical factors
            critical_factors = await self.identify_critical_factors(current_profile)
            
            # Analyze interactions
            interaction_analysis = await self.analyze_factor_interactions(current_profile)
            
            # Generate summary statistics
            sensitivity_scores = [r.sensitivity_score for r in sensitivity_results]
            
            report = {
                'profile_summary': {
                    'current_score': current_profile.get('credit_score', 650),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'sensitivity_analysis': {
                    'factor_sensitivities': [
                        {
                            'factor': r.factor,
                            'sensitivity_score': r.sensitivity_score,
                            'impact_range': r.impact_range,
                            'confidence': r.confidence
                        }
                        for r in sensitivity_results
                    ],
                    'most_sensitive_factor': sensitivity_results[0].factor if sensitivity_results else None,
                    'least_sensitive_factor': sensitivity_results[-1].factor if sensitivity_results else None,
                    'average_sensitivity': np.mean(sensitivity_scores) if sensitivity_scores else 0,
                    'sensitivity_variance': np.var(sensitivity_scores) if sensitivity_scores else 0
                },
                'critical_factors': {
                    'factors': critical_factors,
                    'count': len(critical_factors),
                    'recommendation': 'Focus improvement efforts on these high-impact factors'
                },
                'factor_interactions': interaction_analysis,
                'recommendations': await self._generate_sensitivity_recommendations(
                    sensitivity_results, critical_factors, interaction_analysis
                )
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating sensitivity report: {e}")
            return {'error': str(e)}
    
    async def _generate_sensitivity_recommendations(self, sensitivity_results: List[SensitivityResult],
                                                  critical_factors: List[str],
                                                  interaction_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on sensitivity analysis"""
        
        try:
            recommendations = []
            
            # Recommendations based on most sensitive factors
            if sensitivity_results:
                top_factor = sensitivity_results[0]
                recommendations.append(
                    f"Focus on {top_factor.factor.replace('_', ' ')} as it has the highest impact on your credit score"
                )
                
                if top_factor.impact_range[1] > 20:
                    recommendations.append(
                        f"Improving {top_factor.factor.replace('_', ' ')} could increase your score by up to {top_factor.impact_range[1]:.0f} points"
                    )
            
            # Recommendations for critical factors
            if len(critical_factors) > 1:
                recommendations.append(
                    f"Address multiple critical factors ({', '.join(critical_factors[:3])}) for maximum impact"
                )
            
            # Recommendations based on interactions
            if 'strongest_interaction' in interaction_analysis and interaction_analysis['strongest_interaction']:
                strongest = interaction_analysis['strongest_interaction'][1]
                if strongest['interaction_strength'] > 5:
                    recommendations.append(
                        f"Improve both {strongest['factor1']} and {strongest['factor2']} together for synergistic effects"
                    )
            
            # General recommendations
            recommendations.append("Monitor sensitive factors regularly as small changes can have significant impact")
            
            if not critical_factors:
                recommendations.append("Your profile shows balanced factor sensitivity - maintain current practices")
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating sensitivity recommendations: {e}")
            return ["Continue monitoring your credit profile for changes"]
    
    def get_supported_factors(self) -> List[str]:
        """Get list of factors supported for sensitivity analysis"""
        return list(self.perturbation_ranges.keys())
    
    def get_supported_methods(self) -> List[SensitivityMethod]:
        """Get list of supported sensitivity analysis methods"""
        return list(SensitivityMethod)
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """Get sensitivity analyzer statistics"""
        
        return {
            'supported_factors': len(self.perturbation_ranges),
            'supported_methods': len(SensitivityMethod),
            'interaction_pairs': len(self.interaction_matrix),
            'factor_constraints': len(self.factor_constraints),
            'timestamp': datetime.now().isoformat()
        }
