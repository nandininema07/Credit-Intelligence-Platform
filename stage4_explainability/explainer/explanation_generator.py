"""
Explanation generator for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ExplanationType(Enum):
    """Types of explanations"""
    LOCAL = "local"
    GLOBAL = "global"
    COUNTERFACTUAL = "counterfactual"
    FEATURE_IMPORTANCE = "feature_importance"
    WHAT_IF = "what_if"
    NARRATIVE = "narrative"

@dataclass
class ExplanationRequest:
    """Explanation request data structure"""
    request_id: str
    user_id: str
    explanation_type: ExplanationType
    model_prediction: Dict[str, Any]
    instance_data: pd.DataFrame
    context: Dict[str, Any]
    preferences: Dict[str, Any]
    timestamp: datetime

@dataclass
class ExplanationResult:
    """Explanation result data structure"""
    request_id: str
    explanation_type: ExplanationType
    explanation_data: Dict[str, Any]
    narrative: str
    visualization_data: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime

class ExplanationGenerator:
    """Generate comprehensive explanations for credit decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.xai_explainers = {}
        self.narrative_builder = None
        self.visualization_generator = None
        self.explanation_cache = None
        self.supported_types = [
            ExplanationType.LOCAL,
            ExplanationType.GLOBAL,
            ExplanationType.COUNTERFACTUAL,
            ExplanationType.FEATURE_IMPORTANCE,
            ExplanationType.WHAT_IF,
            ExplanationType.NARRATIVE
        ]
        
    def initialize(self, xai_explainers: Dict[str, Any], narrative_builder,
                  visualization_generator, explanation_cache):
        """Initialize explanation generator with components"""
        
        self.xai_explainers = xai_explainers
        self.narrative_builder = narrative_builder
        self.visualization_generator = visualization_generator
        self.explanation_cache = explanation_cache
        
        logger.info("Explanation generator initialized")
    
    async def generate_explanation(self, request: ExplanationRequest) -> ExplanationResult:
        """Generate explanation based on request"""
        
        try:
            # Check cache first
            cached_result = await self.explanation_cache.get_explanation(
                request.request_id, request.explanation_type
            )
            
            if cached_result:
                logger.info(f"Retrieved cached explanation for {request.request_id}")
                return cached_result
            
            # Generate new explanation
            explanation_data = await self._generate_explanation_data(request)
            
            # Generate narrative
            narrative = await self.narrative_builder.build_narrative(
                request.explanation_type, explanation_data, request.context
            )
            
            # Generate visualization data
            visualization_data = await self.visualization_generator.generate_visualization_data(
                request.explanation_type, explanation_data, request.preferences
            )
            
            # Calculate confidence
            confidence = self._calculate_explanation_confidence(explanation_data)
            
            # Create result
            result = ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_data=explanation_data,
                narrative=narrative,
                visualization_data=visualization_data,
                confidence=confidence,
                metadata={
                    'generation_method': self._get_generation_method(request.explanation_type),
                    'processing_time': (datetime.now() - request.timestamp).total_seconds(),
                    'data_points_analyzed': len(request.instance_data),
                    'context_factors': len(request.context)
                },
                timestamp=datetime.now()
            )
            
            # Cache result
            await self.explanation_cache.cache_explanation(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return ExplanationResult(
                request_id=request.request_id,
                explanation_type=request.explanation_type,
                explanation_data={'error': str(e)},
                narrative=f"I encountered an error generating the explanation: {str(e)}",
                visualization_data={},
                confidence=0.0,
                metadata={'error': True},
                timestamp=datetime.now()
            )
    
    async def _generate_explanation_data(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate explanation data based on type"""
        
        explanation_type = request.explanation_type
        
        if explanation_type == ExplanationType.LOCAL:
            return await self._generate_local_explanation(request)
        elif explanation_type == ExplanationType.GLOBAL:
            return await self._generate_global_explanation(request)
        elif explanation_type == ExplanationType.COUNTERFACTUAL:
            return await self._generate_counterfactual_explanation(request)
        elif explanation_type == ExplanationType.FEATURE_IMPORTANCE:
            return await self._generate_feature_importance_explanation(request)
        elif explanation_type == ExplanationType.WHAT_IF:
            return await self._generate_what_if_explanation(request)
        elif explanation_type == ExplanationType.NARRATIVE:
            return await self._generate_narrative_explanation(request)
        else:
            raise ValueError(f"Unsupported explanation type: {explanation_type}")
    
    async def _generate_local_explanation(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate local explanation for specific instance"""
        
        try:
            explanation_data = {
                'explanation_type': 'local',
                'instance_id': request.context.get('instance_id', 'unknown'),
                'prediction': request.model_prediction,
                'explanations': {}
            }
            
            # SHAP explanation
            if 'shap' in self.xai_explainers:
                shap_result = await self.xai_explainers['shap'].explain_instance(
                    request.instance_data
                )
                explanation_data['explanations']['shap'] = shap_result
            
            # LIME explanation
            if 'lime' in self.xai_explainers:
                lime_result = await self.xai_explainers['lime'].explain_instance(
                    request.instance_data
                )
                explanation_data['explanations']['lime'] = lime_result
            
            # Feature attribution
            if 'feature_attribution' in self.xai_explainers:
                attribution_result = await self.xai_explainers['feature_attribution'].analyze_instance(
                    request.instance_data
                )
                explanation_data['explanations']['feature_attribution'] = attribution_result
            
            # Combine explanations
            explanation_data['combined_importance'] = self._combine_local_explanations(
                explanation_data['explanations']
            )
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating local explanation: {e}")
            return {'error': str(e)}
    
    async def _generate_global_explanation(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate global explanation for model behavior"""
        
        try:
            explanation_data = {
                'explanation_type': 'global',
                'model_info': request.context.get('model_info', {}),
                'explanations': {}
            }
            
            # Global feature importance
            if 'global_explainer' in self.xai_explainers:
                global_result = await self.xai_explainers['global_explainer'].generate_global_feature_importance()
                explanation_data['explanations']['global_importance'] = global_result
            
            # Partial dependence plots
            if 'global_explainer' in self.xai_explainers:
                pdp_result = await self.xai_explainers['global_explainer'].generate_partial_dependence_plots()
                explanation_data['explanations']['partial_dependence'] = pdp_result
            
            # Feature interactions
            if 'global_explainer' in self.xai_explainers:
                interaction_result = await self.xai_explainers['global_explainer'].analyze_feature_interactions()
                explanation_data['explanations']['feature_interactions'] = interaction_result
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating global explanation: {e}")
            return {'error': str(e)}
    
    async def _generate_counterfactual_explanation(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate counterfactual explanation"""
        
        try:
            explanation_data = {
                'explanation_type': 'counterfactual',
                'original_instance': request.instance_data.iloc[0].to_dict(),
                'original_prediction': request.model_prediction,
                'counterfactuals': []
            }
            
            if 'counterfactual' in self.xai_explainers:
                # Generate single counterfactual
                cf_result = await self.xai_explainers['counterfactual'].generate_counterfactual(
                    request.instance_data,
                    desired_outcome=request.context.get('desired_outcome')
                )
                explanation_data['counterfactuals'].append(cf_result)
                
                # Generate diverse counterfactuals
                diverse_cf = await self.xai_explainers['counterfactual'].analyze_counterfactual_diversity(
                    request.instance_data
                )
                explanation_data['diverse_counterfactuals'] = diverse_cf
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating counterfactual explanation: {e}")
            return {'error': str(e)}
    
    async def _generate_feature_importance_explanation(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate feature importance explanation"""
        
        try:
            explanation_data = {
                'explanation_type': 'feature_importance',
                'instance_data': request.instance_data.iloc[0].to_dict(),
                'importance_methods': {}
            }
            
            # Multiple importance methods
            if 'feature_attribution' in self.xai_explainers:
                attribution_result = await self.xai_explainers['feature_attribution'].analyze_instance(
                    request.instance_data
                )
                explanation_data['importance_methods']['attribution'] = attribution_result
            
            if 'global_explainer' in self.xai_explainers:
                global_importance = await self.xai_explainers['global_explainer'].generate_global_feature_importance()
                explanation_data['importance_methods']['global'] = global_importance
            
            # Create consensus ranking
            explanation_data['consensus_importance'] = self._create_importance_consensus(
                explanation_data['importance_methods']
            )
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating feature importance explanation: {e}")
            return {'error': str(e)}
    
    async def _generate_what_if_explanation(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate what-if scenario explanation"""
        
        try:
            explanation_data = {
                'explanation_type': 'what_if',
                'base_instance': request.instance_data.iloc[0].to_dict(),
                'base_prediction': request.model_prediction,
                'scenarios': []
            }
            
            # Get scenario parameters from context
            scenarios = request.context.get('scenarios', [])
            
            if not scenarios:
                # Generate default scenarios
                scenarios = self._generate_default_scenarios(request.instance_data)
            
            # Analyze each scenario
            for scenario in scenarios:
                scenario_result = await self._analyze_scenario(
                    request.instance_data, scenario, request.context
                )
                explanation_data['scenarios'].append(scenario_result)
            
            # Rank scenarios by impact
            explanation_data['ranked_scenarios'] = self._rank_scenarios_by_impact(
                explanation_data['scenarios']
            )
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating what-if explanation: {e}")
            return {'error': str(e)}
    
    async def _generate_narrative_explanation(self, request: ExplanationRequest) -> Dict[str, Any]:
        """Generate narrative explanation combining multiple methods"""
        
        try:
            explanation_data = {
                'explanation_type': 'narrative',
                'components': {}
            }
            
            # Generate multiple explanation types
            local_request = ExplanationRequest(
                request_id=f"{request.request_id}_local",
                user_id=request.user_id,
                explanation_type=ExplanationType.LOCAL,
                model_prediction=request.model_prediction,
                instance_data=request.instance_data,
                context=request.context,
                preferences=request.preferences,
                timestamp=request.timestamp
            )
            
            local_data = await self._generate_local_explanation(local_request)
            explanation_data['components']['local'] = local_data
            
            # Feature importance
            importance_request = ExplanationRequest(
                request_id=f"{request.request_id}_importance",
                user_id=request.user_id,
                explanation_type=ExplanationType.FEATURE_IMPORTANCE,
                model_prediction=request.model_prediction,
                instance_data=request.instance_data,
                context=request.context,
                preferences=request.preferences,
                timestamp=request.timestamp
            )
            
            importance_data = await self._generate_feature_importance_explanation(importance_request)
            explanation_data['components']['importance'] = importance_data
            
            # Combine into comprehensive narrative
            explanation_data['comprehensive_analysis'] = self._combine_narrative_components(
                explanation_data['components']
            )
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error generating narrative explanation: {e}")
            return {'error': str(e)}
    
    def _combine_local_explanations(self, explanations: Dict[str, Any]) -> Dict[str, float]:
        """Combine multiple local explanations into consensus"""
        
        try:
            combined_importance = {}
            method_count = 0
            
            for method, explanation in explanations.items():
                if 'error' in explanation:
                    continue
                
                method_count += 1
                
                # Extract feature importance from different methods
                if method == 'shap' and 'feature_importance' in explanation:
                    for feature, importance in explanation['feature_importance'].items():
                        combined_importance[feature] = combined_importance.get(feature, 0) + abs(importance)
                
                elif method == 'lime' and 'feature_importance' in explanation:
                    for feature, importance in explanation['feature_importance'].items():
                        combined_importance[feature] = combined_importance.get(feature, 0) + abs(importance)
                
                elif method == 'feature_attribution' and 'importance_scores' in explanation:
                    for feature, importance in explanation['importance_scores'].items():
                        combined_importance[feature] = combined_importance.get(feature, 0) + abs(importance)
            
            # Average across methods
            if method_count > 0:
                for feature in combined_importance:
                    combined_importance[feature] /= method_count
            
            return combined_importance
            
        except Exception as e:
            logger.error(f"Error combining local explanations: {e}")
            return {}
    
    def _create_importance_consensus(self, importance_methods: Dict[str, Any]) -> Dict[str, Any]:
        """Create consensus ranking from multiple importance methods"""
        
        try:
            feature_scores = {}
            
            for method, data in importance_methods.items():
                if 'error' in data:
                    continue
                
                # Extract importance scores from different methods
                if 'importance_scores' in data:
                    scores = data['importance_scores']
                elif 'importance_methods' in data:
                    scores = data['importance_methods'].get('consensus', {})
                else:
                    continue
                
                for feature, score in scores.items():
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(score)
            
            # Calculate consensus scores
            consensus = {}
            for feature, scores in feature_scores.items():
                consensus[feature] = np.mean(scores)
            
            # Sort by importance
            sorted_features = sorted(consensus.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'consensus_scores': consensus,
                'ranked_features': sorted_features,
                'top_features': sorted_features[:5]
            }
            
        except Exception as e:
            logger.error(f"Error creating importance consensus: {e}")
            return {}
    
    def _generate_default_scenarios(self, instance_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate default what-if scenarios"""
        
        scenarios = []
        instance = instance_data.iloc[0]
        
        # Scenario 1: Improve payment history
        scenarios.append({
            'name': 'Perfect Payment History',
            'description': 'What if you had perfect payment history',
            'changes': {'payment_history_score': 100}
        })
        
        # Scenario 2: Reduce credit utilization
        if 'credit_utilization' in instance:
            current_util = instance['credit_utilization']
            scenarios.append({
                'name': 'Lower Credit Utilization',
                'description': 'What if you reduced credit utilization to 10%',
                'changes': {'credit_utilization': min(10, current_util * 0.5)}
            })
        
        # Scenario 3: Increase income
        if 'annual_income' in instance:
            current_income = instance['annual_income']
            scenarios.append({
                'name': 'Higher Income',
                'description': 'What if your income increased by 20%',
                'changes': {'annual_income': current_income * 1.2}
            })
        
        return scenarios
    
    async def _analyze_scenario(self, base_instance: pd.DataFrame, scenario: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific what-if scenario"""
        
        try:
            # Create modified instance
            modified_instance = base_instance.copy()
            
            for feature, new_value in scenario['changes'].items():
                if feature in modified_instance.columns:
                    modified_instance[feature] = new_value
            
            # Get prediction for modified instance (would need model access)
            # For now, simulate prediction change
            scenario_result = {
                'scenario': scenario,
                'modified_instance': modified_instance.iloc[0].to_dict(),
                'predicted_change': self._estimate_prediction_change(scenario['changes']),
                'impact_analysis': self._analyze_scenario_impact(scenario['changes'])
            }
            
            return scenario_result
            
        except Exception as e:
            logger.error(f"Error analyzing scenario: {e}")
            return {'error': str(e)}
    
    def _estimate_prediction_change(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate prediction change for scenario (simplified)"""
        
        # Simplified impact estimation
        impact_weights = {
            'payment_history_score': 0.35,
            'credit_utilization': -0.30,  # Negative because lower is better
            'annual_income': 0.15,
            'debt_to_income_ratio': -0.20
        }
        
        total_impact = 0
        for feature, new_value in changes.items():
            if feature in impact_weights:
                # Simplified calculation
                if feature == 'credit_utilization':
                    impact = impact_weights[feature] * (30 - new_value) / 30  # Assume 30% baseline
                else:
                    impact = impact_weights[feature] * 0.1  # 10% improvement assumption
                total_impact += impact
        
        return {
            'estimated_score_change': total_impact * 100,  # Convert to score points
            'confidence': 0.7,  # Moderate confidence for estimation
            'direction': 'increase' if total_impact > 0 else 'decrease'
        }
    
    def _analyze_scenario_impact(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of scenario changes"""
        
        impact_analysis = {
            'primary_factors_changed': list(changes.keys()),
            'change_difficulty': {},
            'timeline_estimate': {},
            'recommendations': []
        }
        
        # Assess difficulty and timeline for each change
        for feature, new_value in changes.items():
            if feature == 'payment_history_score':
                impact_analysis['change_difficulty'][feature] = 'moderate'
                impact_analysis['timeline_estimate'][feature] = '6-12 months'
                impact_analysis['recommendations'].append('Set up automatic payments')
            
            elif feature == 'credit_utilization':
                impact_analysis['change_difficulty'][feature] = 'easy'
                impact_analysis['timeline_estimate'][feature] = '1-2 months'
                impact_analysis['recommendations'].append('Pay down credit card balances')
            
            elif feature == 'annual_income':
                impact_analysis['change_difficulty'][feature] = 'difficult'
                impact_analysis['timeline_estimate'][feature] = '6+ months'
                impact_analysis['recommendations'].append('Seek promotion or additional income sources')
        
        return impact_analysis
    
    def _rank_scenarios_by_impact(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank scenarios by their predicted impact"""
        
        try:
            ranked_scenarios = []
            
            for scenario in scenarios:
                if 'predicted_change' in scenario:
                    impact_score = abs(scenario['predicted_change'].get('estimated_score_change', 0))
                    scenario['impact_score'] = impact_score
                    ranked_scenarios.append(scenario)
            
            # Sort by impact score
            ranked_scenarios.sort(key=lambda x: x['impact_score'], reverse=True)
            
            return ranked_scenarios
            
        except Exception as e:
            logger.error(f"Error ranking scenarios: {e}")
            return scenarios
    
    def _combine_narrative_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Combine narrative components into comprehensive analysis"""
        
        try:
            comprehensive = {
                'key_findings': [],
                'primary_factors': [],
                'improvement_opportunities': [],
                'risk_factors': []
            }
            
            # Extract key findings from local explanation
            if 'local' in components and 'combined_importance' in components['local']:
                importance = components['local']['combined_importance']
                top_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for factor, score in top_factors:
                    comprehensive['primary_factors'].append({
                        'factor': factor,
                        'importance_score': score,
                        'impact': 'high' if score > 0.3 else 'medium' if score > 0.1 else 'low'
                    })
            
            # Extract improvement opportunities
            if 'importance' in components and 'consensus_importance' in components['importance']:
                consensus = components['importance']['consensus_importance']
                if 'top_features' in consensus:
                    for feature, score in consensus['top_features'][:3]:
                        comprehensive['improvement_opportunities'].append({
                            'feature': feature,
                            'potential_impact': score,
                            'actionable': self._is_feature_actionable(feature)
                        })
            
            return comprehensive
            
        except Exception as e:
            logger.error(f"Error combining narrative components: {e}")
            return {}
    
    def _is_feature_actionable(self, feature: str) -> bool:
        """Determine if a feature is actionable by the user"""
        
        actionable_features = [
            'credit_utilization', 'payment_history', 'debt_to_income_ratio',
            'number_of_accounts', 'credit_inquiries'
        ]
        
        return any(actionable in feature.lower() for actionable in actionable_features)
    
    def _calculate_explanation_confidence(self, explanation_data: Dict[str, Any]) -> float:
        """Calculate confidence score for explanation"""
        
        try:
            if 'error' in explanation_data:
                return 0.0
            
            confidence_factors = []
            
            # Check data quality
            if 'explanations' in explanation_data:
                method_count = len([exp for exp in explanation_data['explanations'].values() 
                                 if 'error' not in exp])
                confidence_factors.append(min(1.0, method_count / 3))  # Up to 3 methods
            
            # Check consensus
            if 'combined_importance' in explanation_data:
                importance_values = list(explanation_data['combined_importance'].values())
                if importance_values:
                    max_importance = max(importance_values)
                    confidence_factors.append(min(1.0, max_importance * 2))  # Scale importance
            
            # Default confidence
            if not confidence_factors:
                confidence_factors.append(0.7)
            
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.5
    
    def _get_generation_method(self, explanation_type: ExplanationType) -> str:
        """Get generation method description"""
        
        method_descriptions = {
            ExplanationType.LOCAL: 'SHAP, LIME, and feature attribution analysis',
            ExplanationType.GLOBAL: 'Global feature importance and partial dependence analysis',
            ExplanationType.COUNTERFACTUAL: 'Counterfactual generation and diversity analysis',
            ExplanationType.FEATURE_IMPORTANCE: 'Multi-method feature importance consensus',
            ExplanationType.WHAT_IF: 'Scenario simulation and impact analysis',
            ExplanationType.NARRATIVE: 'Comprehensive multi-method analysis'
        }
        
        return method_descriptions.get(explanation_type, 'Unknown method')
    
    async def generate_batch_explanations(self, requests: List[ExplanationRequest]) -> List[ExplanationResult]:
        """Generate explanations for multiple requests"""
        
        results = []
        
        for request in requests:
            try:
                result = await self.generate_explanation(request)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch explanation for {request.request_id}: {e}")
                results.append(ExplanationResult(
                    request_id=request.request_id,
                    explanation_type=request.explanation_type,
                    explanation_data={'error': str(e)},
                    narrative=f"Error generating explanation: {str(e)}",
                    visualization_data={},
                    confidence=0.0,
                    metadata={'error': True},
                    timestamp=datetime.now()
                ))
        
        return results
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """Get explanation generator statistics"""
        
        return {
            'supported_explanation_types': [t.value for t in self.supported_types],
            'available_explainers': list(self.xai_explainers.keys()),
            'components_initialized': {
                'narrative_builder': self.narrative_builder is not None,
                'visualization_generator': self.visualization_generator is not None,
                'explanation_cache': self.explanation_cache is not None
            },
            'timestamp': datetime.now().isoformat()
        }
