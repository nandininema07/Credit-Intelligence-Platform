"""
Narrative builder for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from enum import Enum

logger = logging.getLogger(__name__)

class NarrativeStyle(Enum):
    """Narrative styles"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"

class NarrativeBuilder:
    """Build natural language narratives for explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.narrative_templates = {}
        self.style_preferences = {}
        self.domain_vocabulary = {}
        self._initialize_templates()
        
    def _initialize_templates(self):
        """Initialize narrative templates"""
        
        self.narrative_templates = {
            'local': {
                'simple': {
                    'intro': "Based on your information, here's what most affects your {prediction_type}:",
                    'factor_template': "• {factor_name} has a {impact_level} impact ({impact_score:.1%})",
                    'conclusion': "The top {top_count} factors account for {total_impact:.1%} of the decision."
                },
                'detailed': {
                    'intro': "I've analyzed your specific case using multiple explanation methods. Here's a comprehensive breakdown of what drives your {prediction_type}:",
                    'factor_template': "• {factor_name}: This factor shows a {impact_level} impact with a score of {impact_score:.1%}. {factor_explanation}",
                    'method_comparison': "Different analysis methods show {consensus_level} agreement on these factors.",
                    'conclusion': "Overall, your {prediction_type} is primarily influenced by {primary_factors}, which together account for {total_impact:.1%} of the decision."
                },
                'technical': {
                    'intro': "Local explanation analysis using SHAP, LIME, and feature attribution methods:",
                    'factor_template': "• {factor_name}: Attribution score {impact_score:.3f}, confidence {confidence:.2f}",
                    'method_details': "Method consensus: {method_agreement:.2f}, variance: {method_variance:.3f}",
                    'conclusion': "Feature importance ranking based on {method_count} explanation methods with {overall_confidence:.2f} confidence."
                }
            },
            'global': {
                'simple': {
                    'intro': "Looking at the overall model behavior, here are the most important factors:",
                    'factor_template': "• {factor_name} is important in {importance_percentage:.0f}% of decisions",
                    'conclusion': "These patterns hold across all similar cases."
                },
                'detailed': {
                    'intro': "Global model analysis reveals consistent patterns in how decisions are made:",
                    'factor_template': "• {factor_name}: Globally important in {importance_percentage:.1f}% of cases. {global_explanation}",
                    'interaction_note': "Key feature interactions: {interaction_summary}",
                    'conclusion': "This analysis is based on {sample_size} similar cases and shows {stability_level} stability."
                }
            },
            'counterfactual': {
                'simple': {
                    'intro': "To achieve {target_outcome}, you would need to change:",
                    'change_template': "• {feature_name}: from {current_value} to {target_value}",
                    'conclusion': "These changes have a {success_probability:.0f}% chance of achieving your goal."
                },
                'detailed': {
                    'intro': "Counterfactual analysis shows the minimum changes needed to reach {target_outcome}:",
                    'change_template': "• {feature_name}: Change from {current_value} to {target_value} ({change_magnitude} change). {change_explanation}",
                    'feasibility': "Feasibility assessment: {feasibility_analysis}",
                    'alternatives': "Alternative paths: {alternative_scenarios}",
                    'conclusion': "The recommended approach has {success_probability:.1f}% likelihood with {effort_level} effort required."
                }
            },
            'feature_importance': {
                'simple': {
                    'intro': "The most important factors for your case are:",
                    'ranking_template': "{rank}. {factor_name} ({importance:.1%})",
                    'conclusion': "Focus on the top {focus_count} factors for maximum impact."
                },
                'detailed': {
                    'intro': "Comprehensive feature importance analysis using multiple methods:",
                    'ranking_template': "{rank}. {factor_name} ({importance:.1%}): {factor_description} - {actionability}",
                    'consensus': "Method consensus: {consensus_strength}",
                    'conclusion': "Prioritize {actionable_factors} as they are both important and within your control."
                }
            },
            'what_if': {
                'simple': {
                    'intro': "Here's how different changes would affect your {outcome_type}:",
                    'scenario_template': "• {scenario_name}: {predicted_change} ({confidence_level} confidence)",
                    'conclusion': "The most impactful change would be {best_scenario}."
                },
                'detailed': {
                    'intro': "What-if scenario analysis shows potential outcomes for different changes:",
                    'scenario_template': "• {scenario_name}: {predicted_change} with {timeline} timeline. {implementation_notes}",
                    'ranking': "Scenarios ranked by impact: {scenario_ranking}",
                    'conclusion': "Recommended approach: {recommended_scenario} based on {selection_criteria}."
                }
            }
        }
        
        # Domain vocabulary for credit explanations
        self.domain_vocabulary = {
            'credit_score': {
                'simple': 'credit score',
                'detailed': 'credit score rating',
                'technical': 'creditworthiness metric'
            },
            'payment_history': {
                'simple': 'payment history',
                'detailed': 'track record of making payments on time',
                'technical': 'payment performance indicator'
            },
            'credit_utilization': {
                'simple': 'credit card usage',
                'detailed': 'percentage of available credit being used',
                'technical': 'credit utilization ratio'
            },
            'debt_to_income': {
                'simple': 'debt compared to income',
                'detailed': 'ratio of monthly debt payments to monthly income',
                'technical': 'debt-to-income ratio (DTI)'
            }
        }
    
    async def build_narrative(self, explanation_type, explanation_data: Dict[str, Any],
                            context: Dict[str, Any]) -> str:
        """Build narrative for explanation"""
        
        try:
            # Determine narrative style
            style = self._determine_narrative_style(context)
            
            # Get appropriate template
            template_key = explanation_type.value if hasattr(explanation_type, 'value') else str(explanation_type)
            
            if template_key not in self.narrative_templates:
                return f"Narrative template not found for {template_key}"
            
            style_templates = self.narrative_templates[template_key].get(style, 
                                                                       self.narrative_templates[template_key].get('simple', {}))
            
            if not style_templates:
                return "No suitable narrative template found"
            
            # Build narrative based on explanation type
            if template_key == 'local':
                return await self._build_local_narrative(explanation_data, style_templates, context, style)
            elif template_key == 'global':
                return await self._build_global_narrative(explanation_data, style_templates, context, style)
            elif template_key == 'counterfactual':
                return await self._build_counterfactual_narrative(explanation_data, style_templates, context, style)
            elif template_key == 'feature_importance':
                return await self._build_importance_narrative(explanation_data, style_templates, context, style)
            elif template_key == 'what_if':
                return await self._build_whatif_narrative(explanation_data, style_templates, context, style)
            else:
                return await self._build_generic_narrative(explanation_data, style_templates, context, style)
                
        except Exception as e:
            logger.error(f"Error building narrative: {e}")
            return f"I encountered an error building the explanation narrative: {str(e)}"
    
    def _determine_narrative_style(self, context: Dict[str, Any]) -> str:
        """Determine appropriate narrative style"""
        
        # Check user preferences
        user_style = context.get('preferred_explanation_style', 'balanced')
        technical_comfort = context.get('technical_comfort', 'medium')
        
        if user_style == 'simple' or technical_comfort == 'low':
            return 'simple'
        elif user_style == 'detailed' or technical_comfort == 'medium':
            return 'detailed'
        elif technical_comfort == 'high':
            return 'technical'
        else:
            return 'detailed'  # Default
    
    async def _build_local_narrative(self, explanation_data: Dict[str, Any],
                                   templates: Dict[str, str], context: Dict[str, Any],
                                   style: str) -> str:
        """Build narrative for local explanation"""
        
        try:
            narrative_parts = []
            
            # Introduction
            prediction_type = context.get('prediction_type', 'credit decision')
            intro = templates.get('intro', '').format(prediction_type=prediction_type)
            narrative_parts.append(intro)
            
            # Factor explanations
            if 'combined_importance' in explanation_data:
                importance = explanation_data['combined_importance']
                sorted_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                for i, (factor, score) in enumerate(sorted_factors[:5]):  # Top 5 factors
                    impact_level = self._get_impact_level(score)
                    factor_name = self._format_factor_name(factor, style)
                    
                    factor_text = templates.get('factor_template', '').format(
                        factor_name=factor_name,
                        impact_level=impact_level,
                        impact_score=score,
                        factor_explanation=self._get_factor_explanation(factor, style),
                        confidence=0.8  # Default confidence
                    )
                    narrative_parts.append(factor_text)
                
                # Method comparison (for detailed style)
                if style == 'detailed' and 'method_comparison' in templates:
                    consensus_level = self._assess_method_consensus(explanation_data.get('explanations', {}))
                    method_text = templates['method_comparison'].format(consensus_level=consensus_level)
                    narrative_parts.append(method_text)
                
                # Conclusion
                top_count = min(3, len(sorted_factors))
                total_impact = sum(score for _, score in sorted_factors[:top_count])
                primary_factors = ', '.join([self._format_factor_name(factor, style) 
                                           for factor, _ in sorted_factors[:top_count]])
                
                conclusion = templates.get('conclusion', '').format(
                    top_count=top_count,
                    total_impact=total_impact,
                    primary_factors=primary_factors
                )
                narrative_parts.append(conclusion)
            
            return '\n\n'.join(narrative_parts)
            
        except Exception as e:
            logger.error(f"Error building local narrative: {e}")
            return "Error building local explanation narrative"
    
    async def _build_global_narrative(self, explanation_data: Dict[str, Any],
                                    templates: Dict[str, str], context: Dict[str, Any],
                                    style: str) -> str:
        """Build narrative for global explanation"""
        
        try:
            narrative_parts = []
            
            # Introduction
            intro = templates.get('intro', '')
            narrative_parts.append(intro)
            
            # Global importance factors
            if 'explanations' in explanation_data and 'global_importance' in explanation_data['explanations']:
                global_data = explanation_data['explanations']['global_importance']
                
                if 'importance_methods' in global_data:
                    importance = global_data['importance_methods'].get('consensus', {})
                    
                    for factor, importance_score in list(importance.items())[:5]:
                        factor_name = self._format_factor_name(factor, style)
                        importance_percentage = importance_score * 100
                        
                        factor_text = templates.get('factor_template', '').format(
                            factor_name=factor_name,
                            importance_percentage=importance_percentage,
                            global_explanation=self._get_global_factor_explanation(factor, style)
                        )
                        narrative_parts.append(factor_text)
            
            # Feature interactions (for detailed style)
            if style == 'detailed' and 'interaction_note' in templates:
                if 'explanations' in explanation_data and 'feature_interactions' in explanation_data['explanations']:
                    interactions = explanation_data['explanations']['feature_interactions']
                    interaction_summary = self._summarize_interactions(interactions.get('feature_interactions', []))
                    
                    interaction_text = templates['interaction_note'].format(
                        interaction_summary=interaction_summary
                    )
                    narrative_parts.append(interaction_text)
            
            # Conclusion
            conclusion = templates.get('conclusion', '').format(
                sample_size='thousands of',  # Placeholder
                stability_level='high'  # Placeholder
            )
            narrative_parts.append(conclusion)
            
            return '\n\n'.join(narrative_parts)
            
        except Exception as e:
            logger.error(f"Error building global narrative: {e}")
            return "Error building global explanation narrative"
    
    async def _build_counterfactual_narrative(self, explanation_data: Dict[str, Any],
                                            templates: Dict[str, str], context: Dict[str, Any],
                                            style: str) -> str:
        """Build narrative for counterfactual explanation"""
        
        try:
            narrative_parts = []
            
            # Introduction
            target_outcome = context.get('desired_outcome', 'your target outcome')
            intro = templates.get('intro', '').format(target_outcome=target_outcome)
            narrative_parts.append(intro)
            
            # Changes needed
            if 'counterfactuals' in explanation_data and explanation_data['counterfactuals']:
                cf_data = explanation_data['counterfactuals'][0]  # First counterfactual
                
                if 'changes_made' in cf_data:
                    for feature, change_info in cf_data['changes_made'].items():
                        feature_name = self._format_factor_name(feature, style)
                        current_value = self._format_value(change_info['original_value'])
                        target_value = self._format_value(change_info['new_value'])
                        change_magnitude = self._assess_change_magnitude(change_info['relative_change'])
                        
                        change_text = templates.get('change_template', '').format(
                            feature_name=feature_name,
                            current_value=current_value,
                            target_value=target_value,
                            change_magnitude=change_magnitude,
                            change_explanation=self._get_change_explanation(feature, change_info, style)
                        )
                        narrative_parts.append(change_text)
                
                # Success probability
                success_prob = cf_data.get('success', False)
                success_probability = 85 if success_prob else 60  # Simplified
                
                conclusion = templates.get('conclusion', '').format(
                    success_probability=success_probability,
                    effort_level='moderate'  # Placeholder
                )
                narrative_parts.append(conclusion)
            
            return '\n\n'.join(narrative_parts)
            
        except Exception as e:
            logger.error(f"Error building counterfactual narrative: {e}")
            return "Error building counterfactual explanation narrative"
    
    async def _build_importance_narrative(self, explanation_data: Dict[str, Any],
                                        templates: Dict[str, str], context: Dict[str, Any],
                                        style: str) -> str:
        """Build narrative for feature importance explanation"""
        
        try:
            narrative_parts = []
            
            # Introduction
            intro = templates.get('intro', '')
            narrative_parts.append(intro)
            
            # Feature ranking
            if 'consensus_importance' in explanation_data:
                consensus = explanation_data['consensus_importance']
                
                if 'top_features' in consensus:
                    for i, (factor, importance) in enumerate(consensus['top_features'][:5], 1):
                        factor_name = self._format_factor_name(factor, style)
                        factor_description = self._get_factor_explanation(factor, style)
                        actionability = 'actionable' if self._is_actionable(factor) else 'not directly controllable'
                        
                        ranking_text = templates.get('ranking_template', '').format(
                            rank=i,
                            factor_name=factor_name,
                            importance=importance,
                            factor_description=factor_description,
                            actionability=actionability
                        )
                        narrative_parts.append(ranking_text)
                
                # Actionable factors conclusion
                actionable_factors = [self._format_factor_name(f, style) 
                                    for f, _ in consensus.get('top_features', [])[:5] 
                                    if self._is_actionable(f)]
                
                if actionable_factors:
                    conclusion = templates.get('conclusion', '').format(
                        actionable_factors=', '.join(actionable_factors[:3]),
                        focus_count=min(3, len(actionable_factors))
                    )
                    narrative_parts.append(conclusion)
            
            return '\n\n'.join(narrative_parts)
            
        except Exception as e:
            logger.error(f"Error building importance narrative: {e}")
            return "Error building feature importance narrative"
    
    async def _build_whatif_narrative(self, explanation_data: Dict[str, Any],
                                    templates: Dict[str, str], context: Dict[str, Any],
                                    style: str) -> str:
        """Build narrative for what-if explanation"""
        
        try:
            narrative_parts = []
            
            # Introduction
            outcome_type = context.get('outcome_type', 'credit score')
            intro = templates.get('intro', '').format(outcome_type=outcome_type)
            narrative_parts.append(intro)
            
            # Scenarios
            if 'ranked_scenarios' in explanation_data:
                scenarios = explanation_data['ranked_scenarios'][:5]  # Top 5 scenarios
                
                for scenario in scenarios:
                    scenario_name = scenario.get('scenario', {}).get('name', 'Scenario')
                    predicted_change = self._format_prediction_change(
                        scenario.get('predicted_change', {})
                    )
                    confidence_level = self._get_confidence_level(
                        scenario.get('predicted_change', {}).get('confidence', 0.5)
                    )
                    timeline = scenario.get('impact_analysis', {}).get('timeline_estimate', {})
                    timeline_text = self._format_timeline(timeline)
                    
                    scenario_text = templates.get('scenario_template', '').format(
                        scenario_name=scenario_name,
                        predicted_change=predicted_change,
                        confidence_level=confidence_level,
                        timeline=timeline_text,
                        implementation_notes=self._get_implementation_notes(scenario, style)
                    )
                    narrative_parts.append(scenario_text)
                
                # Best scenario conclusion
                if scenarios:
                    best_scenario = scenarios[0].get('scenario', {}).get('name', 'the first option')
                    conclusion = templates.get('conclusion', '').format(
                        best_scenario=best_scenario,
                        recommended_scenario=best_scenario,
                        selection_criteria='impact and feasibility'
                    )
                    narrative_parts.append(conclusion)
            
            return '\n\n'.join(narrative_parts)
            
        except Exception as e:
            logger.error(f"Error building what-if narrative: {e}")
            return "Error building what-if scenario narrative"
    
    async def _build_generic_narrative(self, explanation_data: Dict[str, Any],
                                     templates: Dict[str, str], context: Dict[str, Any],
                                     style: str) -> str:
        """Build generic narrative for unknown explanation types"""
        
        return f"Analysis complete. The explanation data contains {len(explanation_data)} key components that influence the decision."
    
    def _get_impact_level(self, score: float) -> str:
        """Convert impact score to descriptive level"""
        
        if score > 0.3:
            return 'strong'
        elif score > 0.15:
            return 'moderate'
        elif score > 0.05:
            return 'mild'
        else:
            return 'minimal'
    
    def _format_factor_name(self, factor: str, style: str) -> str:
        """Format factor name based on style"""
        
        # Clean up factor name
        clean_factor = factor.replace('_', ' ').title()
        
        # Use domain vocabulary if available
        factor_key = factor.lower()
        if factor_key in self.domain_vocabulary:
            vocab = self.domain_vocabulary[factor_key]
            if style in vocab:
                return vocab[style]
        
        return clean_factor
    
    def _get_factor_explanation(self, factor: str, style: str) -> str:
        """Get explanation for a factor"""
        
        explanations = {
            'payment_history': {
                'simple': 'This tracks whether you pay bills on time',
                'detailed': 'This represents your track record of making payments by their due dates, which is the most important factor in credit scoring',
                'technical': 'Payment performance metric based on historical payment behavior patterns'
            },
            'credit_utilization': {
                'simple': 'This is how much of your available credit you use',
                'detailed': 'This measures the percentage of your available credit limits that you are currently using across all accounts',
                'technical': 'Credit utilization ratio calculated as total balances divided by total credit limits'
            }
        }
        
        factor_key = factor.lower().replace(' ', '_')
        if factor_key in explanations:
            return explanations[factor_key].get(style, explanations[factor_key]['simple'])
        
        return 'This factor influences your credit profile'
    
    def _get_global_factor_explanation(self, factor: str, style: str) -> str:
        """Get global explanation for a factor"""
        
        base_explanation = self._get_factor_explanation(factor, style)
        return f"{base_explanation} across all similar profiles"
    
    def _assess_method_consensus(self, explanations: Dict[str, Any]) -> str:
        """Assess consensus between explanation methods"""
        
        method_count = len([exp for exp in explanations.values() if 'error' not in exp])
        
        if method_count >= 3:
            return 'strong'
        elif method_count >= 2:
            return 'moderate'
        else:
            return 'limited'
    
    def _summarize_interactions(self, interactions: List[Dict[str, Any]]) -> str:
        """Summarize feature interactions"""
        
        if not interactions:
            return 'No significant feature interactions detected'
        
        top_interactions = interactions[:2]  # Top 2 interactions
        interaction_names = []
        
        for interaction in top_interactions:
            entities = interaction.get('entities', [])
            if len(entities) >= 2:
                interaction_names.append(f"{entities[0]} and {entities[1]}")
        
        if interaction_names:
            return ', '.join(interaction_names)
        else:
            return 'Complex multi-factor interactions'
    
    def _format_value(self, value: Any) -> str:
        """Format value for display"""
        
        if isinstance(value, float):
            if value < 1:
                return f"{value:.1%}"
            else:
                return f"{value:.1f}"
        elif isinstance(value, int):
            return str(value)
        else:
            return str(value)
    
    def _assess_change_magnitude(self, relative_change: float) -> str:
        """Assess magnitude of change"""
        
        abs_change = abs(relative_change)
        
        if abs_change > 0.5:
            return 'large'
        elif abs_change > 0.2:
            return 'moderate'
        elif abs_change > 0.05:
            return 'small'
        else:
            return 'minimal'
    
    def _get_change_explanation(self, feature: str, change_info: Dict[str, Any], style: str) -> str:
        """Get explanation for a specific change"""
        
        change_magnitude = self._assess_change_magnitude(change_info.get('relative_change', 0))
        
        explanations = {
            'simple': f'This is a {change_magnitude} change that could help',
            'detailed': f'This represents a {change_magnitude} adjustment in {self._format_factor_name(feature, style).lower()}, which should have a positive impact',
            'technical': f'Relative change of {change_info.get("relative_change", 0):.2f} in {feature}'
        }
        
        return explanations.get(style, explanations['simple'])
    
    def _is_actionable(self, factor: str) -> bool:
        """Check if factor is actionable by user"""
        
        actionable_factors = [
            'payment_history', 'credit_utilization', 'debt_to_income',
            'number_of_accounts', 'credit_inquiries', 'account_age'
        ]
        
        factor_lower = factor.lower().replace(' ', '_')
        return any(actionable in factor_lower for actionable in actionable_factors)
    
    def _format_prediction_change(self, predicted_change: Dict[str, Any]) -> str:
        """Format prediction change for display"""
        
        score_change = predicted_change.get('estimated_score_change', 0)
        direction = predicted_change.get('direction', 'change')
        
        if abs(score_change) < 1:
            return f"minimal {direction}"
        elif abs(score_change) < 20:
            return f"{abs(score_change):.0f} point {direction}"
        else:
            return f"{abs(score_change):.0f} point {direction}"
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Convert confidence score to level"""
        
        if confidence > 0.8:
            return 'high'
        elif confidence > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _format_timeline(self, timeline: Dict[str, Any]) -> str:
        """Format timeline information"""
        
        if not timeline:
            return '3-6 months'
        
        # Extract first timeline value
        first_timeline = next(iter(timeline.values()), '3-6 months')
        return first_timeline
    
    def _get_implementation_notes(self, scenario: Dict[str, Any], style: str) -> str:
        """Get implementation notes for scenario"""
        
        impact_analysis = scenario.get('impact_analysis', {})
        recommendations = impact_analysis.get('recommendations', [])
        
        if recommendations and style != 'simple':
            return f"Key action: {recommendations[0]}"
        else:
            return "Implementation details available upon request"
    
    def get_narrative_statistics(self) -> Dict[str, Any]:
        """Get narrative builder statistics"""
        
        template_counts = {}
        for explanation_type, styles in self.narrative_templates.items():
            template_counts[explanation_type] = len(styles)
        
        return {
            'supported_explanation_types': list(self.narrative_templates.keys()),
            'supported_styles': ['simple', 'detailed', 'technical'],
            'template_counts': template_counts,
            'domain_vocabulary_terms': len(self.domain_vocabulary),
            'timestamp': datetime.now().isoformat()
        }
