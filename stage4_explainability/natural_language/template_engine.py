"""
Template engine for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re
import json
from string import Template
from enum import Enum

logger = logging.getLogger(__name__)

class TemplateType(Enum):
    """Template types"""
    EXPLANATION = "explanation"
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    ALERT = "alert"

class TemplateEngine:
    """Template engine for generating structured explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = {}
        self.template_variables = {}
        self.custom_functions = {}
        self._initialize_templates()
        
    def _initialize_templates(self):
        """Initialize default templates"""
        
        # Explanation templates
        self.templates[TemplateType.EXPLANATION] = {
            'local_explanation': Template("""
Your credit score of $score is influenced by several key factors:

$top_factors

$detailed_analysis

$impact_summary
            """.strip()),
            
            'global_explanation': Template("""
Based on analysis of similar profiles, the most important factors are:

$global_factors

$pattern_analysis

$benchmark_comparison
            """.strip()),
            
            'counterfactual_explanation': Template("""
To achieve your target score of $target_score, you would need to:

$required_changes

$success_probability

$timeline_estimate
            """.strip()),
            
            'feature_importance': Template("""
Factor importance ranking for your profile:

$importance_ranking

$actionable_factors

$improvement_potential
            """.strip())
        }
        
        # Recommendation templates
        self.templates[TemplateType.RECOMMENDATION] = {
            'improvement_plan': Template("""
Recommended improvement plan:

Priority 1: $priority_1_action
Expected impact: $priority_1_impact

Priority 2: $priority_2_action  
Expected impact: $priority_2_impact

Priority 3: $priority_3_action
Expected impact: $priority_3_impact

Timeline: $estimated_timeline
            """.strip()),
            
            'quick_wins': Template("""
Quick wins to improve your score:

$quick_action_1 - $quick_impact_1
$quick_action_2 - $quick_impact_2
$quick_action_3 - $quick_impact_3

These changes could show results in $quick_timeline.
            """.strip()),
            
            'long_term_strategy': Template("""
Long-term credit building strategy:

$strategy_overview

Key milestones:
$milestone_1
$milestone_2
$milestone_3

Expected outcome: $long_term_outcome
            """.strip())
        }
        
        # Comparison templates
        self.templates[TemplateType.COMPARISON] = {
            'peer_comparison': Template("""
Compared to similar profiles:

Your score: $your_score
Peer average: $peer_average
Difference: $score_difference ($performance_level)

$strength_areas

$improvement_areas
            """.strip()),
            
            'historical_comparison': Template("""
Your credit progress over time:

$time_period_1: $score_1 ($trend_1)
$time_period_2: $score_2 ($trend_2)
$time_period_3: $score_3 ($trend_3)

Overall trend: $overall_trend
Key changes: $significant_changes
            """.strip()),
            
            'scenario_comparison': Template("""
Scenario analysis results:

$scenario_1_name: $scenario_1_outcome
$scenario_2_name: $scenario_2_outcome
$scenario_3_name: $scenario_3_outcome

Best option: $recommended_scenario
Reason: $recommendation_reason
            """.strip())
        }
        
        # Summary templates
        self.templates[TemplateType.SUMMARY] = {
            'executive_summary': Template("""
Credit Profile Summary

Current Score: $current_score ($score_category)
Key Strengths: $top_strengths
Main Concerns: $top_concerns
Improvement Potential: $improvement_potential

Next Steps: $recommended_actions
            """.strip()),
            
            'factor_summary': Template("""
Factor Analysis Summary

Most Important: $most_important_factor ($importance_score)
Most Actionable: $most_actionable_factor
Biggest Risk: $biggest_risk_factor

$summary_insight
            """.strip()),
            
            'progress_summary': Template("""
Progress Summary

Period: $analysis_period
Score Change: $score_change
Trend: $trend_direction

Achievements: $positive_changes
Areas for Focus: $focus_areas
            """.strip())
        }
        
        # Alert templates
        self.templates[TemplateType.ALERT] = {
            'risk_alert': Template("""
⚠️ Credit Risk Alert

Issue: $risk_issue
Impact: $risk_impact
Urgency: $urgency_level

Immediate Actions:
$immediate_action_1
$immediate_action_2

Timeline: $action_timeline
            """.strip()),
            
            'opportunity_alert': Template("""
💡 Credit Opportunity

Opportunity: $opportunity_description
Potential Benefit: $potential_benefit
Effort Required: $effort_level

Action Steps:
$action_step_1
$action_step_2

Expected Timeline: $opportunity_timeline
            """.strip()),
            
            'milestone_alert': Template("""
🎉 Milestone Achievement

Achievement: $milestone_description
Previous Score: $previous_score
Current Score: $current_score
Improvement: $improvement_amount

What This Means: $milestone_impact
Next Goal: $next_milestone
            """.strip())
        }
        
        # Template variables and functions
        self._initialize_template_functions()
    
    def _initialize_template_functions(self):
        """Initialize custom template functions"""
        
        self.custom_functions = {
            'format_score': self._format_score,
            'format_percentage': self._format_percentage,
            'format_currency': self._format_currency,
            'get_score_category': self._get_score_category,
            'get_trend_description': self._get_trend_description,
            'get_urgency_level': self._get_urgency_level,
            'format_factor_list': self._format_factor_list,
            'calculate_improvement_potential': self._calculate_improvement_potential
        }
    
    async def render_template(self, template_type: TemplateType, template_name: str,
                            data: Dict[str, Any], custom_variables: Dict[str, Any] = None) -> str:
        """Render template with data"""
        
        try:
            # Get template
            if template_type not in self.templates:
                raise ValueError(f"Template type {template_type} not found")
            
            if template_name not in self.templates[template_type]:
                raise ValueError(f"Template {template_name} not found in {template_type}")
            
            template = self.templates[template_type][template_name]
            
            # Prepare template variables
            template_vars = await self._prepare_template_variables(data, custom_variables)
            
            # Render template
            rendered_text = template.safe_substitute(template_vars)
            
            # Post-process rendered text
            final_text = await self._post_process_template(rendered_text)
            
            return final_text
            
        except Exception as e:
            logger.error(f"Error rendering template: {e}")
            return f"Error rendering {template_name} template"
    
    async def _prepare_template_variables(self, data: Dict[str, Any],
                                        custom_variables: Dict[str, Any] = None) -> Dict[str, str]:
        """Prepare variables for template substitution"""
        
        try:
            variables = {}
            
            # Basic data mapping
            if 'score' in data:
                variables['score'] = self._format_score(data['score'])
                variables['current_score'] = variables['score']
                variables['score_category'] = self._get_score_category(data['score'])
            
            if 'target_score' in data:
                variables['target_score'] = self._format_score(data['target_score'])
            
            # Factor analysis
            if 'combined_importance' in data:
                importance = data['combined_importance']
                variables.update(await self._process_importance_data(importance))
            
            # Counterfactual data
            if 'counterfactuals' in data and data['counterfactuals']:
                cf_data = data['counterfactuals'][0]
                variables.update(await self._process_counterfactual_data(cf_data))
            
            # Scenario data
            if 'ranked_scenarios' in data:
                scenarios = data['ranked_scenarios']
                variables.update(await self._process_scenario_data(scenarios))
            
            # Global explanation data
            if 'explanations' in data:
                explanations = data['explanations']
                variables.update(await self._process_explanation_data(explanations))
            
            # Custom variables override
            if custom_variables:
                variables.update(custom_variables)
            
            # Ensure all variables are strings
            for key, value in variables.items():
                if not isinstance(value, str):
                    variables[key] = str(value)
            
            return variables
            
        except Exception as e:
            logger.error(f"Error preparing template variables: {e}")
            return {'error': 'Error processing template data'}
    
    async def _process_importance_data(self, importance: Dict[str, float]) -> Dict[str, str]:
        """Process importance data for templates"""
        
        try:
            variables = {}
            
            # Sort factors by importance
            sorted_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            # Top factors
            top_factors = sorted_factors[:3]
            factor_list = []
            
            for i, (factor, score) in enumerate(top_factors, 1):
                clean_factor = factor.replace('_', ' ').title()
                percentage = score * 100
                factor_list.append(f"{i}. {clean_factor} ({percentage:.0f}%)")
            
            variables['top_factors'] = '\n'.join(factor_list)
            variables['importance_ranking'] = '\n'.join(factor_list)
            
            # Most important factor
            if sorted_factors:
                most_important = sorted_factors[0]
                variables['most_important_factor'] = most_important[0].replace('_', ' ').title()
                variables['importance_score'] = f"{most_important[1]*100:.0f}%"
            
            # Actionable factors
            actionable = [
                factor.replace('_', ' ').title() 
                for factor, _ in sorted_factors[:5]
                if self._is_actionable_factor(factor)
            ]
            variables['actionable_factors'] = ', '.join(actionable[:3])
            variables['most_actionable_factor'] = actionable[0] if actionable else 'Payment History'
            
            return variables
            
        except Exception as e:
            logger.error(f"Error processing importance data: {e}")
            return {}
    
    async def _process_counterfactual_data(self, cf_data: Dict[str, Any]) -> Dict[str, str]:
        """Process counterfactual data for templates"""
        
        try:
            variables = {}
            
            # Required changes
            if 'changes_made' in cf_data:
                changes = cf_data['changes_made']
                change_list = []
                
                for feature, change_info in changes.items():
                    clean_feature = feature.replace('_', ' ').title()
                    original = change_info['original_value']
                    new_value = change_info['new_value']
                    
                    change_desc = f"• Change {clean_feature} from {self._format_value(original)} to {self._format_value(new_value)}"
                    change_list.append(change_desc)
                
                variables['required_changes'] = '\n'.join(change_list)
            
            # Success probability
            success = cf_data.get('success', False)
            confidence = cf_data.get('confidence', 0.5)
            
            if success:
                variables['success_probability'] = f"High likelihood of success ({confidence*100:.0f}% confidence)"
            else:
                variables['success_probability'] = f"Moderate likelihood of success ({confidence*100:.0f}% confidence)"
            
            # Timeline estimate
            num_changes = len(cf_data.get('changes_made', {}))
            if num_changes <= 2:
                variables['timeline_estimate'] = "3-6 months"
            elif num_changes <= 4:
                variables['timeline_estimate'] = "6-12 months"
            else:
                variables['timeline_estimate'] = "12+ months"
            
            return variables
            
        except Exception as e:
            logger.error(f"Error processing counterfactual data: {e}")
            return {}
    
    async def _process_scenario_data(self, scenarios: List[Dict[str, Any]]) -> Dict[str, str]:
        """Process scenario data for templates"""
        
        try:
            variables = {}
            
            # Top scenarios
            for i, scenario in enumerate(scenarios[:3], 1):
                scenario_info = scenario.get('scenario', {})
                predicted_change = scenario.get('predicted_change', {})
                
                scenario_name = scenario_info.get('name', f'Scenario {i}')
                score_change = predicted_change.get('estimated_score_change', 0)
                
                variables[f'scenario_{i}_name'] = scenario_name
                variables[f'scenario_{i}_outcome'] = f"{score_change:+.0f} points"
            
            # Best scenario
            if scenarios:
                best_scenario = scenarios[0]
                variables['recommended_scenario'] = best_scenario.get('scenario', {}).get('name', 'Top scenario')
                variables['recommendation_reason'] = "Highest predicted improvement with reasonable effort"
            
            return variables
            
        except Exception as e:
            logger.error(f"Error processing scenario data: {e}")
            return {}
    
    async def _process_explanation_data(self, explanations: Dict[str, Any]) -> Dict[str, str]:
        """Process explanation data for templates"""
        
        try:
            variables = {}
            
            # Global importance
            if 'global_importance' in explanations:
                global_data = explanations['global_importance']
                
                if 'importance_methods' in global_data and 'consensus' in global_data['importance_methods']:
                    consensus = global_data['importance_methods']['consensus']
                    sorted_global = sorted(consensus.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    global_factors = []
                    for factor, importance in sorted_global:
                        clean_factor = factor.replace('_', ' ').title()
                        percentage = importance * 100
                        global_factors.append(f"• {clean_factor} ({percentage:.0f}%)")
                    
                    variables['global_factors'] = '\n'.join(global_factors)
            
            # Feature interactions
            if 'feature_interactions' in explanations:
                variables['pattern_analysis'] = "Analysis reveals important interactions between factors that influence credit decisions."
            
            return variables
            
        except Exception as e:
            logger.error(f"Error processing explanation data: {e}")
            return {}
    
    async def _post_process_template(self, rendered_text: str) -> str:
        """Post-process rendered template"""
        
        try:
            # Remove empty lines and extra whitespace
            lines = [line.strip() for line in rendered_text.split('\n')]
            lines = [line for line in lines if line]
            
            # Join with proper spacing
            processed_text = '\n'.join(lines)
            
            # Handle any remaining template variables that weren't substituted
            processed_text = re.sub(r'\$\w+', '[Data not available]', processed_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error post-processing template: {e}")
            return rendered_text
    
    def add_custom_template(self, template_type: TemplateType, name: str, template_string: str):
        """Add custom template"""
        
        try:
            if template_type not in self.templates:
                self.templates[template_type] = {}
            
            self.templates[template_type][name] = Template(template_string)
            logger.info(f"Added custom template: {template_type.value}.{name}")
            
        except Exception as e:
            logger.error(f"Error adding custom template: {e}")
    
    def get_available_templates(self) -> Dict[str, List[str]]:
        """Get list of available templates"""
        
        available = {}
        for template_type, templates in self.templates.items():
            available[template_type.value] = list(templates.keys())
        
        return available
    
    async def validate_template(self, template_type: TemplateType, template_name: str,
                              sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template with sample data"""
        
        try:
            # Attempt to render template
            rendered = await self.render_template(template_type, template_name, sample_data)
            
            # Check for unsubstituted variables
            unsubstituted = re.findall(r'\$\w+', rendered)
            
            # Check template length
            word_count = len(rendered.split())
            
            validation_result = {
                'valid': len(unsubstituted) == 0,
                'unsubstituted_variables': unsubstituted,
                'word_count': word_count,
                'character_count': len(rendered),
                'preview': rendered[:200] + '...' if len(rendered) > 200 else rendered
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'word_count': 0,
                'character_count': 0
            }
    
    # Helper functions for template processing
    def _format_score(self, score: float) -> str:
        """Format credit score"""
        return f"{int(score)}"
    
    def _format_percentage(self, value: float) -> str:
        """Format percentage"""
        return f"{value*100:.1f}%"
    
    def _format_currency(self, value: float) -> str:
        """Format currency"""
        return f"${value:,.2f}"
    
    def _get_score_category(self, score: float) -> str:
        """Get score category description"""
        if score >= 800:
            return "Excellent"
        elif score >= 740:
            return "Very Good"
        elif score >= 670:
            return "Good"
        elif score >= 580:
            return "Fair"
        else:
            return "Poor"
    
    def _get_trend_description(self, change: float) -> str:
        """Get trend description"""
        if change > 10:
            return "Strong Improvement"
        elif change > 0:
            return "Improving"
        elif change == 0:
            return "Stable"
        elif change > -10:
            return "Declining"
        else:
            return "Significant Decline"
    
    def _get_urgency_level(self, risk_score: float) -> str:
        """Get urgency level"""
        if risk_score > 0.8:
            return "Critical"
        elif risk_score > 0.6:
            return "High"
        elif risk_score > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _format_factor_list(self, factors: List[str]) -> str:
        """Format list of factors"""
        if not factors:
            return "None identified"
        
        formatted = [factor.replace('_', ' ').title() for factor in factors]
        
        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        else:
            return f"{', '.join(formatted[:-1])}, and {formatted[-1]}"
    
    def _calculate_improvement_potential(self, current_score: float, factors: Dict[str, float]) -> str:
        """Calculate improvement potential"""
        
        total_impact = sum(factors.values())
        
        if total_impact > 0.7:
            potential_gain = min(50, (850 - current_score) * 0.6)
            return f"High ({potential_gain:.0f}+ points possible)"
        elif total_impact > 0.4:
            potential_gain = min(30, (850 - current_score) * 0.4)
            return f"Moderate ({potential_gain:.0f}+ points possible)"
        else:
            potential_gain = min(20, (850 - current_score) * 0.2)
            return f"Limited ({potential_gain:.0f}+ points possible)"
    
    def _format_value(self, value: Any) -> str:
        """Format value for display"""
        if isinstance(value, float):
            if 0 < value < 1:
                return f"{value:.1%}"
            else:
                return f"{value:.1f}"
        elif isinstance(value, int):
            return str(value)
        else:
            return str(value)
    
    def _is_actionable_factor(self, factor: str) -> bool:
        """Check if factor is actionable"""
        actionable_terms = [
            'payment', 'utilization', 'debt', 'balance', 'inquiry', 'account', 'credit'
        ]
        factor_lower = factor.lower()
        return any(term in factor_lower for term in actionable_terms)
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get template engine statistics"""
        
        template_counts = {}
        for template_type, templates in self.templates.items():
            template_counts[template_type.value] = len(templates)
        
        return {
            'template_types': len(self.templates),
            'total_templates': sum(template_counts.values()),
            'template_counts_by_type': template_counts,
            'custom_functions': len(self.custom_functions),
            'timestamp': datetime.now().isoformat()
        }
