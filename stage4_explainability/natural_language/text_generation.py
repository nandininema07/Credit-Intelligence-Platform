"""
Text generation for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re
import random
from enum import Enum

logger = logging.getLogger(__name__)

class TextStyle(Enum):
    """Text generation styles"""
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    SIMPLE = "simple"

class TextGenerator:
    """Generate natural language text for explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.style_templates = {}
        self.vocabulary_levels = {}
        self.sentence_patterns = {}
        self._initialize_generation_rules()
        
    def _initialize_generation_rules(self):
        """Initialize text generation rules and patterns"""
        
        # Style-specific templates
        self.style_templates = {
            TextStyle.FORMAL: {
                'intro_phrases': [
                    "Based on our analysis,",
                    "The evaluation indicates that",
                    "Our assessment reveals that",
                    "The data demonstrates that"
                ],
                'transition_phrases': [
                    "Furthermore,",
                    "Additionally,",
                    "Moreover,",
                    "In addition to this,"
                ],
                'conclusion_phrases': [
                    "In conclusion,",
                    "To summarize,",
                    "In summary,",
                    "Therefore,"
                ]
            },
            TextStyle.CONVERSATIONAL: {
                'intro_phrases': [
                    "Here's what I found:",
                    "Let me explain what's happening:",
                    "So here's the deal:",
                    "Here's what the data tells us:"
                ],
                'transition_phrases': [
                    "Also,",
                    "Plus,",
                    "And another thing,",
                    "What's more,"
                ],
                'conclusion_phrases': [
                    "Bottom line:",
                    "So to wrap up:",
                    "The key takeaway is:",
                    "Here's what matters most:"
                ]
            },
            TextStyle.TECHNICAL: {
                'intro_phrases': [
                    "Statistical analysis indicates:",
                    "Model evaluation demonstrates:",
                    "Algorithmic assessment shows:",
                    "Computational analysis reveals:"
                ],
                'transition_phrases': [
                    "Subsequent analysis shows:",
                    "Further examination reveals:",
                    "Additional metrics indicate:",
                    "Complementary analysis demonstrates:"
                ],
                'conclusion_phrases': [
                    "Analytical summary:",
                    "Model conclusion:",
                    "Statistical inference:",
                    "Algorithmic determination:"
                ]
            },
            TextStyle.SIMPLE: {
                'intro_phrases': [
                    "Here's what we found:",
                    "This is what happened:",
                    "The main thing is:",
                    "What you need to know:"
                ],
                'transition_phrases': [
                    "Also,",
                    "Next,",
                    "Then,",
                    "And,"
                ],
                'conclusion_phrases': [
                    "So,",
                    "In the end,",
                    "What this means:",
                    "The main point:"
                ]
            }
        }
        
        # Vocabulary levels for different audiences
        self.vocabulary_levels = {
            'basic': {
                'credit_utilization': 'how much of your credit you use',
                'debt_to_income_ratio': 'how much debt you have compared to your income',
                'payment_history': 'whether you pay bills on time',
                'credit_inquiries': 'times someone checked your credit',
                'algorithm': 'computer program',
                'statistical_significance': 'meaningful difference'
            },
            'intermediate': {
                'credit_utilization': 'credit utilization rate',
                'debt_to_income_ratio': 'debt-to-income ratio',
                'payment_history': 'payment track record',
                'credit_inquiries': 'credit checks',
                'algorithm': 'analytical model',
                'statistical_significance': 'statistically significant'
            },
            'advanced': {
                'credit_utilization': 'credit utilization ratio',
                'debt_to_income_ratio': 'debt-to-income ratio (DTI)',
                'payment_history': 'payment performance metrics',
                'credit_inquiries': 'hard credit inquiries',
                'algorithm': 'machine learning algorithm',
                'statistical_significance': 'statistical significance'
            }
        }
        
        # Sentence patterns for different explanation types
        self.sentence_patterns = {
            'factor_importance': [
                "{factor} is {importance_level} important because {reason}.",
                "{factor} has a {impact_level} impact on your {outcome} due to {explanation}.",
                "Your {outcome} is {direction} affected by {factor}, which {description}."
            ],
            'comparison': [
                "{item1} is {comparison_type} than {item2} by {magnitude}.",
                "Compared to {baseline}, {item} shows {difference}.",
                "{item1} and {item2} differ in that {difference_explanation}."
            ],
            'recommendation': [
                "To improve your {target}, consider {action} because {benefit}.",
                "You could {action} to {expected_outcome}.",
                "The most effective approach would be to {recommendation} since {justification}."
            ]
        }
    
    async def generate_explanation_text(self, explanation_data: Dict[str, Any],
                                      style: TextStyle = TextStyle.CONVERSATIONAL,
                                      vocabulary_level: str = 'intermediate') -> str:
        """Generate natural language explanation text"""
        
        try:
            # Determine text structure
            text_sections = []
            
            # Introduction
            intro = await self._generate_introduction(explanation_data, style)
            text_sections.append(intro)
            
            # Main content
            main_content = await self._generate_main_content(
                explanation_data, style, vocabulary_level
            )
            text_sections.extend(main_content)
            
            # Conclusion
            conclusion = await self._generate_conclusion(explanation_data, style)
            text_sections.append(conclusion)
            
            # Combine sections
            full_text = self._combine_text_sections(text_sections, style)
            
            # Post-process text
            final_text = await self._post_process_text(full_text, style, vocabulary_level)
            
            return final_text
            
        except Exception as e:
            logger.error(f"Error generating explanation text: {e}")
            return "I encountered an error generating the explanation text."
    
    async def _generate_introduction(self, explanation_data: Dict[str, Any],
                                   style: TextStyle) -> str:
        """Generate introduction text"""
        
        try:
            intro_phrases = self.style_templates[style]['intro_phrases']
            intro_phrase = random.choice(intro_phrases)
            
            # Determine explanation type
            explanation_type = explanation_data.get('explanation_type', 'analysis')
            
            if explanation_type == 'local':
                intro_text = f"{intro_phrase} your specific situation shows several key factors affecting your credit profile."
            elif explanation_type == 'global':
                intro_text = f"{intro_phrase} the overall model behavior reveals important patterns in credit decisions."
            elif explanation_type == 'counterfactual':
                intro_text = f"{intro_phrase} there are specific changes that could help you achieve your desired outcome."
            elif explanation_type == 'feature_importance':
                intro_text = f"{intro_phrase} certain factors have more influence on your credit profile than others."
            elif explanation_type == 'what_if':
                intro_text = f"{intro_phrase} different scenarios would have varying impacts on your credit situation."
            else:
                intro_text = f"{intro_phrase} the analysis provides insights into your credit profile."
            
            return intro_text
            
        except Exception as e:
            logger.error(f"Error generating introduction: {e}")
            return "Let me explain what the analysis shows."
    
    async def _generate_main_content(self, explanation_data: Dict[str, Any],
                                   style: TextStyle, vocabulary_level: str) -> List[str]:
        """Generate main content sections"""
        
        try:
            content_sections = []
            explanation_type = explanation_data.get('explanation_type', 'analysis')
            
            if explanation_type == 'local':
                content_sections = await self._generate_local_content(
                    explanation_data, style, vocabulary_level
                )
            elif explanation_type == 'global':
                content_sections = await self._generate_global_content(
                    explanation_data, style, vocabulary_level
                )
            elif explanation_type == 'counterfactual':
                content_sections = await self._generate_counterfactual_content(
                    explanation_data, style, vocabulary_level
                )
            elif explanation_type == 'feature_importance':
                content_sections = await self._generate_importance_content(
                    explanation_data, style, vocabulary_level
                )
            elif explanation_type == 'what_if':
                content_sections = await self._generate_whatif_content(
                    explanation_data, style, vocabulary_level
                )
            else:
                content_sections = ["The analysis shows various factors affecting your credit profile."]
            
            return content_sections
            
        except Exception as e:
            logger.error(f"Error generating main content: {e}")
            return ["The analysis provides insights into your situation."]
    
    async def _generate_local_content(self, explanation_data: Dict[str, Any],
                                    style: TextStyle, vocabulary_level: str) -> List[str]:
        """Generate content for local explanations"""
        
        try:
            content = []
            
            if 'combined_importance' in explanation_data:
                importance = explanation_data['combined_importance']
                sorted_factors = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                # Top factors
                top_factors = sorted_factors[:3]
                factor_texts = []
                
                for factor, score in top_factors:
                    factor_name = self._translate_term(factor, vocabulary_level)
                    importance_level = self._get_importance_description(score)
                    
                    factor_text = f"{factor_name} is {importance_level} important in your case"
                    if score > 0.3:
                        factor_text += " and has a significant impact on your credit profile"
                    elif score > 0.15:
                        factor_text += " and plays a moderate role in your credit assessment"
                    else:
                        factor_text += " and contributes to your overall credit evaluation"
                    
                    factor_texts.append(factor_text + ".")
                
                if factor_texts:
                    content.append(" ".join(factor_texts))
                
                # Impact summary
                total_impact = sum(score for _, score in top_factors)
                if total_impact > 0.7:
                    content.append(f"These top factors account for most of the decision in your case.")
                else:
                    content.append(f"These factors, along with others, contribute to your overall credit profile.")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating local content: {e}")
            return ["Your specific situation involves several important factors."]
    
    async def _generate_global_content(self, explanation_data: Dict[str, Any],
                                     style: TextStyle, vocabulary_level: str) -> List[str]:
        """Generate content for global explanations"""
        
        try:
            content = []
            
            if 'explanations' in explanation_data and 'global_importance' in explanation_data['explanations']:
                global_data = explanation_data['explanations']['global_importance']
                
                if 'importance_methods' in global_data:
                    consensus = global_data['importance_methods'].get('consensus', {})
                    
                    if consensus:
                        sorted_factors = sorted(consensus.items(), key=lambda x: x[1], reverse=True)[:3]
                        
                        factor_descriptions = []
                        for factor, importance in sorted_factors:
                            factor_name = self._translate_term(factor, vocabulary_level)
                            percentage = importance * 100
                            factor_descriptions.append(f"{factor_name} ({percentage:.0f}%)")
                        
                        if factor_descriptions:
                            content.append(f"Across all similar cases, the most important factors are: {', '.join(factor_descriptions)}.")
                
                # Feature interactions
                if 'feature_interactions' in explanation_data['explanations']:
                    interactions = explanation_data['explanations']['feature_interactions']
                    if 'feature_interactions' in interactions and interactions['feature_interactions']:
                        content.append("The analysis also reveals important interactions between different factors that influence credit decisions.")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating global content: {e}")
            return ["The overall analysis shows consistent patterns across similar cases."]
    
    async def _generate_counterfactual_content(self, explanation_data: Dict[str, Any],
                                             style: TextStyle, vocabulary_level: str) -> List[str]:
        """Generate content for counterfactual explanations"""
        
        try:
            content = []
            
            if 'counterfactuals' in explanation_data and explanation_data['counterfactuals']:
                cf_data = explanation_data['counterfactuals'][0]
                
                if 'changes_made' in cf_data:
                    changes = cf_data['changes_made']
                    change_descriptions = []
                    
                    for feature, change_info in changes.items():
                        factor_name = self._translate_term(feature, vocabulary_level)
                        original_value = change_info['original_value']
                        new_value = change_info['new_value']
                        
                        change_desc = f"change your {factor_name} from {self._format_value(original_value)} to {self._format_value(new_value)}"
                        change_descriptions.append(change_desc)
                    
                    if change_descriptions:
                        if len(change_descriptions) == 1:
                            content.append(f"You would need to {change_descriptions[0]}.")
                        else:
                            content.append(f"You would need to: {', '.join(change_descriptions[:-1])}, and {change_descriptions[-1]}.")
                
                # Success probability
                success = cf_data.get('success', False)
                if success:
                    content.append("These changes have a high likelihood of achieving your desired outcome.")
                else:
                    content.append("While these changes would help, additional improvements might be needed for your target outcome.")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating counterfactual content: {e}")
            return ["Specific changes to your profile could help achieve your desired outcome."]
    
    async def _generate_importance_content(self, explanation_data: Dict[str, Any],
                                         style: TextStyle, vocabulary_level: str) -> List[str]:
        """Generate content for feature importance explanations"""
        
        try:
            content = []
            
            if 'consensus_importance' in explanation_data:
                consensus = explanation_data['consensus_importance']
                
                if 'top_features' in consensus:
                    top_features = consensus['top_features'][:5]
                    
                    # Create ranking text
                    ranking_items = []
                    for i, (feature, importance) in enumerate(top_features, 1):
                        factor_name = self._translate_term(feature, vocabulary_level)
                        percentage = importance * 100
                        ranking_items.append(f"{i}. {factor_name} ({percentage:.0f}%)")
                    
                    if ranking_items:
                        content.append(f"The factors ranked by importance are: {', '.join(ranking_items)}.")
                    
                    # Actionability note
                    actionable_factors = [
                        self._translate_term(feature, vocabulary_level) 
                        for feature, _ in top_features 
                        if self._is_actionable(feature)
                    ]
                    
                    if actionable_factors:
                        content.append(f"Among these, you have direct control over: {', '.join(actionable_factors[:3])}.")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating importance content: {e}")
            return ["Different factors have varying levels of importance for your credit profile."]
    
    async def _generate_whatif_content(self, explanation_data: Dict[str, Any],
                                     style: TextStyle, vocabulary_level: str) -> List[str]:
        """Generate content for what-if explanations"""
        
        try:
            content = []
            
            if 'ranked_scenarios' in explanation_data:
                scenarios = explanation_data['ranked_scenarios'][:3]  # Top 3 scenarios
                
                scenario_descriptions = []
                for scenario in scenarios:
                    scenario_name = scenario.get('scenario', {}).get('name', 'Scenario')
                    predicted_change = scenario.get('predicted_change', {})
                    
                    change_text = self._format_prediction_change(predicted_change)
                    scenario_descriptions.append(f"{scenario_name}: {change_text}")
                
                if scenario_descriptions:
                    content.append(f"The scenario analysis shows: {'; '.join(scenario_descriptions)}.")
                
                # Best scenario recommendation
                if scenarios:
                    best_scenario = scenarios[0]
                    best_name = best_scenario.get('scenario', {}).get('name', 'the top scenario')
                    content.append(f"The most impactful approach would be {best_name.lower()}.")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating what-if content: {e}")
            return ["Different scenarios would have varying impacts on your credit situation."]
    
    async def _generate_conclusion(self, explanation_data: Dict[str, Any],
                                 style: TextStyle) -> str:
        """Generate conclusion text"""
        
        try:
            conclusion_phrases = self.style_templates[style]['conclusion_phrases']
            conclusion_phrase = random.choice(conclusion_phrases)
            
            explanation_type = explanation_data.get('explanation_type', 'analysis')
            
            if explanation_type == 'local':
                conclusion = f"{conclusion_phrase} focusing on the top factors in your specific case will have the most impact."
            elif explanation_type == 'global':
                conclusion = f"{conclusion_phrase} these patterns are consistent across similar credit profiles."
            elif explanation_type == 'counterfactual':
                conclusion = f"{conclusion_phrase} the recommended changes offer the best path to your desired outcome."
            elif explanation_type == 'feature_importance':
                conclusion = f"{conclusion_phrase} prioritizing the most important factors will maximize your improvement efforts."
            elif explanation_type == 'what_if':
                conclusion = f"{conclusion_phrase} the scenario analysis helps you choose the most effective approach."
            else:
                conclusion = f"{conclusion_phrase} this analysis provides valuable insights for improving your credit profile."
            
            return conclusion
            
        except Exception as e:
            logger.error(f"Error generating conclusion: {e}")
            return "This analysis provides insights to help you understand and improve your credit situation."
    
    def _combine_text_sections(self, sections: List[str], style: TextStyle) -> str:
        """Combine text sections into cohesive text"""
        
        try:
            if not sections:
                return "Analysis complete."
            
            # Add appropriate spacing and transitions
            combined_text = sections[0]
            
            for i, section in enumerate(sections[1:], 1):
                if i < len(sections) - 1:  # Not the last section
                    transition_phrases = self.style_templates[style]['transition_phrases']
                    transition = random.choice(transition_phrases)
                    combined_text += f" {transition} {section}"
                else:  # Last section (conclusion)
                    combined_text += f" {section}"
            
            return combined_text
            
        except Exception as e:
            logger.error(f"Error combining text sections: {e}")
            return " ".join(sections)
    
    async def _post_process_text(self, text: str, style: TextStyle, vocabulary_level: str) -> str:
        """Post-process generated text"""
        
        try:
            # Clean up spacing
            processed_text = re.sub(r'\s+', ' ', text).strip()
            
            # Ensure proper capitalization
            processed_text = processed_text[0].upper() + processed_text[1:] if processed_text else ""
            
            # Add final punctuation if missing
            if processed_text and not processed_text.endswith(('.', '!', '?')):
                processed_text += '.'
            
            # Replace technical terms based on vocabulary level
            for term, replacement in self.vocabulary_levels.get(vocabulary_level, {}).items():
                processed_text = processed_text.replace(term, replacement)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error post-processing text: {e}")
            return text
    
    def _translate_term(self, term: str, vocabulary_level: str) -> str:
        """Translate technical term to appropriate vocabulary level"""
        
        # Clean term
        clean_term = term.lower().replace('_', ' ')
        
        # Check vocabulary mapping
        vocab = self.vocabulary_levels.get(vocabulary_level, {})
        
        for tech_term, simple_term in vocab.items():
            if tech_term in clean_term:
                return simple_term
        
        # Default: clean up the term
        return clean_term.title()
    
    def _get_importance_description(self, score: float) -> str:
        """Convert importance score to descriptive text"""
        
        if score > 0.4:
            return "extremely"
        elif score > 0.3:
            return "very"
        elif score > 0.2:
            return "quite"
        elif score > 0.1:
            return "moderately"
        else:
            return "somewhat"
    
    def _format_value(self, value: Any) -> str:
        """Format value for display in text"""
        
        if isinstance(value, float):
            if 0 < value < 1:
                return f"{value:.1%}"
            else:
                return f"{value:.1f}"
        elif isinstance(value, int):
            return str(value)
        else:
            return str(value)
    
    def _format_prediction_change(self, predicted_change: Dict[str, Any]) -> str:
        """Format prediction change for text"""
        
        score_change = predicted_change.get('estimated_score_change', 0)
        direction = predicted_change.get('direction', 'change')
        
        if abs(score_change) < 5:
            return f"minimal {direction}"
        elif abs(score_change) < 20:
            return f"{abs(score_change):.0f} point {direction}"
        else:
            return f"significant {direction} of {abs(score_change):.0f} points"
    
    def _is_actionable(self, factor: str) -> bool:
        """Check if factor is actionable by user"""
        
        actionable_terms = [
            'payment', 'utilization', 'debt', 'balance', 'inquiry', 'account'
        ]
        
        factor_lower = factor.lower()
        return any(term in factor_lower for term in actionable_terms)
    
    async def generate_summary_text(self, explanation_data: Dict[str, Any],
                                  max_words: int = 50) -> str:
        """Generate concise summary text"""
        
        try:
            explanation_type = explanation_data.get('explanation_type', 'analysis')
            
            if explanation_type == 'local' and 'combined_importance' in explanation_data:
                importance = explanation_data['combined_importance']
                top_factor = max(importance.items(), key=lambda x: x[1])[0]
                clean_factor = self._translate_term(top_factor, 'basic')
                return f"Your credit profile is most influenced by {clean_factor}."
            
            elif explanation_type == 'counterfactual' and 'counterfactuals' in explanation_data:
                cf_data = explanation_data['counterfactuals'][0]
                changes_count = len(cf_data.get('changes_made', {}))
                return f"You would need to change {changes_count} key factors to reach your goal."
            
            elif explanation_type == 'what_if' and 'ranked_scenarios' in explanation_data:
                scenarios = explanation_data['ranked_scenarios']
                if scenarios:
                    best_scenario = scenarios[0].get('scenario', {}).get('name', 'the top option')
                    return f"{best_scenario} would have the biggest impact."
            
            return "The analysis shows key factors affecting your credit profile."
            
        except Exception as e:
            logger.error(f"Error generating summary text: {e}")
            return "Analysis complete with key insights identified."
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get text generation statistics"""
        
        return {
            'supported_styles': [style.value for style in TextStyle],
            'vocabulary_levels': list(self.vocabulary_levels.keys()),
            'template_categories': list(self.style_templates[TextStyle.CONVERSATIONAL].keys()),
            'sentence_patterns': list(self.sentence_patterns.keys()),
            'timestamp': datetime.now().isoformat()
        }
