"""
Response generator for Stage 4 explainability chatbot.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import random
import json
from dataclasses import dataclass
from .chat_engine import ChatResponse

logger = logging.getLogger(__name__)

@dataclass
class ResponseTemplate:
    """Response template data structure"""
    template_id: str
    intent: str
    template: str
    variables: List[str]
    confidence_threshold: float = 0.6

class ResponseGenerator:
    """Generate responses for credit explanation chatbot"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.response_templates = {}
        self.fallback_responses = []
        self.context_aware_responses = {}
        self._initialize_templates()
        
    def _initialize_templates(self):
        """Initialize response templates"""
        
        self.response_templates = {
            'credit_score_inquiry': [
                ResponseTemplate(
                    'credit_score_basic',
                    'credit_score_inquiry',
                    "Your credit score of {score} is in the {range} category. This score is calculated based on factors like payment history ({payment_history}%), credit utilization ({utilization}%), and length of credit history.",
                    ['score', 'range', 'payment_history', 'utilization']
                ),
                ResponseTemplate(
                    'credit_score_detailed',
                    'credit_score_inquiry',
                    "Based on your credit profile, your score of {score} reflects {primary_factor} as the most significant factor. {explanation}",
                    ['score', 'primary_factor', 'explanation']
                )
            ],
            'loan_decision_explanation': [
                ResponseTemplate(
                    'loan_approved',
                    'loan_decision_explanation',
                    "Your loan application was approved because {approval_reasons}. The key factors that supported your application were: {key_factors}.",
                    ['approval_reasons', 'key_factors']
                ),
                ResponseTemplate(
                    'loan_denied',
                    'loan_decision_explanation',
                    "Your loan application was not approved primarily due to {denial_reasons}. To improve your chances in the future, consider: {recommendations}.",
                    ['denial_reasons', 'recommendations']
                )
            ],
            'feature_importance': [
                ResponseTemplate(
                    'feature_ranking',
                    'feature_importance',
                    "The most important factors affecting your credit profile are: 1) {factor1} ({importance1}%), 2) {factor2} ({importance2}%), 3) {factor3} ({importance3}%).",
                    ['factor1', 'importance1', 'factor2', 'importance2', 'factor3', 'importance3']
                ),
                ResponseTemplate(
                    'feature_explanation',
                    'feature_importance',
                    "{feature} is particularly important in your case because {explanation}. This factor accounts for {percentage}% of the decision.",
                    ['feature', 'explanation', 'percentage']
                )
            ],
            'what_if_scenario': [
                ResponseTemplate(
                    'scenario_result',
                    'what_if_scenario',
                    "If you {scenario_change}, your credit score would likely {direction} to approximately {new_score}. This change would {impact_description}.",
                    ['scenario_change', 'direction', 'new_score', 'impact_description']
                ),
                ResponseTemplate(
                    'multiple_scenarios',
                    'what_if_scenario',
                    "I've analyzed several scenarios for you: {scenario_list}. The most impactful change would be {best_scenario}.",
                    ['scenario_list', 'best_scenario']
                )
            ],
            'counterfactual_analysis': [
                ResponseTemplate(
                    'counterfactual_simple',
                    'counterfactual_analysis',
                    "To achieve {target_outcome}, you would need to change: {required_changes}. These changes would have a {confidence}% likelihood of success.",
                    ['target_outcome', 'required_changes', 'confidence']
                ),
                ResponseTemplate(
                    'counterfactual_detailed',
                    'counterfactual_analysis',
                    "Based on counterfactual analysis, the minimum changes needed are: {changes}. This would move your score from {current_score} to {target_score}.",
                    ['changes', 'current_score', 'target_score']
                )
            ],
            'improvement_suggestions': [
                ResponseTemplate(
                    'improvement_list',
                    'improvement_suggestions',
                    "Here are personalized recommendations to improve your credit: {recommendations}. Focus on {priority_action} first for maximum impact.",
                    ['recommendations', 'priority_action']
                ),
                ResponseTemplate(
                    'improvement_timeline',
                    'improvement_suggestions',
                    "Your improvement plan: {timeline}. You could see results in {timeframe} by following these steps.",
                    ['timeline', 'timeframe']
                )
            ],
            'general_explanation': [
                ResponseTemplate(
                    'explanation_simple',
                    'general_explanation',
                    "{concept} works by {explanation}. In your specific case, {personalized_info}.",
                    ['concept', 'explanation', 'personalized_info']
                ),
                ResponseTemplate(
                    'explanation_detailed',
                    'general_explanation',
                    "Let me explain {topic} in detail: {detailed_explanation}. The key points are: {key_points}.",
                    ['topic', 'detailed_explanation', 'key_points']
                )
            ],
            'greeting': [
                ResponseTemplate(
                    'greeting_basic',
                    'greeting',
                    "Hello! I'm here to help you understand your credit profile and financial decisions. What would you like to know about?",
                    []
                ),
                ResponseTemplate(
                    'greeting_personalized',
                    'greeting',
                    "Welcome back! I can help explain your credit analysis, discuss improvement strategies, or answer questions about your financial profile. How can I assist you today?",
                    []
                )
            ],
            'help': [
                ResponseTemplate(
                    'help_general',
                    'help',
                    "I can help you with: credit score explanations, loan decision analysis, feature importance, what-if scenarios, improvement suggestions, and counterfactual analysis. What interests you most?",
                    []
                ),
                ResponseTemplate(
                    'help_specific',
                    'help',
                    "Here are some things you can ask me: {help_options}. Just ask in natural language!",
                    ['help_options']
                )
            ],
            'clarification': [
                ResponseTemplate(
                    'clarification_request',
                    'clarification',
                    "I want to make sure I understand correctly. Are you asking about {clarification_topic}? Could you provide more details about {specific_aspect}?",
                    ['clarification_topic', 'specific_aspect']
                ),
                ResponseTemplate(
                    'clarification_explanation',
                    'clarification',
                    "Let me clarify {topic}: {clear_explanation}. Does this answer your question, or would you like me to explain any part in more detail?",
                    ['topic', 'clear_explanation']
                )
            ]
        }
        
        self.fallback_responses = [
            "I'm not sure I understand that question. Could you rephrase it or ask about your credit score, loan decisions, or improvement suggestions?",
            "That's an interesting question! Could you provide more context so I can give you a better answer?",
            "I'd like to help with that. Could you be more specific about what aspect of your credit profile you're interested in?",
            "I can help explain credit-related topics. Try asking about your credit score, loan decisions, or what factors are most important."
        ]
        
        self.context_aware_responses = {
            'first_interaction': "Welcome! I'm your credit explanation assistant. I can help you understand your credit profile, explain decisions, and suggest improvements.",
            'returning_user': "Welcome back! Based on our previous conversation, would you like to continue discussing {previous_topic} or explore something new?",
            'high_confidence': "I'm confident in this analysis: {response}",
            'low_confidence': "Based on available information, {response}. However, I'd recommend getting additional details for a more precise analysis.",
            'error_recovery': "I apologize for the confusion. Let me try to help in a different way: {alternative_response}"
        }
    
    async def generate_response(self, message, context: Dict[str, Any], 
                              knowledge_base) -> 'ChatResponse':
        """Generate response based on message and context"""
        
        try:
            intent = message.intent
            confidence = message.confidence
            entities = message.entities or {}
            
            # Get appropriate template
            template = self._select_template(intent, confidence, context)
            
            if template:
                # Fill template with context data
                response_text = await self._fill_template(template, context, entities, knowledge_base)
            else:
                # Use fallback response
                response_text = self._get_fallback_response(intent, context)
            
            # Add context-aware modifications
            response_text = self._add_context_awareness(response_text, context, confidence)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(intent, context, entities)
            
            from .chat_engine import ChatResponse
            return ChatResponse(
                response=response_text,
                intent=intent,
                confidence=confidence,
                entities=entities,
                context=context,
                suggestions=suggestions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            from .chat_engine import ChatResponse
            return ChatResponse(
                response="I apologize, but I encountered an error. Please try asking your question again.",
                intent="error",
                confidence=0.0,
                entities={},
                context=context,
                suggestions=["Try rephrasing your question", "Ask about credit score factors"],
                timestamp=datetime.now()
            )
    
    def _select_template(self, intent: str, confidence: float, 
                        context: Dict[str, Any]) -> Optional[ResponseTemplate]:
        """Select appropriate response template"""
        
        try:
            if intent not in self.response_templates:
                return None
            
            templates = self.response_templates[intent]
            
            # Filter templates by confidence threshold
            suitable_templates = [
                t for t in templates 
                if confidence >= t.confidence_threshold
            ]
            
            if not suitable_templates:
                suitable_templates = templates  # Use any template if none meet threshold
            
            # Select based on context
            if context.get('detailed_explanation_requested'):
                detailed_templates = [t for t in suitable_templates if 'detailed' in t.template_id]
                if detailed_templates:
                    return detailed_templates[0]
            
            # Default to first suitable template
            return suitable_templates[0]
            
        except Exception as e:
            logger.error(f"Error selecting template: {e}")
            return None
    
    async def _fill_template(self, template: ResponseTemplate, context: Dict[str, Any],
                           entities: Dict[str, Any], knowledge_base) -> str:
        """Fill template with context data"""
        
        try:
            template_vars = {}
            
            # Extract variables from context and entities
            for var in template.variables:
                if var in context:
                    template_vars[var] = context[var]
                elif var in entities:
                    template_vars[var] = entities[var]
                else:
                    # Try to generate variable from knowledge base or context
                    template_vars[var] = await self._generate_variable(var, context, entities, knowledge_base)
            
            # Fill template
            try:
                response = template.template.format(**template_vars)
            except KeyError as e:
                # Handle missing variables
                missing_var = str(e).strip("'")
                template_vars[missing_var] = f"[{missing_var}]"
                response = template.template.format(**template_vars)
            
            return response
            
        except Exception as e:
            logger.error(f"Error filling template: {e}")
            return template.template  # Return unfilled template as fallback
    
    async def _generate_variable(self, var_name: str, context: Dict[str, Any],
                               entities: Dict[str, Any], knowledge_base) -> str:
        """Generate variable value from context or knowledge base"""
        
        try:
            # Credit score related variables
            if var_name == 'score':
                if 'credit_score' in context:
                    return str(context['credit_score'])
                elif 'CREDIT_SCORE' in entities:
                    return str(entities['CREDIT_SCORE'][0].text if entities['CREDIT_SCORE'] else 'your score')
                return 'your credit score'
            
            elif var_name == 'range':
                score = context.get('credit_score', 700)
                return self._get_score_range(score)
            
            elif var_name in ['payment_history', 'utilization']:
                return context.get(var_name, 'N/A')
            
            # Feature importance variables
            elif var_name.startswith('factor'):
                factor_num = int(var_name[-1]) - 1
                factors = context.get('top_factors', ['Payment History', 'Credit Utilization', 'Credit Age'])
                return factors[factor_num] if factor_num < len(factors) else 'Other Factor'
            
            elif var_name.startswith('importance'):
                importance_num = int(var_name[-1]) - 1
                importances = context.get('factor_importances', [35, 30, 15])
                return str(importances[importance_num]) if importance_num < len(importances) else '10'
            
            # Scenario variables
            elif var_name == 'scenario_change':
                return context.get('scenario_description', 'make the suggested changes')
            
            elif var_name == 'new_score':
                return str(context.get('predicted_score', 'a higher value'))
            
            elif var_name == 'direction':
                current = context.get('credit_score', 700)
                new = context.get('predicted_score', 720)
                return 'increase' if new > current else 'decrease' if new < current else 'remain similar'
            
            # Improvement variables
            elif var_name == 'recommendations':
                return context.get('improvement_suggestions', 'focus on timely payments and reducing credit utilization')
            
            elif var_name == 'priority_action':
                return context.get('top_recommendation', 'improving payment history')
            
            # Default fallback
            return context.get(var_name, f'[{var_name}]')
            
        except Exception as e:
            logger.error(f"Error generating variable {var_name}: {e}")
            return f'[{var_name}]'
    
    def _get_score_range(self, score: int) -> str:
        """Get credit score range description"""
        
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
    
    def _get_fallback_response(self, intent: str, context: Dict[str, Any]) -> str:
        """Get fallback response when template fails"""
        
        try:
            # Intent-specific fallbacks
            if intent == 'credit_score_inquiry':
                return "I'd be happy to explain your credit score. Could you provide your current score or specific questions about it?"
            
            elif intent == 'loan_decision_explanation':
                return "I can help explain loan decisions. Was your application approved or denied, and what specific aspects would you like me to explain?"
            
            elif intent == 'feature_importance':
                return "I can explain which factors are most important for your credit profile. Would you like me to analyze your specific situation?"
            
            elif intent == 'improvement_suggestions':
                return "I'd be happy to provide personalized improvement suggestions. Could you share some details about your current credit situation?"
            
            # Use random fallback
            return random.choice(self.fallback_responses)
            
        except Exception:
            return self.fallback_responses[0]
    
    def _add_context_awareness(self, response: str, context: Dict[str, Any], 
                             confidence: float) -> str:
        """Add context-aware modifications to response"""
        
        try:
            # Add confidence indicators
            if confidence < 0.5:
                response = self.context_aware_responses['low_confidence'].format(response=response)
            elif confidence > 0.9:
                response = self.context_aware_responses['high_confidence'].format(response=response)
            
            # Add personalization based on context
            if context.get('user_name'):
                response = f"{context['user_name']}, {response.lower()}"
            
            # Add follow-up prompts
            if context.get('conversation_depth', 0) < 2:
                response += " Is there anything specific you'd like me to explain further?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding context awareness: {e}")
            return response
    
    async def _generate_suggestions(self, intent: str, context: Dict[str, Any],
                                  entities: Dict[str, Any]) -> List[str]:
        """Generate follow-up suggestions"""
        
        try:
            suggestions = []
            
            if intent == 'credit_score_inquiry':
                suggestions = [
                    "What factors most impact my score?",
                    "How can I improve my credit score?",
                    "Show me what-if scenarios",
                    "Explain feature importance"
                ]
            
            elif intent == 'loan_decision_explanation':
                suggestions = [
                    "What changes would improve my chances?",
                    "Show me counterfactual analysis",
                    "Explain the decision factors",
                    "How can I strengthen my application?"
                ]
            
            elif intent == 'feature_importance':
                suggestions = [
                    "How can I improve the top factors?",
                    "Show me what-if scenarios",
                    "Explain why these factors matter",
                    "What's my improvement timeline?"
                ]
            
            elif intent == 'what_if_scenario':
                suggestions = [
                    "Try different scenarios",
                    "Show me improvement recommendations",
                    "Explain the analysis method",
                    "What's the most impactful change?"
                ]
            
            elif intent == 'improvement_suggestions':
                suggestions = [
                    "Create an improvement timeline",
                    "Show me what-if scenarios",
                    "Explain why these help",
                    "What should I prioritize first?"
                ]
            
            else:
                # General suggestions
                suggestions = [
                    "Explain my credit score",
                    "Show feature importance",
                    "Give improvement suggestions",
                    "Analyze what-if scenarios"
                ]
            
            # Personalize based on context
            if context.get('credit_score'):
                score = context['credit_score']
                if score < 600:
                    suggestions.append("Focus on payment history improvement")
                elif score > 750:
                    suggestions.append("Optimize for excellent credit")
            
            return suggestions[:4]  # Limit to 4 suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return ["Ask me about your credit", "Request an explanation"]
    
    async def generate_welcome_message(self, user_id: str, 
                                     initial_context: Dict[str, Any] = None) -> str:
        """Generate welcome message for new session"""
        
        try:
            context = initial_context or {}
            
            if context.get('returning_user'):
                previous_topic = context.get('previous_topic', 'your credit profile')
                return self.context_aware_responses['returning_user'].format(previous_topic=previous_topic)
            else:
                return self.context_aware_responses['first_interaction']
                
        except Exception as e:
            logger.error(f"Error generating welcome message: {e}")
            return "Hello! I'm here to help explain your credit profile and financial decisions."
    
    async def generate_credit_response(self, credit_data: Dict[str, Any],
                                     context: Dict[str, Any], knowledge_base) -> 'ChatResponse':
        """Generate credit-specific response"""
        
        try:
            # Analyze credit data
            score = credit_data.get('credit_score', 0)
            factors = credit_data.get('top_factors', [])
            
            # Create response based on credit data
            if score > 0:
                score_range = self._get_score_range(score)
                response = f"Based on your credit data, your score of {score} is in the {score_range} range. "
                
                if factors:
                    response += f"The key factors affecting your score are: {', '.join(factors[:3])}. "
                
                if score < 600:
                    response += "I recommend focusing on payment history and reducing credit utilization to improve your score."
                elif score > 750:
                    response += "Your credit profile is strong. Consider optimizing for even better terms on future credit applications."
                else:
                    response += "There are several opportunities to improve your credit score with targeted actions."
            else:
                response = "I'd be happy to analyze your credit profile. Could you provide your credit score and any specific concerns?"
            
            
            return ChatResponse(
                response=response,
                intent="credit_analysis",
                confidence=0.9,
                entities={},
                context=context,
                suggestions=await self._generate_suggestions("credit_score_inquiry", context, {}),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating credit response: {e}")
            from .chat_engine import ChatResponse
            return ChatResponse(
                response="I encountered an error analyzing your credit data. Please try again.",
                intent="error",
                confidence=0.0,
                entities={},
                context=context,
                suggestions=["Provide credit score", "Ask about specific factors"],
                timestamp=datetime.now()
            )
    
    def add_custom_template(self, intent: str, template: ResponseTemplate):
        """Add custom response template"""
        
        try:
            if intent not in self.response_templates:
                self.response_templates[intent] = []
            
            self.response_templates[intent].append(template)
            logger.info(f"Added custom template for intent: {intent}")
            
        except Exception as e:
            logger.error(f"Error adding custom template: {e}")
    
    def get_response_statistics(self) -> Dict[str, Any]:
        """Get response generation statistics"""
        
        template_counts = {}
        for intent, templates in self.response_templates.items():
            template_counts[intent] = len(templates)
        
        return {
            'supported_intents': list(self.response_templates.keys()),
            'template_counts': template_counts,
            'total_templates': sum(template_counts.values()),
            'fallback_responses': len(self.fallback_responses),
            'timestamp': datetime.now().isoformat()
        }
