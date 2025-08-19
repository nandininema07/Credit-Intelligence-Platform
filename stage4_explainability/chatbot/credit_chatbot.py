"""
Credit chatbot for explaining credit decisions and answering questions.
Provides natural language explanations for credit risk assessments.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Chat message structure"""
    user_id: str
    message: str
    timestamp: datetime
    message_type: str = "user"  # user, bot, system
    
@dataclass
class ChatResponse:
    """Chat response structure"""
    response: str
    confidence: float
    response_type: str  # explanation, clarification, error
    related_data: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class CreditChatbot:
    """Credit risk chatbot for explanations and Q&A"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_history = {}
        self.credit_data = {}
        self.explanation_templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load response templates"""
        return {
            'credit_score_explanation': """
            Your credit score of {score} is in the {risk_category} category. 
            This score is calculated based on several key factors:
            
            {top_factors}
            
            {improvement_suggestions}
            """,
            
            'feature_explanation': """
            {feature_name} has a {impact_type} impact on your credit score.
            Current value: {feature_value}
            Impact: {impact_magnitude} points
            
            {feature_description}
            """,
            
            'comparison_explanation': """
            Compared to similar companies in your industry:
            - Your score: {your_score}
            - Industry average: {industry_avg}
            - You are performing {performance_level}
            
            {comparison_details}
            """,
            
            'improvement_suggestions': """
            To improve your credit score, consider focusing on:
            {suggestions}
            """,
            
            'risk_factors': """
            The main risk factors affecting your score are:
            {risk_factors}
            
            These factors contribute to {total_risk_impact} points of risk.
            """
        }
    
    def _extract_intent(self, message: str) -> str:
        """Extract user intent from message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['why', 'explain', 'reason']):
            if 'score' in message_lower:
                return 'explain_score'
            elif any(word in message_lower for word in ['factor', 'feature', 'metric']):
                return 'explain_feature'
            else:
                return 'general_explanation'
        
        elif any(word in message_lower for word in ['improve', 'better', 'increase']):
            return 'improvement_advice'
        
        elif any(word in message_lower for word in ['compare', 'benchmark', 'industry']):
            return 'comparison'
        
        elif any(word in message_lower for word in ['risk', 'danger', 'concern']):
            return 'risk_analysis'
        
        elif any(word in message_lower for word in ['what is', 'define', 'meaning']):
            return 'definition'
        
        else:
            return 'general_query'
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from user message"""
        entities = {}
        
        # Extract company mentions
        company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        companies = re.findall(company_pattern, message)
        if companies:
            entities['companies'] = companies
        
        # Extract feature/metric names
        feature_keywords = [
            'liquidity', 'profitability', 'leverage', 'debt', 'equity',
            'revenue', 'earnings', 'cash flow', 'market cap', 'volatility',
            'sentiment', 'trend', 'ratio', 'margin'
        ]
        
        mentioned_features = [kw for kw in feature_keywords if kw in message.lower()]
        if mentioned_features:
            entities['features'] = mentioned_features
        
        # Extract numbers (scores, percentages, etc.)
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', message)
        if numbers:
            entities['numbers'] = numbers
        
        return entities
    
    async def process_message(self, user_id: str, message: str, 
                            credit_data: Dict[str, Any] = None) -> ChatResponse:
        """Process user message and generate response"""
        
        # Store message in conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        chat_message = ChatMessage(user_id, message, datetime.now())
        self.conversation_history[user_id].append(chat_message)
        
        # Store credit data if provided
        if credit_data:
            self.credit_data[user_id] = credit_data
        
        # Extract intent and entities
        intent = self._extract_intent(message)
        entities = self._extract_entities(message)
        
        # Generate response based on intent
        try:
            response = await self._generate_response(user_id, intent, entities, message)
            
            # Store bot response
            bot_message = ChatMessage(user_id, response.response, datetime.now(), "bot")
            self.conversation_history[user_id].append(bot_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return ChatResponse(
                response="I apologize, but I encountered an error processing your request. Please try again.",
                confidence=0.0,
                response_type="error"
            )
    
    async def _generate_response(self, user_id: str, intent: str, 
                               entities: Dict[str, Any], original_message: str) -> ChatResponse:
        """Generate response based on intent"""
        
        user_credit_data = self.credit_data.get(user_id, {})
        
        if intent == 'explain_score':
            return await self._explain_credit_score(user_credit_data)
        
        elif intent == 'explain_feature':
            return await self._explain_feature(entities, user_credit_data)
        
        elif intent == 'improvement_advice':
            return await self._provide_improvement_advice(user_credit_data)
        
        elif intent == 'comparison':
            return await self._provide_comparison(user_credit_data)
        
        elif intent == 'risk_analysis':
            return await self._analyze_risk_factors(user_credit_data)
        
        elif intent == 'definition':
            return await self._provide_definition(entities, original_message)
        
        else:
            return await self._handle_general_query(original_message, user_credit_data)
    
    async def _explain_credit_score(self, credit_data: Dict[str, Any]) -> ChatResponse:
        """Explain credit score"""
        if not credit_data:
            return ChatResponse(
                response="I don't have your credit information available. Please provide your credit assessment first.",
                confidence=0.8,
                response_type="clarification"
            )
        
        score = credit_data.get('credit_score', 0)
        risk_category = credit_data.get('risk_category', 'Unknown')
        shap_values = credit_data.get('shap_values', {})
        
        # Get top factors
        if shap_values:
            sorted_factors = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
            top_factors = []
            
            for feature, impact in sorted_factors[:3]:
                impact_type = "positive" if impact > 0 else "negative"
                feature_name = feature.replace('_', ' ').title()
                top_factors.append(f"• {feature_name}: {impact_type} impact ({abs(impact):.1f} points)")
            
            top_factors_text = "\n".join(top_factors)
        else:
            top_factors_text = "Detailed factor analysis not available."
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(shap_values)
        
        response_text = self.explanation_templates['credit_score_explanation'].format(
            score=int(score),
            risk_category=risk_category,
            top_factors=top_factors_text,
            improvement_suggestions=improvement_suggestions
        )
        
        return ChatResponse(
            response=response_text,
            confidence=0.9,
            response_type="explanation",
            related_data={'score': score, 'risk_category': risk_category}
        )
    
    async def _explain_feature(self, entities: Dict[str, Any], 
                             credit_data: Dict[str, Any]) -> ChatResponse:
        """Explain specific feature impact"""
        features = entities.get('features', [])
        shap_values = credit_data.get('shap_values', {})
        feature_values = credit_data.get('feature_values', {})
        
        if not features:
            return ChatResponse(
                response="Which specific factor would you like me to explain? For example: liquidity, profitability, or debt ratios.",
                confidence=0.7,
                response_type="clarification"
            )
        
        feature = features[0]  # Focus on first mentioned feature
        
        # Find matching feature in data
        matching_features = [k for k in shap_values.keys() if feature.lower() in k.lower()]
        
        if not matching_features:
            return ChatResponse(
                response=f"I don't have specific information about {feature} in your credit assessment.",
                confidence=0.6,
                response_type="clarification"
            )
        
        feature_key = matching_features[0]
        impact = shap_values.get(feature_key, 0)
        value = feature_values.get(feature_key, 0)
        
        impact_type = "positive" if impact > 0 else "negative"
        feature_description = self._get_feature_description(feature_key)
        
        response_text = self.explanation_templates['feature_explanation'].format(
            feature_name=feature_key.replace('_', ' ').title(),
            impact_type=impact_type,
            feature_value=f"{value:.2f}",
            impact_magnitude=f"{abs(impact):.1f}",
            feature_description=feature_description
        )
        
        return ChatResponse(
            response=response_text,
            confidence=0.85,
            response_type="explanation",
            related_data={'feature': feature_key, 'impact': impact, 'value': value}
        )
    
    def _generate_improvement_suggestions(self, shap_values: Dict[str, float]) -> str:
        """Generate improvement suggestions based on SHAP values"""
        if not shap_values:
            return "Specific improvement suggestions are not available without detailed analysis."
        
        # Find most negative impacts
        negative_factors = [(k, v) for k, v in shap_values.items() if v < 0]
        negative_factors.sort(key=lambda x: x[1])  # Most negative first
        
        suggestions = []
        for feature, impact in negative_factors[:3]:
            suggestion = self._get_improvement_suggestion(feature)
            if suggestion:
                suggestions.append(f"• {suggestion}")
        
        if not suggestions:
            return "Your current metrics are performing well. Continue monitoring key financial ratios."
        
        return "\n".join(suggestions)
    
    def _get_improvement_suggestion(self, feature: str) -> str:
        """Get improvement suggestion for specific feature"""
        suggestions = {
            'liquidity': "Improve cash flow management and maintain higher cash reserves",
            'profitability': "Focus on increasing profit margins and operational efficiency",
            'leverage': "Consider reducing debt levels or increasing equity",
            'debt': "Work on debt reduction strategies and improve debt service coverage",
            'volatility': "Implement risk management strategies to reduce earnings volatility",
            'sentiment': "Improve public relations and stakeholder communication",
            'trend': "Focus on sustainable growth and consistent performance"
        }
        
        for key, suggestion in suggestions.items():
            if key in feature.lower():
                return suggestion
        
        return f"Monitor and improve {feature.replace('_', ' ').lower()} metrics"
    
    def _get_feature_description(self, feature: str) -> str:
        """Get description for a feature"""
        descriptions = {
            'liquidity': "Measures your ability to meet short-term obligations with liquid assets.",
            'profitability': "Indicates how efficiently your company generates profit from operations.",
            'leverage': "Shows your reliance on debt financing relative to equity.",
            'debt_to_equity': "Compares total debt to shareholders' equity, indicating financial leverage.",
            'current_ratio': "Measures ability to pay short-term debts with current assets.",
            'net_profit_margin': "Shows what percentage of revenue becomes profit after all expenses.",
            'sentiment': "Reflects market and public perception based on news and social media.",
            'volatility': "Measures the degree of variation in your financial metrics over time."
        }
        
        for key, description in descriptions.items():
            if key in feature.lower():
                return description
        
        return "This metric contributes to your overall credit risk assessment."
    
    async def _provide_improvement_advice(self, credit_data: Dict[str, Any]) -> ChatResponse:
        """Provide improvement advice"""
        shap_values = credit_data.get('shap_values', {})
        
        if not shap_values:
            general_advice = """
            To improve your credit score, focus on these key areas:
            • Maintain strong liquidity ratios
            • Improve profitability margins
            • Manage debt levels effectively
            • Ensure consistent financial performance
            • Monitor market sentiment and communication
            """
            
            return ChatResponse(
                response=general_advice,
                confidence=0.7,
                response_type="explanation"
            )
        
        suggestions = self._generate_improvement_suggestions(shap_values)
        
        response_text = self.explanation_templates['improvement_suggestions'].format(
            suggestions=suggestions
        )
        
        return ChatResponse(
            response=response_text,
            confidence=0.85,
            response_type="explanation"
        )
    
    async def _provide_comparison(self, credit_data: Dict[str, Any]) -> ChatResponse:
        """Provide industry comparison"""
        score = credit_data.get('credit_score', 500)
        
        # Simulated industry benchmarks
        industry_avg = 650
        performance_level = "above average" if score > industry_avg else "below average"
        
        comparison_details = f"""
        Key differences from industry peers:
        • Your liquidity position: {'Strong' if score > 700 else 'Needs improvement'}
        • Profitability performance: {'Above peer group' if score > 650 else 'Below peer group'}
        • Risk management: {'Effective' if score > 600 else 'Requires attention'}
        """
        
        response_text = self.explanation_templates['comparison_explanation'].format(
            your_score=int(score),
            industry_avg=industry_avg,
            performance_level=performance_level,
            comparison_details=comparison_details
        )
        
        return ChatResponse(
            response=response_text,
            confidence=0.75,
            response_type="explanation"
        )
    
    async def _analyze_risk_factors(self, credit_data: Dict[str, Any]) -> ChatResponse:
        """Analyze risk factors"""
        shap_values = credit_data.get('shap_values', {})
        
        if not shap_values:
            return ChatResponse(
                response="I need your detailed credit assessment to analyze specific risk factors.",
                confidence=0.6,
                response_type="clarification"
            )
        
        # Find negative contributing factors
        risk_factors = [(k, v) for k, v in shap_values.items() if v < 0]
        risk_factors.sort(key=lambda x: x[1])  # Most negative first
        
        if not risk_factors:
            return ChatResponse(
                response="Great news! Your current assessment shows no significant risk factors. All metrics are contributing positively to your credit score.",
                confidence=0.9,
                response_type="explanation"
            )
        
        risk_text = []
        total_risk = 0
        
        for feature, impact in risk_factors[:5]:
            feature_name = feature.replace('_', ' ').title()
            risk_text.append(f"• {feature_name}: -{abs(impact):.1f} points")
            total_risk += abs(impact)
        
        risk_factors_text = "\n".join(risk_text)
        
        response_text = self.explanation_templates['risk_factors'].format(
            risk_factors=risk_factors_text,
            total_risk_impact=int(total_risk)
        )
        
        return ChatResponse(
            response=response_text,
            confidence=0.85,
            response_type="explanation",
            related_data={'total_risk_impact': total_risk}
        )
    
    async def _provide_definition(self, entities: Dict[str, Any], message: str) -> ChatResponse:
        """Provide definitions for financial terms"""
        definitions = {
            'credit score': "A numerical representation of creditworthiness, typically ranging from 300-850, with higher scores indicating lower risk.",
            'liquidity': "The ability to quickly convert assets to cash or meet short-term obligations without significant loss.",
            'leverage': "The use of borrowed money to finance operations or investments, measured by debt-to-equity ratios.",
            'volatility': "The degree of variation in financial metrics over time, indicating stability or instability.",
            'shap values': "SHAP (SHapley Additive exPlanations) values show how much each feature contributes to a model's prediction.",
            'risk category': "A classification system (Low, Medium, High, Very High Risk) based on credit score ranges."
        }
        
        # Find relevant definition
        message_lower = message.lower()
        for term, definition in definitions.items():
            if term in message_lower:
                return ChatResponse(
                    response=f"{term.title()}: {definition}",
                    confidence=0.95,
                    response_type="explanation"
                )
        
        return ChatResponse(
            response="I can help define financial terms like credit score, liquidity, leverage, volatility, and risk categories. What specific term would you like me to explain?",
            confidence=0.7,
            response_type="clarification"
        )
    
    async def _handle_general_query(self, message: str, credit_data: Dict[str, Any]) -> ChatResponse:
        """Handle general queries"""
        return ChatResponse(
            response="I'm here to help explain your credit assessment and answer questions about financial metrics. You can ask me about your credit score, specific factors affecting it, improvement suggestions, or definitions of financial terms.",
            confidence=0.6,
            response_type="clarification"
        )
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get conversation summary for a user"""
        if user_id not in self.conversation_history:
            return {}
        
        messages = self.conversation_history[user_id]
        user_messages = [m for m in messages if m.message_type == "user"]
        bot_messages = [m for m in messages if m.message_type == "bot"]
        
        return {
            'total_messages': len(messages),
            'user_messages': len(user_messages),
            'bot_messages': len(bot_messages),
            'conversation_start': messages[0].timestamp if messages else None,
            'last_interaction': messages[-1].timestamp if messages else None
        }
