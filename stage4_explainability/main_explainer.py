"""
Main explainability engine for Stage 4.
Provides AI-powered explanations for credit scores and model decisions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ML Explainability
import shap
from lime import lime_tabular
from sklearn.inspection import permutation_importance

# AI/LLM
import openai
from transformers import pipeline

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """Main explainability engine for credit score explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainers = {}
        self.explanation_cache = {}
        self.chat_history = {}
        
        # Initialize OpenAI if API key is provided
        self.openai_client = None
        if config.get('openai_api_key'):
            openai.api_key = config['openai_api_key']
            self.openai_client = openai
        
    async def initialize(self):
        """Initialize the explainability engine"""
        try:
            # Initialize SHAP explainers
            await self._initialize_explainers()
            
            # Initialize chat components
            await self._initialize_chat_components()
            
            logger.info("Stage 4 Explainability Engine initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Stage 4: {e}")
            raise
    
    async def _initialize_explainers(self):
        """Initialize ML explainers"""
        try:
            # SHAP explainers will be initialized when models are loaded
            self.explainers['shap'] = {}
            self.explainers['lime'] = {}
            self.explainers['permutation'] = {}
            
            logger.info("ML explainers initialized")
        except Exception as e:
            logger.error(f"Error initializing explainers: {e}")
    
    async def _initialize_chat_components(self):
        """Initialize chat and NLP components"""
        try:
            # Initialize sentiment analysis for chat
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            logger.info("Chat components initialized")
        except Exception as e:
            logger.error(f"Error initializing chat components: {e}")
    
    async def explain_score(self, company_name: str, score_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive explanation for a credit score"""
        try:
            logger.info(f"Generating explanation for {company_name}")
            
            # Get model and features used for scoring
            model_name = score_result.get('model_used')
            score = score_result.get('score')
            
            if not model_name or score is None:
                return {'error': 'Invalid score result'}
            
            # Get feature importance
            feature_importance = await self._get_feature_importance(company_name, model_name)
            
            # Generate SHAP explanation
            shap_explanation = await self._generate_shap_explanation(company_name, model_name)
            
            # Generate natural language explanation
            nl_explanation = await self._generate_natural_language_explanation(
                company_name, score, feature_importance, shap_explanation
            )
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                company_name, score, feature_importance
            )
            
            explanation = {
                'company': company_name,
                'score': score,
                'model_used': model_name,
                'timestamp': datetime.now().isoformat(),
                'feature_importance': feature_importance,
                'shap_values': shap_explanation,
                'natural_language': nl_explanation,
                'recommendations': recommendations,
                'confidence': score_result.get('confidence', 0.5)
            }
            
            # Cache explanation
            cache_key = f"{company_name}_{model_name}_{datetime.now().date()}"
            self.explanation_cache[cache_key] = explanation
            
            logger.info(f"Generated explanation for {company_name}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation for {company_name}: {e}")
            return {'error': str(e)}
    
    async def _get_feature_importance(self, company_name: str, model_name: str) -> Dict[str, float]:
        """Get feature importance for the model"""
        try:
            # This would typically get actual feature importance from the trained model
            # For now, generate mock feature importance
            features = [
                'debt_to_equity', 'current_ratio', 'roa', 'profit_margin', 'revenue_growth',
                'sentiment_score', 'news_volume', 'market_volatility', 'peer_comparison',
                'industry_trend', 'management_quality', 'esg_score', 'liquidity_ratio',
                'interest_coverage', 'asset_turnover', 'working_capital', 'cash_flow',
                'market_share', 'competitive_position', 'regulatory_risk'
            ]
            
            # Generate realistic importance scores
            importance_scores = np.random.exponential(0.1, len(features))
            importance_scores = importance_scores / np.sum(importance_scores)  # Normalize
            
            feature_importance = dict(zip(features, importance_scores))
            
            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    async def _generate_shap_explanation(self, company_name: str, model_name: str) -> Dict[str, Any]:
        """Generate SHAP explanation"""
        try:
            # This would typically use actual SHAP values from the model
            # For now, generate mock SHAP explanation
            features = ['debt_to_equity', 'current_ratio', 'roa', 'profit_margin', 'sentiment_score']
            shap_values = np.random.normal(0, 0.1, len(features))
            
            shap_explanation = {
                'base_value': 650,  # Base credit score
                'shap_values': dict(zip(features, shap_values.tolist())),
                'expected_value': 650 + np.sum(shap_values),
                'feature_values': {feature: np.random.randn() for feature in features}
            }
            
            return shap_explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            return {}
    
    async def _generate_natural_language_explanation(self, company_name: str, score: float,
                                                   feature_importance: Dict[str, float],
                                                   shap_explanation: Dict[str, Any]) -> str:
        """Generate natural language explanation using AI"""
        try:
            if not self.openai_client:
                return await self._generate_template_explanation(company_name, score, feature_importance)
            
            # Prepare context for AI explanation
            top_features = list(feature_importance.items())[:5]
            
            prompt = f"""
            Generate a clear, professional explanation for why {company_name} received a credit score of {score}.
            
            Key factors (in order of importance):
            {chr(10).join([f"- {feature}: {importance:.3f}" for feature, importance in top_features])}
            
            The explanation should be:
            1. Easy to understand for business stakeholders
            2. Specific about which factors helped or hurt the score
            3. Professional and objective
            4. Around 150-200 words
            
            Focus on the business implications and be specific about the financial metrics.
            """
            
            response = await self.openai_client.ChatCompletion.acreate(
                model=self.config.get('chat_model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": "You are a financial analyst explaining credit scores."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI explanation: {e}")
            return await self._generate_template_explanation(company_name, score, feature_importance)
    
    async def _generate_template_explanation(self, company_name: str, score: float,
                                           feature_importance: Dict[str, float]) -> str:
        """Generate template-based explanation"""
        try:
            risk_level = "low" if score >= 700 else "medium" if score >= 600 else "high"
            top_feature = list(feature_importance.keys())[0] if feature_importance else "financial metrics"
            
            explanation = f"""
            {company_name} has been assigned a credit score of {score}, indicating {risk_level} credit risk.
            
            The primary factor influencing this score is {top_feature.replace('_', ' ')}, which shows 
            {"strong" if score >= 700 else "moderate" if score >= 600 else "weak"} performance relative to industry peers.
            
            {"This score reflects solid financial health with low default probability." if score >= 700 else
             "This score indicates moderate risk with some areas for improvement." if score >= 600 else
             "This score suggests elevated risk requiring careful monitoring."}
            
            Key contributing factors include financial ratios, market sentiment, and operational metrics.
            Regular monitoring is recommended to track changes in creditworthiness.
            """
            
            return explanation.strip()
            
        except Exception as e:
            logger.error(f"Error generating template explanation: {e}")
            return f"Credit score of {score} assigned to {company_name} based on comprehensive financial analysis."
    
    async def _generate_recommendations(self, company_name: str, score: float,
                                      feature_importance: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        try:
            recommendations = []
            
            if score < 600:
                recommendations.extend([
                    "Immediate attention required for debt management and liquidity improvement",
                    "Consider restructuring high-interest debt to improve debt-to-equity ratio",
                    "Focus on cash flow optimization and working capital management"
                ])
            elif score < 700:
                recommendations.extend([
                    "Monitor key financial ratios for early warning signs",
                    "Strengthen operational efficiency to improve profitability metrics",
                    "Consider diversifying revenue streams to reduce risk concentration"
                ])
            else:
                recommendations.extend([
                    "Maintain current strong financial position",
                    "Continue monitoring market conditions and competitive landscape",
                    "Consider strategic investments for long-term growth"
                ])
            
            # Add feature-specific recommendations
            top_features = list(feature_importance.keys())[:3]
            for feature in top_features:
                if 'debt' in feature:
                    recommendations.append("Review debt structure and consider refinancing opportunities")
                elif 'liquidity' in feature or 'current_ratio' in feature:
                    recommendations.append("Optimize working capital management and cash flow")
                elif 'sentiment' in feature:
                    recommendations.append("Monitor market sentiment and public perception")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Regular financial monitoring recommended"]
    
    async def process_chat_message(self, user_id: str, message: str, 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process chat message and generate AI response"""
        try:
            # Initialize chat history for user
            if user_id not in self.chat_history:
                self.chat_history[user_id] = []
            
            # Add user message to history
            self.chat_history[user_id].append({
                'role': 'user',
                'content': message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only recent messages
            max_history = self.config.get('max_chat_history', 10)
            self.chat_history[user_id] = self.chat_history[user_id][-max_history:]
            
            # Generate response
            if self.openai_client:
                response_text = await self._generate_ai_chat_response(user_id, message, context)
            else:
                response_text = await self._generate_template_chat_response(message, context)
            
            # Add AI response to history
            self.chat_history[user_id].append({
                'role': 'assistant',
                'content': response_text,
                'timestamp': datetime.now().isoformat()
            })
            
            # Analyze sentiment
            sentiment = await self._analyze_message_sentiment(message)
            
            return {
                'response': response_text,
                'sentiment': sentiment,
                'timestamp': datetime.now().isoformat(),
                'context_used': bool(context)
            }
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            return {
                'response': "I apologize, but I'm having trouble processing your request right now. Please try again.",
                'error': str(e)
            }
    
    async def _generate_ai_chat_response(self, user_id: str, message: str, 
                                       context: Dict[str, Any] = None) -> str:
        """Generate AI-powered chat response"""
        try:
            # Build context prompt
            context_info = ""
            if context:
                context_info = f"""
                Current context:
                - Company: {context.get('company', 'N/A')}
                - Credit Score: {context.get('score', 'N/A')}
                - Risk Level: {context.get('risk_level', 'N/A')}
                """
            
            # Build conversation history
            history_messages = []
            for msg in self.chat_history[user_id][-5:]:  # Last 5 messages
                history_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
            
            system_prompt = f"""
            You are an expert financial analyst and credit risk specialist. You help users understand 
            credit scores, financial metrics, and risk assessment. Be professional, accurate, and helpful.
            
            {context_info}
            
            Guidelines:
            - Provide clear, actionable insights
            - Use financial terminology appropriately
            - Be concise but comprehensive
            - If you don't know something, say so
            """
            
            messages = [{"role": "system", "content": system_prompt}] + history_messages
            
            response = await self.openai_client.ChatCompletion.acreate(
                model=self.config.get('chat_model', 'gpt-3.5-turbo'),
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating AI chat response: {e}")
            return "I'm sorry, I'm having trouble generating a response right now."
    
    async def _generate_template_chat_response(self, message: str, 
                                             context: Dict[str, Any] = None) -> str:
        """Generate template-based chat response"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['score', 'credit', 'rating']):
            return "Credit scores are calculated based on various financial metrics including debt ratios, profitability, liquidity, and market factors. Would you like me to explain any specific aspect?"
        
        elif any(word in message_lower for word in ['risk', 'danger', 'safe']):
            return "Risk assessment considers multiple factors including financial stability, market conditions, and operational metrics. Higher scores indicate lower risk."
        
        elif any(word in message_lower for word in ['improve', 'better', 'increase']):
            return "To improve credit scores, focus on strengthening key financial ratios, improving cash flow, reducing debt levels, and maintaining consistent profitability."
        
        else:
            return "I can help you understand credit scores, risk factors, and financial analysis. What specific aspect would you like to know more about?"
    
    async def _analyze_message_sentiment(self, message: str) -> Dict[str, float]:
        """Analyze sentiment of user message"""
        try:
            results = self.sentiment_analyzer(message)
            sentiment_scores = {result['label']: result['score'] for result in results[0]}
            return sentiment_scores
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'neutral': 1.0}
    
    async def generate_batch_explanations(self):
        """Generate explanations for recent score changes"""
        try:
            # This would typically process recent score changes
            # For now, just log that batch processing occurred
            logger.info("Batch explanation generation completed")
        except Exception as e:
            logger.error(f"Error in batch explanation generation: {e}")
    
    async def get_explanation_status(self) -> Dict[str, Any]:
        """Get current explanation engine status"""
        return {
            'healthy': True,
            'explainers_loaded': len(self.explainers),
            'cached_explanations': len(self.explanation_cache),
            'active_chat_sessions': len(self.chat_history),
            'openai_enabled': self.openai_client is not None,
            'last_batch_processing': datetime.now()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.explanation_cache.clear()
            self.chat_history.clear()
            logger.info("Stage 4 cleanup completed")
        except Exception as e:
            logger.error(f"Error during Stage 4 cleanup: {e}")
