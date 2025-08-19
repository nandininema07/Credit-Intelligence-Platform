"""
Chat engine for Stage 4 explainability chatbot.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Chat message data structure"""
    user_id: str
    message: str
    timestamp: datetime
    message_type: str = "user"  # user, bot, system
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None

@dataclass
class ChatResponse:
    """Chat response data structure"""
    response: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    context: Dict[str, Any]
    suggestions: List[str]
    timestamp: datetime

class ChatEngine:
    """Main chat engine for credit explanation chatbot"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.intent_classifier = None
        self.entity_recognizer = None
        self.response_generator = None
        self.context_manager = None
        self.knowledge_base = None
        self.conversation_history = {}
        self.active_sessions = {}
        
    def initialize(self, intent_classifier, entity_recognizer, 
                  response_generator, context_manager, knowledge_base):
        """Initialize chat engine with components"""
        
        self.intent_classifier = intent_classifier
        self.entity_recognizer = entity_recognizer
        self.response_generator = response_generator
        self.context_manager = context_manager
        self.knowledge_base = knowledge_base
        
        logger.info("Chat engine initialized")
    
    async def process_message(self, user_id: str, message: str,
                            session_context: Dict[str, Any] = None) -> ChatResponse:
        """Process incoming chat message"""
        
        try:
            # Create chat message
            chat_message = ChatMessage(
                user_id=user_id,
                message=message,
                timestamp=datetime.now()
            )
            
            # Classify intent
            intent_result = await self.intent_classifier.classify_intent(message)
            chat_message.intent = intent_result['intent']
            chat_message.confidence = intent_result['confidence']
            
            # Extract entities
            entities = await self.entity_recognizer.extract_entities(message)
            chat_message.entities = entities
            
            # Update context
            context = await self.context_manager.update_context(
                user_id, chat_message, session_context
            )
            
            # Generate response
            response = await self.response_generator.generate_response(
                chat_message, context, self.knowledge_base
            )
            
            # Store conversation
            self._store_conversation(user_id, chat_message, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return ChatResponse(
                response="I apologize, but I encountered an error processing your request. Please try again.",
                intent="error",
                confidence=0.0,
                entities={},
                context={},
                suggestions=["Try rephrasing your question", "Ask about credit score factors"],
                timestamp=datetime.now()
            )
    
    async def start_session(self, user_id: str, initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Start a new chat session"""
        
        session_id = f"{user_id}_{datetime.now().timestamp()}"
        
        session_data = {
            'session_id': session_id,
            'user_id': user_id,
            'start_time': datetime.now(),
            'context': initial_context or {},
            'message_count': 0,
            'active': True
        }
        
        self.active_sessions[session_id] = session_data
        
        # Initialize context
        await self.context_manager.initialize_session(user_id, session_data)
        
        # Generate welcome message
        welcome_response = await self.response_generator.generate_welcome_message(
            user_id, initial_context
        )
        
        return {
            'session_id': session_id,
            'welcome_message': welcome_response,
            'status': 'active'
        }
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a chat session"""
        
        if session_id in self.active_sessions:
            session_data = self.active_sessions[session_id]
            session_data['active'] = False
            session_data['end_time'] = datetime.now()
            
            # Generate session summary
            summary = await self._generate_session_summary(session_data)
            
            # Clean up context
            await self.context_manager.cleanup_session(session_data['user_id'])
            
            return {
                'session_id': session_id,
                'status': 'ended',
                'summary': summary
            }
        
        return {'error': 'Session not found'}
    
    async def get_conversation_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history for user"""
        
        if user_id in self.conversation_history:
            history = self.conversation_history[user_id]
            return history[-limit:] if len(history) > limit else history
        
        return []
    
    async def handle_credit_inquiry(self, user_id: str, credit_data: Dict[str, Any]) -> ChatResponse:
        """Handle specific credit-related inquiry"""
        
        try:
            # Process credit data
            context = {
                'credit_data': credit_data,
                'inquiry_type': 'credit_analysis',
                'timestamp': datetime.now()
            }
            
            # Generate credit-specific response
            response = await self.response_generator.generate_credit_response(
                credit_data, context, self.knowledge_base
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error handling credit inquiry: {e}")
            return ChatResponse(
                response="I encountered an error analyzing your credit information. Please try again.",
                intent="error",
                confidence=0.0,
                entities={},
                context=context,
                suggestions=["Provide credit score", "Ask about specific factors"],
                timestamp=datetime.now()
            )
    
    async def get_explanation_suggestions(self, user_id: str, 
                                       explanation_type: str = None) -> List[str]:
        """Get explanation suggestions for user"""
        
        try:
            # Get user context
            context = await self.context_manager.get_user_context(user_id)
            
            # Generate suggestions based on context and explanation type
            suggestions = []
            
            if explanation_type == "credit_score":
                suggestions = [
                    "Why is my credit score this value?",
                    "What factors most impact my credit score?",
                    "How can I improve my credit score?",
                    "What would happen if I paid off my debts?",
                    "Show me feature importance for my score"
                ]
            elif explanation_type == "loan_decision":
                suggestions = [
                    "Why was my loan application approved/denied?",
                    "What factors influenced the decision?",
                    "What changes would improve my chances?",
                    "Show me a counterfactual analysis",
                    "Explain the decision process"
                ]
            else:
                # General suggestions
                suggestions = [
                    "Explain my credit analysis",
                    "What are the key factors?",
                    "How can I improve?",
                    "Show me what-if scenarios",
                    "Generate a detailed report"
                ]
            
            # Personalize based on context
            if context and 'recent_topics' in context:
                recent_topics = context['recent_topics']
                if 'credit_score' in recent_topics:
                    suggestions.append("Tell me more about credit score factors")
                if 'loan' in recent_topics:
                    suggestions.append("Explain loan decision criteria")
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting explanation suggestions: {e}")
            return ["Ask me about your credit analysis", "Request an explanation"]
    
    def _store_conversation(self, user_id: str, message: ChatMessage, response: ChatResponse):
        """Store conversation in history"""
        
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        conversation_entry = {
            'timestamp': message.timestamp.isoformat(),
            'user_message': message.message,
            'bot_response': response.response,
            'intent': message.intent,
            'confidence': message.confidence,
            'entities': message.entities,
            'context': response.context
        }
        
        self.conversation_history[user_id].append(conversation_entry)
        
        # Limit history size
        max_history = self.config.get('max_conversation_history', 1000)
        if len(self.conversation_history[user_id]) > max_history:
            self.conversation_history[user_id] = self.conversation_history[user_id][-max_history:]
    
    async def _generate_session_summary(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for ended session"""
        
        try:
            user_id = session_data['user_id']
            session_history = await self.get_conversation_history(user_id)
            
            # Filter messages from this session
            session_start = session_data['start_time']
            session_messages = [
                msg for msg in session_history 
                if datetime.fromisoformat(msg['timestamp']) >= session_start
            ]
            
            # Analyze session
            total_messages = len(session_messages)
            intents = [msg['intent'] for msg in session_messages if msg['intent']]
            most_common_intent = max(set(intents), key=intents.count) if intents else "unknown"
            
            # Extract topics discussed
            topics = set()
            for msg in session_messages:
                if msg['entities']:
                    for entity_type, entities in msg['entities'].items():
                        if isinstance(entities, list):
                            topics.update(entities)
                        else:
                            topics.add(str(entities))
            
            duration = (session_data.get('end_time', datetime.now()) - session_start).total_seconds()
            
            return {
                'session_duration_seconds': duration,
                'total_messages': total_messages,
                'primary_intent': most_common_intent,
                'topics_discussed': list(topics),
                'user_satisfaction': self._estimate_user_satisfaction(session_messages),
                'summary_generated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return {'error': 'Could not generate session summary'}
    
    def _estimate_user_satisfaction(self, session_messages: List[Dict[str, Any]]) -> str:
        """Estimate user satisfaction based on conversation patterns"""
        
        try:
            if not session_messages:
                return "unknown"
            
            # Simple heuristics for satisfaction
            total_messages = len(session_messages)
            
            # Check for positive indicators
            positive_indicators = 0
            negative_indicators = 0
            
            for msg in session_messages:
                message_text = msg.get('user_message', '').lower()
                
                # Positive indicators
                if any(word in message_text for word in ['thank', 'helpful', 'good', 'great', 'perfect']):
                    positive_indicators += 1
                
                # Negative indicators
                if any(word in message_text for word in ['confused', 'unclear', 'wrong', 'error', 'problem']):
                    negative_indicators += 1
            
            # Check conversation length (very short might indicate frustration)
            if total_messages < 3:
                return "low"
            elif positive_indicators > negative_indicators:
                return "high"
            elif negative_indicators > positive_indicators:
                return "low"
            else:
                return "medium"
                
        except Exception:
            return "unknown"
    
    async def get_chat_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """Get chat analytics"""
        
        try:
            if user_id:
                # User-specific analytics
                history = await self.get_conversation_history(user_id)
                
                return {
                    'user_id': user_id,
                    'total_conversations': len(history),
                    'intents': self._analyze_intents(history),
                    'topics': self._analyze_topics(history),
                    'engagement_score': self._calculate_engagement_score(history)
                }
            else:
                # Global analytics
                all_users = list(self.conversation_history.keys())
                total_conversations = sum(len(self.conversation_history[uid]) for uid in all_users)
                
                return {
                    'total_users': len(all_users),
                    'total_conversations': total_conversations,
                    'active_sessions': len([s for s in self.active_sessions.values() if s['active']]),
                    'average_conversations_per_user': total_conversations / len(all_users) if all_users else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting chat analytics: {e}")
            return {'error': str(e)}
    
    def _analyze_intents(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze intent distribution"""
        
        intent_counts = {}
        for msg in history:
            intent = msg.get('intent', 'unknown')
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        return intent_counts
    
    def _analyze_topics(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze topic distribution"""
        
        topic_counts = {}
        for msg in history:
            entities = msg.get('entities', {})
            for entity_type, entity_values in entities.items():
                if isinstance(entity_values, list):
                    for value in entity_values:
                        topic_counts[str(value)] = topic_counts.get(str(value), 0) + 1
                else:
                    topic_counts[str(entity_values)] = topic_counts.get(str(entity_values), 0) + 1
        
        return topic_counts
    
    def _calculate_engagement_score(self, history: List[Dict[str, Any]]) -> float:
        """Calculate user engagement score"""
        
        if not history:
            return 0.0
        
        # Simple engagement score based on conversation length and frequency
        total_messages = len(history)
        
        # Check for follow-up questions (engagement indicator)
        follow_ups = 0
        for i in range(1, len(history)):
            current_intent = history[i].get('intent', '')
            prev_intent = history[i-1].get('intent', '')
            
            if current_intent == prev_intent or 'follow_up' in current_intent:
                follow_ups += 1
        
        # Normalize to 0-1 scale
        engagement_score = min(1.0, (total_messages + follow_ups * 2) / 20)
        
        return round(engagement_score, 2)
