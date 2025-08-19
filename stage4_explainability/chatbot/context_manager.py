"""
Context manager for Stage 4 explainability chatbot.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ContextManager:
    """Manage conversation context and user session state"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.user_contexts = {}
        self.session_contexts = {}
        self.conversation_history = defaultdict(deque)
        self.context_timeout = config.get('context_timeout_minutes', 30)
        self.max_context_items = config.get('max_context_items', 50)
        
    async def initialize_session(self, user_id: str, session_data: Dict[str, Any]):
        """Initialize context for new session"""
        
        try:
            session_id = session_data['session_id']
            
            # Initialize user context if not exists
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = {
                    'user_id': user_id,
                    'first_interaction': datetime.now(),
                    'last_interaction': datetime.now(),
                    'total_sessions': 0,
                    'preferences': {},
                    'credit_profile': {},
                    'interaction_patterns': {},
                    'topics_discussed': set(),
                    'frequently_asked': defaultdict(int)
                }
            
            # Update user context
            user_context = self.user_contexts[user_id]
            user_context['last_interaction'] = datetime.now()
            user_context['total_sessions'] += 1
            
            # Initialize session context
            self.session_contexts[session_id] = {
                'session_id': session_id,
                'user_id': user_id,
                'start_time': datetime.now(),
                'current_topic': None,
                'conversation_flow': [],
                'entities_mentioned': {},
                'intents_history': [],
                'context_variables': {},
                'explanation_requests': [],
                'user_satisfaction_indicators': [],
                'active': True
            }
            
            # Merge initial context
            if 'context' in session_data:
                self.session_contexts[session_id]['context_variables'].update(session_data['context'])
            
            logger.info(f"Initialized context for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error initializing session context: {e}")
    
    async def update_context(self, user_id: str, message, session_context: Dict[str, Any] = None):
        """Update context based on new message"""
        
        try:
            # Get or create user context
            if user_id not in self.user_contexts:
                await self.initialize_session(user_id, {'session_id': f"{user_id}_default"})
            
            user_context = self.user_contexts[user_id]
            user_context['last_interaction'] = datetime.now()
            
            # Update conversation history
            self.conversation_history[user_id].append({
                'timestamp': message.timestamp,
                'message': message.message,
                'intent': message.intent,
                'entities': message.entities,
                'confidence': message.confidence
            })
            
            # Limit history size
            if len(self.conversation_history[user_id]) > self.max_context_items:
                self.conversation_history[user_id].popleft()
            
            # Update topics discussed
            if message.intent:
                user_context['topics_discussed'].add(message.intent)
                user_context['frequently_asked'][message.intent] += 1
            
            # Update entities mentioned
            if message.entities:
                for entity_type, entities in message.entities.items():
                    if entity_type not in user_context.get('entities_mentioned', {}):
                        user_context['entities_mentioned'] = user_context.get('entities_mentioned', {})
                        user_context['entities_mentioned'][entity_type] = []
                    
                    if isinstance(entities, list):
                        for entity in entities:
                            if hasattr(entity, 'text'):
                                user_context['entities_mentioned'][entity_type].append(entity.text)
                    else:
                        user_context['entities_mentioned'][entity_type].append(str(entities))
            
            # Update session context if available
            session_id = session_context.get('session_id') if session_context else None
            if session_id and session_id in self.session_contexts:
                session_ctx = self.session_contexts[session_id]
                session_ctx['intents_history'].append({
                    'intent': message.intent,
                    'confidence': message.confidence,
                    'timestamp': message.timestamp
                })
                
                # Update current topic
                if message.intent and message.confidence > 0.7:
                    session_ctx['current_topic'] = message.intent
                
                # Track conversation flow
                session_ctx['conversation_flow'].append({
                    'step': len(session_ctx['conversation_flow']) + 1,
                    'intent': message.intent,
                    'entities': list(message.entities.keys()) if message.entities else [],
                    'timestamp': message.timestamp
                })
            
            # Extract and update credit profile information
            await self._update_credit_profile(user_id, message)
            
            # Update interaction patterns
            await self._update_interaction_patterns(user_id, message)
            
            # Build comprehensive context
            context = await self._build_context(user_id, session_id)
            
            return context
            
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return {}
    
    async def _update_credit_profile(self, user_id: str, message):
        """Update user's credit profile from message entities"""
        
        try:
            user_context = self.user_contexts[user_id]
            
            if 'credit_profile' not in user_context:
                user_context['credit_profile'] = {}
            
            credit_profile = user_context['credit_profile']
            
            if message.entities:
                # Extract credit score
                if 'CREDIT_SCORE' in message.entities:
                    scores = message.entities['CREDIT_SCORE']
                    if scores and hasattr(scores[0], 'text'):
                        try:
                            score = int(scores[0].text)
                            credit_profile['credit_score'] = score
                            credit_profile['score_range'] = self._get_score_range(score)
                            credit_profile['last_updated'] = datetime.now()
                        except ValueError:
                            pass
                
                # Extract financial amounts
                for amount_type in ['LOAN_AMOUNT', 'INCOME', 'DEBT_AMOUNT']:
                    if amount_type in message.entities:
                        amounts = message.entities[amount_type]
                        if amounts and hasattr(amounts[0], 'text'):
                            try:
                                amount_str = amounts[0].text.replace(',', '').replace('$', '')
                                amount = float(amount_str)
                                credit_profile[amount_type.lower()] = amount
                            except ValueError:
                                pass
                
                # Extract utilization
                if 'CREDIT_UTILIZATION' in message.entities:
                    utilization = message.entities['CREDIT_UTILIZATION']
                    if utilization and hasattr(utilization[0], 'text'):
                        try:
                            util_str = utilization[0].text.replace('%', '')
                            util_value = float(util_str)
                            credit_profile['credit_utilization'] = util_value
                        except ValueError:
                            pass
                
                # Extract account information
                if 'ACCOUNT_TYPE' in message.entities:
                    account_types = message.entities['ACCOUNT_TYPE']
                    if 'account_types' not in credit_profile:
                        credit_profile['account_types'] = []
                    
                    for account in account_types:
                        if hasattr(account, 'text'):
                            if account.text not in credit_profile['account_types']:
                                credit_profile['account_types'].append(account.text)
            
        except Exception as e:
            logger.error(f"Error updating credit profile: {e}")
    
    def _get_score_range(self, score: int) -> str:
        """Get credit score range"""
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
    
    async def _update_interaction_patterns(self, user_id: str, message):
        """Update user interaction patterns"""
        
        try:
            user_context = self.user_contexts[user_id]
            
            if 'interaction_patterns' not in user_context:
                user_context['interaction_patterns'] = {
                    'preferred_explanation_style': 'balanced',  # simple, detailed, balanced
                    'common_question_types': defaultdict(int),
                    'response_preferences': {},
                    'engagement_level': 'medium',
                    'technical_comfort': 'medium'
                }
            
            patterns = user_context['interaction_patterns']
            
            # Update question type preferences
            if message.intent:
                patterns['common_question_types'][message.intent] += 1
            
            # Infer technical comfort from language used
            message_text = message.message.lower()
            technical_terms = ['algorithm', 'model', 'feature', 'correlation', 'regression', 'analysis']
            
            if any(term in message_text for term in technical_terms):
                if patterns['technical_comfort'] == 'low':
                    patterns['technical_comfort'] = 'medium'
                elif patterns['technical_comfort'] == 'medium':
                    patterns['technical_comfort'] = 'high'
            
            # Infer explanation style preference
            if any(word in message_text for word in ['detail', 'explain more', 'elaborate']):
                patterns['preferred_explanation_style'] = 'detailed'
            elif any(word in message_text for word in ['simple', 'basic', 'quick']):
                patterns['preferred_explanation_style'] = 'simple'
            
        except Exception as e:
            logger.error(f"Error updating interaction patterns: {e}")
    
    async def _build_context(self, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """Build comprehensive context for response generation"""
        
        try:
            user_context = self.user_contexts.get(user_id, {})
            session_context = self.session_contexts.get(session_id, {}) if session_id else {}
            
            # Recent conversation history
            recent_history = list(self.conversation_history[user_id])[-5:]  # Last 5 messages
            
            # Build context
            context = {
                # User information
                'user_id': user_id,
                'user_name': user_context.get('user_name'),
                'returning_user': user_context.get('total_sessions', 0) > 1,
                'total_sessions': user_context.get('total_sessions', 0),
                
                # Credit profile
                'credit_score': user_context.get('credit_profile', {}).get('credit_score'),
                'score_range': user_context.get('credit_profile', {}).get('score_range'),
                'credit_utilization': user_context.get('credit_profile', {}).get('credit_utilization'),
                'loan_amount': user_context.get('credit_profile', {}).get('loan_amount'),
                'income': user_context.get('credit_profile', {}).get('income'),
                'debt_amount': user_context.get('credit_profile', {}).get('debt_amount'),
                'account_types': user_context.get('credit_profile', {}).get('account_types', []),
                
                # Conversation context
                'current_topic': session_context.get('current_topic'),
                'recent_intents': [msg['intent'] for msg in recent_history if msg['intent']],
                'recent_topics': list(user_context.get('topics_discussed', set()))[-5:],
                'conversation_depth': len(recent_history),
                'frequently_asked': dict(user_context.get('frequently_asked', {})),
                
                # Interaction preferences
                'preferred_explanation_style': user_context.get('interaction_patterns', {}).get('preferred_explanation_style', 'balanced'),
                'technical_comfort': user_context.get('interaction_patterns', {}).get('technical_comfort', 'medium'),
                'engagement_level': user_context.get('interaction_patterns', {}).get('engagement_level', 'medium'),
                
                # Session information
                'session_id': session_id,
                'session_start': session_context.get('start_time'),
                'conversation_flow': session_context.get('conversation_flow', []),
                
                # Additional context variables
                **session_context.get('context_variables', {})
            }
            
            # Add derived context
            context['is_new_user'] = user_context.get('total_sessions', 0) <= 1
            context['has_credit_data'] = bool(context['credit_score'])
            context['primary_interest'] = self._get_primary_interest(user_context)
            context['context_timestamp'] = datetime.now()
            
            return context
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            return {'user_id': user_id, 'error': 'context_build_failed'}
    
    def _get_primary_interest(self, user_context: Dict[str, Any]) -> str:
        """Determine user's primary interest based on interaction history"""
        
        try:
            frequently_asked = user_context.get('frequently_asked', {})
            
            if not frequently_asked:
                return 'general'
            
            # Get most frequent intent
            primary_intent = max(frequently_asked.items(), key=lambda x: x[1])[0]
            
            # Map intents to interests
            intent_mapping = {
                'credit_score_inquiry': 'credit_score',
                'loan_decision_explanation': 'loan_decisions',
                'feature_importance': 'understanding_factors',
                'improvement_suggestions': 'credit_improvement',
                'what_if_scenario': 'scenario_analysis',
                'counterfactual_analysis': 'advanced_analysis'
            }
            
            return intent_mapping.get(primary_intent, 'general')
            
        except Exception:
            return 'general'
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context"""
        
        return self.user_contexts.get(user_id, {})
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Update user preferences"""
        
        try:
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = {}
            
            if 'preferences' not in self.user_contexts[user_id]:
                self.user_contexts[user_id]['preferences'] = {}
            
            self.user_contexts[user_id]['preferences'].update(preferences)
            logger.info(f"Updated preferences for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    async def add_context_variable(self, session_id: str, key: str, value: Any):
        """Add context variable to session"""
        
        try:
            if session_id in self.session_contexts:
                self.session_contexts[session_id]['context_variables'][key] = value
                logger.info(f"Added context variable {key} to session {session_id}")
            
        except Exception as e:
            logger.error(f"Error adding context variable: {e}")
    
    async def get_conversation_summary(self, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """Get conversation summary"""
        
        try:
            user_context = self.user_contexts.get(user_id, {})
            session_context = self.session_contexts.get(session_id, {}) if session_id else {}
            
            # Get conversation history
            history = list(self.conversation_history[user_id])
            
            # Filter by session if specified
            if session_id and session_context:
                session_start = session_context.get('start_time')
                if session_start:
                    history = [
                        msg for msg in history 
                        if msg['timestamp'] >= session_start
                    ]
            
            # Analyze conversation
            intents = [msg['intent'] for msg in history if msg['intent']]
            entities = {}
            
            for msg in history:
                if msg['entities']:
                    for entity_type, entity_list in msg['entities'].items():
                        if entity_type not in entities:
                            entities[entity_type] = []
                        entities[entity_type].extend(entity_list)
            
            summary = {
                'user_id': user_id,
                'session_id': session_id,
                'total_messages': len(history),
                'unique_intents': list(set(intents)),
                'most_common_intent': max(set(intents), key=intents.count) if intents else None,
                'entities_mentioned': {k: len(v) for k, v in entities.items()},
                'conversation_duration': None,
                'topics_covered': list(user_context.get('topics_discussed', set())),
                'user_engagement': self._calculate_engagement_score(history),
                'summary_timestamp': datetime.now()
            }
            
            # Calculate duration
            if history:
                start_time = history[0]['timestamp']
                end_time = history[-1]['timestamp']
                summary['conversation_duration'] = (end_time - start_time).total_seconds()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting conversation summary: {e}")
            return {'error': str(e)}
    
    def _calculate_engagement_score(self, history: List[Dict[str, Any]]) -> float:
        """Calculate user engagement score"""
        
        try:
            if not history:
                return 0.0
            
            # Factors for engagement
            total_messages = len(history)
            avg_confidence = np.mean([msg['confidence'] for msg in history if msg['confidence']])
            unique_intents = len(set(msg['intent'] for msg in history if msg['intent']))
            
            # Calculate engagement score (0-1)
            message_score = min(1.0, total_messages / 10)  # Up to 10 messages = full score
            confidence_score = avg_confidence if not np.isnan(avg_confidence) else 0.5
            diversity_score = min(1.0, unique_intents / 5)  # Up to 5 different intents = full score
            
            engagement_score = (message_score + confidence_score + diversity_score) / 3
            
            return round(engagement_score, 2)
            
        except Exception:
            return 0.5
    
    async def cleanup_session(self, user_id: str, session_id: str = None):
        """Clean up session context"""
        
        try:
            if session_id and session_id in self.session_contexts:
                self.session_contexts[session_id]['active'] = False
                # Keep session data for analysis but mark as inactive
                logger.info(f"Cleaned up session {session_id}")
            
            # Clean up old contexts
            await self._cleanup_old_contexts()
            
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")
    
    async def _cleanup_old_contexts(self):
        """Clean up old inactive contexts"""
        
        try:
            current_time = datetime.now()
            timeout_delta = timedelta(minutes=self.context_timeout)
            
            # Clean up old session contexts
            expired_sessions = []
            for session_id, session_data in self.session_contexts.items():
                if not session_data.get('active', True):
                    continue
                
                last_activity = session_data.get('start_time', current_time)
                if current_time - last_activity > timeout_delta:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                self.session_contexts[session_id]['active'] = False
                logger.info(f"Expired session {session_id}")
            
            # Clean up old conversation history
            for user_id in list(self.conversation_history.keys()):
                history = self.conversation_history[user_id]
                
                # Remove messages older than timeout
                while history and (current_time - history[0]['timestamp']) > timeout_delta:
                    history.popleft()
                
                # Remove empty histories
                if not history:
                    del self.conversation_history[user_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up old contexts: {e}")
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context management statistics"""
        
        try:
            active_sessions = sum(1 for s in self.session_contexts.values() if s.get('active', True))
            total_users = len(self.user_contexts)
            total_conversations = sum(len(history) for history in self.conversation_history.values())
            
            return {
                'total_users': total_users,
                'active_sessions': active_sessions,
                'total_sessions': len(self.session_contexts),
                'total_conversations': total_conversations,
                'average_conversations_per_user': total_conversations / total_users if total_users > 0 else 0,
                'context_timeout_minutes': self.context_timeout,
                'max_context_items': self.max_context_items,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting context statistics: {e}")
            return {'error': str(e)}
