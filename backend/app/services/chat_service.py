"""
Chat and AI assistant service integrating Stage 4 explainability
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import uuid

from app.models.chat import (
    ChatMessage, ChatRequest, ChatResponse, ChatSession,
    ChatAnalytics, QuestionSuggestion, ChatExplanation
)

# Import Stage 4 components
from stage4_explainability.chatbot.chat_engine import ChatEngine
from stage4_explainability.explainer.explanation_generator import ExplanationGenerator
from stage4_explainability.natural_language.language_models import LanguageModel

logger = logging.getLogger(__name__)

class ChatService:
    """Service for chat and AI assistant functionality"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.chat_engine = ChatEngine()
        self.explainer = ExplanationGenerator()
        self.language_model = LanguageModel()
    
    async def process_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        company_id: Optional[int] = None
    ) -> ChatResponse:
        """Process a chat message and generate AI response"""
        try:
            # Create or get session
            if not session_id:
                session_id = str(uuid.uuid4())
                await self.create_session(user_id=user_id, session_id=session_id)
            
            # Save user message
            user_message = ChatMessage(
                content=message,
                message_type="user",
                session_id=session_id,
                user_id=user_id,
                context=context,
                created_at=datetime.utcnow()
            )
            
            self.db.add(user_message)
            await self.db.commit()
            await self.db.refresh(user_message)
            
            # Generate AI response using Stage 4 chat engine
            ai_response_data = await self.chat_engine.process_query(
                query=message,
                session_id=session_id,
                context=context,
                company_id=company_id
            )
            
            # Save AI message
            ai_message = ChatMessage(
                content=ai_response_data['response'],
                message_type="assistant",
                session_id=session_id,
                user_id=user_id,
                model_used=ai_response_data.get('model_used', 'gpt-3.5-turbo'),
                tokens_used=ai_response_data.get('tokens_used'),
                response_time_ms=ai_response_data.get('response_time_ms'),
                context=ai_response_data.get('context'),
                metadata=ai_response_data.get('metadata'),
                created_at=datetime.utcnow()
            )
            
            self.db.add(ai_message)
            await self.db.commit()
            await self.db.refresh(ai_message)
            
            # Update session
            await self._update_session_activity(session_id)
            
            # Create response
            return ChatResponse(
                message=ai_response_data['response'],
                message_id=ai_message.id,
                session_id=session_id,
                timestamp=ai_message.created_at,
                confidence=ai_response_data.get('confidence'),
                sources=ai_response_data.get('sources', []),
                related_companies=ai_response_data.get('related_companies', []),
                suggested_actions=ai_response_data.get('suggested_actions', []),
                follow_up_questions=ai_response_data.get('follow_up_questions', [])
            )
            
        except Exception as e:
            logger.error(f"Error processing chat message: {str(e)}")
            raise
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 20
    ) -> List[ChatSession]:
        """Get user's chat sessions"""
        try:
            # Mock implementation - would query chat_sessions table
            return [
                ChatSession(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    title="Credit Risk Analysis",
                    is_active=True,
                    message_count=15,
                    last_activity=datetime.utcnow(),
                    created_at=datetime.utcnow(),
                    context={"topic": "credit_analysis"},
                    preferences={"language": "en", "detail_level": "medium"}
                )
            ]
            
        except Exception as e:
            logger.error(f"Error fetching user sessions: {str(e)}")
            raise
    
    async def create_session(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            session = ChatSession(
                id=session_id,
                user_id=user_id,
                title=title or "New Chat",
                is_active=True,
                message_count=0,
                last_activity=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            
            # Mock implementation - would save to database
            logger.info(f"Created chat session {session_id} for user {user_id}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error creating chat session: {str(e)}")
            raise
    
    async def get_session_messages(
        self,
        session_id: str,
        user_id: str,
        limit: int = 50
    ) -> List[ChatMessage]:
        """Get messages from a chat session"""
        try:
            query = select(ChatMessage).where(
                and_(
                    ChatMessage.session_id == session_id,
                    ChatMessage.user_id == user_id
                )
            ).order_by(ChatMessage.created_at).limit(limit)
            
            result = await self.db.execute(query)
            messages = result.scalars().all()
            
            return list(messages)
            
        except Exception as e:
            logger.error(f"Error fetching session messages: {str(e)}")
            raise
    
    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete a chat session"""
        try:
            # Mock implementation - would delete from database
            logger.info(f"Deleted chat session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting chat session: {str(e)}")
            raise
    
    async def get_question_suggestions(
        self,
        category: Optional[str] = None,
        company_id: Optional[int] = None,
        limit: int = 5
    ) -> List[QuestionSuggestion]:
        """Get suggested questions for the user"""
        try:
            suggestions = [
                QuestionSuggestion(
                    question="Why did Apple's credit score change today?",
                    category="score_explanation",
                    relevance_score=0.95,
                    context_required=True
                ),
                QuestionSuggestion(
                    question="Compare Tesla and Ford's credit risk profiles",
                    category="comparison",
                    relevance_score=0.88,
                    context_required=False
                ),
                QuestionSuggestion(
                    question="What are the key risk factors for Meta?",
                    category="risk_analysis",
                    relevance_score=0.82,
                    context_required=True
                ),
                QuestionSuggestion(
                    question="Show me companies with improving credit scores",
                    category="market_analysis",
                    relevance_score=0.79,
                    context_required=False
                ),
                QuestionSuggestion(
                    question="Explain the SHAP analysis for Microsoft",
                    category="explainability",
                    relevance_score=0.75,
                    context_required=True
                )
            ]
            
            # Filter by category if specified
            if category:
                suggestions = [s for s in suggestions if s.category == category]
            
            # Sort by relevance and limit
            suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching question suggestions: {str(e)}")
            raise
    
    async def generate_explanation(
        self,
        explanation_type: str,
        company_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ChatExplanation:
        """Generate detailed explanation about specific topics"""
        try:
            # Use Stage 4 explainer to generate explanation
            explanation_result = await self.explainer.generate_explanation(
                explanation_type=explanation_type,
                company_id=company_id,
                context=context or {}
            )
            
            return ChatExplanation(
                explanation_type=explanation_type,
                company_id=company_id,
                explanation=explanation_result['explanation'],
                key_points=explanation_result['key_points'],
                visualizations=explanation_result.get('visualizations'),
                data_sources=explanation_result.get('data_sources', []),
                confidence_level=explanation_result.get('confidence_level', 'medium')
            )
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    async def get_session_analytics(
        self,
        session_id: str,
        user_id: str
    ) -> ChatAnalytics:
        """Get analytics for a chat session"""
        try:
            # Mock implementation - would calculate from actual messages
            return ChatAnalytics(
                session_id=session_id,
                total_messages=24,
                user_messages=12,
                assistant_messages=12,
                avg_response_time_ms=1250.5,
                topics_discussed=["credit_scores", "risk_analysis", "company_comparison"],
                companies_mentioned=["Apple Inc.", "Tesla Inc.", "Microsoft Corp."],
                sentiment_score=0.75,
                satisfaction_rating=4
            )
            
        except Exception as e:
            logger.error(f"Error fetching session analytics: {str(e)}")
            raise
    
    async def submit_feedback(
        self,
        message_id: int,
        user_id: str,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """Submit feedback for a chat response"""
        try:
            # Mock implementation - would save feedback to database
            logger.info(f"Received feedback for message {message_id}: rating={rating}")
            
            if feedback_text:
                logger.info(f"Feedback text: {feedback_text}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            raise
    
    async def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        try:
            # Mock implementation - would update session in database
            logger.debug(f"Updated activity for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error updating session activity: {str(e)}")
            raise
