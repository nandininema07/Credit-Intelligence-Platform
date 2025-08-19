"""
Chat and AI assistant API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from app.database import get_db
from app.models.chat import (
    ChatMessage, ChatRequest, ChatResponse, ChatSession,
    ChatAnalytics, QuestionSuggestion, ChatExplanation
)
from app.services.chat_service import ChatService
from app.utils.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Send a message to the AI assistant"""
    try:
        chat_service = ChatService(db)
        response = await chat_service.process_message(
            message=request.content,
            session_id=request.session_id,
            user_id=current_user.get("id"),
            context=request.context,
            company_id=request.company_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions", response_model=List[ChatSession])
async def get_chat_sessions(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get user's chat sessions"""
    try:
        chat_service = ChatService(db)
        sessions = await chat_service.get_user_sessions(
            user_id=current_user.get("id"),
            limit=limit
        )
        return sessions
        
    except Exception as e:
        logger.error(f"Error fetching chat sessions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/sessions", response_model=ChatSession)
async def create_chat_session(
    title: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new chat session"""
    try:
        chat_service = ChatService(db)
        session = await chat_service.create_session(
            user_id=current_user.get("id"),
            title=title
        )
        return session
        
    except Exception as e:
        logger.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str = Path(...),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get messages from a chat session"""
    try:
        chat_service = ChatService(db)
        messages = await chat_service.get_session_messages(
            session_id=session_id,
            user_id=current_user.get("id"),
            limit=limit
        )
        return messages
        
    except Exception as e:
        logger.error(f"Error fetching session messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a chat session"""
    try:
        chat_service = ChatService(db)
        success = await chat_service.delete_session(
            session_id=session_id,
            user_id=current_user.get("id")
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
            
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/suggestions", response_model=List[QuestionSuggestion])
async def get_question_suggestions(
    category: Optional[str] = Query(None),
    company_id: Optional[int] = Query(None),
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get suggested questions for the user"""
    try:
        chat_service = ChatService(db)
        suggestions = await chat_service.get_question_suggestions(
            category=category,
            company_id=company_id,
            limit=limit
        )
        return suggestions
        
    except Exception as e:
        logger.error(f"Error fetching question suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/explain", response_model=ChatExplanation)
async def get_explanation(
    explanation_type: str = Query(...),
    company_id: Optional[int] = Query(None),
    context: Optional[dict] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get detailed explanation about specific topics"""
    try:
        chat_service = ChatService(db)
        explanation = await chat_service.generate_explanation(
            explanation_type=explanation_type,
            company_id=company_id,
            context=context
        )
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/analytics/{session_id}", response_model=ChatAnalytics)
async def get_session_analytics(
    session_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get analytics for a chat session"""
    try:
        chat_service = ChatService(db)
        analytics = await chat_service.get_session_analytics(
            session_id=session_id,
            user_id=current_user.get("id")
        )
        return analytics
        
    except Exception as e:
        logger.error(f"Error fetching session analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/feedback")
async def submit_feedback(
    message_id: int = Query(...),
    rating: int = Query(..., ge=1, le=5),
    feedback_text: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for a chat response"""
    try:
        chat_service = ChatService(db)
        success = await chat_service.submit_feedback(
            message_id=message_id,
            user_id=current_user.get("id"),
            rating=rating,
            feedback_text=feedback_text
        )
        
        return {"message": "Feedback submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
