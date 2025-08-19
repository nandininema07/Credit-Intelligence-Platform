"""
Chat and AI assistant-related Pydantic models
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field

from .common import BaseModel, TimestampMixin

class ChatMessageBase(BaseModel):
    """Base chat message model"""
    content: str = Field(..., min_length=1, max_length=4000)
    message_type: str = Field(default="user")  # "user", "assistant", "system"

class ChatRequest(ChatMessageBase):
    """Chat request from user"""
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    company_id: Optional[int] = None
    
class ChatMessage(ChatMessageBase, TimestampMixin):
    """Complete chat message model"""
    id: int
    session_id: str
    user_id: Optional[str] = None
    
    # AI-specific fields
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    response_time_ms: Optional[int] = None
    
    # Context and metadata
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    message_id: int
    session_id: str
    timestamp: datetime
    
    # AI insights
    confidence: Optional[float] = Field(None, ge=0, le=1)
    sources: List[str] = Field(default_factory=list)
    related_companies: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    
    # Follow-up questions
    follow_up_questions: List[str] = Field(default_factory=list)

class ChatSession(BaseModel, TimestampMixin):
    """Chat session model"""
    id: str
    user_id: Optional[str] = None
    title: Optional[str] = None
    is_active: bool = True
    message_count: int = 0
    last_activity: datetime
    
    # Session context
    context: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None

class ChatAnalytics(BaseModel):
    """Chat analytics and insights"""
    session_id: str
    total_messages: int
    user_messages: int
    assistant_messages: int
    avg_response_time_ms: float
    topics_discussed: List[str]
    companies_mentioned: List[str]
    sentiment_score: Optional[float] = None
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=5)

class QuestionSuggestion(BaseModel):
    """Suggested questions for users"""
    question: str
    category: str  # "company_analysis", "score_explanation", "comparison", "general"
    relevance_score: float = Field(..., ge=0, le=1)
    context_required: bool = False

class ChatExplanation(BaseModel):
    """Detailed explanation response"""
    explanation_type: str  # "score_change", "risk_factors", "comparison", "prediction"
    company_id: Optional[int] = None
    explanation: str
    key_points: List[str]
    visualizations: Optional[Dict[str, Any]] = None
    data_sources: List[str] = Field(default_factory=list)
    confidence_level: str  # "high", "medium", "low"
