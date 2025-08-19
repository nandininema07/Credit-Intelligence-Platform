"""
Common Pydantic models and base classes
"""

from pydantic import BaseModel as PydanticBaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Generic, TypeVar
from datetime import datetime
from enum import Enum

T = TypeVar('T')

class BaseModel(PydanticBaseModel):
    """Base model with common configuration"""
    model_config = ConfigDict(
        from_attributes=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True
    )

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response"""
    items: List[T]
    total: int
    page: int = Field(ge=1)
    size: int = Field(ge=1, le=100)
    pages: int
    has_next: bool
    has_prev: bool

class TimeSeriesData(BaseModel):
    """Time series data point"""
    timestamp: datetime
    value: float
    metadata: Optional[Dict[str, Any]] = None

class ScoreRange(str, Enum):
    """Credit score ranges"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 80-89
    FAIR = "fair"           # 70-79
    POOR = "poor"           # 60-69
    VERY_POOR = "very_poor" # 0-59

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    MUTED = "muted"

class TrendDirection(str, Enum):
    """Trend direction"""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class DataSource(str, Enum):
    """Data sources"""
    FINANCIAL_STATEMENTS = "financial_statements"
    MARKET_DATA = "market_data"
    NEWS_SENTIMENT = "news_sentiment"
    REGULATORY_FILINGS = "regulatory_filings"
    SOCIAL_MEDIA = "social_media"
    PEER_COMPARISON = "peer_comparison"

class IndustryCategory(str, Enum):
    """Industry categories"""
    TECHNOLOGY = "technology"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    CONSUMER_GOODS = "consumer_goods"
    ENERGY = "energy"
    MANUFACTURING = "manufacturing"
    TELECOMMUNICATIONS = "telecommunications"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    MATERIALS = "materials"
