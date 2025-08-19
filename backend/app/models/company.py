"""
Company-related Pydantic models
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field, validator

from .common import BaseModel, TimestampMixin, IndustryCategory, ScoreRange

class CompanyBase(BaseModel):
    """Base company model"""
    name: str = Field(..., min_length=1, max_length=200)
    ticker: Optional[str] = Field(None, max_length=10)
    industry: IndustryCategory
    sector: Optional[str] = Field(None, max_length=100)
    market_cap: Optional[float] = Field(None, ge=0)
    country: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = Field(None, max_length=1000)
    website: Optional[str] = None
    
    @validator('ticker')
    def ticker_uppercase(cls, v):
        return v.upper() if v else v

class CompanyCreate(CompanyBase):
    """Model for creating a company"""
    pass

class CompanyUpdate(BaseModel):
    """Model for updating a company"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    ticker: Optional[str] = Field(None, max_length=10)
    industry: Optional[IndustryCategory] = None
    sector: Optional[str] = Field(None, max_length=100)
    market_cap: Optional[float] = Field(None, ge=0)
    country: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = Field(None, max_length=1000)
    website: Optional[str] = None

class Company(CompanyBase, TimestampMixin):
    """Complete company model"""
    id: int
    is_active: bool = True
    last_score_update: Optional[datetime] = None
    
    # Computed fields
    current_score: Optional[float] = Field(None, ge=0, le=100)
    score_range: Optional[ScoreRange] = None
    daily_change: Optional[float] = None
    weekly_change: Optional[float] = None
    monthly_change: Optional[float] = None
    
    # Metadata
    data_sources: List[str] = Field(default_factory=list)
    last_data_update: Optional[datetime] = None
    data_quality_score: Optional[float] = Field(None, ge=0, le=1)

class CompanyResponse(Company):
    """Company response model with additional computed fields"""
    peer_companies: List[str] = Field(default_factory=list)
    recent_alerts_count: int = 0
    trend_direction: Optional[str] = None
    risk_factors: List[str] = Field(default_factory=list)
    
    # Financial metrics
    financial_metrics: Optional[Dict[str, float]] = None
    market_metrics: Optional[Dict[str, float]] = None
    
class CompanySearch(BaseModel):
    """Company search parameters"""
    query: Optional[str] = None
    industry: Optional[IndustryCategory] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    min_market_cap: Optional[float] = Field(None, ge=0)
    max_market_cap: Optional[float] = Field(None, ge=0)
    min_score: Optional[float] = Field(None, ge=0, le=100)
    max_score: Optional[float] = Field(None, ge=0, le=100)
    score_range: Optional[ScoreRange] = None
    has_alerts: Optional[bool] = None
    
class CompanyBulkUpdate(BaseModel):
    """Bulk update model for companies"""
    company_ids: List[int]
    updates: CompanyUpdate
    
class CompanyStats(BaseModel):
    """Company statistics"""
    total_companies: int
    active_companies: int
    by_industry: Dict[str, int]
    by_score_range: Dict[str, int]
    by_country: Dict[str, int]
    average_score: float
    score_distribution: Dict[str, int]
