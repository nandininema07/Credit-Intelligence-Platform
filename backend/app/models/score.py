"""
Credit score-related Pydantic models
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import Field, validator

from .common import BaseModel, TimestampMixin, TimeSeriesData, ScoreRange, TrendDirection, DataSource

class ScoreBase(BaseModel):
    """Base score model"""
    company_id: int
    score: float = Field(..., ge=0, le=100)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    model_version: str = Field(..., min_length=1)
    
    @validator('score')
    def validate_score(cls, v):
        return round(v, 2)

class ScoreCreate(ScoreBase):
    """Model for creating a credit score"""
    features: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class CreditScore(ScoreBase, TimestampMixin):
    """Complete credit score model"""
    id: int
    score_range: ScoreRange
    previous_score: Optional[float] = None
    score_change: Optional[float] = None
    score_change_percent: Optional[float] = None
    
    # Feature contributions
    feature_contributions: Optional[Dict[str, float]] = None
    top_positive_factors: List[str] = Field(default_factory=list)
    top_negative_factors: List[str] = Field(default_factory=list)
    
    # Data quality metrics
    data_completeness: Optional[float] = Field(None, ge=0, le=1)
    data_freshness_hours: Optional[int] = Field(None, ge=0)
    data_sources_used: List[DataSource] = Field(default_factory=list)

class ScoreResponse(CreditScore):
    """Score response with additional computed fields"""
    company_name: str
    company_ticker: Optional[str] = None
    trend_direction: TrendDirection
    percentile_rank: Optional[float] = Field(None, ge=0, le=100)
    industry_average: Optional[float] = None
    peer_comparison: Optional[Dict[str, float]] = None

class ScoreHistory(BaseModel):
    """Historical score data"""
    company_id: int
    scores: List[TimeSeriesData]
    period_start: datetime
    period_end: datetime
    average_score: float
    min_score: float
    max_score: float
    volatility: float
    trend: TrendDirection

class ScorePrediction(BaseModel):
    """Score prediction model"""
    company_id: int
    predicted_score: float = Field(..., ge=0, le=100)
    prediction_date: datetime
    confidence_interval: Dict[str, float]  # {"lower": 75.2, "upper": 82.8}
    prediction_horizon_days: int
    model_used: str
    key_assumptions: List[str] = Field(default_factory=list)

class ScoreExplanation(BaseModel):
    """Score explanation using SHAP/LIME"""
    company_id: int
    score: float
    explanation_type: str  # "shap", "lime", "permutation"
    feature_importances: Dict[str, float]
    explanation_text: str
    visualization_data: Optional[Dict[str, Any]] = None
    generated_at: datetime

class ScoreAlert(BaseModel):
    """Score-based alert thresholds"""
    company_id: int
    threshold_type: str  # "absolute", "change", "percentile"
    threshold_value: float
    comparison_operator: str  # "gt", "lt", "eq", "gte", "lte"
    alert_message: str
    is_active: bool = True

class ScoreComparison(BaseModel):
    """Compare scores between companies"""
    companies: List[Dict[str, Any]]  # [{"id": 1, "name": "Apple", "score": 89.5}]
    comparison_date: datetime
    metrics: Dict[str, Any]  # statistical comparisons
    ranking: List[int]  # company_ids in rank order

class ScoreBenchmark(BaseModel):
    """Industry/peer benchmarks"""
    industry: str
    sector: Optional[str] = None
    benchmark_score: float
    percentiles: Dict[str, float]  # {"p25": 65.2, "p50": 78.1, "p75": 87.3}
    company_count: int
    last_updated: datetime

class ScoreDistribution(BaseModel):
    """Score distribution analysis"""
    total_companies: int
    score_ranges: Dict[ScoreRange, int]
    statistics: Dict[str, float]  # mean, median, std, etc.
    by_industry: Dict[str, Dict[str, float]]
    by_market_cap: Dict[str, Dict[str, float]]

class ScoreModelMetrics(BaseModel):
    """Model performance metrics"""
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    feature_count: int
    training_date: datetime
    validation_metrics: Dict[str, float]
