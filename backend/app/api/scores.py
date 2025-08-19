"""
Credit score API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from app.database import get_db
from app.models.score import (
    CreditScore, ScoreCreate, ScoreResponse, ScoreHistory,
    ScorePrediction, ScoreExplanation, ScoreComparison,
    ScoreBenchmark, ScoreDistribution, PaginatedResponse
)
from app.services.score_service import ScoreService
from app.utils.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=PaginatedResponse[ScoreResponse])
async def get_scores(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    company_id: Optional[int] = Query(None),
    min_score: Optional[float] = Query(None, ge=0, le=100),
    max_score: Optional[float] = Query(None, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get paginated list of credit scores"""
    try:
        score_service = ScoreService(db)
        result = await score_service.get_scores(
            page=page,
            size=size,
            company_id=company_id,
            min_score=min_score,
            max_score=max_score
        )
        return result
        
    except Exception as e:
        logger.error(f"Error fetching scores: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/company/{company_id}/current", response_model=ScoreResponse)
async def get_current_score(
    company_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get current credit score for a company"""
    try:
        score_service = ScoreService(db)
        score = await score_service.get_current_score(company_id)
        
        if not score:
            raise HTTPException(status_code=404, detail="Score not found")
            
        return score
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching current score for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/company/{company_id}/history", response_model=ScoreHistory)
async def get_score_history(
    company_id: int = Path(..., ge=1),
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get historical credit scores for a company"""
    try:
        score_service = ScoreService(db)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        history = await score_service.get_score_history(
            company_id=company_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Error fetching score history for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/company/{company_id}/prediction", response_model=ScorePrediction)
async def get_score_prediction(
    company_id: int = Path(..., ge=1),
    horizon_days: int = Query(30, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get credit score prediction for a company"""
    try:
        score_service = ScoreService(db)
        prediction = await score_service.get_score_prediction(
            company_id=company_id,
            horizon_days=horizon_days
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error fetching score prediction for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/company/{company_id}/explanation", response_model=ScoreExplanation)
async def get_score_explanation(
    company_id: int = Path(..., ge=1),
    explanation_type: str = Query("shap", regex="^(shap|lime|permutation)$"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get detailed explanation of credit score using AI explainability"""
    try:
        score_service = ScoreService(db)
        explanation = await score_service.get_score_explanation(
            company_id=company_id,
            explanation_type=explanation_type
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating score explanation for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/company/{company_id}/compare", response_model=ScoreComparison)
async def compare_scores(
    company_id: int = Path(..., ge=1),
    compare_with: List[int] = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Compare credit scores between companies"""
    try:
        if len(compare_with) > 10:
            raise HTTPException(status_code=400, detail="Cannot compare with more than 10 companies")
            
        score_service = ScoreService(db)
        comparison = await score_service.compare_scores(
            company_id=company_id,
            compare_with=compare_with
        )
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing scores: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/benchmarks/{industry}", response_model=ScoreBenchmark)
async def get_industry_benchmark(
    industry: str = Path(...),
    sector: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get industry benchmark scores"""
    try:
        score_service = ScoreService(db)
        benchmark = await score_service.get_industry_benchmark(
            industry=industry,
            sector=sector
        )
        
        return benchmark
        
    except Exception as e:
        logger.error(f"Error fetching benchmark for industry {industry}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/distribution", response_model=ScoreDistribution)
async def get_score_distribution(
    industry: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get score distribution analysis"""
    try:
        score_service = ScoreService(db)
        distribution = await score_service.get_score_distribution(industry=industry)
        return distribution
        
    except Exception as e:
        logger.error(f"Error fetching score distribution: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/top-movers", response_model=List[ScoreResponse])
async def get_top_movers(
    direction: str = Query("both", regex="^(up|down|both)$"),
    limit: int = Query(10, ge=1, le=50),
    timeframe: str = Query("1d", regex="^(1d|1w|1m)$"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get companies with biggest score changes"""
    try:
        score_service = ScoreService(db)
        movers = await score_service.get_top_movers(
            direction=direction,
            limit=limit,
            timeframe=timeframe
        )
        
        return movers
        
    except Exception as e:
        logger.error(f"Error fetching top movers: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/", response_model=ScoreResponse)
async def create_score(
    score: ScoreCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new credit score (typically called by ML pipeline)"""
    try:
        score_service = ScoreService(db)
        new_score = await score_service.create_score(score)
        return new_score
        
    except Exception as e:
        logger.error(f"Error creating score: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/market-pulse", response_model=dict)
async def get_market_pulse(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get market pulse data for dashboard"""
    try:
        score_service = ScoreService(db)
        pulse_data = await score_service.get_market_pulse()
        return pulse_data
        
    except Exception as e:
        logger.error(f"Error fetching market pulse: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/simulate", response_model=dict)
async def simulate_score_change(
    company_id: int = Query(..., ge=1),
    scenario: dict = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Simulate score changes based on scenario inputs"""
    try:
        score_service = ScoreService(db)
        simulation = await score_service.simulate_score_change(
            company_id=company_id,
            scenario=scenario
        )
        
        return simulation
        
    except Exception as e:
        logger.error(f"Error simulating score change: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
