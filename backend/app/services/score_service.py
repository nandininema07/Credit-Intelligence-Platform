"""
Credit score service integrating ML models and explainability (Stages 2-4)
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, desc
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import numpy as np
import asyncio

from app.models.score import (
    CreditScore, ScoreCreate, ScoreResponse, ScoreHistory,
    ScorePrediction, ScoreExplanation, ScoreComparison,
    ScoreBenchmark, ScoreDistribution, PaginatedResponse
)
from app.models.common import TimeSeriesData, TrendDirection, DataSource

# Import Stage 2-4 components
from stage2_feature_engineering.feature_store.feature_store import FeatureStore
from stage3_model_training.models.ensemble_models import EnsembleModel
from stage4_explainability.explainer.explanation_generator import ExplanationGenerator

logger = logging.getLogger(__name__)

class ScoreService:
    """Service for credit score operations integrating ML pipeline"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.feature_store = FeatureStore()
        self.model = EnsembleModel()
        self.explainer = ExplanationGenerator()
    
    async def get_scores(
        self,
        page: int = 1,
        size: int = 20,
        company_id: Optional[int] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None
    ) -> PaginatedResponse[ScoreResponse]:
        """Get paginated credit scores"""
        try:
            # Build query
            query = select(CreditScore)
            
            if company_id:
                query = query.where(CreditScore.company_id == company_id)
            if min_score is not None:
                query = query.where(CreditScore.score >= min_score)
            if max_score is not None:
                query = query.where(CreditScore.score <= max_score)
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await self.db.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination and ordering
            offset = (page - 1) * size
            query = query.order_by(desc(CreditScore.created_at)).offset(offset).limit(size)
            
            result = await self.db.execute(query)
            scores = result.scalars().all()
            
            # Enrich with additional data
            score_responses = []
            for score in scores:
                response = await self._enrich_score_data(score)
                score_responses.append(response)
            
            pages = (total + size - 1) // size
            
            return PaginatedResponse(
                items=score_responses,
                total=total,
                page=page,
                size=size,
                pages=pages,
                has_next=page < pages,
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error fetching scores: {str(e)}")
            raise
    
    async def get_current_score(self, company_id: int) -> Optional[ScoreResponse]:
        """Get current credit score for a company"""
        try:
            query = select(CreditScore).where(
                CreditScore.company_id == company_id
            ).order_by(desc(CreditScore.created_at)).limit(1)
            
            result = await self.db.execute(query)
            score = result.scalar_one_or_none()
            
            if not score:
                return None
            
            return await self._enrich_score_data(score)
            
        except Exception as e:
            logger.error(f"Error fetching current score for company {company_id}: {str(e)}")
            raise
    
    async def get_score_history(
        self,
        company_id: int,
        start_date: datetime,
        end_date: datetime
    ) -> ScoreHistory:
        """Get historical credit scores for a company"""
        try:
            query = select(CreditScore).where(
                and_(
                    CreditScore.company_id == company_id,
                    CreditScore.created_at >= start_date,
                    CreditScore.created_at <= end_date
                )
            ).order_by(CreditScore.created_at)
            
            result = await self.db.execute(query)
            scores = result.scalars().all()
            
            # Convert to time series data
            time_series = []
            score_values = []
            
            for score in scores:
                time_series.append(TimeSeriesData(
                    timestamp=score.created_at,
                    value=score.score,
                    metadata={
                        'confidence': score.confidence,
                        'model_version': score.model_version
                    }
                ))
                score_values.append(score.score)
            
            # Calculate statistics
            if score_values:
                avg_score = np.mean(score_values)
                min_score = np.min(score_values)
                max_score = np.max(score_values)
                volatility = np.std(score_values)
                
                # Determine trend
                if len(score_values) >= 2:
                    trend_slope = (score_values[-1] - score_values[0]) / len(score_values)
                    if trend_slope > 1:
                        trend = TrendDirection.UP
                    elif trend_slope < -1:
                        trend = TrendDirection.DOWN
                    else:
                        trend = TrendDirection.STABLE
                else:
                    trend = TrendDirection.STABLE
            else:
                avg_score = min_score = max_score = volatility = 0
                trend = TrendDirection.STABLE
            
            return ScoreHistory(
                company_id=company_id,
                scores=time_series,
                period_start=start_date,
                period_end=end_date,
                average_score=avg_score,
                min_score=min_score,
                max_score=max_score,
                volatility=volatility,
                trend=trend
            )
            
        except Exception as e:
            logger.error(f"Error fetching score history: {str(e)}")
            raise
    
    async def get_score_prediction(
        self,
        company_id: int,
        horizon_days: int = 30
    ) -> ScorePrediction:
        """Get credit score prediction using ML models"""
        try:
            # Get current features from feature store
            features = await self.feature_store.get_company_features(company_id)
            
            # Generate prediction using ensemble model
            prediction_result = await self.model.predict_score(
                features=features,
                horizon_days=horizon_days
            )
            
            prediction_date = datetime.utcnow() + timedelta(days=horizon_days)
            
            return ScorePrediction(
                company_id=company_id,
                predicted_score=prediction_result['score'],
                prediction_date=prediction_date,
                confidence_interval=prediction_result['confidence_interval'],
                prediction_horizon_days=horizon_days,
                model_used=prediction_result['model_version'],
                key_assumptions=prediction_result.get('assumptions', [])
            )
            
        except Exception as e:
            logger.error(f"Error generating score prediction: {str(e)}")
            raise
    
    async def get_score_explanation(
        self,
        company_id: int,
        explanation_type: str = "shap"
    ) -> ScoreExplanation:
        """Get detailed explanation of credit score"""
        try:
            # Get current score and features
            current_score = await self.get_current_score(company_id)
            if not current_score:
                raise ValueError("No current score found for company")
            
            features = await self.feature_store.get_company_features(company_id)
            
            # Generate explanation using Stage 4 explainability
            explanation_result = await self.explainer.explain_score(
                company_id=company_id,
                score=current_score.score,
                features=features,
                method=explanation_type
            )
            
            return ScoreExplanation(
                company_id=company_id,
                score=current_score.score,
                explanation_type=explanation_type,
                feature_importances=explanation_result['feature_importances'],
                explanation_text=explanation_result['explanation_text'],
                visualization_data=explanation_result.get('visualization_data'),
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating score explanation: {str(e)}")
            raise
    
    async def compare_scores(
        self,
        company_id: int,
        compare_with: List[int]
    ) -> ScoreComparison:
        """Compare credit scores between companies"""
        try:
            all_company_ids = [company_id] + compare_with
            companies_data = []
            
            # Get current scores for all companies
            for cid in all_company_ids:
                score = await self.get_current_score(cid)
                if score:
                    companies_data.append({
                        'id': cid,
                        'name': score.company_name,
                        'score': score.score,
                        'ticker': score.company_ticker
                    })
            
            # Calculate comparison metrics
            scores = [c['score'] for c in companies_data]
            metrics = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'range': np.max(scores) - np.min(scores)
            }
            
            # Rank companies by score
            sorted_companies = sorted(companies_data, key=lambda x: x['score'], reverse=True)
            ranking = [c['id'] for c in sorted_companies]
            
            return ScoreComparison(
                companies=companies_data,
                comparison_date=datetime.utcnow(),
                metrics=metrics,
                ranking=ranking
            )
            
        except Exception as e:
            logger.error(f"Error comparing scores: {str(e)}")
            raise
    
    async def get_industry_benchmark(
        self,
        industry: str,
        sector: Optional[str] = None
    ) -> ScoreBenchmark:
        """Get industry benchmark scores"""
        try:
            # Mock implementation - would query actual industry data
            benchmark_data = {
                'technology': {'benchmark': 82.5, 'p25': 75.2, 'p50': 82.1, 'p75': 89.3, 'count': 245},
                'financial_services': {'benchmark': 78.9, 'p25': 71.8, 'p50': 78.5, 'p75': 86.2, 'count': 189},
                'healthcare': {'benchmark': 80.1, 'p25': 73.4, 'p50': 79.8, 'p75': 87.1, 'count': 156},
                'energy': {'benchmark': 72.3, 'p25': 65.1, 'p50': 72.0, 'p75': 79.8, 'count': 98}
            }
            
            industry_lower = industry.lower().replace(' ', '_')
            data = benchmark_data.get(industry_lower, benchmark_data['technology'])
            
            return ScoreBenchmark(
                industry=industry,
                sector=sector,
                benchmark_score=data['benchmark'],
                percentiles={
                    'p25': data['p25'],
                    'p50': data['p50'],
                    'p75': data['p75']
                },
                company_count=data['count'],
                last_updated=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error fetching industry benchmark: {str(e)}")
            raise
    
    async def get_score_distribution(
        self,
        industry: Optional[str] = None
    ) -> ScoreDistribution:
        """Get score distribution analysis"""
        try:
            # Mock implementation with realistic data
            total_companies = 1250
            
            score_ranges = {
                'excellent': 125,
                'good': 340,
                'fair': 285,
                'poor': 180,
                'very_poor': 70
            }
            
            statistics = {
                'mean': 78.5,
                'median': 79.2,
                'std': 12.8,
                'skewness': -0.15,
                'kurtosis': 2.8
            }
            
            by_industry = {
                'technology': {'mean': 82.5, 'count': 245},
                'financial_services': {'mean': 78.9, 'count': 189},
                'healthcare': {'mean': 80.1, 'count': 156},
                'energy': {'mean': 72.3, 'count': 98}
            }
            
            by_market_cap = {
                'large_cap': {'mean': 84.2, 'count': 156},
                'mid_cap': {'mean': 79.8, 'count': 387},
                'small_cap': {'mean': 75.1, 'count': 707}
            }
            
            return ScoreDistribution(
                total_companies=total_companies,
                score_ranges=score_ranges,
                statistics=statistics,
                by_industry=by_industry,
                by_market_cap=by_market_cap
            )
            
        except Exception as e:
            logger.error(f"Error fetching score distribution: {str(e)}")
            raise
    
    async def get_top_movers(
        self,
        direction: str = "both",
        limit: int = 10,
        timeframe: str = "1d"
    ) -> List[ScoreResponse]:
        """Get companies with biggest score changes"""
        try:
            # Mock implementation - would calculate actual score changes
            mock_movers = [
                {'company_id': 1, 'name': 'Tesla Inc.', 'score': 89.5, 'change': 15.2},
                {'company_id': 2, 'name': 'Apple Inc.', 'score': 92.1, 'change': 12.5},
                {'company_id': 3, 'name': 'Microsoft Corp.', 'score': 91.8, 'change': 8.3},
                {'company_id': 4, 'name': 'Meta Platforms', 'score': 76.2, 'change': -9.4},
                {'company_id': 5, 'name': 'Netflix Inc.', 'score': 78.9, 'change': -7.8}
            ]
            
            # Filter by direction
            if direction == "up":
                filtered_movers = [m for m in mock_movers if m['change'] > 0]
            elif direction == "down":
                filtered_movers = [m for m in mock_movers if m['change'] < 0]
            else:
                filtered_movers = mock_movers
            
            # Sort by absolute change and limit
            filtered_movers.sort(key=lambda x: abs(x['change']), reverse=True)
            filtered_movers = filtered_movers[:limit]
            
            # Convert to ScoreResponse objects
            score_responses = []
            for mover in filtered_movers:
                # Mock ScoreResponse
                response = ScoreResponse(
                    id=mover['company_id'],
                    company_id=mover['company_id'],
                    score=mover['score'],
                    confidence=0.95,
                    model_version="v1.2.3",
                    score_range='good',
                    company_name=mover['name'],
                    trend_direction='up' if mover['change'] > 0 else 'down',
                    score_change=mover['change'],
                    created_at=datetime.utcnow(),
                    feature_contributions={},
                    top_positive_factors=[],
                    top_negative_factors=[],
                    data_sources_used=[]
                )
                score_responses.append(response)
            
            return score_responses
            
        except Exception as e:
            logger.error(f"Error fetching top movers: {str(e)}")
            raise
    
    async def create_score(self, score_data: ScoreCreate) -> ScoreResponse:
        """Create a new credit score"""
        try:
            score = CreditScore(**score_data.dict())
            score.created_at = datetime.utcnow()
            
            # Calculate score range
            if score.score >= 90:
                score.score_range = 'excellent'
            elif score.score >= 80:
                score.score_range = 'good'
            elif score.score >= 70:
                score.score_range = 'fair'
            elif score.score >= 60:
                score.score_range = 'poor'
            else:
                score.score_range = 'very_poor'
            
            self.db.add(score)
            await self.db.commit()
            await self.db.refresh(score)
            
            return await self._enrich_score_data(score)
            
        except Exception as e:
            logger.error(f"Error creating score: {str(e)}")
            await self.db.rollback()
            raise
    
    async def get_market_pulse(self) -> Dict[str, Any]:
        """Get market pulse data for dashboard"""
        try:
            # Mock market pulse data
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall_sentiment': 'positive',
                'market_volatility': 0.18,
                'risk_appetite': 'moderate',
                'top_sectors': [
                    {'name': 'Technology', 'avg_score': 82.5, 'change': 2.3},
                    {'name': 'Healthcare', 'avg_score': 80.1, 'change': 1.8},
                    {'name': 'Financial Services', 'avg_score': 78.9, 'change': -0.5}
                ],
                'market_indicators': {
                    'vix': 18.2,
                    'credit_spreads': 1.45,
                    'default_rate': 0.023
                },
                'alerts_summary': {
                    'critical': 3,
                    'high': 12,
                    'medium': 28,
                    'low': 45
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching market pulse: {str(e)}")
            raise
    
    async def simulate_score_change(
        self,
        company_id: int,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate score changes based on scenario inputs"""
        try:
            # Get current score
            current_score = await self.get_current_score(company_id)
            if not current_score:
                raise ValueError("No current score found for company")
            
            # Mock simulation based on scenario
            base_score = current_score.score
            
            # Apply scenario adjustments
            adjustments = {
                'revenue_change': scenario.get('revenue_change', 0) * 0.1,
                'debt_change': scenario.get('debt_change', 0) * -0.05,
                'market_conditions': scenario.get('market_conditions', 0) * 0.03
            }
            
            total_adjustment = sum(adjustments.values())
            simulated_score = max(0, min(100, base_score + total_adjustment))
            
            return {
                'company_id': company_id,
                'current_score': base_score,
                'simulated_score': simulated_score,
                'score_change': simulated_score - base_score,
                'scenario': scenario,
                'adjustments': adjustments,
                'confidence': 0.85,
                'simulation_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error simulating score change: {str(e)}")
            raise
    
    async def _enrich_score_data(self, score: CreditScore) -> ScoreResponse:
        """Enrich score data with additional computed fields"""
        try:
            # Mock company data - would join with companies table
            company_names = {
                1: 'Apple Inc.',
                2: 'Microsoft Corp.',
                3: 'Tesla Inc.',
                4: 'Amazon.com Inc.',
                5: 'Google (Alphabet)'
            }
            
            company_tickers = {
                1: 'AAPL',
                2: 'MSFT', 
                3: 'TSLA',
                4: 'AMZN',
                5: 'GOOGL'
            }
            
            # Calculate trend direction
            trend_direction = TrendDirection.STABLE
            if score.score_change:
                if score.score_change > 2:
                    trend_direction = TrendDirection.UP
                elif score.score_change < -2:
                    trend_direction = TrendDirection.DOWN
            
            # Calculate percentile rank (mock)
            percentile_rank = min(95, max(5, (score.score / 100) * 90 + 5))
            
            response_data = score.__dict__.copy()
            response_data.update({
                'company_name': company_names.get(score.company_id, f'Company {score.company_id}'),
                'company_ticker': company_tickers.get(score.company_id),
                'trend_direction': trend_direction,
                'percentile_rank': percentile_rank,
                'industry_average': 78.5,  # Mock industry average
                'peer_comparison': {
                    'better_than': 0.65,
                    'worse_than': 0.35
                }
            })
            
            return ScoreResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Error enriching score data: {str(e)}")
            raise
