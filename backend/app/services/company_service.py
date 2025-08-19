"""
Company management service integrating all pipeline stages
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta

from app.models.company import (
    Company, CompanyCreate, CompanyUpdate, CompanyResponse, 
    CompanySearch, CompanyStats, PaginatedResponse
)
from app.models.common import ScoreRange, IndustryCategory

logger = logging.getLogger(__name__)

class CompanyService:
    """Service for company-related operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_companies(
        self, 
        page: int = 1, 
        size: int = 20,
        search_params: Optional[CompanySearch] = None
    ) -> PaginatedResponse[CompanyResponse]:
        """Get paginated companies with filtering"""
        try:
            # Build query with filters
            query = select(Company).where(Company.is_active == True)
            
            if search_params:
                if search_params.query:
                    query = query.where(
                        or_(
                            Company.name.ilike(f"%{search_params.query}%"),
                            Company.ticker.ilike(f"%{search_params.query}%")
                        )
                    )
                
                if search_params.industry:
                    query = query.where(Company.industry == search_params.industry)
                
                if search_params.sector:
                    query = query.where(Company.sector == search_params.sector)
                
                if search_params.country:
                    query = query.where(Company.country == search_params.country)
                
                if search_params.min_score is not None:
                    query = query.where(Company.current_score >= search_params.min_score)
                
                if search_params.max_score is not None:
                    query = query.where(Company.current_score <= search_params.max_score)
            
            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await self.db.execute(count_query)
            total = total_result.scalar()
            
            # Apply pagination
            offset = (page - 1) * size
            query = query.offset(offset).limit(size)
            
            # Execute query
            result = await self.db.execute(query)
            companies = result.scalars().all()
            
            # Convert to response models with additional data
            company_responses = []
            for company in companies:
                response = await self._enrich_company_data(company)
                company_responses.append(response)
            
            pages = (total + size - 1) // size
            
            return PaginatedResponse(
                items=company_responses,
                total=total,
                page=page,
                size=size,
                pages=pages,
                has_next=page < pages,
                has_prev=page > 1
            )
            
        except Exception as e:
            logger.error(f"Error fetching companies: {str(e)}")
            raise
    
    async def get_company_by_id(self, company_id: int) -> Optional[CompanyResponse]:
        """Get company by ID with enriched data"""
        try:
            query = select(Company).where(
                and_(Company.id == company_id, Company.is_active == True)
            )
            result = await self.db.execute(query)
            company = result.scalar_one_or_none()
            
            if not company:
                return None
            
            return await self._enrich_company_data(company)
            
        except Exception as e:
            logger.error(f"Error fetching company {company_id}: {str(e)}")
            raise
    
    async def get_company_by_ticker(self, ticker: str) -> Optional[Company]:
        """Get company by ticker symbol"""
        try:
            query = select(Company).where(
                and_(Company.ticker == ticker.upper(), Company.is_active == True)
            )
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Error fetching company by ticker {ticker}: {str(e)}")
            raise
    
    async def search_companies(self, query: str, limit: int = 10) -> List[CompanyResponse]:
        """Search companies by name or ticker"""
        try:
            search_query = select(Company).where(
                and_(
                    Company.is_active == True,
                    or_(
                        Company.name.ilike(f"%{query}%"),
                        Company.ticker.ilike(f"%{query}%")
                    )
                )
            ).limit(limit)
            
            result = await self.db.execute(search_query)
            companies = result.scalars().all()
            
            # Enrich with additional data
            enriched_companies = []
            for company in companies:
                enriched = await self._enrich_company_data(company)
                enriched_companies.append(enriched)
            
            return enriched_companies
            
        except Exception as e:
            logger.error(f"Error searching companies: {str(e)}")
            raise
    
    async def create_company(self, company_data: CompanyCreate) -> CompanyResponse:
        """Create a new company"""
        try:
            company = Company(**company_data.dict())
            company.created_at = datetime.utcnow()
            
            self.db.add(company)
            await self.db.commit()
            await self.db.refresh(company)
            
            return await self._enrich_company_data(company)
            
        except Exception as e:
            logger.error(f"Error creating company: {str(e)}")
            await self.db.rollback()
            raise
    
    async def update_company(
        self, 
        company_id: int, 
        company_update: CompanyUpdate
    ) -> Optional[CompanyResponse]:
        """Update company information"""
        try:
            query = select(Company).where(Company.id == company_id)
            result = await self.db.execute(query)
            company = result.scalar_one_or_none()
            
            if not company:
                return None
            
            # Update fields
            update_data = company_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(company, field, value)
            
            company.updated_at = datetime.utcnow()
            
            await self.db.commit()
            await self.db.refresh(company)
            
            return await self._enrich_company_data(company)
            
        except Exception as e:
            logger.error(f"Error updating company {company_id}: {str(e)}")
            await self.db.rollback()
            raise
    
    async def delete_company(self, company_id: int) -> bool:
        """Soft delete a company"""
        try:
            query = select(Company).where(Company.id == company_id)
            result = await self.db.execute(query)
            company = result.scalar_one_or_none()
            
            if not company:
                return False
            
            company.is_active = False
            company.updated_at = datetime.utcnow()
            
            await self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting company {company_id}: {str(e)}")
            await self.db.rollback()
            raise
    
    async def get_peer_companies(self, company_id: int, limit: int = 5) -> List[CompanyResponse]:
        """Get peer companies for comparison"""
        try:
            # Get the target company
            target_company = await self.get_company_by_id(company_id)
            if not target_company:
                return []
            
            # Find peers in same industry/sector with similar market cap
            query = select(Company).where(
                and_(
                    Company.id != company_id,
                    Company.is_active == True,
                    Company.industry == target_company.industry
                )
            )
            
            # Add market cap similarity filter if available
            if target_company.market_cap:
                lower_bound = target_company.market_cap * 0.5
                upper_bound = target_company.market_cap * 2.0
                query = query.where(
                    and_(
                        Company.market_cap >= lower_bound,
                        Company.market_cap <= upper_bound
                    )
                )
            
            query = query.limit(limit)
            
            result = await self.db.execute(query)
            peers = result.scalars().all()
            
            # Enrich peer data
            enriched_peers = []
            for peer in peers:
                enriched = await self._enrich_company_data(peer)
                enriched_peers.append(enriched)
            
            return enriched_peers
            
        except Exception as e:
            logger.error(f"Error fetching peers for company {company_id}: {str(e)}")
            raise
    
    async def get_risk_factors(self, company_id: int) -> List[str]:
        """Get key risk factors for a company"""
        try:
            # This would integrate with Stage 4 explainability
            # For now, return mock data based on company characteristics
            company = await self.get_company_by_id(company_id)
            if not company:
                return []
            
            risk_factors = []
            
            # Score-based risk factors
            if company.current_score and company.current_score < 60:
                risk_factors.append("Low credit score indicates high default risk")
            
            # Industry-specific risks
            industry_risks = {
                IndustryCategory.TECHNOLOGY: [
                    "High R&D spending volatility",
                    "Rapid technological obsolescence risk"
                ],
                IndustryCategory.ENERGY: [
                    "Commodity price volatility",
                    "Environmental regulatory changes"
                ],
                IndustryCategory.FINANCIAL_SERVICES: [
                    "Interest rate sensitivity",
                    "Regulatory capital requirements"
                ]
            }
            
            if company.industry in industry_risks:
                risk_factors.extend(industry_risks[company.industry])
            
            # Market cap based risks
            if company.market_cap and company.market_cap < 1e9:  # < $1B
                risk_factors.append("Small market capitalization increases volatility")
            
            return risk_factors[:5]  # Return top 5 risk factors
            
        except Exception as e:
            logger.error(f"Error fetching risk factors for company {company_id}: {str(e)}")
            raise
    
    async def get_company_stats(self) -> CompanyStats:
        """Get company statistics"""
        try:
            # Total and active companies
            total_query = select(func.count(Company.id))
            active_query = select(func.count(Company.id)).where(Company.is_active == True)
            
            total_result = await self.db.execute(total_query)
            active_result = await self.db.execute(active_query)
            
            total_companies = total_result.scalar()
            active_companies = active_result.scalar()
            
            # By industry
            industry_query = select(
                Company.industry,
                func.count(Company.id).label('count')
            ).where(Company.is_active == True).group_by(Company.industry)
            
            industry_result = await self.db.execute(industry_query)
            by_industry = {row.industry: row.count for row in industry_result}
            
            # By score range (mock implementation)
            by_score_range = {
                "excellent": 125,
                "good": 340,
                "fair": 285,
                "poor": 180,
                "very_poor": 70
            }
            
            # By country (mock implementation)
            by_country = {
                "United States": 650,
                "United Kingdom": 180,
                "Germany": 120,
                "Japan": 95,
                "Canada": 85,
                "Others": 120
            }
            
            return CompanyStats(
                total_companies=total_companies,
                active_companies=active_companies,
                by_industry=by_industry,
                by_score_range=by_score_range,
                by_country=by_country,
                average_score=78.5,
                score_distribution={
                    "0-20": 15,
                    "21-40": 45,
                    "41-60": 180,
                    "61-80": 485,
                    "81-100": 275
                }
            )
            
        except Exception as e:
            logger.error(f"Error fetching company stats: {str(e)}")
            raise
    
    async def get_industry_top_performers(
        self, 
        industry: str, 
        limit: int = 10
    ) -> List[CompanyResponse]:
        """Get top performing companies in an industry"""
        try:
            query = select(Company).where(
                and_(
                    Company.is_active == True,
                    Company.industry == industry,
                    Company.current_score.isnot(None)
                )
            ).order_by(Company.current_score.desc()).limit(limit)
            
            result = await self.db.execute(query)
            companies = result.scalars().all()
            
            # Enrich with additional data
            enriched_companies = []
            for company in companies:
                enriched = await self._enrich_company_data(company)
                enriched_companies.append(enriched)
            
            return enriched_companies
            
        except Exception as e:
            logger.error(f"Error fetching top performers for industry {industry}: {str(e)}")
            raise
    
    async def get_watchlist_companies(self, user_id: str) -> List[CompanyResponse]:
        """Get user's watchlist companies"""
        try:
            # Mock implementation - in real system, this would query user_watchlist table
            # For now, return top companies
            query = select(Company).where(
                and_(
                    Company.is_active == True,
                    Company.current_score.isnot(None)
                )
            ).order_by(Company.current_score.desc()).limit(10)
            
            result = await self.db.execute(query)
            companies = result.scalars().all()
            
            enriched_companies = []
            for company in companies:
                enriched = await self._enrich_company_data(company)
                enriched_companies.append(enriched)
            
            return enriched_companies
            
        except Exception as e:
            logger.error(f"Error fetching watchlist for user {user_id}: {str(e)}")
            raise
    
    async def add_to_watchlist(self, user_id: str, company_id: int) -> bool:
        """Add company to user's watchlist"""
        try:
            # Mock implementation - would insert into user_watchlist table
            logger.info(f"Added company {company_id} to watchlist for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to watchlist: {str(e)}")
            raise
    
    async def remove_from_watchlist(self, user_id: str, company_id: int) -> bool:
        """Remove company from user's watchlist"""
        try:
            # Mock implementation - would delete from user_watchlist table
            logger.info(f"Removed company {company_id} from watchlist for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing from watchlist: {str(e)}")
            raise
    
    async def _enrich_company_data(self, company: Company) -> CompanyResponse:
        """Enrich company data with additional computed fields"""
        try:
            # Convert to response model
            response_data = company.__dict__.copy()
            
            # Add computed fields
            response_data['peer_companies'] = []  # Would be populated from peer analysis
            response_data['recent_alerts_count'] = 0  # Would query alerts table
            response_data['trend_direction'] = self._calculate_trend_direction(company)
            response_data['risk_factors'] = await self.get_risk_factors(company.id)
            
            # Financial and market metrics (mock data)
            response_data['financial_metrics'] = {
                'revenue_growth': 0.12,
                'profit_margin': 0.15,
                'debt_to_equity': 0.45,
                'current_ratio': 2.1
            }
            
            response_data['market_metrics'] = {
                'beta': 1.2,
                'pe_ratio': 18.5,
                'market_cap_rank': 150,
                'volatility': 0.25
            }
            
            return CompanyResponse(**response_data)
            
        except Exception as e:
            logger.error(f"Error enriching company data: {str(e)}")
            raise
    
    def _calculate_trend_direction(self, company: Company) -> Optional[str]:
        """Calculate trend direction based on score changes"""
        try:
            if company.daily_change is None:
                return None
            
            if company.daily_change > 2:
                return "up"
            elif company.daily_change < -2:
                return "down"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {str(e)}")
            return None
