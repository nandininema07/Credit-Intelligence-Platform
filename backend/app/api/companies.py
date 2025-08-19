"""
Company management API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from app.database import get_db
from app.models.company import (
    Company, CompanyCreate, CompanyUpdate, CompanyResponse, 
    CompanySearch, CompanyStats, PaginatedResponse
)
from app.services.company_service import CompanyService
from app.utils.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=PaginatedResponse[CompanyResponse])
async def get_companies(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None),
    industry: Optional[str] = Query(None),
    sector: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    min_score: Optional[float] = Query(None, ge=0, le=100),
    max_score: Optional[float] = Query(None, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get paginated list of companies with optional filtering"""
    try:
        company_service = CompanyService(db)
        
        search_params = CompanySearch(
            query=search,
            industry=industry,
            sector=sector,
            country=country,
            min_score=min_score,
            max_score=max_score
        )
        
        result = await company_service.get_companies(
            page=page,
            size=size,
            search_params=search_params
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching companies: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats", response_model=CompanyStats)
async def get_company_stats(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get company statistics and distribution"""
    try:
        company_service = CompanyService(db)
        stats = await company_service.get_company_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching company stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/search", response_model=List[CompanyResponse])
async def search_companies(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Search companies by name or ticker"""
    try:
        company_service = CompanyService(db)
        companies = await company_service.search_companies(query=q, limit=limit)
        return companies
        
    except Exception as e:
        logger.error(f"Error searching companies: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{company_id}", response_model=CompanyResponse)
async def get_company(
    company_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get company by ID with detailed information"""
    try:
        company_service = CompanyService(db)
        company = await company_service.get_company_by_id(company_id)
        
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
            
        return company
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{company_id}/peers", response_model=List[CompanyResponse])
async def get_company_peers(
    company_id: int = Path(..., ge=1),
    limit: int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get peer companies for comparison"""
    try:
        company_service = CompanyService(db)
        peers = await company_service.get_peer_companies(company_id, limit=limit)
        return peers
        
    except Exception as e:
        logger.error(f"Error fetching peers for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/{company_id}/risk-factors", response_model=List[str])
async def get_company_risk_factors(
    company_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get key risk factors for a company"""
    try:
        company_service = CompanyService(db)
        risk_factors = await company_service.get_risk_factors(company_id)
        return risk_factors
        
    except Exception as e:
        logger.error(f"Error fetching risk factors for company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/", response_model=CompanyResponse)
async def create_company(
    company: CompanyCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new company"""
    try:
        company_service = CompanyService(db)
        
        # Check if company already exists
        existing = await company_service.get_company_by_ticker(company.ticker)
        if existing:
            raise HTTPException(
                status_code=400, 
                detail=f"Company with ticker {company.ticker} already exists"
            )
        
        new_company = await company_service.create_company(company)
        return new_company
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating company: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/{company_id}", response_model=CompanyResponse)
async def update_company(
    company_id: int = Path(..., ge=1),
    company_update: CompanyUpdate = ...,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update company information"""
    try:
        company_service = CompanyService(db)
        
        updated_company = await company_service.update_company(company_id, company_update)
        if not updated_company:
            raise HTTPException(status_code=404, detail="Company not found")
            
        return updated_company
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{company_id}")
async def delete_company(
    company_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Soft delete a company (mark as inactive)"""
    try:
        company_service = CompanyService(db)
        
        success = await company_service.delete_company(company_id)
        if not success:
            raise HTTPException(status_code=404, detail="Company not found")
            
        return {"message": "Company deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting company {company_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/industry/{industry}/top-performers", response_model=List[CompanyResponse])
async def get_industry_top_performers(
    industry: str = Path(...),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get top performing companies in an industry"""
    try:
        company_service = CompanyService(db)
        top_performers = await company_service.get_industry_top_performers(
            industry=industry, 
            limit=limit
        )
        return top_performers
        
    except Exception as e:
        logger.error(f"Error fetching top performers for industry {industry}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/watchlist", response_model=List[CompanyResponse])
async def get_watchlist(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get user's watchlist companies"""
    try:
        company_service = CompanyService(db)
        # In a real implementation, this would be user-specific
        watchlist = await company_service.get_watchlist_companies(current_user.get("id"))
        return watchlist
        
    except Exception as e:
        logger.error(f"Error fetching watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/{company_id}/watchlist")
async def add_to_watchlist(
    company_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Add company to user's watchlist"""
    try:
        company_service = CompanyService(db)
        success = await company_service.add_to_watchlist(
            user_id=current_user.get("id"),
            company_id=company_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Company not found")
            
        return {"message": "Company added to watchlist"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding company {company_id} to watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/{company_id}/watchlist")
async def remove_from_watchlist(
    company_id: int = Path(..., ge=1),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Remove company from user's watchlist"""
    try:
        company_service = CompanyService(db)
        success = await company_service.remove_from_watchlist(
            user_id=current_user.get("id"),
            company_id=company_id
        )
        
        return {"message": "Company removed from watchlist"}
        
    except Exception as e:
        logger.error(f"Error removing company {company_id} from watchlist: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
