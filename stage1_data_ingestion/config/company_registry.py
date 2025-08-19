"""
Company registry for managing tracked companies and their metadata.
Handles company information, sector classification, and monitoring configuration.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class Company:
    """Company information"""
    id: str
    name: str
    ticker: str
    sector: str
    industry: str
    country: str
    market_cap: Optional[float] = None
    employees: Optional[int] = None
    founded_year: Optional[int] = None
    website: Optional[str] = None
    description: Optional[str] = None
    risk_tier: str = "medium"  # low, medium, high
    monitoring_enabled: bool = True
    last_updated: Optional[datetime] = None

class CompanyRegistry:
    """Registry for managing tracked companies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.companies = {}
        self.sectors = {}
        self.load_default_companies()
        
    def load_default_companies(self):
        """Load default company list"""
        default_companies = [
            Company(
                id="AAPL",
                name="Apple Inc.",
                ticker="AAPL",
                sector="Technology",
                industry="Consumer Electronics",
                country="US",
                market_cap=3000000000000,
                risk_tier="low"
            ),
            Company(
                id="MSFT",
                name="Microsoft Corporation",
                ticker="MSFT",
                sector="Technology",
                industry="Software",
                country="US",
                market_cap=2800000000000,
                risk_tier="low"
            ),
            Company(
                id="GOOGL",
                name="Alphabet Inc.",
                ticker="GOOGL",
                sector="Technology",
                industry="Internet Services",
                country="US",
                market_cap=1800000000000,
                risk_tier="low"
            ),
            Company(
                id="TSLA",
                name="Tesla Inc.",
                ticker="TSLA",
                sector="Automotive",
                industry="Electric Vehicles",
                country="US",
                market_cap=800000000000,
                risk_tier="medium"
            ),
            Company(
                id="JPM",
                name="JPMorgan Chase & Co.",
                ticker="JPM",
                sector="Financial Services",
                industry="Banking",
                country="US",
                market_cap=450000000000,
                risk_tier="medium"
            )
        ]
        
        for company in default_companies:
            self.companies[company.id] = company
            
        logger.info(f"Loaded {len(default_companies)} default companies")
    
    def add_company(self, company: Company) -> bool:
        """Add company to registry"""
        try:
            company.last_updated = datetime.now()
            self.companies[company.id] = company
            logger.info(f"Added company: {company.name} ({company.id})")
            return True
        except Exception as e:
            logger.error(f"Error adding company {company.id}: {e}")
            return False
    
    def get_company(self, company_id: str) -> Optional[Company]:
        """Get company by ID"""
        return self.companies.get(company_id)
    
    def get_companies_by_sector(self, sector: str) -> List[Company]:
        """Get companies by sector"""
        return [
            company for company in self.companies.values()
            if company.sector.lower() == sector.lower()
        ]
    
    def get_companies_by_risk_tier(self, risk_tier: str) -> List[Company]:
        """Get companies by risk tier"""
        return [
            company for company in self.companies.values()
            if company.risk_tier.lower() == risk_tier.lower()
        ]
    
    def get_monitored_companies(self) -> List[Company]:
        """Get companies with monitoring enabled"""
        return [
            company for company in self.companies.values()
            if company.monitoring_enabled
        ]
    
    def update_company(self, company_id: str, updates: Dict[str, Any]) -> bool:
        """Update company information"""
        if company_id not in self.companies:
            logger.warning(f"Company {company_id} not found for update")
            return False
        
        try:
            company = self.companies[company_id]
            for field, value in updates.items():
                if hasattr(company, field):
                    setattr(company, field, value)
            
            company.last_updated = datetime.now()
            logger.info(f"Updated company {company_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating company {company_id}: {e}")
            return False
    
    def remove_company(self, company_id: str) -> bool:
        """Remove company from registry"""
        if company_id in self.companies:
            del self.companies[company_id]
            logger.info(f"Removed company {company_id}")
            return True
        return False
    
    def enable_monitoring(self, company_id: str) -> bool:
        """Enable monitoring for company"""
        return self.update_company(company_id, {"monitoring_enabled": True})
    
    def disable_monitoring(self, company_id: str) -> bool:
        """Disable monitoring for company"""
        return self.update_company(company_id, {"monitoring_enabled": False})
    
    def get_company_list(self) -> List[Dict[str, Any]]:
        """Get list of all companies"""
        return [
            {
                "id": company.id,
                "name": company.name,
                "ticker": company.ticker,
                "sector": company.sector,
                "industry": company.industry,
                "country": company.country,
                "risk_tier": company.risk_tier,
                "monitoring_enabled": company.monitoring_enabled,
                "last_updated": company.last_updated.isoformat() if company.last_updated else None
            }
            for company in self.companies.values()
        ]
    
    def get_sector_summary(self) -> Dict[str, Any]:
        """Get summary by sector"""
        sector_stats = {}
        
        for company in self.companies.values():
            sector = company.sector
            if sector not in sector_stats:
                sector_stats[sector] = {
                    "count": 0,
                    "monitored": 0,
                    "risk_tiers": {"low": 0, "medium": 0, "high": 0}
                }
            
            sector_stats[sector]["count"] += 1
            if company.monitoring_enabled:
                sector_stats[sector]["monitored"] += 1
            
            sector_stats[sector]["risk_tiers"][company.risk_tier] += 1
        
        return sector_stats
    
    def search_companies(self, query: str) -> List[Company]:
        """Search companies by name or ticker"""
        query = query.lower()
        results = []
        
        for company in self.companies.values():
            if (query in company.name.lower() or 
                query in company.ticker.lower() or
                query in company.industry.lower()):
                results.append(company)
        
        return results
    
    def load_from_file(self, filepath: str) -> bool:
        """Load companies from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            for company_data in data:
                company = Company(**company_data)
                self.add_company(company)
            
            logger.info(f"Loaded companies from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading companies from file: {e}")
            return False
    
    def save_to_file(self, filepath: str) -> bool:
        """Save companies to JSON file"""
        try:
            company_data = []
            for company in self.companies.values():
                data = {
                    "id": company.id,
                    "name": company.name,
                    "ticker": company.ticker,
                    "sector": company.sector,
                    "industry": company.industry,
                    "country": company.country,
                    "market_cap": company.market_cap,
                    "employees": company.employees,
                    "founded_year": company.founded_year,
                    "website": company.website,
                    "description": company.description,
                    "risk_tier": company.risk_tier,
                    "monitoring_enabled": company.monitoring_enabled,
                    "last_updated": company.last_updated.isoformat() if company.last_updated else None
                }
                company_data.append(data)
            
            with open(filepath, 'w') as f:
                json.dump(company_data, f, indent=2)
            
            logger.info(f"Saved companies to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving companies to file: {e}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_companies = len(self.companies)
        monitored_companies = len(self.get_monitored_companies())
        
        return {
            "total_companies": total_companies,
            "monitored_companies": monitored_companies,
            "monitoring_rate": monitored_companies / total_companies if total_companies > 0 else 0,
            "sectors": len(set(company.sector for company in self.companies.values())),
            "countries": len(set(company.country for company in self.companies.values())),
            "risk_distribution": {
                "low": len(self.get_companies_by_risk_tier("low")),
                "medium": len(self.get_companies_by_risk_tier("medium")),
                "high": len(self.get_companies_by_risk_tier("high"))
            }
        }
