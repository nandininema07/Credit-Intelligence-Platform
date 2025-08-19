"""
Scrapers module for multilingual data ingestion.
Contains specialized scrapers for different data sources.
"""

from .news_scrapers import NewsScrapers
from .social_scrapers import SocialScrapers
from .financial_scrapers import FinancialScrapers
from .regulatory_scrapers import RegulatoryScrapers
from .international_scrapers import InternationalScrapers
from .alternative_scrapers import AlternativeScrapers

__all__ = [
    'NewsScrapers',
    'SocialScrapers', 
    'FinancialScrapers',
    'RegulatoryScrapers',
    'InternationalScrapers',
    'AlternativeScrapers'
]
