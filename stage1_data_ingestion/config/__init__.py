"""
Configuration module for Stage 1 data ingestion.
Provides configuration management and company registry.
"""

from .company_registry import CompanyRegistry, Company
from .sources_config import SourcesConfig

__all__ = [
    'CompanyRegistry',
    'Company',
    'SourcesConfig'
]
