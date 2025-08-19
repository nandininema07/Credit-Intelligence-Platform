"""
Stage 1: Data Ingestion Pipeline
Handles multilingual data ingestion, processing, and storage.
"""

from .main_pipeline import DataIngestionPipeline
from .config import CompanyRegistry, Company, SourcesConfig
from .storage.postgres_manager import PostgreSQLManager
from .data_processing import TextProcessor, LanguageDetector, EntityExtractor, DataCleaner
from .monitoring import HealthChecker, MetricsCollector, PipelineAlerting

__all__ = [
    'DataIngestionPipeline',
    'CompanyRegistry',
    'Company', 
    'SourcesConfig',
    'PostgreSQLManager',
    'TextProcessor',
    'LanguageDetector',
    'EntityExtractor',
    'DataCleaner',
    'HealthChecker',
    'MetricsCollector',
    'PipelineAlerting'
]
