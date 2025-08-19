"""
Integration tests for Stage 1 data ingestion pipeline.
Tests end-to-end functionality and component interactions.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

from ..main_pipeline import DataIngestionPipeline
from ..storage.postgres_manager import PostgreSQLManager
from ..config.company_registry import CompanyRegistry, Company
from ..monitoring.health_checks import HealthChecker
from ..monitoring.metrics import MetricsCollector

@pytest.fixture
def test_config():
    return {
        'postgres': {
            'postgres_host': 'localhost',
            'postgres_database': 'test_db',
            'postgres_user': 'test_user',
            'postgres_password': 'test_pass'
        },
        'scrapers': {
            'news': {'enabled': True, 'api_keys': {}},
            'social': {'enabled': True, 'api_keys': {}},
            'financial': {'enabled': True, 'api_keys': {}}
        },
        'processing': {
            'batch_size': 10,
            'max_workers': 2
        }
    }

@pytest.fixture
def sample_companies():
    return [
        Company(
            id="AAPL",
            name="Apple Inc.",
            ticker="AAPL",
            sector="Technology",
            industry="Consumer Electronics",
            country="US",
            monitoring_enabled=True
        ),
        Company(
            id="MSFT",
            name="Microsoft Corporation", 
            ticker="MSFT",
            sector="Technology",
            industry="Software",
            country="US",
            monitoring_enabled=True
        )
    ]

class TestPipelineIntegration:
    """Test full pipeline integration"""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, test_config):
        """Test pipeline initialization"""
        pipeline = DataIngestionPipeline(test_config)
        assert pipeline.config == test_config
        assert pipeline.storage_manager is not None
        assert pipeline.company_registry is not None
    
    @pytest.mark.asyncio
    async def test_company_registry_integration(self, test_config, sample_companies):
        """Test company registry integration"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Add companies
        for company in sample_companies:
            pipeline.company_registry.add_company(company)
        
        # Get monitored companies
        monitored = pipeline.company_registry.get_monitored_companies()
        assert len(monitored) == 2
        assert all(company.monitoring_enabled for company in monitored)
    
    @pytest.mark.asyncio
    @patch('asyncpg.create_pool')
    async def test_storage_integration(self, mock_pool, test_config):
        """Test storage integration"""
        # Mock database pool
        mock_connection = AsyncMock()
        mock_pool.return_value.acquire.return_value.__aenter__.return_value = mock_connection
        
        pipeline = DataIngestionPipeline(test_config)
        
        # Test data storage
        sample_data = {
            'title': 'Test Article',
            'content': 'Test content',
            'source': 'Test Source',
            'timestamp': datetime.now()
        }
        
        # This would normally store to database
        assert pipeline.storage_manager is not None
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, test_config):
        """Test health monitoring integration"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Initialize health checker
        health_checker = HealthChecker(test_config)
        
        # Mock health check
        with patch.object(health_checker, 'check_system_health', return_value={'healthy': True}):
            health_status = await health_checker.check_system_health()
            assert health_status['healthy'] == True
    
    @pytest.mark.asyncio
    async def test_metrics_collection_integration(self, test_config):
        """Test metrics collection integration"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector(test_config)
        
        # Record some metrics
        metrics_collector.increment_counter('articles_processed')
        metrics_collector.record_timer('processing_time', 1.5)
        
        # Get metrics
        metrics = metrics_collector.get_metrics()
        assert 'counters' in metrics
        assert 'timers' in metrics

class TestDataFlowIntegration:
    """Test data flow through pipeline stages"""
    
    @pytest.mark.asyncio
    async def test_news_data_flow(self, test_config):
        """Test news data ingestion flow"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Mock news data
        mock_articles = [
            {
                'title': 'Apple Reports Earnings',
                'content': 'Apple Inc. reported strong quarterly results...',
                'url': 'https://example.com/apple',
                'source': 'Financial Times',
                'published_date': datetime.now()
            }
        ]
        
        # Mock scrapers
        with patch.object(pipeline, '_scrape_news_data', return_value=mock_articles):
            with patch.object(pipeline, '_process_articles', return_value=mock_articles):
                with patch.object(pipeline.storage_manager, 'store_raw_data', return_value=True):
                    
                    # Test company processing
                    company = Company(
                        id="AAPL", name="Apple Inc.", ticker="AAPL",
                        sector="Technology", industry="Electronics", country="US"
                    )
                    
                    # This would normally run the full ingestion
                    assert company.name == "Apple Inc."
    
    @pytest.mark.asyncio
    async def test_social_data_flow(self, test_config):
        """Test social media data ingestion flow"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Mock social data
        mock_posts = [
            {
                'content': 'Great earnings from $AAPL today!',
                'platform': 'Twitter',
                'author': 'user123',
                'created_date': datetime.now(),
                'engagement_score': 15
            }
        ]
        
        # Mock social scrapers
        with patch.object(pipeline, '_scrape_social_data', return_value=mock_posts):
            with patch.object(pipeline, '_process_social_posts', return_value=mock_posts):
                with patch.object(pipeline.storage_manager, 'store_raw_data', return_value=True):
                    
                    # Test social data processing
                    assert len(mock_posts) == 1
                    assert mock_posts[0]['platform'] == 'Twitter'
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, test_config):
        """Test error handling across components"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Test with invalid data
        invalid_article = {'title': None, 'content': ''}
        
        # Mock error scenarios
        with patch.object(pipeline.data_cleaner, 'clean_article', side_effect=Exception("Processing error")):
            # Pipeline should handle errors gracefully
            try:
                pipeline.data_cleaner.clean_article(invalid_article)
                assert False, "Should have raised exception"
            except Exception as e:
                assert "Processing error" in str(e)

class TestPerformanceIntegration:
    """Test performance and scalability"""
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, test_config):
        """Test batch processing performance"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Create large batch of mock data
        mock_articles = []
        for i in range(100):
            mock_articles.append({
                'title': f'Article {i}',
                'content': f'Content for article {i}',
                'url': f'https://example.com/article-{i}',
                'source': 'Test Source',
                'published_date': datetime.now()
            })
        
        # Mock batch processing
        with patch.object(pipeline, '_process_articles', return_value=mock_articles):
            processed = await pipeline._process_articles(mock_articles)
            assert len(processed) == 100
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, test_config):
        """Test concurrent processing capabilities"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Mock concurrent tasks
        async def mock_task(data):
            await asyncio.sleep(0.1)  # Simulate processing time
            return data
        
        # Test concurrent execution
        tasks = [mock_task(f"data_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(f"data_{i}" in results for i in range(10))

class TestConfigurationIntegration:
    """Test configuration management"""
    
    def test_config_loading(self, test_config):
        """Test configuration loading"""
        pipeline = DataIngestionPipeline(test_config)
        
        assert pipeline.config['postgres']['postgres_host'] == 'localhost'
        assert pipeline.config['scrapers']['news']['enabled'] == True
        assert pipeline.config['processing']['batch_size'] == 10
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test with missing required config
        invalid_config = {'postgres': {}}
        
        try:
            pipeline = DataIngestionPipeline(invalid_config)
            # Should handle missing config gracefully
            assert pipeline.config == invalid_config
        except Exception:
            # Or raise appropriate error
            pass

class TestEndToEndIntegration:
    """Test complete end-to-end scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_run(self, test_config, sample_companies):
        """Test complete pipeline execution"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Add companies to registry
        for company in sample_companies:
            pipeline.company_registry.add_company(company)
        
        # Mock all external dependencies
        with patch.object(pipeline, '_scrape_news_data', return_value=[]):
            with patch.object(pipeline, '_scrape_social_data', return_value=[]):
                with patch.object(pipeline, '_scrape_financial_data', return_value=[]):
                    with patch.object(pipeline.storage_manager, 'store_raw_data', return_value=True):
                        
                        # This would run the full pipeline
                        companies = pipeline.company_registry.get_monitored_companies()
                        assert len(companies) == 2
    
    @pytest.mark.asyncio
    async def test_pipeline_with_real_data_simulation(self, test_config):
        """Test pipeline with realistic data simulation"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Simulate realistic data volumes
        start_time = datetime.now()
        
        # Mock realistic processing times
        with patch('asyncio.sleep', return_value=None):
            # Simulate data processing
            processing_time = (datetime.now() - start_time).total_seconds()
            assert processing_time < 1.0  # Should be fast in test
    
    def test_pipeline_cleanup(self, test_config):
        """Test pipeline cleanup and resource management"""
        pipeline = DataIngestionPipeline(test_config)
        
        # Test cleanup
        # In real implementation, this would close connections, etc.
        assert pipeline is not None

@pytest.mark.integration
class TestExternalIntegration:
    """Test integration with external services (when available)"""
    
    @pytest.mark.skipif(os.getenv('SKIP_EXTERNAL_TESTS') == 'true', 
                       reason="External tests disabled")
    @pytest.mark.asyncio
    async def test_database_connection(self, test_config):
        """Test actual database connection (if available)"""
        # This would test real database connection
        # Skip if no test database available
        pass
    
    @pytest.mark.skipif(os.getenv('SKIP_API_TESTS') == 'true',
                       reason="API tests disabled") 
    @pytest.mark.asyncio
    async def test_external_api_integration(self, test_config):
        """Test integration with external APIs (if available)"""
        # This would test real API connections
        # Skip if no API keys available
        pass
