"""
Main data ingestion pipeline for multilingual credit intelligence data.
Orchestrates scraping, processing, and storage of financial data.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from dataclasses import asdict

from .scrapers.news_scrapers import NewsScrapers
from .scrapers.social_scrapers import SocialScrapers
from .scrapers.financial_scrapers import FinancialScrapers
from .scrapers.regulatory_scrapers import RegulatoryScrapers
from .scrapers.international_scrapers import InternationalScrapers
from .scrapers.alternative_scrapers import AlternativeScrapers

from .data_processing.text_processor import TextProcessor
from .data_processing.language_detector import LanguageDetector
from .data_processing.entity_extractor import EntityExtractor
from .data_processing.data_cleaner import DataCleaner

from .storage.postgres_manager import PostgreSQLManager
from .config.company_registry import CompanyRegistry
from .monitoring.health_checks import HealthChecker
from .monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Main data ingestion pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.company_registry = CompanyRegistry(config.get('company_registry', {}))
        
        # Initialize scrapers
        self.news_scrapers = NewsScrapers(config.get('news_scrapers', {}))
        self.social_scrapers = SocialScrapers(config.get('social_scrapers', {}))
        self.financial_scrapers = FinancialScrapers(config.get('financial_scrapers', {}))
        self.regulatory_scrapers = RegulatoryScrapers(config.get('regulatory_scrapers', {}))
        self.international_scrapers = InternationalScrapers(config.get('international_scrapers', {}))
        self.alternative_scrapers = AlternativeScrapers(config.get('alternative_scrapers', {}))
        
        # Initialize processors
        self.text_processor = TextProcessor(config.get('text_processing', {}))
        self.language_detector = LanguageDetector(config.get('language_detection', {}))
        self.entity_extractor = EntityExtractor(config.get('entity_extraction', {}))
        self.data_cleaner = DataCleaner(config.get('data_cleaning', {}))
        
        # Initialize storage and monitoring
        self.storage = PostgreSQLManager(config.get('postgres', {}))
        self.health_checker = HealthChecker(config.get('health_checks', {}))
        self.metrics = MetricsCollector(config.get('metrics', {}))
        
        self.running = False
        self.pipeline_tasks = []
        
    async def initialize(self):
        """Initialize all pipeline components"""
        try:
            await self.storage.initialize()
            await self.company_registry.load_companies()
            await self.text_processor.initialize()
            await self.language_detector.initialize()
            await self.entity_extractor.initialize()
            
            logger.info("Data ingestion pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    async def start_pipeline(self):
        """Start the continuous data ingestion pipeline"""
        if self.running:
            logger.warning("Pipeline is already running")
            return
            
        self.running = True
        logger.info("Starting data ingestion pipeline")
        
        # Start different pipeline components
        self.pipeline_tasks = [
            asyncio.create_task(self._news_ingestion_loop()),
            asyncio.create_task(self._social_ingestion_loop()),
            asyncio.create_task(self._financial_ingestion_loop()),
            asyncio.create_task(self._regulatory_ingestion_loop()),
            asyncio.create_task(self._international_ingestion_loop()),
            asyncio.create_task(self._alternative_ingestion_loop()),
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._metrics_collection_loop())
        ]
        
        try:
            await asyncio.gather(*self.pipeline_tasks)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            await self.stop_pipeline()
    
    async def stop_pipeline(self):
        """Stop the data ingestion pipeline"""
        self.running = False
        logger.info("Stopping data ingestion pipeline")
        
        # Cancel all tasks
        for task in self.pipeline_tasks:
            if not task.done():
                task.cancel()
                
        # Wait for tasks to complete
        if self.pipeline_tasks:
            await asyncio.gather(*self.pipeline_tasks, return_exceptions=True)
            
        self.pipeline_tasks = []
        logger.info("Data ingestion pipeline stopped")
    
    async def _news_ingestion_loop(self):
        """Continuous news data ingestion"""
        interval = self.config.get('news_interval', 300)  # 5 minutes
        
        while self.running:
            try:
                companies = await self.company_registry.get_active_companies()
                
                for company in companies:
                    await self._ingest_news_data(company)
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in news ingestion loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _social_ingestion_loop(self):
        """Continuous social media data ingestion"""
        interval = self.config.get('social_interval', 600)  # 10 minutes
        
        while self.running:
            try:
                companies = await self.company_registry.get_active_companies()
                
                for company in companies:
                    await self._ingest_social_data(company)
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in social ingestion loop: {e}")
                await asyncio.sleep(60)
    
    async def _financial_ingestion_loop(self):
        """Continuous financial data ingestion"""
        interval = self.config.get('financial_interval', 900)  # 15 minutes
        
        while self.running:
            try:
                companies = await self.company_registry.get_active_companies()
                
                for company in companies:
                    await self._ingest_financial_data(company)
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in financial ingestion loop: {e}")
                await asyncio.sleep(60)
    
    async def _regulatory_ingestion_loop(self):
        """Continuous regulatory data ingestion"""
        interval = self.config.get('regulatory_interval', 3600)  # 1 hour
        
        while self.running:
            try:
                companies = await self.company_registry.get_active_companies()
                
                for company in companies:
                    await self._ingest_regulatory_data(company)
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in regulatory ingestion loop: {e}")
                await asyncio.sleep(300)
    
    async def _international_ingestion_loop(self):
        """Continuous international data ingestion"""
        interval = self.config.get('international_interval', 1800)  # 30 minutes
        
        while self.running:
            try:
                companies = await self.company_registry.get_active_companies()
                
                for company in companies:
                    await self._ingest_international_data(company)
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in international ingestion loop: {e}")
                await asyncio.sleep(300)
    
    async def _alternative_ingestion_loop(self):
        """Continuous alternative data ingestion"""
        interval = self.config.get('alternative_interval', 7200)  # 2 hours
        
        while self.running:
            try:
                companies = await self.company_registry.get_active_companies()
                
                for company in companies:
                    await self._ingest_alternative_data(company)
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alternative ingestion loop: {e}")
                await asyncio.sleep(600)
    
    async def _ingest_news_data(self, company: Dict[str, Any]):
        """Ingest news data for a company"""
        try:
            company_name = company['name']
            search_terms = company.get('search_terms', [company_name])
            
            for search_term in search_terms:
                # Scrape news from multiple sources
                articles = await self.news_scrapers.scrape_all_sources(
                    query=search_term,
                    languages=company.get('languages', ['en'])
                )
                
                # Process each article
                for article in articles:
                    await self._process_and_store_article(article, company_name, 'news')
                    
            self.metrics.increment_counter('news_ingestion_completed', {'company': company_name})
            
        except Exception as e:
            logger.error(f"Error ingesting news data for {company.get('name', 'unknown')}: {e}")
            self.metrics.increment_counter('news_ingestion_errors', {'company': company.get('name', 'unknown')})
    
    async def _ingest_social_data(self, company: Dict[str, Any]):
        """Ingest social media data for a company"""
        try:
            company_name = company['name']
            
            # Twitter data
            tweets = await self.social_scrapers.scrape_twitter(
                query=company_name,
                count=self.config.get('twitter_count', 100)
            )
            
            for tweet in tweets:
                await self._process_and_store_social_post(tweet, company_name, 'twitter')
            
            # Reddit data
            reddit_posts = await self.social_scrapers.scrape_reddit(
                subreddits=['investing', 'stocks', 'finance'],
                query=company_name
            )
            
            for post in reddit_posts:
                await self._process_and_store_social_post(post, company_name, 'reddit')
                
            self.metrics.increment_counter('social_ingestion_completed', {'company': company_name})
            
        except Exception as e:
            logger.error(f"Error ingesting social data for {company.get('name', 'unknown')}: {e}")
            self.metrics.increment_counter('social_ingestion_errors', {'company': company.get('name', 'unknown')})
    
    async def _ingest_financial_data(self, company: Dict[str, Any]):
        """Ingest financial data for a company"""
        try:
            company_name = company['name']
            ticker = company.get('ticker')
            
            if not ticker:
                logger.warning(f"No ticker found for {company_name}")
                return
            
            # Stock data
            stock_data = await self.financial_scrapers.get_stock_data(ticker)
            if stock_data:
                await self.storage.store_raw_data(
                    data=stock_data,
                    data_type='stock_data',
                    company=company_name,
                    source='financial_api'
                )
            
            # Financial statements
            financials = await self.financial_scrapers.get_financial_statements(ticker)
            if financials:
                await self.storage.store_raw_data(
                    data=financials,
                    data_type='financial_statements',
                    company=company_name,
                    source='financial_api'
                )
                
            self.metrics.increment_counter('financial_ingestion_completed', {'company': company_name})
            
        except Exception as e:
            logger.error(f"Error ingesting financial data for {company.get('name', 'unknown')}: {e}")
            self.metrics.increment_counter('financial_ingestion_errors', {'company': company.get('name', 'unknown')})
    
    async def _ingest_regulatory_data(self, company: Dict[str, Any]):
        """Ingest regulatory data for a company"""
        try:
            company_name = company['name']
            
            # SEC filings
            sec_filings = await self.regulatory_scrapers.get_sec_filings(company_name)
            for filing in sec_filings:
                await self.storage.store_raw_data(
                    data=filing,
                    data_type='sec_filing',
                    company=company_name,
                    source='sec'
                )
            
            # Other regulatory sources
            regulatory_data = await self.regulatory_scrapers.get_regulatory_updates(company_name)
            for data in regulatory_data:
                await self.storage.store_raw_data(
                    data=data,
                    data_type='regulatory_update',
                    company=company_name,
                    source='regulatory'
                )
                
            self.metrics.increment_counter('regulatory_ingestion_completed', {'company': company_name})
            
        except Exception as e:
            logger.error(f"Error ingesting regulatory data for {company.get('name', 'unknown')}: {e}")
            self.metrics.increment_counter('regulatory_ingestion_errors', {'company': company.get('name', 'unknown')})
    
    async def _ingest_international_data(self, company: Dict[str, Any]):
        """Ingest international data for a company"""
        try:
            company_name = company['name']
            
            # International news sources
            intl_articles = await self.international_scrapers.scrape_international_sources(company_name)
            for article in intl_articles:
                await self._process_and_store_article(article, company_name, 'international_news')
                
            self.metrics.increment_counter('international_ingestion_completed', {'company': company_name})
            
        except Exception as e:
            logger.error(f"Error ingesting international data for {company.get('name', 'unknown')}: {e}")
            self.metrics.increment_counter('international_ingestion_errors', {'company': company.get('name', 'unknown')})
    
    async def _ingest_alternative_data(self, company: Dict[str, Any]):
        """Ingest alternative data for a company"""
        try:
            company_name = company['name']
            
            # Satellite data, patent data, job postings, etc.
            alt_data = await self.alternative_scrapers.scrape_alternative_sources(company_name)
            for data in alt_data:
                await self.storage.store_raw_data(
                    data=data,
                    data_type='alternative_data',
                    company=company_name,
                    source='alternative'
                )
                
            self.metrics.increment_counter('alternative_ingestion_completed', {'company': company_name})
            
        except Exception as e:
            logger.error(f"Error ingesting alternative data for {company.get('name', 'unknown')}: {e}")
            self.metrics.increment_counter('alternative_ingestion_errors', {'company': company.get('name', 'unknown')})
    
    async def _process_and_store_article(self, article, company_name: str, data_type: str):
        """Process and store a news article"""
        try:
            # Detect language
            language = await self.language_detector.detect_language(article.content)
            article.language = language
            
            # Clean and process text
            processed_text = await self.text_processor.process_text(
                article.content, 
                language=language
            )
            
            # Extract entities
            entities = await self.entity_extractor.extract_entities(
                processed_text, 
                language=language
            )
            article.entities = entities
            
            # Clean data
            cleaned_article = await self.data_cleaner.clean_article_data(asdict(article))
            
            # Store in database
            await self.storage.store_raw_data(
                data=cleaned_article,
                data_type=data_type,
                company=company_name,
                source=article.source,
                language=language
            )
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
    
    async def _process_and_store_social_post(self, post, company_name: str, platform: str):
        """Process and store a social media post"""
        try:
            # Detect language
            language = await self.language_detector.detect_language(post.get('text', ''))
            
            # Process text
            processed_text = await self.text_processor.process_text(
                post.get('text', ''), 
                language=language
            )
            
            # Extract entities
            entities = await self.entity_extractor.extract_entities(
                processed_text, 
                language=language
            )
            
            # Clean data
            cleaned_post = await self.data_cleaner.clean_social_data(post)
            cleaned_post['entities'] = entities
            cleaned_post['processed_text'] = processed_text
            
            # Store in database
            await self.storage.store_raw_data(
                data=cleaned_post,
                data_type=f'social_{platform}',
                company=company_name,
                source=platform,
                language=language
            )
            
        except Exception as e:
            logger.error(f"Error processing social post: {e}")
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        interval = self.config.get('health_check_interval', 300)  # 5 minutes
        
        while self.running:
            try:
                health_status = await self.health_checker.check_pipeline_health()
                
                if not health_status['healthy']:
                    logger.warning(f"Pipeline health issues detected: {health_status}")
                    
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection"""
        interval = self.config.get('metrics_interval', 60)  # 1 minute
        
        while self.running:
            try:
                await self.metrics.collect_pipeline_metrics()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(60)
    
    async def run_single_ingestion(self, company_name: str, data_types: List[str] = None):
        """Run a single ingestion cycle for testing/debugging"""
        if data_types is None:
            data_types = ['news', 'social', 'financial', 'regulatory']
            
        company = await self.company_registry.get_company(company_name)
        if not company:
            logger.error(f"Company {company_name} not found in registry")
            return
            
        tasks = []
        
        if 'news' in data_types:
            tasks.append(self._ingest_news_data(company))
        if 'social' in data_types:
            tasks.append(self._ingest_social_data(company))
        if 'financial' in data_types:
            tasks.append(self._ingest_financial_data(company))
        if 'regulatory' in data_types:
            tasks.append(self._ingest_regulatory_data(company))
        if 'international' in data_types:
            tasks.append(self._ingest_international_data(company))
        if 'alternative' in data_types:
            tasks.append(self._ingest_alternative_data(company))
            
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Single ingestion completed for {company_name}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'running': self.running,
            'active_tasks': len([t for t in self.pipeline_tasks if not t.done()]),
            'health_status': await self.health_checker.check_pipeline_health(),
            'metrics': await self.metrics.get_current_metrics(),
            'storage_stats': await self.storage.get_data_statistics()
        }
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        await self.stop_pipeline()
        await self.storage.close()
        logger.info("Pipeline cleanup completed")
