import asyncio
import json
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import List

# Import from organized modules
from .config.sources_config import ConfigManager
from .config.company_registry import CompanyRegistry
from .storage.database_manager import DatabaseManager
from .storage.s3_manager import S3Manager
from .data_processing.data_models import DataPoint

# Import all scrapers from organized modules
from .scrapers.news_scrapers import NewsApiScraper, RSSFeedScraper
from .scrapers.social_scrapers import TwitterScraper, RedditScraper
from .scrapers.financial_scrapers import FinancialDataScraper
from .scrapers.regulatory_scrapers import SECFilingScraper, WebScraper
from .scrapers.international_scrapers import InternationalScraper
from .scrapers.alternative_scrapers import AlternativeDataScraper, MacroeconomicScraper

# Import specialized scrapers (your existing implementation)
from .specialized_scrapers import RegulatoryScraper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLakeManager:
    """Combined data lake manager for PostgreSQL and optional S3 operations"""
    def __init__(self, config: dict):
        logger.info("🔧 Initializing DataLakeManager...")
        self.config = config
        self.s3_manager = None
        self.db_manager = None
        
        # Initialize PostgreSQL database (required)
        logger.info("Attempting to initialize PostgreSQL database...")
        try:
            self.db_manager = DatabaseManager(config)
            logger.info("✅ PostgreSQL DatabaseManager initialized successfully")
        except Exception as e:
            logger.error(f"PostgreSQL initialization failed: {e}")
            raise Exception(f"Database connection required but failed: {e}")
        
        # Initialize S3 (optional for backup/archival)
        logger.info("☁️ Checking for AWS S3 configuration...")
        try:
            if config['aws']['access_key']:
                logger.info("AWS credentials found, initializing S3Manager...")
                self.s3_manager = S3Manager(config)
                logger.info("S3Manager initialized successfully (optional backup)")
            else:
                logger.info("No AWS credentials found, S3 backup disabled")
        except Exception as e:
            logger.warning(f"S3Manager initialization failed (optional): {e}")

    def store_to_s3(self, data_points: List[DataPoint]) -> bool:
        if not self.s3_manager:
            logger.debug("S3 storage skipped - no credentials configured")
            return True  # S3 is optional
        return self.s3_manager.store_data(data_points)
    
    def store_to_database(self, data_points: List[DataPoint]) -> bool:
        if not self.db_manager:
            logger.error("Database storage failed - no database connection")
            return False  # Database is required
        return self.db_manager.store_to_database(data_points)
    
    def store_data(self, data_points: List[DataPoint]) -> bool:
        """Store data with PostgreSQL as primary and S3 as optional backup"""
        success = True
        
        # Primary storage: PostgreSQL (required)
        db_success = self.store_to_database(data_points)
        if not db_success:
            logger.error("Failed to store data to PostgreSQL database")
            success = False
        
        # Secondary storage: S3 (optional backup)
        s3_success = self.store_to_s3(data_points)
        if not s3_success and self.s3_manager:
            logger.warning("Failed to backup data to S3")
        
        return success


class DataIngestionPipeline:
    """Basic data ingestion pipeline with core scrapers"""
    
    def __init__(self, config_path: str = "config/config.json"):
        logger.info("Initializing DataIngestionPipeline...")
        logger.info(f"Loading configuration from: {config_path}")

        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        logger.info("Configuration loaded successfully")

        logger.info("Initializing company registry...")
        self.company_registry = CompanyRegistry()
        logger.info("Company registry initialized")

        logger.info("Initializing data lake manager...")
        self.data_lake = DataLakeManager(self.config)
        logger.info("Data lake manager initialized")

        # Initialize core scrapers
        logger.info("Initializing core scrapers...")
        
        logger.info("Initializing news scraper...")
        self.news_scraper = NewsApiScraper(self.config['apis']['newsapi_key'])
        
        logger.info("Initializing RSS scraper...")
        self.rss_scraper = RSSFeedScraper(self.config['sources']['rss_feeds'])

        logger.info("Initializing Twitter scraper...")
        self.twitter_scraper = TwitterScraper(self.config['apis']['twitter_bearer_token'])
        
        logger.info("Initializing Reddit scraper...")
        self.reddit_scraper = RedditScraper(
            self.config['apis']['reddit_client_id'],
            self.config['apis']['reddit_client_secret']
        )

        logger.info("Initializing financial scraper...")
        self.financial_scraper = FinancialDataScraper(
            self.config['apis']['alpha_vantage_key'],
            self.config['apis']['fred_key']
        )
        
        logger.info("Initializing SEC scraper...")
        self.sec_scraper = SECFilingScraper()
        
        logger.info("Initializing web scraper...")
        self.web_scraper = WebScraper()
        
        logger.info("All core scrapers initialized successfully")
    
    async def run_scraping_cycle(self) -> List[DataPoint]:
        """Run one complete scraping cycle across all core sources"""
        logger.info("Starting scraping cycle...")
        all_data_points = []
        
        logger.info("Getting company tickers...")
        tickers = self.company_registry.get_all_tickers()
        logger.info(f"Found {len(tickers)} company tickers to process")
        
        # Define core scraping tasks
        logger.info("Setting up scraping tasks...")
        tasks = [
            self.news_scraper.scrape_news(tickers, ['en', 'es', 'fr', 'de', 'ja', 'zh']),
            self.rss_scraper.scrape_feeds(self.company_registry),
            self.twitter_scraper.scrape_tweets(tickers),
            self.reddit_scraper.scrape_reddit(tickers),
            self.financial_scraper.scrape_yahoo_finance(tickers),
            self.financial_scraper.scrape_alpha_vantage(tickers),
            self.sec_scraper.scrape_recent_filings(tickers),
            self.web_scraper.scrape_company_websites(self.company_registry.companies)
        ]
        logger.info(f"Created {len(tasks)} scraping tasks")
        
        # Run tasks concurrently
        logger.info("Executing scraping tasks concurrently...")
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"Received {len(results)} task results")

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Scraping task {i+1} failed: {result}")
                else:
                    data_count = len(result) if result else 0
                    logger.info(f"Task {i+1} completed: {data_count} data points")
                    all_data_points.extend(result)

            logger.info(f"Total collected: {len(all_data_points)} data points")

            # Store data
            if all_data_points:
                logger.info("Storing collected data...")
                success = self.data_lake.store_data(all_data_points)
                if success:
                    logger.info("Data stored successfully")
                else:
                    logger.error("Failed to store data")
            else:
                logger.warning("No data points collected to store")

            return all_data_points
        
        except Exception as e:
            logger.error(f"Critical error in scraping cycle: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    def start_continuous_monitoring(self):
        """Start the 24/7 monitoring system"""
        logger.info("Starting continuous data monitoring...")
        
        def run_cycle():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_scraping_cycle())
            finally:
                loop.close()
        
        # Schedule regular scraping
        schedule.every(self.config['processing']['update_frequency']).seconds.do(run_cycle)
        
        # Run immediately
        run_cycle()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


class EnhancedDataIngestionPipeline(DataIngestionPipeline):
    """Enhanced pipeline with all scrapers including specialized ones"""
    
    def __init__(self, config_path: str = "config/config.json"):
        super().__init__(config_path)
        
        # Initialize enhanced scrapers
        try:
            # International and alternative scrapers from organized modules
            self.international_scraper = InternationalScraper()
            self.alternative_scraper = AlternativeDataScraper()
            self.macro_scraper = MacroeconomicScraper(
                self.config['apis']['fred_key'],
                self.config['apis']['world_bank_key']
            )
            
            # Regulatory scraper from specialized_scrapers.py
            self.regulatory_scraper = RegulatoryScraper()
            
        except ImportError as e:
            logger.warning(f"Could not import specialized scrapers: {e}")
            self.international_scraper = None
            self.alternative_scraper = None
            self.regulatory_scraper = None
            self.macro_scraper = None

    def _create_regulatory_scraper(self):
        """Create regulatory data scraper"""
        class RegulatoryScraper:
            async def scrape_regulatory_data(self, companies: List[str]) -> List[DataPoint]:
                data_points = []

                # SEC filings
                filing_types = ['10-K', '10-Q', '8-K']
                for company in companies[:20]:  # Limit to top 20
                    for filing_type in filing_types:
                        data_point = DataPoint(
                            source_type='regulatory_filing',
                            source_name='sec_edgar',
                            company_ticker=company,
                            company_name=None,
                            content_type='regulatory_filing',
                            language='en',
                            title=f'{company} {filing_type} Filing',
                            content=f'SEC {filing_type} filing submitted by {company}',
                            url=f'https://www.sec.gov/edgar/search/#/ciks={company}',
                            published_date=datetime.utcnow() - timedelta(days=30),
                            metadata={'filing_type': filing_type, 'regulator': 'SEC'}
                        )
                        data_points.append(data_point)
                
                return data_points
        
        return RegulatoryScraper()
    
    
    async def run_enhanced_scraping_cycle(self) -> List[DataPoint]:
        """Enhanced scraping cycle with all data sources"""
        all_data_points = []
        tickers = self.company_registry.get_all_tickers()
        
        # Base scraping tasks (from parent class)
        base_tasks = [
            self.news_scraper.scrape_news(tickers, ['en', 'es', 'fr', 'de', 'ja', 'zh']),
            self.rss_scraper.scrape_feeds(self.company_registry),
            self.twitter_scraper.scrape_tweets(tickers),
            self.reddit_scraper.scrape_reddit(tickers),
            self.financial_scraper.scrape_yahoo_finance(tickers),
            self.financial_scraper.scrape_alpha_vantage(tickers),
            self.sec_scraper.scrape_recent_filings(tickers),
            self.web_scraper.scrape_company_websites(self.company_registry.companies)
        ]
        
        # Enhanced tasks
        enhanced_tasks = []
        
        # Add international scraping tasks
        if self.international_scraper:
            enhanced_tasks.extend([
                self.international_scraper.scrape_international_news('de', self.company_registry),
                self.international_scraper.scrape_international_news('fr', self.company_registry),
                self.international_scraper.scrape_international_news('es', self.company_registry)
            ])
        
        # Add alternative data tasks
        if self.alternative_scraper:
            enhanced_tasks.extend([
                self.alternative_scraper.scrape_satellite_data(tickers),
                self.alternative_scraper.scrape_shipping_data(tickers),
                self.alternative_scraper.scrape_patent_data(tickers),
                self.alternative_scraper.scrape_job_postings(tickers)
            ])
        
        # Add regulatory tasks
        if self.regulatory_scraper:
            enhanced_tasks.extend([
                self.regulatory_scraper.scrape_regulatory_data('US', tickers),
                self.regulatory_scraper.scrape_regulatory_data('UK', tickers),
                self.regulatory_scraper.scrape_regulatory_data('EU', tickers)
            ])
        
        # Add macroeconomic tasks
        if self.macro_scraper:
            enhanced_tasks.append(
                self.macro_scraper.scrape_economic_indicators(['US', 'EU', 'UK'])
            )
        
        # Combine all tasks
        all_tasks = base_tasks + enhanced_tasks
        
        try:
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Enhanced scraping task failed: {result}")
                else:
                    all_data_points.extend(result)
            
            logger.info(f"Enhanced pipeline collected {len(all_data_points)} data points from {len(all_tasks)} sources")
            
            # Store enhanced data with categorization
            if all_data_points:
                await self._store_enhanced_data(all_data_points)
            
            return all_data_points
        
        except Exception as e:
            logger.error(f"Error in enhanced scraping cycle: {e}")
            return []
    
    async def _store_enhanced_data(self, data_points: List[DataPoint]):
        """Store data with enhanced categorization and metrics"""
        # Categorize data by type for better organization
        categories = {
            'news': [],
            'social': [],
            'financial': [],
            'regulatory': [],
            'alternative': [],
            'macroeconomic': []
        }
        
        for dp in data_points:
            if dp.content_type in ['news', 'financial_news']:
                categories['news'].append(dp)
            elif dp.content_type == 'social':
                categories['social'].append(dp)
            elif dp.content_type in ['financial_metrics', 'company_overview']:
                categories['financial'].append(dp)
            elif dp.content_type in ['regulatory_filing', 'regulatory_news']:
                categories['regulatory'].append(dp)
            elif dp.content_type == 'alternative_data':
                categories['alternative'].append(dp)
            elif dp.content_type == 'economic_indicator':
                categories['macroeconomic'].append(dp)
        
        # Log statistics
        for category, data in categories.items():
            if data:
                logger.info(f"Collected {len(data)} {category} data points")
        
        # Store all data
        self.data_lake.store_to_s3(data_points)
        self.data_lake.store_to_database(data_points)
    
    def start_enhanced_monitoring(self):
        """Start enhanced 24/7 monitoring with all data sources"""
        logger.info("Starting enhanced continuous data monitoring with international and alternative data sources...")
        
        def run_enhanced_cycle():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.run_enhanced_scraping_cycle())
            finally:
                loop.close()
        
        # Schedule enhanced scraping
        schedule.every(self.config['processing']['update_frequency']).seconds.do(run_enhanced_cycle)
        
        # Run immediately
        run_enhanced_cycle()
        
        # Keep running with enhanced monitoring
        while True:
            schedule.run_pending()
            time.sleep(60)


def main():
    """Main entry point - uses Enhanced Pipeline by default"""
    logger.info("🎯 ===== STARTING CREDIT INTELLIGENCE PIPELINE =====")
    logger.info("🔧 Initializing Enhanced Data Ingestion Pipeline...")
    
    try:
        logger.info("Creating pipeline instance...")
        pipeline = EnhancedDataIngestionPipeline()
        logger.info("Pipeline initialized successfully")
        
        # For testing, run one enhanced cycle
        logger.info("Setting up async event loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        logger.info("Event loop created")

        logger.info("Starting enhanced scraping cycle...")
        data_points = loop.run_until_complete(pipeline.run_enhanced_scraping_cycle())
        logger.info(f"Enhanced test run completed with {len(data_points)} data points")
        
        # For production, start enhanced continuous monitoring
        # logger.info("Starting continuous monitoring mode...")
        # pipeline.start_enhanced_monitoring()
        
        logger.info("===== PIPELINE EXECUTION COMPLETED SUCCESSFULLY =====")
    
    except KeyboardInterrupt:
        logger.info("Enhanced pipeline stopped by user")
    except Exception as e:
        logger.error(f"Enhanced pipeline failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.error("===== PIPELINE EXECUTION FAILED =====")
        input("Press Enter to continue...")  # Keep window open to see error


def run_basic_pipeline():
    """Alternative entry point for basic pipeline only"""
    pipeline = DataIngestionPipeline()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data_points = loop.run_until_complete(pipeline.run_scraping_cycle())
        logger.info(f"Basic test run completed with {len(data_points)} data points")
    
    except KeyboardInterrupt:
        logger.info("Basic pipeline stopped by user")
    except Exception as e:
        logger.error(f"Basic pipeline failed: {e}")


if __name__ == "__main__":
    # Default to enhanced pipeline
    main()
    
    # Uncomment below to run basic pipeline instead
    # run_basic_pipeline()