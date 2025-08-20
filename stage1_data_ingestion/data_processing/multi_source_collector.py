"""
Multi-source data collector for comprehensive financial data ingestion.
Integrates NewsAPI, Twitter, Reddit, Alpha Vantage, FRED, Finnhub, Polygon, and other APIs.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
import feedparser
import yfinance as yf
from newsapi import NewsApiClient
import tweepy
import praw
import requests
import hashlib
from textblob import TextBlob
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

@dataclass
class DataPoint:
    """Standardized data point structure"""
    source_type: str
    source_name: str
    company_ticker: Optional[str]
    company_name: Optional[str]
    content_type: str
    language: Optional[str]
    title: Optional[str]
    content: str
    url: Optional[str]
    published_date: Optional[datetime]
    sentiment_score: Optional[float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        # Keep datetime objects as-is for proper database handling
        # The database layer will handle the conversion if needed
        return data

class DataProcessor:
    """Utility class for data processing"""
    
    @staticmethod
    def calculate_sentiment(text: str) -> float:
        """Calculate sentiment score using TextBlob"""
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    @staticmethod
    def extract_companies_from_text(text: str, company_tickers: List[str]) -> List[str]:
        """Extract mentioned company tickers from text"""
        mentioned = []
        text_upper = text.upper()
        for ticker in company_tickers:
            base_ticker = ticker.split('.')[0]  # Remove exchange suffix
            if base_ticker in text_upper or f"${base_ticker}" in text_upper:
                mentioned.append(ticker)
        return mentioned
    
    @staticmethod
    def generate_id(source_type: str, url: str, published_date: datetime) -> str:
        """Generate unique ID for data point"""
        if published_date:
            # Ensure timezone-aware
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=timezone.utc)
            date_str = published_date.isoformat()
        else:
            date_str = datetime.now(timezone.utc).isoformat()
        
        content = f"{source_type}_{url}_{date_str}"
        return hashlib.sha256(content.encode()).hexdigest()

class NewsApiScraper:
    """NewsAPI.org scraper for news articles"""
    
    def __init__(self, api_key: str):
        self.client = NewsApiClient(api_key=api_key) if api_key else None
        
    async def scrape_news(self, companies: List[str], languages: List[str] = ['en']) -> List[DataPoint]:
        """Scrape news articles for companies"""
        if not self.client:
            logger.warning("NewsAPI key not provided")
            return []
            
        data_points = []
        for company in companies:
            for language in languages:
                try:
                    # Add delay between requests to avoid rate limiting
                    await asyncio.sleep(0.5)  # 500ms delay between requests
                    
                    articles = self.client.get_everything(
                        q=f"{company} OR ${company}",
                        language=language,
                        sort_by='publishedAt',
                        from_param=(datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d'),
                        page_size=20
                    )
                    
                    for article in articles.get('articles', []):
                        content = f"{article.get('description', '')} {article.get('content', '')}"
                        data_point = DataPoint(
                            source_type='news_api',
                            source_name=article.get('source', {}).get('name', 'unknown'),
                            company_ticker=company,
                            company_name=None,
                            content_type='news',
                            language=language,
                            title=article.get('title'),
                            content=content,
                            url=article.get('url'),
                            published_date=datetime.fromisoformat(article.get('publishedAt').replace('Z', '+00:00')) if article.get('publishedAt') else None,
                            sentiment_score=DataProcessor.calculate_sentiment(content),
                            metadata={'author': article.get('author'), 'source': article.get('source')}
                        )
                        data_points.append(data_point)
                        
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        logger.warning(f"NewsAPI rate limit hit for {company}, skipping...")
                        await asyncio.sleep(60)  # Wait 1 minute before next request
                    else:
                        logger.error(f"Error scraping NewsAPI for {company}: {e}")
                    
        return data_points

class TwitterScraper:
    """Twitter API v2 scraper"""
    
    def __init__(self, bearer_token: str):
        self.client = tweepy.Client(bearer_token=bearer_token) if bearer_token else None
        
    async def scrape_tweets(self, companies: List[str]) -> List[DataPoint]:
        """Scrape recent tweets mentioning companies"""
        if not self.client:
            logger.warning("Twitter bearer token not provided")
            return []
            
        data_points = []
        for company in companies:
            try:
                # Add delay between requests to avoid rate limiting
                await asyncio.sleep(1)  # 1 second delay between companies
                
                tweets = self.client.search_recent_tweets(
                    query=f"{company} OR ${company} -is:retweet lang:en",
                    max_results=50,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang', 'context_annotations'],
                    user_fields=['username', 'verified', 'public_metrics'],
                    expansions=['author_id']
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        data_point = DataPoint(
                            source_type='twitter',
                            source_name='twitter',
                            company_ticker=company,
                            company_name=None,
                            content_type='social',
                            language=tweet.lang,
                            title=None,
                            content=tweet.text,
                            url=f"https://twitter.com/i/web/status/{tweet.id}",
                            published_date=tweet.created_at,
                            sentiment_score=DataProcessor.calculate_sentiment(tweet.text),
                            metadata={
                                'author_id': tweet.author_id,
                                'metrics': tweet.public_metrics,
                                'context': getattr(tweet, 'context_annotations', [])
                            }
                        )
                        data_points.append(data_point)
                        
            except Exception as e:
                if "429" in str(e):
                    logger.warning(f"Twitter rate limit hit for {company}, skipping...")
                    # Add longer delay for rate limit
                    await asyncio.sleep(60)  # Wait 1 minute before next request
                else:
                    logger.error(f"Error scraping Twitter for {company}: {e}")
                
        return data_points

class RedditScraper:
    """Reddit API scraper"""
    
    def __init__(self, client_id: str, client_secret: str):
        if client_id and client_secret:
            try:
                import praw
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="credtech_scraper/1.0"
                )
                # Test connection
                self.reddit.user.me()
                logger.info("Reddit API connection successful")
            except Exception as e:
                logger.warning(f"Reddit API connection failed: {e}")
                self.reddit = None
        else:
            self.reddit = None
            logger.warning("Reddit credentials not provided")
            
    async def scrape_reddit(self, companies: List[str]) -> List[DataPoint]:
        """Scrape Reddit posts mentioning companies"""
        if not self.reddit:
            logger.warning("Reddit client not available")
            return []
            
        data_points = []
        for company in companies:
            try:
                # Add delay between requests
                await asyncio.sleep(1)
                
                # Always use sync PRAW in thread pool to avoid async/sync issues
                loop = asyncio.get_event_loop()
                posts = await loop.run_in_executor(None, self._scrape_reddit_sync, company)
                data_points.extend(posts)
                    
            except Exception as e:
                logger.error(f"Error scraping Reddit for {company}: {e}")
                
        return data_points
    
    def _scrape_reddit_sync(self, company: str) -> List[DataPoint]:
        """Synchronous Reddit scraping for compatibility"""
        data_points = []
        try:
            subreddit = self.reddit.subreddit("investing+stocks+wallstreetbets")
            for post in subreddit.search(company, limit=20, sort='hot'):
                if post.score > 5:  # Only high-quality posts
                    data_points.append(DataPoint(
                        source_type='reddit',
                        source_name='reddit',
                        company_ticker=company,
                        company_name=None,
                        content_type='post',
                        language='en',
                        title=post.title,
                        content=post.title + '\n' + post.selftext,
                        url=post.url,
                        published_date=datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                        sentiment_score=DataProcessor.calculate_sentiment(post.title + '\n' + post.selftext),
                        metadata={
                            'subreddit': str(post.subreddit),
                            'score': post.score,
                            'comments': post.num_comments,
                            'author': str(post.author) if post.author else 'deleted'
                        }
                    ))
        except Exception as e:
            logger.error(f"Error in sync Reddit scraping for {company}: {e}")
        return data_points

class YahooFinanceScraper:
    """Yahoo Finance data scraper"""
    
    async def scrape_yahoo_finance(self, tickers: List[str]) -> List[DataPoint]:
        """Scrape Yahoo Finance data and news"""
        data_points = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Get recent news
                news = stock.news
                for article in news[:5]:
                    data_point = DataPoint(
                        source_type='yahoo_finance',
                        source_name='yahoo_finance_news',
                        company_ticker=ticker,
                        company_name=info.get('longName'),
                        content_type='financial_news',
                        language='en',
                        title=article.get('title'),
                        content=article.get('summary', ''),
                        url=article.get('link'),
                        published_date=datetime.fromtimestamp(article.get('providerPublishTime', time.time())),
                        sentiment_score=DataProcessor.calculate_sentiment(article.get('summary', '')),
                        metadata={'publisher': article.get('publisher')}
                    )
                    data_points.append(data_point)
                
                # Financial metrics
                financials_data = {
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'revenue': info.get('totalRevenue'),
                    'profit_margin': info.get('profitMargins'),
                    'current_price': info.get('currentPrice'),
                    'target_price': info.get('targetMeanPrice')
                }
                
                data_point = DataPoint(
                    source_type='yahoo_finance',
                    source_name='yahoo_finance_metrics',
                    company_ticker=ticker,
                    company_name=info.get('longName'),
                    content_type='financial_metrics',
                    language='en',
                    title=f"{ticker} Financial Metrics",
                    content=json.dumps(financials_data),
                    url=None,
                    published_date=datetime.utcnow(),
                    sentiment_score=None,
                    metadata=financials_data
                )
                data_points.append(data_point)
                
            except Exception as e:
                logger.error(f"Error scraping Yahoo Finance for {ticker}: {e}")
                
        return data_points

class AlphaVantageScraper:
    """Alpha Vantage API scraper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
    async def scrape_alpha_vantage(self, tickers: List[str]) -> List[DataPoint]:
        """Scrape Alpha Vantage company data"""
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
            return []
            
        data_points = []
        
        async with aiohttp.ClientSession() as session:
            for ticker in tickers[:10]:  # Limit due to API rate limits
                try:
                    # Company overview
                    params = {
                        'function': 'OVERVIEW',
                        'symbol': ticker,
                        'apikey': self.api_key
                    }
                    
                    async with session.get(self.base_url, params=params) as response:
                        data = await response.json()
                        
                        if 'Symbol' in data:
                            data_point = DataPoint(
                                source_type='alpha_vantage',
                                source_name='alpha_vantage_overview',
                                company_ticker=ticker,
                                company_name=data.get('Name'),
                                content_type='company_overview',
                                language='en',
                                title=f"{ticker} Company Overview",
                                content=json.dumps(data),
                                url=None,
                                published_date=datetime.utcnow(),
                                sentiment_score=None,
                                metadata=data
                            )
                            data_points.append(data_point)
                            
                    # Add delay to respect rate limits
                    await asyncio.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
                    
                except Exception as e:
                    logger.error(f"Error scraping Alpha Vantage for {ticker}: {e}")
                    
        return data_points

class FREDScraper:
    """Federal Reserve Economic Data (FRED) scraper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
    async def scrape_economic_indicators(self) -> List[DataPoint]:
        """Scrape key economic indicators from FRED"""
        if not self.api_key:
            logger.warning("FRED API key not provided")
            return []
            
        data_points = []
        indicators = {
            'GDP': 'Gross Domestic Product',
            'UNRATE': 'Unemployment Rate',
            'CPIAUCSL': 'Consumer Price Index',
            'FEDFUNDS': 'Federal Funds Rate',
            'DGS10': '10-Year Treasury Rate',
            'DEXUSEU': 'USD/EUR Exchange Rate'
        }
        
        async with aiohttp.ClientSession() as session:
            for indicator, description in indicators.items():
                try:
                    params = {
                        'series_id': indicator,
                        'api_key': self.api_key,
                        'file_type': 'json',
                        'limit': 1,
                        'sort_order': 'desc'
                    }
                    
                    url = f"{self.base_url}/series/observations"
                    async with session.get(url, params=params) as response:
                        data = await response.json()
                        
                        if 'observations' in data and data['observations']:
                            obs = data['observations'][0]
                            data_point = DataPoint(
                                source_type='fred',
                                source_name='federal_reserve',
                                company_ticker=None,
                                company_name=None,
                                content_type='economic_indicator',
                                language='en',
                                title=f"{indicator} - {description}",
                                content=f"Latest {description}: {obs.get('value')}",
                                url=f"https://fred.stlouisfed.org/series/{indicator}",
                                published_date=datetime.strptime(obs.get('date'), '%Y-%m-%d') if obs.get('date') else datetime.utcnow(),
                                sentiment_score=None,
                                metadata={
                                    'indicator': indicator,
                                    'value': obs.get('value'),
                                    'date': obs.get('date')
                                }
                            )
                            data_points.append(data_point)
                            
                except Exception as e:
                    logger.error(f"Error scraping FRED indicator {indicator}: {e}")
                    
        return data_points

class FinnhubScraper:
    """Finnhub API scraper for financial data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        
    async def scrape_finnhub_data(self, tickers: List[str]) -> List[DataPoint]:
        """Scrape Finnhub financial data and news"""
        if not self.api_key:
            logger.warning("Finnhub API key not provided")
            return []
            
        data_points = []
        
        async with aiohttp.ClientSession() as session:
            for ticker in tickers[:20]:  # Limit API calls
                try:
                    # Company news
                    params = {
                        'symbol': ticker,
                        'token': self.api_key,
                        'from': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                        'to': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    async with session.get(f"{self.base_url}/company-news", params=params) as response:
                        news_data = await response.json()
                        
                        for article in news_data[:5]:  # Limit articles
                            data_point = DataPoint(
                                source_type='finnhub',
                                source_name='finnhub_news',
                                company_ticker=ticker,
                                company_name=None,
                                content_type='financial_news',
                                language='en',
                                title=article.get('headline'),
                                content=article.get('summary', ''),
                                url=article.get('url'),
                                published_date=datetime.fromtimestamp(article.get('datetime', time.time())),
                                sentiment_score=DataProcessor.calculate_sentiment(article.get('summary', '')),
                                metadata={
                                    'category': article.get('category'),
                                    'source': article.get('source')
                                }
                            )
                            data_points.append(data_point)
                    
                    # Add delay for rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error scraping Finnhub for {ticker}: {e}")
                    
        return data_points

class PolygonScraper:
    """Polygon.io API scraper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    async def scrape_polygon_data(self, tickers: List[str]) -> List[DataPoint]:
        """Scrape Polygon market data and news"""
        if not self.api_key:
            logger.warning("Polygon API key not provided")
            return []
            
        data_points = []
        
        async with aiohttp.ClientSession() as session:
            for ticker in tickers[:15]:  # Limit API calls
                try:
                    # Market news
                    params = {
                        'ticker': ticker,
                        'apikey': self.api_key,
                        'limit': 5
                    }
                    
                    async with session.get(f"{self.base_url}/v2/reference/news", params=params) as response:
                        data = await response.json()
                        
                        if 'results' in data:
                            for article in data['results']:
                                data_point = DataPoint(
                                    source_type='polygon',
                                    source_name='polygon_news',
                                    company_ticker=ticker,
                                    company_name=None,
                                    content_type='financial_news',
                                    language='en',
                                    title=article.get('title'),
                                    content=article.get('description', ''),
                                    url=article.get('article_url'),
                                    published_date=datetime.fromisoformat(article.get('published_utc').replace('Z', '+00:00')) if article.get('published_utc') else None,
                                    sentiment_score=DataProcessor.calculate_sentiment(article.get('description', '')),
                                    metadata={
                                        'author': article.get('author'),
                                        'publisher': article.get('publisher')
                                    }
                                )
                                data_points.append(data_point)
                    
                    await asyncio.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error scraping Polygon for {ticker}: {e}")
                    
        return data_points

class RSSFeedScraper:
    """RSS feed scraper for financial news"""
    
    def __init__(self):
        self.feeds = [
            {'name': 'Reuters Business', 'url': 'https://feeds.reuters.com/reuters/businessNews'},
            {'name': 'Bloomberg Markets', 'url': 'https://feeds.bloomberg.com/markets/news.rss'},
            {'name': 'WSJ Markets', 'url': 'https://www.wsj.com/xml/rss/3_7085.xml'},
            {'name': 'MarketWatch', 'url': 'https://feeds.marketwatch.com/marketwatch/topstories/'},
            {'name': 'CNBC Finance', 'url': 'https://www.cnbc.com/id/100003114/device/rss/rss.html'}
        ]
        
    async def scrape_rss_feeds(self, company_tickers: List[str]) -> List[DataPoint]:
        """Scrape RSS feeds for financial news"""
        data_points = []
        
        for feed in self.feeds:
            try:
                parsed_feed = feedparser.parse(feed['url'])
                
                for entry in parsed_feed.entries[:10]:  # Limit entries per feed
                    content = f"{entry.get('summary', '')} {entry.get('description', '')}"
                    mentioned_companies = DataProcessor.extract_companies_from_text(content, company_tickers)
                    
                    if mentioned_companies:  # Only include if companies are mentioned
                        for company in mentioned_companies:
                            data_point = DataPoint(
                                source_type='rss_feed',
                                source_name=feed['name'],
                                company_ticker=company,
                                company_name=None,
                                content_type='financial_news',
                                language='en',
                                title=entry.get('title'),
                                content=content,
                                url=entry.get('link'),
                                published_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') and entry.published_parsed else None,
                                sentiment_score=DataProcessor.calculate_sentiment(content),
                                metadata={
                                    'feed_name': feed['name'],
                                    'author': entry.get('author')
                                }
                            )
                            data_points.append(data_point)
                            
            except Exception as e:
                logger.error(f"Error scraping RSS feed {feed['name']}: {e}")
                
        return data_points

class MultiSourceDataCollector:
    """Main collector orchestrating all data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get API keys from environment variables using ConfigManager
        from shared.utils.config_manager import ConfigManager
        config_manager = ConfigManager()
        api_keys = config_manager.get_all_api_keys()
        
        # Initialize all scrapers with API keys from environment variables
        self.news_scraper = NewsApiScraper(api_keys.get('newsapi'))
        self.twitter_scraper = TwitterScraper(api_keys.get('twitter', {}).get('bearer_token'))
        self.reddit_scraper = RedditScraper(
            api_keys.get('reddit', {}).get('client_id'),
            api_keys.get('reddit', {}).get('client_secret')
        )
        self.yahoo_scraper = YahooFinanceScraper()
        self.alpha_vantage_scraper = AlphaVantageScraper(api_keys.get('alpha_vantage'))
        self.fred_scraper = FREDScraper(api_keys.get('fred'))
        self.finnhub_scraper = FinnhubScraper(api_keys.get('finnhub'))
        self.polygon_scraper = PolygonScraper(api_keys.get('polygon'))
        self.rss_scraper = RSSFeedScraper()
        
        # Initialize SEC filing scraper
        try:
            from ..scrapers.sec_filing_scraper import SECDataCollector
            self.sec_collector = SECDataCollector(config)
        except ImportError:
            logger.warning("SEC filing scraper not available")
            self.sec_collector = None
        
        # Default company tickers
        self.company_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO', 'PEP'
        ]
        
    async def collect_all_data(self) -> List[DataPoint]:
        """Collect data from all configured sources"""
        logger.info("Starting multi-source data collection...")
        
        all_data_points = []
        
        # Define collection tasks
        tasks = [
            self.news_scraper.scrape_news(self.company_tickers, ['en']),
            self.twitter_scraper.scrape_tweets(self.company_tickers),
            self.reddit_scraper.scrape_reddit(self.company_tickers),
            self.yahoo_scraper.scrape_yahoo_finance(self.company_tickers),
            self.alpha_vantage_scraper.scrape_alpha_vantage(self.company_tickers),
            self.fred_scraper.scrape_economic_indicators(),
            self.finnhub_scraper.scrape_finnhub_data(self.company_tickers),
            self.polygon_scraper.scrape_polygon_data(self.company_tickers),
            self.rss_scraper.scrape_rss_feeds(self.company_tickers)
        ]
        
        # Add SEC filing collection if available
        if self.sec_collector:
            tasks.append(self.sec_collector.collect_sec_data(self.company_tickers))
        
        try:
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Task {i} failed: {result}")
                else:
                    all_data_points.extend(result)
                    logger.info(f"Task {i} collected {len(result)} data points")
            
            logger.info(f"Total collected: {len(all_data_points)} data points from {len(tasks)} sources")
            
            return all_data_points
            
        except Exception as e:
            logger.error(f"Error in multi-source data collection: {e}")
            return []
    
    def get_collection_stats(self, data_points: List[DataPoint]) -> Dict[str, Any]:
        """Get statistics about collected data"""
        stats = {
            'total_points': len(data_points),
            'by_source': {},
            'by_content_type': {},
            'by_company': {},
            'languages': set(),
            'date_range': {'earliest': None, 'latest': None}
        }
        
        for dp in data_points:
            # By source
            stats['by_source'][dp.source_type] = stats['by_source'].get(dp.source_type, 0) + 1
            
            # By content type
            stats['by_content_type'][dp.content_type] = stats['by_content_type'].get(dp.content_type, 0) + 1
            
            # By company
            if dp.company_ticker:
                stats['by_company'][dp.company_ticker] = stats['by_company'].get(dp.company_ticker, 0) + 1
            
            # Languages
            if dp.language:
                stats['languages'].add(dp.language)
            
            # Date range - ensure timezone-aware comparison
            if dp.published_date:
                # Ensure timezone-aware
                if dp.published_date.tzinfo is None:
                    dp.published_date = dp.published_date.replace(tzinfo=timezone.utc)
                
                if not stats['date_range']['earliest'] or dp.published_date < stats['date_range']['earliest']:
                    stats['date_range']['earliest'] = dp.published_date
                if not stats['date_range']['latest'] or dp.published_date > stats['date_range']['latest']:
                    stats['date_range']['latest'] = dp.published_date
        
        stats['languages'] = list(stats['languages'])
        return stats
