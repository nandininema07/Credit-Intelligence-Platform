"""
International news and data scrapers for global market coverage.
Handles multilingual sources from different regions and countries.
"""

import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from bs4 import BeautifulSoup
import feedparser
import json

logger = logging.getLogger(__name__)

@dataclass
class InternationalData:
    """Data class for international data"""
    title: str
    content: str
    url: str
    source: str
    country: str
    language: str
    published_date: datetime
    category: str
    relevance_score: Optional[float] = None

class InternationalScrapers:
    """International data scrapers for global coverage"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.international_sources = self._get_international_sources()
        
    def _get_international_sources(self) -> Dict[str, Dict[str, Any]]:
        """Define international news sources by region"""
        return {
            'europe': {
                'uk': {
                    'sources': ['BBC Business', 'Financial Times', 'The Guardian Business'],
                    'rss_feeds': [
                        'http://feeds.bbci.co.uk/news/business/rss.xml',
                        'https://www.ft.com/rss/home/uk'
                    ],
                    'language': 'en'
                },
                'germany': {
                    'sources': ['Handelsblatt', 'FAZ', 'Der Spiegel'],
                    'rss_feeds': [
                        'https://www.handelsblatt.com/contentexport/feed/schlagzeilen',
                        'https://www.faz.net/rss/aktuell/wirtschaft/'
                    ],
                    'language': 'de'
                },
                'france': {
                    'sources': ['Le Monde', 'Les Echos', 'Le Figaro'],
                    'rss_feeds': [
                        'https://www.lemonde.fr/economie/rss_full.xml',
                        'https://www.lesechos.fr/rss/rss_une.xml'
                    ],
                    'language': 'fr'
                }
            },
            'asia': {
                'japan': {
                    'sources': ['Nikkei', 'Japan Times', 'Asahi Shimbun'],
                    'rss_feeds': [
                        'https://www.nikkei.com/news/feed/'
                    ],
                    'language': 'ja'
                },
                'china': {
                    'sources': ['Xinhua', 'China Daily', 'SCMP'],
                    'rss_feeds': [
                        'http://rss.sina.com.cn/finance.xml'
                    ],
                    'language': 'zh'
                },
                'india': {
                    'sources': ['Economic Times', 'Business Standard', 'Mint'],
                    'rss_feeds': [
                        'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
                        'https://www.business-standard.com/rss/home_page_top_stories.rss'
                    ],
                    'language': 'en'
                }
            },
            'americas': {
                'brazil': {
                    'sources': ['Folha', 'O Globo', 'Valor Economico'],
                    'rss_feeds': [
                        'https://www1.folha.uol.com.br/rss/mercado.xml'
                    ],
                    'language': 'pt'
                },
                'mexico': {
                    'sources': ['El Universal', 'Reforma', 'El Economista'],
                    'rss_feeds': [
                        'https://www.eluniversal.com.mx/rss.xml'
                    ],
                    'language': 'es'
                }
            }
        }
    
    async def scrape_region_news(self, region: str, country: str, 
                               query: str = None) -> List[InternationalData]:
        """Scrape news from a specific region and country"""
        if region not in self.international_sources:
            logger.warning(f"Region {region} not configured")
            return []
            
        if country not in self.international_sources[region]:
            logger.warning(f"Country {country} not configured for region {region}")
            return []
            
        country_config = self.international_sources[region][country]
        articles = []
        
        # Scrape RSS feeds
        for feed_url in country_config.get('rss_feeds', []):
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit per feed
                    # Filter by query if provided
                    if query and query.lower() not in entry.get('title', '').lower():
                        continue
                        
                    article = InternationalData(
                        title=entry.get('title', ''),
                        content=entry.get('summary', ''),
                        url=entry.get('link', ''),
                        source=feed.feed.get('title', 'RSS Feed'),
                        country=country,
                        language=country_config['language'],
                        published_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                        category='Business'
                    )
                    articles.append(article)
                    
            except Exception as e:
                logger.error(f"Error scraping RSS feed {feed_url}: {e}")
                continue
                
        logger.info(f"Scraped {len(articles)} articles from {country}")
        return articles
    
    async def scrape_european_markets(self, query: str = None) -> List[InternationalData]:
        """Scrape European market data and news"""
        all_articles = []
        
        european_countries = ['uk', 'germany', 'france']
        tasks = [
            self.scrape_region_news('europe', country, query) 
            for country in european_countries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
                
        return all_articles
    
    async def scrape_asian_markets(self, query: str = None) -> List[InternationalData]:
        """Scrape Asian market data and news"""
        all_articles = []
        
        asian_countries = ['japan', 'china', 'india']
        tasks = [
            self.scrape_region_news('asia', country, query) 
            for country in asian_countries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
                
        return all_articles
    
    async def scrape_emerging_markets(self, query: str = None) -> List[InternationalData]:
        """Scrape emerging market news"""
        all_articles = []
        
        emerging_countries = ['brazil', 'mexico']
        tasks = [
            self.scrape_region_news('americas', country, query) 
            for country in emerging_countries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
                
        return all_articles
    
    async def scrape_central_bank_data(self) -> List[Dict[str, Any]]:
        """Scrape central bank announcements and data"""
        central_banks = []
        
        bank_feeds = {
            'Federal Reserve': 'https://www.federalreserve.gov/feeds/press_all.xml',
            'ECB': 'https://www.ecb.europa.eu/rss/press.html',
            'Bank of England': 'https://www.bankofengland.co.uk/news/rss',
            'Bank of Japan': 'https://www.boj.or.jp/en/rss/index.htm'
        }
        
        for bank_name, feed_url in bank_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:5]:  # Latest 5 from each bank
                    bank_data = {
                        'bank': bank_name,
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'url': entry.get('link', ''),
                        'published_date': datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                        'type': 'Central Bank Announcement'
                    }
                    central_banks.append(bank_data)
                    
            except Exception as e:
                logger.error(f"Error scraping {bank_name}: {e}")
                continue
                
        logger.info(f"Scraped {len(central_banks)} central bank announcements")
        return central_banks
    
    async def scrape_all_international_sources(self, query: str = None) -> Dict[str, Any]:
        """Scrape all international sources"""
        results = {}
        
        tasks = [
            self.scrape_european_markets(query),
            self.scrape_asian_markets(query),
            self.scrape_emerging_markets(query),
            self.scrape_central_bank_data()
        ]
        
        task_names = ['european_markets', 'asian_markets', 'emerging_markets', 'central_banks']
        
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(task_results):
            if isinstance(result, list):
                results[task_names[i]] = result
            elif isinstance(result, Exception):
                logger.error(f"Error in {task_names[i]}: {result}")
                results[task_names[i]] = []
                
        return results
    
    def get_currency_exchange_rates(self) -> Dict[str, float]:
        """Get current currency exchange rates"""
        try:
            url = "https://api.exchangerate-api.com/v4/latest/USD"
            response = requests.get(url)
            data = response.json()
            return data.get('rates', {})
        except Exception as e:
            logger.error(f"Error getting exchange rates: {e}")
            return {}
