"""
News scrapers for various news APIs and RSS feeds.
Handles multilingual news data collection from global sources.
"""

import requests
import feedparser
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import asyncio
import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class for news articles"""
    title: str
    content: str
    url: str
    source: str
    published_date: datetime
    language: str
    author: Optional[str] = None
    category: Optional[str] = None
    sentiment_score: Optional[float] = None
    entities: Optional[List[str]] = None

class NewsScrapers:
    """News scrapers for multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.rss_feeds = config.get('rss_feeds', [])
        self.session = requests.Session()
        
    async def scrape_newsapi(self, query: str, language: str = 'en', 
                           days_back: int = 7) -> List[NewsArticle]:
        """Scrape news from NewsAPI"""
        if 'newsapi' not in self.api_keys:
            logger.warning("NewsAPI key not found")
            return []
            
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'language': language,
            'sortBy': 'publishedAt',
            'from': (datetime.now() - timedelta(days=days_back)).isoformat(),
            'apiKey': self.api_keys['newsapi']
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    
            articles = []
            for article_data in data.get('articles', []):
                article = NewsArticle(
                    title=article_data.get('title', ''),
                    content=article_data.get('content', ''),
                    url=article_data.get('url', ''),
                    source=article_data.get('source', {}).get('name', ''),
                    published_date=datetime.fromisoformat(
                        article_data.get('publishedAt', '').replace('Z', '+00:00')
                    ),
                    language=language,
                    author=article_data.get('author')
                )
                articles.append(article)
                
            logger.info(f"Scraped {len(articles)} articles from NewsAPI")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping NewsAPI: {e}")
            return []
    
    async def scrape_rss_feeds(self, feeds: Optional[List[str]] = None) -> List[NewsArticle]:
        """Scrape RSS feeds"""
        feeds = feeds or self.rss_feeds
        articles = []
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    article = NewsArticle(
                        title=entry.get('title', ''),
                        content=entry.get('summary', ''),
                        url=entry.get('link', ''),
                        source=feed.feed.get('title', 'RSS Feed'),
                        published_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                        language='en'  # Default, should be detected later
                    )
                    articles.append(article)
                    
            except Exception as e:
                logger.error(f"Error scraping RSS feed {feed_url}: {e}")
                continue
                
        logger.info(f"Scraped {len(articles)} articles from RSS feeds")
        return articles
    
    async def scrape_reuters(self, query: str) -> List[NewsArticle]:
        """Scrape Reuters news"""
        base_url = "https://www.reuters.com/search/news"
        params = {'blob': query}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            articles = []
            
            # Parse Reuters search results
            for article_elem in soup.find_all('div', class_='search-result-content'):
                title_elem = article_elem.find('h3')
                if not title_elem:
                    continue
                    
                link_elem = title_elem.find('a')
                if not link_elem:
                    continue
                    
                article = NewsArticle(
                    title=title_elem.get_text(strip=True),
                    content='',  # Would need additional request to get full content
                    url=f"https://www.reuters.com{link_elem.get('href', '')}",
                    source='Reuters',
                    published_date=datetime.now(),  # Would need to parse from page
                    language='en'
                )
                articles.append(article)
                
            logger.info(f"Scraped {len(articles)} articles from Reuters")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping Reuters: {e}")
            return []
    
    async def scrape_bloomberg(self, query: str) -> List[NewsArticle]:
        """Scrape Bloomberg news"""
        # Note: Bloomberg has strict scraping policies, this is a simplified example
        try:
            search_url = f"https://www.bloomberg.com/search?query={query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            articles = []
            
            # This would need to be adapted based on Bloomberg's actual HTML structure
            for article_elem in soup.find_all('div', class_='storyItem'):
                title_elem = article_elem.find('h1') or article_elem.find('h2')
                if not title_elem:
                    continue
                    
                article = NewsArticle(
                    title=title_elem.get_text(strip=True),
                    content='',
                    url='',  # Would need to extract from element
                    source='Bloomberg',
                    published_date=datetime.now(),
                    language='en'
                )
                articles.append(article)
                
            logger.info(f"Scraped {len(articles)} articles from Bloomberg")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping Bloomberg: {e}")
            return []
    
    async def scrape_all_sources(self, query: str, languages: List[str] = ['en']) -> List[NewsArticle]:
        """Scrape all configured news sources"""
        all_articles = []
        
        # Scrape different sources concurrently
        tasks = []
        
        for language in languages:
            tasks.append(self.scrape_newsapi(query, language))
            
        tasks.extend([
            self.scrape_rss_feeds(),
            self.scrape_reuters(query),
            self.scrape_bloomberg(query)
        ])
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in news scraping task: {result}")
                
        logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles
    
    def get_international_feeds(self) -> Dict[str, List[str]]:
        """Get RSS feeds for international sources"""
        return {
            'english': [
                'http://feeds.bbci.co.uk/news/business/rss.xml',
                'https://feeds.reuters.com/reuters/businessNews',
                'https://rss.cnn.com/rss/money_news_international.rss'
            ],
            'spanish': [
                'https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/economia/portada',
                'https://www.expansion.com/rss/portada.xml'
            ],
            'french': [
                'https://www.lemonde.fr/economie/rss_full.xml',
                'https://www.lesechos.fr/rss/rss_une.xml'
            ],
            'german': [
                'https://www.handelsblatt.com/contentexport/feed/schlagzeilen',
                'https://www.faz.net/rss/aktuell/wirtschaft/'
            ],
            'chinese': [
                'http://rss.sina.com.cn/finance.xml'
            ],
            'japanese': [
                'https://www.nikkei.com/news/feed/'
            ]
        }
