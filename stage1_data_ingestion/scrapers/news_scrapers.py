import asyncio
import feedparser
import requests
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from typing import List
import logging
from ..data_processing.data_models import DataPoint
from ..data_processing.text_processor import DataProcessor

class NewsApiScraper:
    def __init__(self, api_key: str):
        self.client = NewsApiClient(api_key=api_key) if api_key else None
    
    async def scrape_news(self, companies: List[str], languages: List[str] = ['en']) -> List[DataPoint]:
        if not self.client:
            return []
        
        data_points = []
        
        for company in companies:
            for language in languages:
                try:
                    articles = self.client.get_everything(
                        q=company,
                        language=language,
                        sort_by='publishedAt',
                        from_param=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    )
                    
                    for article in articles.get('articles', []):
                        data_point = DataPoint(
                            source_type='news_api',
                            source_name=article.get('source', {}).get('name', 'unknown'),
                            company_ticker=company,
                            company_name=None,
                            content_type='news',
                            language=language,
                            title=article.get('title'),
                            content=article.get('description', '') + ' ' + article.get('content', ''),
                            url=article.get('url'),
                            published_date=datetime.fromisoformat(article.get('publishedAt').replace('Z', '+00:00')) if article.get('publishedAt') else None,
                            metadata={'author': article.get('author'), 'source': article.get('source')}
                        )
                        data_points.append(data_point)
                
                except Exception as e:
                    logger.error(f"Error scraping news for {company}: {e}")
        
        return data_points

class RSSFeedScraper:
    def __init__(self, feeds: List[str]):
        self.feeds = feeds
    
    async def scrape_feeds(self, company_registry: CompanyRegistry) -> List[DataPoint]:
        data_points = []
        
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    content = entry.get('summary', '') + ' ' + entry.get('description', '')
                    mentioned_companies = DataProcessor.extract_companies_from_text(content, company_registry)
                    
                    for company in mentioned_companies:
                        data_point = DataPoint(
                            source_type='rss',
                            source_name=feed.feed.get('title', urlparse(feed_url).netloc),
                            company_ticker=company,
                            company_name=None,
                            content_type='news',
                            language=DataProcessor.detect_language(content),
                            title=entry.get('title'),
                            content=content,
                            url=entry.get('link'),
                            published_date=datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') and entry.published_parsed else None,
                            metadata={'tags': entry.get('tags', []), 'author': entry.get('author')}
                        )
                        data_points.append(data_point)
            
            except Exception as e:
                logger.error(f"Error scraping RSS feed {feed_url}: {e}")
        
        return data_points
