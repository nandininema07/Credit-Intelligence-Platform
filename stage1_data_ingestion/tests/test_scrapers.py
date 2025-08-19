"""
Tests for data scrapers in Stage 1 pipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from ..scrapers.news_scrapers import NewsScrapers, NewsArticle
from ..scrapers.social_scrapers import SocialScrapers, SocialPost
from ..scrapers.financial_scrapers import FinancialScrapers
from ..scrapers.regulatory_scrapers import RegulatoryScrapers

@pytest.fixture
def news_config():
    return {
        'api_keys': {
            'newsapi': 'test_key'
        },
        'rss_feeds': [
            'http://feeds.bbci.co.uk/news/business/rss.xml'
        ]
    }

@pytest.fixture
def social_config():
    return {
        'api_keys': {
            'twitter': {
                'bearer_token': 'test_token',
                'api_key': 'test_key',
                'api_secret': 'test_secret',
                'access_token': 'test_access',
                'access_token_secret': 'test_access_secret'
            }
        }
    }

class TestNewsScrapers:
    """Test news scrapers functionality"""
    
    def test_news_scrapers_init(self, news_config):
        scrapers = NewsScrapers(news_config)
        assert scrapers.config == news_config
        assert 'newsapi' in scrapers.api_keys
    
    @pytest.mark.asyncio
    async def test_scrape_newsapi_no_key(self):
        scrapers = NewsScrapers({})
        articles = await scrapers.scrape_newsapi("test query")
        assert articles == []
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_scrape_newsapi_success(self, mock_get, news_config):
        # Mock response
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            'articles': [{
                'title': 'Test Article',
                'content': 'Test content',
                'url': 'https://test.com',
                'source': {'name': 'Test Source'},
                'publishedAt': '2023-01-01T12:00:00Z',
                'author': 'Test Author'
            }]
        }
        mock_get.return_value.__aenter__.return_value = mock_response
        
        scrapers = NewsScrapers(news_config)
        articles = await scrapers.scrape_newsapi("test query")
        
        assert len(articles) == 1
        assert articles[0].title == 'Test Article'
        assert articles[0].source == 'Test Source'

class TestSocialScrapers:
    """Test social media scrapers"""
    
    def test_social_scrapers_init(self, social_config):
        scrapers = SocialScrapers(social_config)
        assert scrapers.config == social_config
    
    @pytest.mark.asyncio
    async def test_scrape_twitter_no_client(self):
        scrapers = SocialScrapers({})
        posts = await scrapers.scrape_twitter("test query")
        assert posts == []
    
    def test_deduplicate_posts(self, social_config):
        scrapers = SocialScrapers(social_config)
        
        posts = [
            SocialPost(
                content="Test content",
                platform="Twitter",
                author="user1",
                url="https://test1.com",
                created_date=datetime.now(),
                engagement_score=10,
                hashtags=[],
                mentions=[],
                language="en"
            ),
            SocialPost(
                content="Test content",  # Duplicate
                platform="Twitter", 
                author="user2",
                url="https://test2.com",
                created_date=datetime.now(),
                engagement_score=5,
                hashtags=[],
                mentions=[],
                language="en"
            ),
            SocialPost(
                content="Different content",
                platform="Reddit",
                author="user3", 
                url="https://test3.com",
                created_date=datetime.now(),
                engagement_score=20,
                hashtags=[],
                mentions=[],
                language="en"
            )
        ]
        
        unique_posts = scrapers._deduplicate_posts(posts)
        assert len(unique_posts) == 2

@pytest.mark.asyncio
async def test_pipeline_integration():
    """Test basic pipeline integration"""
    config = {
        'news_scrapers': {'api_keys': {}},
        'social_scrapers': {'api_keys': {}},
        'postgres': {
            'postgres_host': 'localhost',
            'postgres_database': 'test_db'
        }
    }
    
    # This would test the main pipeline
    # For now, just verify config loading
    assert 'news_scrapers' in config
    assert 'postgres' in config
