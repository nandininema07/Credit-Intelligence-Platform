# Imports
import pytest
import asyncio
from unittest.mock import Mock, patch

from stage1_data_ingestion.stage1_pipeline import DataPoint
from ..scrapers.news_scrapers import NewsApiScraper
from ..scrapers.social_scrapers import TwitterScraper

# Test functions for all scrapers
@pytest.mark.asyncio
async def test_news_api_scraper():
    scraper = NewsApiScraper("test_api_key")
    data = await scraper.scrape_news(["AAPL", "GOOGL"], ["en"])
    assert isinstance(data, list)
    assert all(isinstance(item, DataPoint) for item in data)

@pytest.mark.asyncio
async def test_twitter_scraper():
    scraper = TwitterScraper("test_bearer_token")
    data = await scraper.scrape_tweets(["AAPL", "GOOGL"])
    assert isinstance(data, list)
    assert all(isinstance(item, DataPoint) for item in data)

