import asyncio
import tweepy
import praw
from datetime import datetime
from typing import List
import logging
from ..data_processing.data_models import DataPoint
from ..data_processing.text_processor import DataProcessor

class TwitterScraper:
    def __init__(self, bearer_token: str):
        self.client = tweepy.Client(bearer_token=bearer_token) if bearer_token else None
    
    async def scrape_tweets(self, companies: List[str]) -> List[DataPoint]:
        if not self.client:
            return []
        
        data_points = []
        
        for company in companies:
            try:
                tweets = self.client.search_recent_tweets(
                    query=f"${company} OR {company}",
                    max_results=100,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang']
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
                            metadata={
                                'author_id': tweet.author_id,
                                'metrics': tweet.public_metrics
                            }
                        )
                        data_points.append(data_point)
            
            except Exception as e:
                logger.error(f"Error scraping Twitter for {company}: {e}")
        
        return data_points

class RedditScraper:
    def __init__(self, client_id: str, client_secret: str):
        if client_id and client_secret:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent="credtech_scraper"
            )
        else:
            self.reddit = None
    
    async def scrape_reddit(self, companies: List[str]) -> List[DataPoint]:
        if not self.reddit:
            return []
        
        data_points = []
        subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'financialindependence']
        
        for company in companies:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    for submission in subreddit.search(company, limit=20, time_filter='day'):
                        data_point = DataPoint(
                            source_type='reddit',
                            source_name=f"r/{subreddit_name}",
                            company_ticker=company,
                            company_name=None,
                            content_type='social',
                            language=DataProcessor.detect_language(submission.title + ' ' + submission.selftext),
                            title=submission.title,
                            content=submission.selftext,
                            url=submission.url,
                            published_date=datetime.fromtimestamp(submission.created_utc),
                            metadata={
                                'score': submission.score,
                                'num_comments': submission.num_comments,
                                'author': str(submission.author)
                            }
                        )
                        data_points.append(data_point)
                
                except Exception as e:
                    logger.error(f"Error scraping Reddit for {company}: {e}")
        
        return data_points
