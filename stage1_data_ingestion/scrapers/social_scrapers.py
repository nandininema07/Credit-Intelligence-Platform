"""
Social media scrapers for Twitter, Reddit, and other platforms.
Handles social sentiment data collection for credit analysis.
"""

import tweepy
import praw
import requests
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    """Data class for social media posts"""
    content: str
    platform: str
    author: str
    url: str
    created_date: datetime
    engagement_score: int
    hashtags: List[str]
    mentions: List[str]
    language: str
    sentiment_score: Optional[float] = None
    influence_score: Optional[float] = None

class SocialScrapers:
    """Social media scrapers for multiple platforms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.setup_clients()
        
    def setup_clients(self):
        """Initialize API clients"""
        # Twitter API v2 client
        if 'twitter' in self.api_keys:
            self.twitter_client = tweepy.Client(
                bearer_token=self.api_keys['twitter']['bearer_token'],
                consumer_key=self.api_keys['twitter']['api_key'],
                consumer_secret=self.api_keys['twitter']['api_secret'],
                access_token=self.api_keys['twitter']['access_token'],
                access_token_secret=self.api_keys['twitter']['access_token_secret']
            )
        else:
            self.twitter_client = None
            
        # Reddit client
        if 'reddit' in self.api_keys:
            self.reddit_client = praw.Reddit(
                client_id=self.api_keys['reddit']['client_id'],
                client_secret=self.api_keys['reddit']['client_secret'],
                user_agent=self.api_keys['reddit']['user_agent']
            )
        else:
            self.reddit_client = None
    
    async def scrape_twitter(self, query: str, max_results: int = 100) -> List[SocialPost]:
        """Scrape Twitter for company mentions"""
        if not self.twitter_client:
            logger.warning("Twitter client not configured")
            return []
            
        try:
            tweets = tweepy.Paginator(
                self.twitter_client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang', 'entities'],
                max_results=min(max_results, 100)
            ).flatten(limit=max_results)
            
            posts = []
            for tweet in tweets:
                # Extract hashtags and mentions
                hashtags = []
                mentions = []
                if tweet.entities:
                    hashtags = [tag['tag'] for tag in tweet.entities.get('hashtags', [])]
                    mentions = [mention['username'] for mention in tweet.entities.get('mentions', [])]
                
                engagement = tweet.public_metrics['like_count'] + tweet.public_metrics['retweet_count']
                
                post = SocialPost(
                    content=tweet.text,
                    platform='Twitter',
                    author=str(tweet.author_id),
                    url=f"https://twitter.com/user/status/{tweet.id}",
                    created_date=tweet.created_at,
                    engagement_score=engagement,
                    hashtags=hashtags,
                    mentions=mentions,
                    language=tweet.lang or 'en'
                )
                posts.append(post)
                
            logger.info(f"Scraped {len(posts)} tweets")
            return posts
            
        except Exception as e:
            logger.error(f"Error scraping Twitter: {e}")
            return []
    
    async def scrape_reddit(self, subreddits: List[str], query: str, 
                          limit: int = 100) -> List[SocialPost]:
        """Scrape Reddit for company discussions"""
        if not self.reddit_client:
            logger.warning("Reddit client not configured")
            return []
            
        posts = []
        
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Search in subreddit
                for submission in subreddit.search(query, limit=limit//len(subreddits)):
                    post = SocialPost(
                        content=f"{submission.title}\n{submission.selftext}",
                        platform='Reddit',
                        author=submission.author.name if submission.author else '[deleted]',
                        url=f"https://reddit.com{submission.permalink}",
                        created_date=datetime.fromtimestamp(submission.created_utc),
                        engagement_score=submission.score + submission.num_comments,
                        hashtags=[],  # Reddit doesn't use hashtags
                        mentions=[],
                        language='en'  # Default, would need detection
                    )
                    posts.append(post)
                    
                    # Also get top comments
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:5]:  # Top 5 comments
                        if hasattr(comment, 'body') and comment.body != '[deleted]':
                            comment_post = SocialPost(
                                content=comment.body,
                                platform='Reddit',
                                author=comment.author.name if comment.author else '[deleted]',
                                url=f"https://reddit.com{comment.permalink}",
                                created_date=datetime.fromtimestamp(comment.created_utc),
                                engagement_score=comment.score,
                                hashtags=[],
                                mentions=[],
                                language='en'
                            )
                            posts.append(comment_post)
                            
            logger.info(f"Scraped {len(posts)} Reddit posts")
            return posts
            
        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
            return []
    
    async def scrape_linkedin_posts(self, company_name: str) -> List[SocialPost]:
        """Scrape LinkedIn company posts (limited without official API)"""
        # Note: LinkedIn has strict scraping policies
        # This is a placeholder for potential LinkedIn integration
        logger.info("LinkedIn scraping requires official API access")
        return []
    
    async def scrape_youtube_comments(self, video_ids: List[str]) -> List[SocialPost]:
        """Scrape YouTube comments for financial videos"""
        if 'youtube' not in self.api_keys:
            logger.warning("YouTube API key not found")
            return []
            
        posts = []
        api_key = self.api_keys['youtube']
        
        try:
            for video_id in video_ids:
                url = "https://www.googleapis.com/youtube/v3/commentThreads"
                params = {
                    'part': 'snippet',
                    'videoId': video_id,
                    'key': api_key,
                    'maxResults': 100
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        data = await response.json()
                        
                for item in data.get('items', []):
                    comment = item['snippet']['topLevelComment']['snippet']
                    
                    post = SocialPost(
                        content=comment['textDisplay'],
                        platform='YouTube',
                        author=comment['authorDisplayName'],
                        url=f"https://youtube.com/watch?v={video_id}",
                        created_date=datetime.fromisoformat(comment['publishedAt'].replace('Z', '+00:00')),
                        engagement_score=comment['likeCount'],
                        hashtags=[],
                        mentions=[],
                        language='en'
                    )
                    posts.append(post)
                    
            logger.info(f"Scraped {len(posts)} YouTube comments")
            return posts
            
        except Exception as e:
            logger.error(f"Error scraping YouTube: {e}")
            return []
    
    async def scrape_telegram_channels(self, channels: List[str], 
                                     keywords: List[str]) -> List[SocialPost]:
        """Scrape Telegram channels for financial discussions"""
        # Note: Requires Telegram API setup
        logger.info("Telegram scraping requires API setup")
        return []
    
    async def get_trending_hashtags(self, platform: str = 'twitter') -> List[str]:
        """Get trending hashtags related to finance"""
        if platform == 'twitter' and self.twitter_client:
            try:
                trends = self.twitter_client.get_place_trends(1)  # Worldwide trends
                financial_keywords = ['finance', 'stock', 'market', 'economy', 'crypto']
                
                trending_hashtags = []
                for trend in trends[0]['trends']:
                    name = trend['name'].lower()
                    if any(keyword in name for keyword in financial_keywords):
                        trending_hashtags.append(trend['name'])
                        
                return trending_hashtags[:10]  # Top 10
                
            except Exception as e:
                logger.error(f"Error getting trending hashtags: {e}")
                
        return []
    
    async def scrape_all_platforms(self, query: str, 
                                 subreddits: List[str] = None) -> List[SocialPost]:
        """Scrape all configured social platforms"""
        if subreddits is None:
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting']
            
        all_posts = []
        
        # Run scrapers concurrently
        tasks = [
            self.scrape_twitter(query, max_results=200),
            self.scrape_reddit(subreddits, query, limit=100)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error in social scraping task: {result}")
                
        # Remove duplicates based on content similarity
        unique_posts = self._deduplicate_posts(all_posts)
        
        logger.info(f"Total unique social posts: {len(unique_posts)}")
        return unique_posts
    
    def _deduplicate_posts(self, posts: List[SocialPost]) -> List[SocialPost]:
        """Remove duplicate posts based on content similarity"""
        unique_posts = []
        seen_content = set()
        
        for post in posts:
            # Simple deduplication based on first 100 characters
            content_hash = hash(post.content[:100].lower().strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_posts.append(post)
                
        return unique_posts
    
    def calculate_influence_score(self, post: SocialPost) -> float:
        """Calculate influence score based on platform and engagement"""
        base_scores = {
            'Twitter': 1.0,
            'Reddit': 0.8,
            'LinkedIn': 1.2,
            'YouTube': 0.6
        }
        
        base_score = base_scores.get(post.platform, 0.5)
        engagement_multiplier = min(post.engagement_score / 100, 5.0)  # Cap at 5x
        
        return base_score * (1 + engagement_multiplier)
