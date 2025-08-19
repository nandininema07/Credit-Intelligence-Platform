import json
import os
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file in the root directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        return {
            "aws": {
                "region": "us-east-1",
                "s3_bucket": "credtech-data-lake",
                "access_key": os.getenv("AWS_ACCESS_KEY", ""),
                "secret_key": os.getenv("AWS_SECRET_KEY", "")
            },
            "database": {
                "url": os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/credtech"),
                "host": os.getenv("DB_HOST", "localhost"),
                "port": os.getenv("DB_PORT", "5432"),
                "name": os.getenv("DB_NAME", "credtech"),
                "user": os.getenv("DB_USER", "user"),
                "password": os.getenv("DB_PASSWORD", "pass")
            },
            "apis": {
                "newsapi_key": os.getenv("NEWSAPI_KEY", ""),
                "twitter_bearer_token": os.getenv("TWITTER_BEARER_TOKEN", ""),
                "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", ""),
                "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
                "alpha_vantage_key": os.getenv("ALPHA_VANTAGE_KEY", ""),
                "fred_key": os.getenv("FRED_KEY", ""),
                "world_bank_key": os.getenv("WORLD_BANK_KEY", "")
            },
            "sources": {
                "news_sources": ["reuters", "bloomberg", "wsj", "ft", "cnbc", "marketwatch"],
                "rss_feeds": [
                    "https://feeds.reuters.com/reuters/businessNews",
                    "https://feeds.bloomberg.com/markets/news.rss",
                    "https://www.wsj.com/xml/rss/3_7085.xml"
                ],
                "company_websites": [],
                "social_platforms": ["twitter", "reddit"],
                "financial_apis": ["yahoo", "alpha_vantage", "fred", "world_bank"]
            },
            "companies": {
                "stock_exchanges": ["NYSE", "NASDAQ", "LSE", "TSE", "HKEX", "BSE", "NSE"],
                "sectors": ["technology", "finance", "healthcare", "energy", "retail"]
            },
            "processing": {
                "batch_size": 100,
                "max_workers": 10,
                "update_frequency": 300  # seconds
            }
        }
