import json
from typing import Dict

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
                "access_key": "",
                "secret_key": ""
            },
            "database": {
                "url": "postgresql://user:pass@localhost:5432/credtech"
            },
            "apis": {
                "newsapi_key": "",
                "twitter_bearer_token": "",
                "reddit_client_id": "",
                "reddit_client_secret": "",
                "alpha_vantage_key": "",
                "fred_key": "",
                "world_bank_key": ""
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
