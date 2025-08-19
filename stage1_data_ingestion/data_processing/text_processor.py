import hashlib
import re
from datetime import datetime
from typing import List
from langdetect import detect
from textblob import TextBlob
from ..config.company_registry import CompanyRegistry

class DataProcessor:
    @staticmethod
    def detect_language(text: str) -> str:
        try:
            return detect(text)
        except:
            return 'unknown'
    
    @staticmethod
    def calculate_sentiment(text: str) -> float:
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    @staticmethod
    def extract_companies_from_text(text: str, company_registry: CompanyRegistry) -> List[str]:
        mentioned_companies = []
        text_upper = text.upper()
        
        for ticker, info in company_registry.companies.items():
            if ticker.split('.')[0] in text_upper or info['name'].upper() in text_upper:
                mentioned_companies.append(ticker)
        
        return mentioned_companies
    
    @staticmethod
    def generate_id(source_type: str, url: str, published_date: datetime) -> str:
        content = f"{source_type}_{url}_{published_date.isoformat() if published_date else datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()

