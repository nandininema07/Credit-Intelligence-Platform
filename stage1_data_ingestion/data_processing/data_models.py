from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, DateTime, Text, Float, Integer, Boolean
from sqlalchemy.orm import declarative_base

# Database Models
Base = declarative_base()

class RawData(Base):
    __tablename__ = 'raw_data'
    
    id = Column(String, primary_key=True)
    source_type = Column(String, nullable=False)
    source_name = Column(String, nullable=False)
    company_ticker = Column(String)
    company_name = Column(String)
    content_type = Column(String, nullable=False)
    language = Column(String)
    title = Column(Text)
    content = Column(Text)
    url = Column(String)
    published_date = Column(DateTime)
    scraped_date = Column(DateTime, default=datetime.utcnow)
    sentiment_score = Column(Float)
    metadata = Column(Text)  # JSON string

@dataclass
class DataPoint:
    source_type: str
    source_name: str
    company_ticker: Optional[str]
    company_name: Optional[str]
    content_type: str
    language: Optional[str]
    title: Optional[str]
    content: str
    url: Optional[str]
    published_date: Optional[datetime]
    metadata: Dict[str, Any]
