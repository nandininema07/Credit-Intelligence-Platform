"""
SQLAlchemy data models for the Credit Intelligence Platform.
Defines database schema and relationships.
"""

from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, ForeignKey, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid

Base = declarative_base()

class Company(Base):
    """Company information model"""
    __tablename__ = 'companies'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, unique=True)
    ticker_symbol = Column(String(20))
    cik = Column(String(20))
    industry = Column(String(100))
    sector = Column(String(100))
    country = Column(String(100))
    market_cap = Column(BigInteger)
    employees = Column(Integer)
    founded_year = Column(Integer)
    company_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    raw_data = relationship("RawData", back_populates="company_ref")
    processed_data = relationship("ProcessedData", back_populates="company_ref")
    credit_scores = relationship("CreditScore", back_populates="company_ref")
    alerts = relationship("Alert", back_populates="company_ref")

class RawData(Base):
    """Raw data storage model"""
    __tablename__ = 'raw_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_type = Column(String(100), nullable=False)
    company = Column(String(200))
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'))
    source = Column(String(100), nullable=False)
    content = Column(JSONB, nullable=False)
    data_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    s3_key = Column(String(500))
    language = Column(String(10))
    processed = Column(Boolean, default=False)
    
    # Relationships
    company_ref = relationship("Company", back_populates="raw_data")
    processed_data = relationship("ProcessedData", back_populates="raw_data_ref")

class ProcessedData(Base):
    """Processed data model"""
    __tablename__ = 'processed_data'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    raw_data_id = Column(UUID(as_uuid=True), ForeignKey('raw_data.id'))
    data_type = Column(String(100), nullable=False)
    company = Column(String(200))
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'))
    features = Column(JSONB, nullable=False)
    sentiment_score = Column(Float)
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    processing_version = Column(String(50))
    
    # Relationships
    raw_data_ref = relationship("RawData", back_populates="processed_data")
    company_ref = relationship("Company", back_populates="processed_data")

class CreditScore(Base):
    """Credit score model"""
    __tablename__ = 'credit_scores'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'), nullable=False)
    score = Column(Integer, nullable=False)
    model_version = Column(String(50), nullable=False)
    features_used = Column(JSONB)
    explanation = Column(JSONB)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    company_ref = relationship("Company", back_populates="credit_scores")

class Alert(Base):
    """Alert model"""
    __tablename__ = 'alerts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'))
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    triggered_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    alert_metadata = Column(JSONB)
    status = Column(String(20), default='active')
    
    # Relationships
    company_ref = relationship("Company", back_populates="alerts")

class DataSource(Base):
    """Data source configuration model"""
    __tablename__ = 'data_sources'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    source_type = Column(String(50), nullable=False)
    url = Column(String(500))
    api_endpoint = Column(String(500))
    config = Column(JSONB)
    is_active = Column(Boolean, default=True)
    last_scraped = Column(DateTime)
    scrape_frequency = Column(Integer)  # Minutes between scrapes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ScrapingJob(Base):
    """Scraping job tracking model"""
    __tablename__ = 'scraping_jobs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_type = Column(String(100), nullable=False)
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    records_processed = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    config = Column(JSONB)
    error_log = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class FeatureStore(Base):
    """Feature store model for ML features"""
    __tablename__ = 'feature_store'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id'), nullable=False)
    feature_name = Column(String(200), nullable=False)
    feature_value = Column(Float)
    feature_metadata = Column(JSONB)
    calculation_date = Column(DateTime, nullable=False)
    data_sources = Column(JSONB)  # List of data sources used
    version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelMetadata(Base):
    """Model metadata and versioning"""
    __tablename__ = 'model_metadata'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50))
    training_date = Column(DateTime, nullable=False)
    performance_metrics = Column(JSONB)
    feature_importance = Column(JSONB)
    hyperparameters = Column(JSONB)
    model_path = Column(String(500))  # S3 path to model file
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DataQuality(Base):
    """Data quality metrics"""
    __tablename__ = 'data_quality'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    data_type = Column(String(100), nullable=False)
    source = Column(String(100), nullable=False)
    quality_score = Column(Float)
    completeness = Column(Float)
    accuracy = Column(Float)
    timeliness = Column(Float)
    consistency = Column(Float)
    issues = Column(JSONB)
    checked_at = Column(DateTime, default=datetime.utcnow)
    records_checked = Column(Integer)
