"""
Database manager for structured data storage and retrieval.
Handles PostgreSQL operations, connection pooling, and data persistence.
"""

import asyncpg
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
import json
from contextlib import asynccontextmanager
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_url = self._build_db_url()
        self.pool = None
        self.engine = None
        self.async_session = None
        
    def _build_db_url(self) -> str:
        """Build database URL from config"""
        db_config = self.config.get('database', {})
        host = db_config.get('host', 'localhost')
        port = db_config.get('port', 5432)
        database = db_config.get('database', 'credit_intelligence')
        username = db_config.get('username', 'postgres')
        password = db_config.get('password', '')
        
        return f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize database connection and create tables"""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.db_url,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=False
            )
            
            # Create session factory
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Create connection pool for direct queries
            self.pool = await asyncpg.create_pool(
                self.db_url.replace('+asyncpg', ''),
                min_size=5,
                max_size=20
            )
            
            # Create tables
            await self._create_tables()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def _create_tables(self):
        """Create database tables"""
        create_statements = [
            """
            CREATE TABLE IF NOT EXISTS raw_data (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                data_type VARCHAR(100) NOT NULL,
                company VARCHAR(200),
                source VARCHAR(100) NOT NULL,
                content JSONB NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                s3_key VARCHAR(500),
                language VARCHAR(10),
                processed BOOLEAN DEFAULT FALSE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS processed_data (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                raw_data_id UUID REFERENCES raw_data(id),
                data_type VARCHAR(100) NOT NULL,
                company VARCHAR(200),
                features JSONB NOT NULL,
                sentiment_score FLOAT,
                confidence_score FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                processing_version VARCHAR(50)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS companies (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(200) NOT NULL UNIQUE,
                ticker_symbol VARCHAR(20),
                cik VARCHAR(20),
                industry VARCHAR(100),
                sector VARCHAR(100),
                country VARCHAR(100),
                market_cap BIGINT,
                employees INTEGER,
                founded_year INTEGER,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS credit_scores (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                company_id UUID REFERENCES companies(id),
                score INTEGER NOT NULL CHECK (score >= 0 AND score <= 100),
                model_version VARCHAR(50) NOT NULL,
                features_used JSONB,
                explanation JSONB,
                confidence FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                company_id UUID REFERENCES companies(id),
                alert_type VARCHAR(100) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                title VARCHAR(500) NOT NULL,
                description TEXT,
                triggered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE,
                metadata JSONB,
                status VARCHAR(20) DEFAULT 'active'
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_raw_data_company_created ON raw_data(company, created_at);
            CREATE INDEX IF NOT EXISTS idx_raw_data_type_created ON raw_data(data_type, created_at);
            CREATE INDEX IF NOT EXISTS idx_processed_data_company ON processed_data(company, created_at);
            CREATE INDEX IF NOT EXISTS idx_credit_scores_company ON credit_scores(company_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_alerts_company_status ON alerts(company_id, status, triggered_at);
            """
        ]
        
        async with self.pool.acquire() as conn:
            for statement in create_statements:
                await conn.execute(statement)
                
        logger.info("Database tables created/verified")
    
    async def insert_raw_data(self, data_type: str, company: str, source: str,
                            content: Dict[str, Any], metadata: Dict[str, Any] = None,
                            s3_key: str = None, language: str = 'en') -> str:
        """Insert raw data into database"""
        try:
            async with self.pool.acquire() as conn:
                row_id = await conn.fetchval(
                    """
                    INSERT INTO raw_data (data_type, company, source, content, metadata, s3_key, language)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """,
                    data_type, company, source, json.dumps(content), 
                    json.dumps(metadata or {}), s3_key, language
                )
                
            logger.debug(f"Inserted raw data record: {row_id}")
            return str(row_id)
            
        except Exception as e:
            logger.error(f"Error inserting raw data: {e}")
            raise
    
    async def insert_processed_data(self, raw_data_id: str, data_type: str, 
                                  company: str, features: Dict[str, Any],
                                  sentiment_score: float = None, 
                                  confidence_score: float = None,
                                  processing_version: str = "1.0") -> str:
        """Insert processed data into database"""
        try:
            async with self.pool.acquire() as conn:
                row_id = await conn.fetchval(
                    """
                    INSERT INTO processed_data (raw_data_id, data_type, company, features, 
                                              sentiment_score, confidence_score, processing_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                    """,
                    raw_data_id, data_type, company, json.dumps(features),
                    sentiment_score, confidence_score, processing_version
                )
                
            logger.debug(f"Inserted processed data record: {row_id}")
            return str(row_id)
            
        except Exception as e:
            logger.error(f"Error inserting processed data: {e}")
            raise
    
    async def get_company_data(self, company: str, data_types: List[str] = None,
                             start_date: datetime = None, end_date: datetime = None,
                             limit: int = 1000) -> List[Dict[str, Any]]:
        """Get data for a specific company"""
        try:
            query = """
                SELECT id, data_type, source, content, metadata, created_at, language
                FROM raw_data 
                WHERE company = $1
            """
            params = [company]
            param_count = 1
            
            if data_types:
                param_count += 1
                query += f" AND data_type = ANY(${param_count})"
                params.append(data_types)
                
            if start_date:
                param_count += 1
                query += f" AND created_at >= ${param_count}"
                params.append(start_date)
                
            if end_date:
                param_count += 1
                query += f" AND created_at <= ${param_count}"
                params.append(end_date)
                
            query += f" ORDER BY created_at DESC LIMIT {limit}"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            data = []
            for row in rows:
                data.append({
                    'id': str(row['id']),
                    'data_type': row['data_type'],
                    'source': row['source'],
                    'content': row['content'],
                    'metadata': row['metadata'],
                    'created_at': row['created_at'],
                    'language': row['language']
                })
                
            logger.info(f"Retrieved {len(data)} records for company {company}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting company data: {e}")
            return []
    
    async def mark_data_processed(self, raw_data_ids: List[str]):
        """Mark raw data as processed"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    "UPDATE raw_data SET processed = TRUE WHERE id = ANY($1)",
                    raw_data_ids
                )
                
            logger.info(f"Marked {len(raw_data_ids)} records as processed")
            
        except Exception as e:
            logger.error(f"Error marking data as processed: {e}")
            raise
    
    async def get_unprocessed_data(self, data_type: str = None, 
                                 limit: int = 1000) -> List[Dict[str, Any]]:
        """Get unprocessed raw data"""
        try:
            query = "SELECT * FROM raw_data WHERE processed = FALSE"
            params = []
            
            if data_type:
                query += " AND data_type = $1"
                params.append(data_type)
                
            query += f" ORDER BY created_at ASC LIMIT {limit}"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            data = []
            for row in rows:
                data.append(dict(row))
                
            logger.info(f"Retrieved {len(data)} unprocessed records")
            return data
            
        except Exception as e:
            logger.error(f"Error getting unprocessed data: {e}")
            return []
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
        if self.engine:
            await self.engine.dispose()
            
        logger.info("Database connections closed")
