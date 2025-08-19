"""
PostgreSQL data manager for storing and retrieving raw and processed data.
Replaces S3 with PostgreSQL for data lake operations.
"""

import asyncio
import asyncpg
import json
import gzip
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO, BytesIO
import pickle
import base64

logger = logging.getLogger(__name__)

class PostgreSQLManager:
    """PostgreSQL data operations manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_string = self._build_connection_string()
        self.pool = None
        
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string"""
        host = self.config.get('postgres_host', 'localhost')
        port = self.config.get('postgres_port', 5432)
        database = self.config.get('postgres_database', 'credit_intelligence')
        user = self.config.get('postgres_user', 'postgres')
        password = self.config.get('postgres_password', '')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize connection pool and create tables"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            await self._create_tables()
            logger.info("PostgreSQL manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL manager: {e}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables for data storage"""
        async with self.pool.acquire() as conn:
            # Raw data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    id SERIAL PRIMARY KEY,
                    data_type VARCHAR(100) NOT NULL,
                    company VARCHAR(200),
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    source VARCHAR(200),
                    language VARCHAR(10),
                    content JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    INDEX (data_type, company, timestamp),
                    INDEX (timestamp),
                    INDEX (company)
                )
            """)
            
            # Processed data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_data (
                    id SERIAL PRIMARY KEY,
                    data_type VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    content BYTEA NOT NULL,
                    format VARCHAR(50) DEFAULT 'parquet',
                    row_count INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    INDEX (data_type, timestamp),
                    INDEX (timestamp)
                )
            """)
            
            # Features table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id SERIAL PRIMARY KEY,
                    company VARCHAR(200) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    feature_name VARCHAR(200) NOT NULL,
                    feature_value FLOAT,
                    feature_metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(company, timestamp, feature_name),
                    INDEX (company, timestamp),
                    INDEX (feature_name, timestamp)
                )
            """)
            
            # Credit scores table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS credit_scores (
                    id SERIAL PRIMARY KEY,
                    company VARCHAR(200) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    score FLOAT NOT NULL,
                    model_version VARCHAR(50),
                    confidence FLOAT,
                    explanation JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    INDEX (company, timestamp),
                    INDEX (timestamp)
                )
            """)
            
            # Alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    company VARCHAR(200) NOT NULL,
                    alert_type VARCHAR(100) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    metadata JSONB,
                    status VARCHAR(20) DEFAULT 'active',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    INDEX (company, timestamp),
                    INDEX (alert_type, timestamp),
                    INDEX (status)
                )
            """)
    
    async def store_raw_data(self, data: Any, data_type: str, 
                           company: str = None, source: str = None,
                           language: str = 'en', timestamp: datetime = None) -> Optional[int]:
        """Store raw data in PostgreSQL"""
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            # Compress data if it's large
            content = data
            if isinstance(data, (dict, list)):
                json_str = json.dumps(data, default=str, ensure_ascii=False)
                if len(json_str) > 1000:  # Compress if > 1KB
                    compressed = gzip.compress(json_str.encode('utf-8'))
                    content = {
                        '_compressed': True,
                        '_data': base64.b64encode(compressed).decode('ascii')
                    }
                else:
                    content = data
            
            async with self.pool.acquire() as conn:
                row_id = await conn.fetchval("""
                    INSERT INTO raw_data (data_type, company, timestamp, source, language, content, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING id
                """, data_type, company, timestamp, source, language, json.dumps(content), json.dumps({
                    'original_size': len(str(data)),
                    'compressed': '_compressed' in content if isinstance(content, dict) else False
                }))
                
            logger.info(f"Stored raw data with ID {row_id}")
            return row_id
            
        except Exception as e:
            logger.error(f"Error storing raw data: {e}")
            return None
    
    async def store_processed_data(self, data: pd.DataFrame, data_type: str,
                                 timestamp: datetime = None) -> Optional[int]:
        """Store processed data as compressed Parquet"""
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            # Convert DataFrame to Parquet bytes
            parquet_buffer = BytesIO()
            data.to_parquet(parquet_buffer, index=False, compression='snappy')
            parquet_bytes = parquet_buffer.getvalue()
            
            async with self.pool.acquire() as conn:
                row_id = await conn.fetchval("""
                    INSERT INTO processed_data (data_type, timestamp, content, format, row_count, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, data_type, timestamp, parquet_bytes, 'parquet', len(data), json.dumps({
                    'columns': list(data.columns),
                    'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
                }))
                
            logger.info(f"Stored processed data with ID {row_id}")
            return row_id
            
        except Exception as e:
            logger.error(f"Error storing processed data: {e}")
            return None
    
    async def get_raw_data(self, data_type: str, company: str = None,
                         start_date: datetime = None, end_date: datetime = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve raw data with filtering"""
        try:
            query = "SELECT * FROM raw_data WHERE data_type = $1"
            params = [data_type]
            
            if company:
                query += " AND company = $" + str(len(params) + 1)
                params.append(company)
                
            if start_date:
                query += " AND timestamp >= $" + str(len(params) + 1)
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= $" + str(len(params) + 1)
                params.append(end_date)
                
            query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            results = []
            for row in rows:
                content = json.loads(row['content'])
                
                # Decompress if needed
                if isinstance(content, dict) and content.get('_compressed'):
                    compressed_data = base64.b64decode(content['_data'])
                    decompressed = gzip.decompress(compressed_data)
                    content = json.loads(decompressed.decode('utf-8'))
                
                results.append({
                    'id': row['id'],
                    'data_type': row['data_type'],
                    'company': row['company'],
                    'timestamp': row['timestamp'],
                    'source': row['source'],
                    'language': row['language'],
                    'content': content,
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at']
                })
                
            logger.info(f"Retrieved {len(results)} raw data records")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving raw data: {e}")
            return []
    
    async def get_processed_data(self, data_type: str, 
                               start_date: datetime = None, end_date: datetime = None,
                               limit: int = 100) -> List[pd.DataFrame]:
        """Retrieve processed data"""
        try:
            query = "SELECT * FROM processed_data WHERE data_type = $1"
            params = [data_type]
            
            if start_date:
                query += " AND timestamp >= $" + str(len(params) + 1)
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= $" + str(len(params) + 1)
                params.append(end_date)
                
            query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            dataframes = []
            for row in rows:
                if row['format'] == 'parquet':
                    df = pd.read_parquet(BytesIO(row['content']))
                    df.attrs['timestamp'] = row['timestamp']
                    df.attrs['metadata'] = json.loads(row['metadata']) if row['metadata'] else {}
                    dataframes.append(df)
                    
            logger.info(f"Retrieved {len(dataframes)} processed data records")
            return dataframes
            
        except Exception as e:
            logger.error(f"Error retrieving processed data: {e}")
            return []
    
    async def store_features(self, company: str, features: Dict[str, float],
                           timestamp: datetime = None) -> bool:
        """Store feature values for a company"""
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for feature_name, feature_value in features.items():
                        await conn.execute("""
                            INSERT INTO features (company, timestamp, feature_name, feature_value)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT (company, timestamp, feature_name)
                            DO UPDATE SET feature_value = EXCLUDED.feature_value
                        """, company, timestamp, feature_name, feature_value)
                        
            logger.info(f"Stored {len(features)} features for {company}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False
    
    async def get_features(self, company: str, feature_names: List[str] = None,
                         start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """Retrieve features for a company"""
        try:
            query = "SELECT * FROM features WHERE company = $1"
            params = [company]
            
            if feature_names:
                query += " AND feature_name = ANY($" + str(len(params) + 1) + ")"
                params.append(feature_names)
                
            if start_date:
                query += " AND timestamp >= $" + str(len(params) + 1)
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= $" + str(len(params) + 1)
                params.append(end_date)
                
            query += " ORDER BY timestamp, feature_name"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            if not rows:
                return pd.DataFrame()
                
            # Convert to DataFrame
            data = []
            for row in rows:
                data.append({
                    'company': row['company'],
                    'timestamp': row['timestamp'],
                    'feature_name': row['feature_name'],
                    'feature_value': row['feature_value'],
                    'feature_metadata': json.loads(row['feature_metadata']) if row['feature_metadata'] else {}
                })
                
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} feature records for {company}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            return pd.DataFrame()
    
    async def store_credit_score(self, company: str, score: float, 
                               model_version: str = None, confidence: float = None,
                               explanation: Dict[str, Any] = None,
                               timestamp: datetime = None) -> Optional[int]:
        """Store credit score"""
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            async with self.pool.acquire() as conn:
                row_id = await conn.fetchval("""
                    INSERT INTO credit_scores (company, timestamp, score, model_version, confidence, explanation)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, company, timestamp, score, model_version, confidence, 
                json.dumps(explanation) if explanation else None)
                
            logger.info(f"Stored credit score {score} for {company}")
            return row_id
            
        except Exception as e:
            logger.error(f"Error storing credit score: {e}")
            return None
    
    async def get_credit_scores(self, company: str = None,
                              start_date: datetime = None, end_date: datetime = None,
                              limit: int = 1000) -> pd.DataFrame:
        """Retrieve credit scores"""
        try:
            query = "SELECT * FROM credit_scores WHERE 1=1"
            params = []
            
            if company:
                query += " AND company = $" + str(len(params) + 1)
                params.append(company)
                
            if start_date:
                query += " AND timestamp >= $" + str(len(params) + 1)
                params.append(start_date)
                
            if end_date:
                query += " AND timestamp <= $" + str(len(params) + 1)
                params.append(end_date)
                
            query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            if not rows:
                return pd.DataFrame()
                
            data = []
            for row in rows:
                data.append({
                    'id': row['id'],
                    'company': row['company'],
                    'timestamp': row['timestamp'],
                    'score': row['score'],
                    'model_version': row['model_version'],
                    'confidence': row['confidence'],
                    'explanation': json.loads(row['explanation']) if row['explanation'] else {},
                    'created_at': row['created_at']
                })
                
            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} credit score records")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving credit scores: {e}")
            return pd.DataFrame()
    
    async def store_alert(self, company: str, alert_type: str, severity: str,
                        message: str, metadata: Dict[str, Any] = None,
                        timestamp: datetime = None) -> Optional[int]:
        """Store alert"""
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            async with self.pool.acquire() as conn:
                row_id = await conn.fetchval("""
                    INSERT INTO alerts (company, alert_type, severity, message, timestamp, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """, company, alert_type, severity, message, timestamp,
                json.dumps(metadata) if metadata else None)
                
            logger.info(f"Stored alert for {company}: {message}")
            return row_id
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            return None
    
    async def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        try:
            async with self.pool.acquire() as conn:
                # Raw data stats
                raw_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT company) as unique_companies,
                        COUNT(DISTINCT data_type) as data_types,
                        MIN(timestamp) as earliest_date,
                        MAX(timestamp) as latest_date
                    FROM raw_data
                """)
                
                # Credit scores stats
                score_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_scores,
                        AVG(score) as avg_score,
                        MIN(score) as min_score,
                        MAX(score) as max_score
                    FROM credit_scores
                """)
                
                # Alerts stats
                alert_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        COUNT(*) FILTER (WHERE status = 'active') as active_alerts
                    FROM alerts
                """)
                
            return {
                'raw_data': dict(raw_stats) if raw_stats else {},
                'credit_scores': dict(score_stats) if score_stats else {},
                'alerts': dict(alert_stats) if alert_stats else {},
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_counts = {}
        
        try:
            async with self.pool.acquire() as conn:
                # Clean raw data
                deleted_counts['raw_data'] = await conn.fetchval("""
                    DELETE FROM raw_data WHERE timestamp < $1
                    RETURNING COUNT(*)
                """, cutoff_date) or 0
                
                # Clean processed data
                deleted_counts['processed_data'] = await conn.fetchval("""
                    DELETE FROM processed_data WHERE timestamp < $1
                    RETURNING COUNT(*)
                """, cutoff_date) or 0
                
                # Clean old features (keep more recent)
                feature_cutoff = datetime.now() - timedelta(days=days_to_keep * 2)
                deleted_counts['features'] = await conn.fetchval("""
                    DELETE FROM features WHERE timestamp < $1
                    RETURNING COUNT(*)
                """, feature_cutoff) or 0
                
            logger.info(f"Cleaned up old data: {deleted_counts}")
            return deleted_counts
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {}
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
