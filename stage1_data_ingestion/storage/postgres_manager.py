"""
PostgreSQL data manager for storing and retrieving raw and processed data.
Replaces S3 with PostgreSQL for data lake operations.
"""

import asyncio
import asyncpg
import json
import gzip
import logging
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta, timezone
import pandas as pd
from io import StringIO, BytesIO
import pickle
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class PostgreSQLManager:
    """PostgreSQL data operations manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection_string = self._build_connection_string()
        self.pool = None
        
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables"""
        # Ensure .env is loaded
        load_dotenv()
        
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '5432')
        database = os.getenv('DB_NAME', 'credit_intelligence')
        user = os.getenv('DB_USER', 'postgres')
        password = os.getenv('DB_PASSWORD', '')
        
        logger.info(f"Building connection string - Host: {host}, Port: {port}, DB: {database}, User: {user}")
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    
    async def initialize(self):
        """Initialize connection pool and create tables"""
        try:
            # Test connection first using individual parameters (same as test_db_connection.py)
            host = os.getenv('DB_HOST', 'localhost')
            port = int(os.getenv('DB_PORT', '5432'))
            database = os.getenv('DB_NAME', 'credit_intelligence')
            user = os.getenv('DB_USER', 'postgres')
            password = os.getenv('DB_PASSWORD', '')
            
            logger.info(f"Testing connection with individual parameters...")
            test_conn = await asyncpg.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            await test_conn.close()
            logger.info("✅ Test connection successful!")
            
            # Now create pool using individual parameters instead of connection string
            self.pool = await asyncpg.create_pool(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
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
        """Create necessary tables for data storage - work with existing schema"""
        async with self.pool.acquire() as conn:
            # Check if raw_data table exists with expected structure
            existing_raw_data = await conn.fetchval("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'raw_data' AND table_schema = 'public'
            """)
            
            if existing_raw_data:
                logger.info("✅ Raw data table already exists with correct structure")
            else:
                # Create raw_data table if it doesn't exist
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
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)
            
            # Create indexes for raw_data (using existing column names)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_type_company_timestamp ON raw_data (data_type, company, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_timestamp ON raw_data (timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_company ON raw_data (company)")
            
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
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create indexes for processed_data
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_processed_data_type_timestamp ON processed_data (data_type, timestamp)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_processed_data_timestamp ON processed_data (timestamp)")
            
            # Features table - check if it exists
            existing_features = await conn.fetchval("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'features' AND table_schema = 'public'
            """)
            
            if existing_features:
                logger.info("✅ Features table already exists with correct structure")
            
            # Create indexes for features (using existing structure with company_ticker)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_features_company_timestamp ON features (company_ticker, calculated_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_features_name_timestamp ON features (feature_name, calculated_at)")
            
            # Credit scores table - check if it exists
            existing_scores = await conn.fetchval("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'credit_scores' AND table_schema = 'public'
            """)
            
            if existing_scores:
                logger.info("✅ Credit scores table already exists with correct structure")
            
            # Create indexes for credit_scores (using existing structure with company_ticker)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_credit_scores_company_timestamp ON credit_scores (company_ticker, calculated_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_credit_scores_timestamp ON credit_scores (calculated_at)")
            
            # Alerts table - check if it exists
            existing_alerts = await conn.fetchval("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'alerts' AND table_schema = 'public'
            """)
            
            if existing_alerts:
                logger.info("✅ Alerts table already exists with correct structure")
            
            # Create indexes for alerts (using existing structure with company_ticker)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_company_timestamp ON alerts (company_ticker, triggered_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type_timestamp ON alerts (alert_type, triggered_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status)")
    
    async def store_raw_data(self, data: Any, data_type: str, 
                           company: str = None, source: str = None,
                           language: str = 'en', timestamp: datetime = None) -> Optional[int]:
        """Store raw data in PostgreSQL"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Convert string timestamp to datetime if needed
        if isinstance(timestamp, str):
            try:
                # Handle different timestamp formats
                if 'T' in timestamp:
                    # ISO format
                    if timestamp.endswith('Z'):
                        timestamp = timestamp.replace('Z', '+00:00')
                    timestamp = datetime.fromisoformat(timestamp)
                else:
                    # Try other common formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d']:
                        try:
                            timestamp = datetime.strptime(timestamp, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # If all formats fail, use current time
                        timestamp = datetime.now()
            except Exception as e:
                logger.warning(f"Could not parse timestamp '{timestamp}': {e}, using current time")
                timestamp = datetime.now()
        
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
            
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

    async def get_data_by_company(self, company: str, data_type: str = None, 
                                 start_date: datetime = None, end_date: datetime = None,
                                 limit: int = 1000) -> List[Dict[str, Any]]:
        """Get data for a specific company"""
        try:
            query = "SELECT * FROM raw_data WHERE company = $1"
            params = [company]
            param_count = 1
            
            if data_type:
                param_count += 1
                query += f" AND data_type = ${param_count}"
                params.append(data_type)
            
            if start_date:
                param_count += 1
                query += f" AND timestamp >= ${param_count}"
                params.append(start_date)
            
            if end_date:
                param_count += 1
                query += f" AND timestamp <= ${param_count}"
                params.append(end_date)
            
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
            
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting data for company {company}: {e}")
            return []
    
    async def bulk_insert(self, table: str, data: List[Dict[str, Any]]) -> bool:
        """Bulk insert data into table - compatibility method for pipeline"""
        if not data:
            return True
        
        try:
            # Convert data to raw_data format
            raw_data_records = []
            for record in data:
                # Extract relevant fields
                content = record.get('content', record)
                metadata = {
                    'source_type': record.get('source_type', 'unknown'),
                    'content_type': record.get('content_type', 'unknown'),
                    'company_ticker': record.get('company_ticker'),
                    'language': record.get('language', 'en'),
                    'url': record.get('url'),
                    'published_date': record.get('published_date'),
                    'sentiment_score': record.get('sentiment_score')
                }
                
                # Convert timestamp to datetime if it's a string
                timestamp = record.get('published_date') or datetime.now()
                if isinstance(timestamp, str):
                    try:
                        # Handle different timestamp formats
                        if 'T' in timestamp:
                            # ISO format
                            if timestamp.endswith('Z'):
                                timestamp = timestamp.replace('Z', '+00:00')
                            timestamp = datetime.fromisoformat(timestamp)
                        else:
                            # Try other common formats
                            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d']:
                                try:
                                    timestamp = datetime.strptime(timestamp, fmt)
                                    break
                                except ValueError:
                                    continue
                            else:
                                # If all formats fail, use current time
                                timestamp = datetime.now()
                    except Exception as e:
                        logger.warning(f"Could not parse timestamp '{timestamp}': {e}, using current time")
                        timestamp = datetime.now()
                
                # Ensure timestamp is timezone-aware
                if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                raw_data_records.append({
                    'data_type': record.get('source_type', 'unknown'),
                    'company': record.get('company_ticker', 'unknown'),
                    'timestamp': timestamp,
                    'source': record.get('source_name', 'unknown'),
                    'language': record.get('language', 'en'),
                    'content': content,
                    'metadata': metadata
                })
            
            # Store each record
            for record in raw_data_records:
                await self.store_raw_data(
                    data=record['content'],
                    data_type=record['data_type'],
                    company=record['company'],
                    source=record['source'],
                    language=record['language'],
                    timestamp=record['timestamp']
                )
            
            logger.info(f"Bulk inserted {len(data)} records into {table}")
            return True
            
        except Exception as e:
            logger.error(f"Error in bulk insert: {e}")
            return False
