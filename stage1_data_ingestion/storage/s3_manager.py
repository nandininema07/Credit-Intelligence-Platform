import boto3
import json
import time
from datetime import datetime
from typing import List, Dict
import logging
from dataclasses import asdict
from ..data_processing.data_models import DataPoint

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self, config: Dict):
        self.config = config
        self.s3_client = None
        self.db_engine = None
        
        if config['aws']['access_key']:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config['aws']['access_key'],
                aws_secret_access_key=config['aws']['secret_key'],
                region_name=config['aws']['region']
            )
        ()
    
    def store_to_s3(self, data_points: List[DataPoint]) -> bool:
        if not self.s3_client:
            return False
        
        try:
            # Group data by date for partitioning
            daily_data = {}
            for dp in data_points:
                date_key = (dp.published_date or datetime.utcnow()).strftime('%Y-%m-%d')
                if date_key not in daily_data:
                    daily_data[date_key] = []
                daily_data[date_key].append(asdict(dp))
            
            # Upload to S3
            for date_key, data in daily_data.items():
                key = f"raw_data/date={date_key}/data_{int(time.time())}.json"
                
                self.s3_client.put_object(
                    Bucket=self.config['aws']['s3_bucket'],
                    Key=key,
                    Body=json.dumps(data, default=str),
                    ContentType='application/json'
                )
            
            logger.info(f"Stored {len(data_points)} data points to S3")
            return True
        
        except Exception as e:
            logger.error(f"Error storing to S3: {e}")
            return False
    
        if not self.db_engine:
            return False
        
        try:
            records = []
            for dp in data_points:
                record = RawData(
                    id=DataProcessor.generate_id(dp.source_type, dp.url or '', dp.published_date or datetime.utcnow()),
                    source_type=dp.source_type,
                    source_name=dp.source_name,
                    company_ticker=dp.company_ticker,
                    company_name=dp.company_name,
                    content_type=dp.content_type,
                    language=dp.language,
                    title=dp.title,
                    content=dp.content,
                    url=dp.url,
                    published_date=dp.published_date,
                    sentiment_score=DataProcessor.calculate_sentiment(dp.content),
                    metadata=json.dumps(dp.metadata, default=str)
                )
                records.append(record)
            
            # Bulk insert
            self.db_session.bulk_save_objects(records)
            self.db_session.commit()
            
            logger.info(f"Stored {len(data_points)} data points to database")
            return True
        
        except Exception as e:
            logger.error(f"Error storing to database: {e}")
            self.db_session.rollback()
            return False