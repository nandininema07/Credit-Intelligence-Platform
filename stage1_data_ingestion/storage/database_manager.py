from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
from datetime import datetime
from typing import List, Dict
import logging
from ..data_processing.data_models import DataPoint, RawData, Base
from ..data_processing.text_processor import DataProcessor

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config: Dict):
        self.config = config
        self.s3_client = None
        self.db_engine = None
        
        
        if config['database']['url']:
            self.db_engine = create_engine(config['database']['url'])
            Base.metadata.create_all(self.db_engine)
            Session = sessionmaker(bind=self.db_engine)
            self.db_session = Session()
    
    def store_to_database(self, data_points: List[DataPoint]) -> bool:
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