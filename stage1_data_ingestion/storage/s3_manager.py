"""
S3 data lake manager for storing and retrieving raw and processed data.
Handles data organization, versioning, and lifecycle management.
"""

import boto3
import asyncio
import aiofiles
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import json
import pickle
import gzip
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
from io import StringIO, BytesIO

logger = logging.getLogger(__name__)

class S3Manager:
    """S3 data lake operations manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bucket_name = config.get('s3_bucket', 'credit-intelligence-data-lake')
        self.region = config.get('aws_region', 'us-east-1')
        
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key')
            )
            self.s3_resource = boto3.resource(
                's3',
                region_name=self.region,
                aws_access_key_id=config.get('aws_access_key_id'),
                aws_secret_access_key=config.get('aws_secret_access_key')
            )
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            self.s3_client = None
            self.s3_resource = None
    
    def create_bucket_if_not_exists(self) -> bool:
        """Create S3 bucket if it doesn't exist"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Bucket {self.bucket_name} already exists")
            return True
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Created bucket {self.bucket_name}")
                    return True
                except ClientError as create_error:
                    logger.error(f"Error creating bucket: {create_error}")
                    return False
            else:
                logger.error(f"Error checking bucket: {e}")
                return False
    
    async def upload_raw_data(self, data: Any, data_type: str, 
                            company: str, timestamp: datetime = None) -> Optional[str]:
        """Upload raw data to S3 with organized structure"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return None
            
        if timestamp is None:
            timestamp = datetime.now()
            
        # Create organized key structure
        date_partition = timestamp.strftime('%Y/%m/%d')
        hour_partition = timestamp.strftime('%H')
        
        key = f"raw_data/{data_type}/{company}/{date_partition}/{hour_partition}/{timestamp.isoformat()}.json.gz"
        
        try:
            # Serialize and compress data
            json_data = json.dumps(data, default=str, ensure_ascii=False)
            compressed_data = gzip.compress(json_data.encode('utf-8'))
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=compressed_data,
                ContentType='application/json',
                ContentEncoding='gzip',
                Metadata={
                    'data_type': data_type,
                    'company': company,
                    'timestamp': timestamp.isoformat(),
                    'size_uncompressed': str(len(json_data))
                }
            )
            
            logger.info(f"Uploaded raw data to s3://{self.bucket_name}/{key}")
            return key
            
        except Exception as e:
            logger.error(f"Error uploading raw data: {e}")
            return None
    
    async def upload_processed_data(self, data: pd.DataFrame, data_type: str,
                                  timestamp: datetime = None) -> Optional[str]:
        """Upload processed data as Parquet for efficient querying"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return None
            
        if timestamp is None:
            timestamp = datetime.now()
            
        date_partition = timestamp.strftime('%Y/%m/%d')
        key = f"processed_data/{data_type}/{date_partition}/{timestamp.isoformat()}.parquet"
        
        try:
            # Convert DataFrame to Parquet
            parquet_buffer = BytesIO()
            data.to_parquet(parquet_buffer, index=False, compression='snappy')
            parquet_buffer.seek(0)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=parquet_buffer.getvalue(),
                ContentType='application/octet-stream',
                Metadata={
                    'data_type': data_type,
                    'timestamp': timestamp.isoformat(),
                    'row_count': str(len(data)),
                    'format': 'parquet'
                }
            )
            
            logger.info(f"Uploaded processed data to s3://{self.bucket_name}/{key}")
            return key
            
        except Exception as e:
            logger.error(f"Error uploading processed data: {e}")
            return None
    
    async def download_data(self, key: str) -> Optional[Any]:
        """Download data from S3"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return None
            
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read()
            
            # Handle different file types
            if key.endswith('.json.gz'):
                # Decompress and parse JSON
                decompressed = gzip.decompress(content)
                return json.loads(decompressed.decode('utf-8'))
            elif key.endswith('.parquet'):
                # Read Parquet
                return pd.read_parquet(BytesIO(content))
            elif key.endswith('.json'):
                # Parse JSON
                return json.loads(content.decode('utf-8'))
            else:
                # Return raw content
                return content
                
        except Exception as e:
            logger.error(f"Error downloading data from {key}: {e}")
            return None
    
    async def list_data_files(self, data_type: str, company: str = None,
                            start_date: datetime = None, end_date: datetime = None) -> List[str]:
        """List data files with optional filtering"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return []
            
        try:
            prefix = f"raw_data/{data_type}/"
            if company:
                prefix += f"{company}/"
                
            if start_date:
                prefix += start_date.strftime('%Y/%m/%d/')
                
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                
                # Filter by date range if specified
                if start_date or end_date:
                    # Extract date from key structure
                    try:
                        date_part = key.split('/')[3:6]  # ['YYYY', 'MM', 'DD']
                        file_date = datetime.strptime('/'.join(date_part), '%Y/%m/%d')
                        
                        if start_date and file_date < start_date:
                            continue
                        if end_date and file_date > end_date:
                            continue
                    except (IndexError, ValueError):
                        continue
                        
                files.append(key)
                
            logger.info(f"Found {len(files)} files matching criteria")
            return files
            
        except Exception as e:
            logger.error(f"Error listing data files: {e}")
            return []
    
    async def delete_old_data(self, data_type: str, days_to_keep: int = 30) -> int:
        """Delete old data files to manage storage costs"""
        if not self.s3_client:
            logger.error("S3 client not available")
            return 0
            
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0
        
        try:
            prefix = f"raw_data/{data_type}/"
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
            
            for page in pages:
                for obj in page.get('Contents', []):
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        self.s3_client.delete_object(
                            Bucket=self.bucket_name,
                            Key=obj['Key']
                        )
                        deleted_count += 1
                        
            logger.info(f"Deleted {deleted_count} old files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return 0
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored data"""
        if not self.s3_client:
            return {}
            
        try:
            stats = {
                'total_objects': 0,
                'total_size_bytes': 0,
                'data_types': {},
                'companies': set(),
                'date_range': {'earliest': None, 'latest': None}
            }
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name)
            
            for page in pages:
                for obj in page.get('Contents', []):
                    stats['total_objects'] += 1
                    stats['total_size_bytes'] += obj['Size']
                    
                    # Parse key structure
                    key_parts = obj['Key'].split('/')
                    if len(key_parts) >= 4:
                        data_type = key_parts[1]
                        company = key_parts[2]
                        
                        if data_type not in stats['data_types']:
                            stats['data_types'][data_type] = 0
                        stats['data_types'][data_type] += 1
                        
                        stats['companies'].add(company)
                        
                        # Track date range
                        last_modified = obj['LastModified'].replace(tzinfo=None)
                        if stats['date_range']['earliest'] is None or last_modified < stats['date_range']['earliest']:
                            stats['date_range']['earliest'] = last_modified
                        if stats['date_range']['latest'] is None or last_modified > stats['date_range']['latest']:
                            stats['date_range']['latest'] = last_modified
            
            stats['companies'] = list(stats['companies'])
            return stats
            
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}
