#!/usr/bin/env python3
"""
Test script to verify MinIO connection and functionality
"""

import os
import boto3
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

def test_minio_connection():
    """Test basic MinIO connection"""
    
    print("🧪 Testing MinIO Connection")
    print("=" * 50)
    
    # Get MinIO configuration from environment
    endpoint = os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
    access_key = os.getenv('MINIO_ACCESS_KEY', 'admin')
    secret_key = os.getenv('MINIO_SECRET_KEY', 'password123')
    bucket_name = os.getenv('MINIO_BUCKET', 'credit-intelligence-data-lake')
    
    print(f"🔗 Endpoint: {endpoint}")
    print(f"👤 Access Key: {access_key}")
    print(f"🔑 Secret Key: {secret_key[:8]}...")
    print(f"📦 Bucket: {bucket_name}")
    
    try:
        # Create MinIO client
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='us-east-1'
        )
        
        # Test connection by listing buckets
        response = s3_client.list_buckets()
        print(f"✅ Connection successful! Found {len(response['Buckets'])} buckets")
        
        # Check if our bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' exists")
        except ClientError:
            print(f"⚠️  Bucket '{bucket_name}' doesn't exist, creating...")
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"✅ Created bucket '{bucket_name}'")
        
        return s3_client, bucket_name
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None, None

def test_data_upload(s3_client, bucket_name):
    """Test uploading raw data"""
    
    print("\n📤 Testing Data Upload")
    print("=" * 50)
    
    try:
        # Create test data
        test_data = {
            'company': 'TEST_COMPANY',
            'timestamp': datetime.now().isoformat(),
            'data_type': 'news',
            'content': 'This is a test news article for credit intelligence platform.',
            'source': 'test_source',
            'sentiment_score': 0.5
        }
        
        # Create key with organized structure
        timestamp = datetime.now()
        date_partition = timestamp.strftime('%Y/%m/%d')
        hour_partition = timestamp.strftime('%H')
        
        key = f"raw_data/news/TEST_COMPANY/{date_partition}/{hour_partition}/{timestamp.isoformat()}.json"
        
        # Upload test data
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(test_data, indent=2),
            ContentType='application/json',
            Metadata={
                'data_type': 'news',
                'company': 'TEST_COMPANY',
                'timestamp': timestamp.isoformat()
            }
        )
        
        print(f"✅ Uploaded test data to {key}")
        return key
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return None

def test_data_download(s3_client, bucket_name, key):
    """Test downloading data"""
    
    print("\n📥 Testing Data Download")
    print("=" * 50)
    
    try:
        # Download the test data
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read().decode('utf-8')
        data = json.loads(content)
        
        print("✅ Download successful!")
        print(f"📄 Content: {data['content'][:50]}...")
        print(f"🏢 Company: {data['company']}")
        print(f"📅 Timestamp: {data['timestamp']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def test_parquet_upload(s3_client, bucket_name):
    """Test uploading processed data as Parquet"""
    
    print("\n📊 Testing Parquet Upload")
    print("=" * 50)
    
    try:
        # Create test DataFrame
        test_df = pd.DataFrame({
            'company': ['TEST_COMPANY'] * 5,
            'timestamp': [datetime.now()] * 5,
            'feature_1': [1.1, 2.2, 3.3, 4.4, 5.5],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'score': [0.8, 0.7, 0.9, 0.6, 0.85]
        })
        
        # Convert to Parquet
        parquet_buffer = test_df.to_parquet(index=False, compression='snappy')
        
        # Upload to MinIO
        timestamp = datetime.now()
        date_partition = timestamp.strftime('%Y/%m/%d')
        key = f"processed_data/test_features/{date_partition}/{timestamp.isoformat()}.parquet"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=parquet_buffer,
            ContentType='application/octet-stream',
            Metadata={
                'data_type': 'test_features',
                'timestamp': timestamp.isoformat(),
                'row_count': str(len(test_df)),
                'format': 'parquet'
            }
        )
        
        print(f"✅ Uploaded Parquet data to {key}")
        print(f"📊 DataFrame shape: {test_df.shape}")
        
        return key
        
    except Exception as e:
        print(f"❌ Parquet upload failed: {e}")
        return None

def test_listing_files(s3_client, bucket_name):
    """Test listing files with filtering"""
    
    print("\n📋 Testing File Listing")
    print("=" * 50)
    
    try:
        # List all files
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' in response:
            print(f"📁 Total files in bucket: {len(response['Contents'])}")
            
            # Group by prefix
            files_by_type = {}
            for obj in response['Contents']:
                key = obj['Key']
                prefix = key.split('/')[0] if '/' in key else 'root'
                
                if prefix not in files_by_type:
                    files_by_type[prefix] = []
                files_by_type[prefix].append(key)
            
            print("\n📂 Files by type:")
            for file_type, files in files_by_type.items():
                print(f"   {file_type}: {len(files)} files")
                for file in files[:3]:  # Show first 3 files
                    print(f"     - {file}")
                if len(files) > 3:
                    print(f"     ... and {len(files) - 3} more")
        else:
            print("📁 Bucket is empty")
        
        return True
        
    except Exception as e:
        print(f"❌ File listing failed: {e}")
        return False

def test_data_statistics(s3_client, bucket_name):
    """Test getting data statistics"""
    
    print("\n📊 Testing Data Statistics")
    print("=" * 50)
    
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        
        if 'Contents' in response:
            total_size = sum(obj['Size'] for obj in response['Contents'])
            total_files = len(response['Contents'])
            
            print(f"📦 Total files: {total_files}")
            print(f"💾 Total size: {total_size / 1024:.2f} KB")
            
            # Get file types
            file_types = {}
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.json'):
                    file_types['JSON'] = file_types.get('JSON', 0) + 1
                elif key.endswith('.parquet'):
                    file_types['Parquet'] = file_types.get('Parquet', 0) + 1
                else:
                    file_types['Other'] = file_types.get('Other', 0) + 1
            
            print("\n📄 File types:")
            for file_type, count in file_types.items():
                print(f"   {file_type}: {count} files")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 MinIO Connection Test for Credit Intelligence Platform")
    print("=" * 60)
    
    # Test connection
    s3_client, bucket_name = test_minio_connection()
    
    if s3_client is None:
        print("\n❌ Cannot proceed without MinIO connection")
        print("Please ensure MinIO server is running and credentials are correct")
        return
    
    # Run all tests
    tests = [
        ("Data Upload", lambda: test_data_upload(s3_client, bucket_name)),
        ("Data Download", lambda: test_data_download(s3_client, bucket_name, 
                                                    test_data_upload(s3_client, bucket_name))),
        ("Parquet Upload", lambda: test_parquet_upload(s3_client, bucket_name)),
        ("File Listing", lambda: test_listing_files(s3_client, bucket_name)),
        ("Data Statistics", lambda: test_data_statistics(s3_client, bucket_name))
    ]
    
    print("\n🧪 Running all tests...")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MinIO is working correctly.")
        print("\n📝 Your platform is ready to use MinIO for data storage.")
    else:
        print("⚠️  Some tests failed. Please check your MinIO configuration.")
    
    print("\n🔗 MinIO Console: http://localhost:9001")
    print("   Login: admin / password123")

if __name__ == "__main__":
    main()

