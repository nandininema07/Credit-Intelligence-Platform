#!/usr/bin/env python3
"""
Script to help set up MinIO server for the Credit Intelligence Platform.
This script provides instructions and Docker commands to run MinIO locally.
"""

import os
import sys
import subprocess
import platform

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker not found or not running")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

def run_minio_docker():
    """Run MinIO using Docker"""
    print("🚀 Starting MinIO server with Docker...")
    
    # Create data directory
    data_dir = os.path.join(os.getcwd(), 'minio_data')
    os.makedirs(data_dir, exist_ok=True)
    print(f"📁 Data directory: {data_dir}")
    
    # Docker command
    docker_cmd = [
        'docker', 'run', '-d',
        '--name', 'minio-server',
        '-p', '9000:9000',
        '-p', '9001:9001',
        '-v', f'{data_dir}:/data',
        '-e', 'MINIO_ROOT_USER=admin',
        '-e', 'MINIO_ROOT_PASSWORD=password123',
        'quay.io/minio/minio:latest',
        'server', '/data', '--console-address', ':9001'
    ]
    
    try:
        # Stop existing container if running
        subprocess.run(['docker', 'stop', 'minio-server'], capture_output=True)
        subprocess.run(['docker', 'rm', 'minio-server'], capture_output=True)
        
        # Start new container
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ MinIO server started successfully!")
            print("\n🌐 Access URLs:")
            print("   - MinIO API: http://localhost:9000")
            print("   - MinIO Console: http://localhost:9001")
            print("\n🔑 Login credentials:")
            print("   - Username: admin")
            print("   - Password: password123")
            return True
        else:
            print(f"❌ Failed to start MinIO: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting MinIO: {e}")
        return False

def test_minio_connection():
    """Test MinIO connection"""
    print("\n🧪 Testing MinIO connection...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Create MinIO client
        s3_client = boto3.client(
            's3',
            endpoint_url='http://localhost:9000',
            aws_access_key_id='admin',
            aws_secret_access_key='password123',
            region_name='us-east-1'
        )
        
        # Test connection
        response = s3_client.list_buckets()
        print("✅ MinIO connection successful!")
        print(f"📦 Found {len(response['Buckets'])} buckets")
        
        # Create bucket if it doesn't exist
        bucket_name = 'credit-intelligence-data-lake'
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✅ Bucket '{bucket_name}' already exists")
        except ClientError:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"✅ Created bucket '{bucket_name}'")
        
        return True
        
    except ImportError:
        print("❌ boto3 not installed. Install with: pip install boto3")
        return False
    except Exception as e:
        print(f"❌ MinIO connection failed: {e}")
        return False

def show_manual_instructions():
    """Show manual setup instructions"""
    print("\n📋 Manual Setup Instructions:")
    print("=" * 50)
    print("\n1. Install Docker:")
    print("   - Windows/Mac: Download from https://docker.com")
    print("   - Linux: sudo apt-get install docker.io")
    print("\n2. Start MinIO server:")
    print("   docker run -d --name minio-server \\")
    print("     -p 9000:9000 -p 9001:9001 \\")
    print("     -v $(pwd)/minio_data:/data \\")
    print("     -e MINIO_ROOT_USER=admin \\")
    print("     -e MINIO_ROOT_PASSWORD=password123 \\")
    print("     quay.io/minio/minio:latest \\")
    print("     server /data --console-address :9001")
    print("\n3. Access MinIO:")
    print("   - API: http://localhost:9000")
    print("   - Console: http://localhost:9001")
    print("   - Login: admin / password123")
    print("\n4. Create bucket:")
    print("   - Go to MinIO Console")
    print("   - Click 'Create Bucket'")
    print("   - Name: credit-intelligence-data-lake")

def show_alternative_setup():
    """Show alternative setup methods"""
    print("\n🔄 Alternative Setup Methods:")
    print("=" * 50)
    print("\n1. Direct Download (Linux/Mac):")
    print("   wget https://dl.min.io/server/minio/release/linux-amd64/minio")
    print("   chmod +x minio")
    print("   ./minio server /data --console-address :9001")
    print("\n2. Using Docker Compose:")
    print("   Create docker-compose.yml:")
    print("   version: '3.8'")
    print("   services:")
    print("     minio:")
    print("       image: quay.io/minio/minio:latest")
    print("       ports:")
    print("         - '9000:9000'")
    print("         - '9001:9001'")
    print("       environment:")
    print("         MINIO_ROOT_USER: admin")
    print("         MINIO_ROOT_PASSWORD: password123")
    print("       volumes:")
    print("         - ./minio_data:/data")
    print("       command: server /data --console-address :9001")
    print("\n   Then run: docker-compose up -d")

def main():
    """Main function"""
    print("🚀 MinIO Setup for Credit Intelligence Platform")
    print("=" * 50)
    
    # Check if Docker is available
    if check_docker():
        print("\n🐳 Docker detected! Attempting automatic setup...")
        
        if run_minio_docker():
            print("\n⏳ Waiting for MinIO to start...")
            import time
            time.sleep(5)
            
            if test_minio_connection():
                print("\n🎉 MinIO setup completed successfully!")
                print("\n📝 Next steps:")
                print("1. Update your .env file with MinIO credentials:")
                print("   MINIO_ENDPOINT=http://localhost:9000")
                print("   MINIO_ACCESS_KEY=admin")
                print("   MINIO_SECRET_KEY=password123")
                print("   MINIO_BUCKET=credit-intelligence-data-lake")
                print("\n2. Test your platform: python test_minio_connection.py")
            else:
                print("\n⚠️  MinIO started but connection test failed.")
                print("Check the MinIO console at http://localhost:9001")
        else:
            print("\n❌ Automatic setup failed. Showing manual instructions...")
            show_manual_instructions()
    else:
        print("\n❌ Docker not available. Showing manual instructions...")
        show_manual_instructions()
        show_alternative_setup()
    
    print("\n📚 For more information:")
    print("   - MinIO Documentation: https://docs.min.io/")
    print("   - Docker Documentation: https://docs.docker.com/")

if __name__ == "__main__":
    main()

