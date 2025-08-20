#!/usr/bin/env python3
"""
Script to help set up the .env file with API keys for the Credit Intelligence Platform.
This script will create a .env file with all the necessary API keys and configuration.
"""

import os
import sys

def create_env_file():
    """Create a .env file with all necessary configuration"""
    
    env_content = """# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/credit_intelligence
DB_HOST=localhost
DB_PORT=5432
DB_NAME=credit_intelligence
DB_USER=postgres
DB_PASSWORD=password
DB_APP_USER=credit_intelligence_user
DB_APP_PASSWORD=secure_password_123

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# External API Keys - Replace these with your actual API keys
NEWSAPI_KEY=your-newsapi-key-here
ALPHA_VANTAGE_KEY=your-alpha-vantage-key-here
FRED_KEY=your-fred-key-here
FINNHUB_KEY=your-finnhub-key-here
POLYGON_KEY=your-polygon-key-here

# Social Media API Keys - Replace these with your actual API keys
TWITTER_BEARER_TOKEN=your-twitter-bearer-token-here
REDDIT_CLIENT_ID=your-reddit-client-id-here
REDDIT_CLIENT_SECRET=your-reddit-client-secret-here

# AI/ML API Keys - Replace these with your actual API keys
OPENAI_API_KEY=your-openai-api-key-here
HUGGINGFACE_TOKEN=your-huggingface-token-here

# Notification Services - Replace these with your actual API keys
SLACK_BOT_TOKEN=your-slack-bot-token-here
SLACK_WEBHOOK_URL=your-slack-webhook-url-here
JIRA_API_TOKEN=your-jira-api-token-here

# Email Configuration - Replace these with your actual credentials
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password-here
SMTP_USE_TLS=true

# MinIO Configuration - Replace these with your actual MinIO credentials
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password123
MINIO_BUCKET=credit-intelligence-data-lake
MINIO_SECURE=false

# Twilio Configuration - Replace these with your actual Twilio credentials
TWILIO_ACCOUNT_SID=your-twilio-account-sid-here
TWILIO_AUTH_TOKEN=your-twilio-auth-token-here
TWILIO_FROM_NUMBER=+1234567890

# Application Settings
SECRET_KEY=your-secret-key-change-in-production
DEBUG=true
LOG_LEVEL=INFO

# TensorFlow Settings
TF_ENABLE_ONEDNN_OPTS=0
"""
    
    # Check if .env file already exists
    if os.path.exists('.env'):
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("❌ Setup cancelled. .env file was not modified.")
            return False
    
    # Write the .env file
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
        print("\n📝 Next steps:")
        print("1. Edit the .env file and replace the placeholder values with your actual API keys")
        print("2. Make sure your database is running and accessible")
        print("3. Set up MinIO server (see setup_minio.py for instructions)")
        print("4. Run the pipeline with: python run_pipeline.py")
        print("\n🔑 Required API keys to configure:")
        print("   - NEWSAPI_KEY: Get from https://newsapi.org/")
        print("   - ALPHA_VANTAGE_KEY: Get from https://www.alphavantage.co/")
        print("   - TWITTER_BEARER_TOKEN: Get from https://developer.twitter.com/")
        print("   - OPENAI_API_KEY: Get from https://platform.openai.com/")
        print("   - TWILIO_ACCOUNT_SID & TWILIO_AUTH_TOKEN: Get from https://www.twilio.com/")
        print("   - And others as needed for your use case")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Credit Intelligence Platform - Environment Setup")
    print("=" * 50)
    
    success = create_env_file()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("Please edit the .env file with your actual API keys before running the pipeline.")
    else:
        print("\n💥 Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
