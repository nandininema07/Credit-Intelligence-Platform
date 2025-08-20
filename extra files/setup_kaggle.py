#!/usr/bin/env python3
"""
Kaggle Setup Script
Helps users set up Kaggle integration for credit intelligence training.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_kaggle_credentials():
    """Set up Kaggle credentials"""
    print("=" * 60)
    print("KAGGLE SETUP FOR CREDIT INTELLIGENCE PLATFORM")
    print("=" * 60)
    print()
    
    # Check if credentials are already set
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_API_KEY')
    
    if kaggle_username and kaggle_key:
        print(f"✅ Kaggle credentials already configured:")
        print(f"   Username: {kaggle_username}")
        print(f"   API Key: {kaggle_key[:8]}...{kaggle_key[-4:]}")
        print()
        
        choice = input("Do you want to update these credentials? (y/N): ").strip().lower()
        if choice != 'y':
            print("Keeping existing credentials.")
            return True
    
    print("To use real credit datasets from Kaggle, you need to:")
    print("1. Create a Kaggle account at https://www.kaggle.com")
    print("2. Go to Account Settings -> API -> Create New API Token")
    print("3. Download the kaggle.json file")
    print("4. Extract your username and API key")
    print()
    
    # Get credentials from user
    username = input("Enter your Kaggle username: ").strip()
    if not username:
        print("❌ Username is required")
        return False
    
    api_key = input("Enter your Kaggle API key: ").strip()
    if not api_key:
        print("❌ API key is required")
        return False
    
    # Validate credentials format
    if len(api_key) < 20:
        print("❌ API key seems too short. Please check your Kaggle API key.")
        return False
    
    # Set environment variables
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_API_KEY'] = api_key
    
    print()
    print("✅ Kaggle credentials configured!")
    print(f"   Username: {username}")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Test connection
    print()
    print("Testing Kaggle connection...")
    if test_kaggle_connection():
        print("✅ Kaggle connection successful!")
        return True
    else:
        print("❌ Kaggle connection failed. Please check your credentials.")
        return False

def test_kaggle_connection():
    """Test Kaggle API connection"""
    try:
        from stage2_feature_engineering.kaggle_data_integration import KaggleDataIntegration
        
        config = {
            'kaggle_username': os.getenv('KAGGLE_USERNAME'),
            'kaggle_key': os.getenv('KAGGLE_API_KEY')
        }
        
        kaggle_data = KaggleDataIntegration(config)
        
        # Test by getting dataset info
        datasets = kaggle_data.get_dataset_info()
        if datasets:
            print(f"   Found {len(datasets)} available datasets")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"   Error: {e}")
        return False

def show_available_datasets():
    """Show available Kaggle datasets"""
    print()
    print("=" * 60)
    print("AVAILABLE KAGGLE CREDIT DATASETS")
    print("=" * 60)
    
    try:
        from stage2_feature_engineering.kaggle_data_integration import KaggleDataIntegration
        
        config = {
            'kaggle_username': os.getenv('KAGGLE_USERNAME', ''),
            'kaggle_key': os.getenv('KAGGLE_API_KEY', '')
        }
        
        kaggle_data = KaggleDataIntegration(config)
        datasets = kaggle_data.get_dataset_info()
        
        for name, info in datasets.items():
            print(f"\n📊 {name.upper().replace('_', ' ')}")
            print(f"   Description: {info['description']}")
            print(f"   Features: {info['feature_count']}")
            print(f"   Target: {info['target']}")
            print(f"   Classes: {info['target_classes']}")
            print(f"   Dataset ID: {info['id']}")
            
    except Exception as e:
        print(f"Error loading dataset info: {e}")

async def download_sample_dataset():
    """Download a sample dataset to test the setup"""
    print()
    print("=" * 60)
    print("DOWNLOADING SAMPLE DATASET")
    print("=" * 60)
    
    try:
        from stage2_feature_engineering.kaggle_data_integration import KaggleDataIntegration
        
        config = {
            'kaggle_username': os.getenv('KAGGLE_USERNAME'),
            'kaggle_key': os.getenv('KAGGLE_API_KEY'),
            'data_cache_path': './kaggle_data/'
        }
        
        kaggle_data = KaggleDataIntegration(config)
        
        # Try to download credit score classification dataset
        print("Downloading credit score classification dataset...")
        X, y = await kaggle_data.get_training_dataset('credit_score_classification', sample_size=1000)
        
        if X is not None and len(X) > 0:
            print(f"✅ Successfully downloaded dataset!")
            print(f"   Samples: {len(X)}")
            print(f"   Features: {len(X.columns)}")
            print(f"   Target distribution: {y.value_counts().to_dict()}")
            return True
        else:
            print("❌ Failed to download dataset")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

def create_environment_file():
    """Create .env file with Kaggle credentials"""
    env_file = Path('.env')
    
    if env_file.exists():
        print(f"⚠️  .env file already exists at {env_file}")
        choice = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if choice != 'y':
            return
    
    try:
        with open(env_file, 'w') as f:
            f.write(f"# Kaggle API Credentials\n")
            f.write(f"KAGGLE_USERNAME={os.getenv('KAGGLE_USERNAME', '')}\n")
            f.write(f"KAGGLE_API_KEY={os.getenv('KAGGLE_API_KEY', '')}\n")
            f.write(f"\n")
            f.write(f"# Other API Keys (optional)\n")
            f.write(f"ALPHA_VANTAGE_API_KEY=\n")
            f.write(f"NEWS_API_KEY=\n")
        
        print(f"✅ Created .env file at {env_file}")
        print("   You can now load these credentials automatically")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

async def main():
    """Main setup function"""
    print("Setting up Kaggle integration for Credit Intelligence Platform...")
    print()
    
    # Step 1: Set up credentials
    if not setup_kaggle_credentials():
        print("❌ Setup failed. Please check your Kaggle credentials.")
        sys.exit(1)
    
    # Step 2: Show available datasets
    show_available_datasets()
    
    # Step 3: Test download
    print()
    choice = input("Do you want to test downloading a sample dataset? (Y/n): ").strip().lower()
    if choice != 'n':
        if await download_sample_dataset():
            print("✅ Dataset download successful!")
        else:
            print("❌ Dataset download failed")
    
    # Step 4: Create environment file
    print()
    choice = input("Do you want to create a .env file to save your credentials? (Y/n): ").strip().lower()
    if choice != 'n':
        create_environment_file()
    
    print()
    print("=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("You can now run the model retraining script:")
    print("   python retrain_models.py")
    print()
    print("The system will automatically:")
    print("1. Download real credit datasets from Kaggle")
    print("2. Process and engineer features")
    print("3. Train models with realistic data")
    print("4. Validate against overfitting")
    print()
    print("Expected results: 70-85% accuracy (realistic for credit models)")

if __name__ == "__main__":
    import asyncio
    
    # Check if we're in an async context
    try:
        asyncio.run(main())
    except RuntimeError:
        # If not in async context, run directly
        main()
