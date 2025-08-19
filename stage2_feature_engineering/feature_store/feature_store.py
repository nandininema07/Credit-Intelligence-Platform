"""
Feature store implementation for ML feature management.
Handles feature storage, retrieval, and versioning.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os

logger = logging.getLogger(__name__)

class FeatureStore:
    """Feature store for ML features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = config.get('storage_path', './feature_store')
        self.database_manager = None
        self.ensure_storage_directory()
        
    def ensure_storage_directory(self):
        """Ensure storage directory exists"""
        os.makedirs(self.storage_path, exist_ok=True)
        
    async def store_features(self, company: str, features: Dict[str, float], 
                           timestamp: datetime = None) -> str:
        """Store features for a company"""
        if timestamp is None:
            timestamp = datetime.now()
            
        feature_id = f"{company}_{timestamp.isoformat()}"
        
        feature_record = {
            'id': feature_id,
            'company': company,
            'features': features,
            'timestamp': timestamp.isoformat(),
            'feature_count': len(features),
            'created_at': datetime.now().isoformat()
        }
        
        # Store to file
        file_path = os.path.join(self.storage_path, f"{feature_id}.json")
        with open(file_path, 'w') as f:
            json.dump(feature_record, f, indent=2)
            
        logger.info(f"Stored {len(features)} features for {company}")
        return feature_id
    
    async def get_features(self, company: str, timestamp: datetime = None) -> Optional[Dict[str, float]]:
        """Get features for a company at a specific time"""
        if timestamp is None:
            # Get latest features
            return await self.get_latest_features(company)
            
        feature_id = f"{company}_{timestamp.isoformat()}"
        file_path = os.path.join(self.storage_path, f"{feature_id}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                feature_record = json.load(f)
                return feature_record.get('features', {})
        
        return None
    
    async def get_latest_features(self, company: str) -> Optional[Dict[str, float]]:
        """Get latest features for a company"""
        company_files = []
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith(f"{company}_") and filename.endswith('.json'):
                company_files.append(filename)
        
        if not company_files:
            return None
            
        # Sort by timestamp (latest first)
        company_files.sort(reverse=True)
        latest_file = company_files[0]
        
        file_path = os.path.join(self.storage_path, latest_file)
        with open(file_path, 'r') as f:
            feature_record = json.load(f)
            return feature_record.get('features', {})
    
    async def get_historical_features(self, company: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get historical features for a company"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        historical_features = []
        
        for filename in os.listdir(self.storage_path):
            if filename.startswith(f"{company}_") and filename.endswith('.json'):
                file_path = os.path.join(self.storage_path, filename)
                
                with open(file_path, 'r') as f:
                    feature_record = json.load(f)
                    
                feature_timestamp = datetime.fromisoformat(feature_record['timestamp'])
                
                if feature_timestamp >= cutoff_date:
                    historical_features.append(feature_record)
        
        # Sort by timestamp
        historical_features.sort(key=lambda x: x['timestamp'])
        return historical_features
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        # This would typically come from model training results
        # For now, return placeholder values
        return {
            'sentiment_mean': 0.15,
            'price_trend_slope': 0.12,
            'liquidity_current_ratio': 0.10,
            'profitability_net_profit_margin': 0.09,
            'leverage_debt_to_equity': 0.08,
            'market_pe_ratio': 0.07,
            'volume_trend_slope': 0.06,
            'earnings_growth_volatility': 0.05
        }
    
    async def create_feature_matrix(self, companies: List[str], 
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> pd.DataFrame:
        """Create feature matrix for multiple companies"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
            
        feature_matrix = []
        
        for company in companies:
            historical_features = await self.get_historical_features(company, 
                                                                   (end_date - start_date).days)
            
            for feature_record in historical_features:
                feature_timestamp = datetime.fromisoformat(feature_record['timestamp'])
                
                if start_date <= feature_timestamp <= end_date:
                    row = {'company': company, 'timestamp': feature_timestamp}
                    row.update(feature_record['features'])
                    feature_matrix.append(row)
        
        df = pd.DataFrame(feature_matrix)
        logger.info(f"Created feature matrix with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def get_feature_statistics(self, feature_name: str, companies: List[str] = None) -> Dict[str, float]:
        """Get statistics for a specific feature"""
        feature_values = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.json'):
                file_path = os.path.join(self.storage_path, filename)
                
                with open(file_path, 'r') as f:
                    feature_record = json.load(f)
                    
                # Filter by companies if specified
                if companies and feature_record['company'] not in companies:
                    continue
                    
                if feature_name in feature_record['features']:
                    feature_values.append(feature_record['features'][feature_name])
        
        if not feature_values:
            return {}
            
        return {
            'count': len(feature_values),
            'mean': float(np.mean(feature_values)),
            'std': float(np.std(feature_values)),
            'min': float(np.min(feature_values)),
            'max': float(np.max(feature_values)),
            'median': float(np.median(feature_values)),
            'p25': float(np.percentile(feature_values, 25)),
            'p75': float(np.percentile(feature_values, 75))
        }
