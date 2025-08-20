"""
Feature scaling and normalization for Stage 2.
Handles different scaling methods for financial and NLP features.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

logger = logging.getLogger(__name__)

class FeatureScaler:
    """Feature scaling with multiple methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.feature_groups = self._setup_feature_groups()
        self.fitted = False
        
    def _setup_feature_groups(self) -> Dict[str, Dict[str, Any]]:
        """Setup feature groups with appropriate scaling methods"""
        default_groups = {
            'sentiment': {
                'method': 'standard',
                'features': ['sentiment_score', 'sentiment_mean', 'sentiment_std'],
                'robust': True
            },
            'financial_ratios': {
                'method': 'robust',
                'features': ['debt_ratio', 'pe_ratio', 'roe', 'roa'],
                'robust': True
            },
            'financial_amounts': {
                'method': 'power',
                'features': ['revenue', 'profit', 'market_cap', 'volume'],
                'robust': False
            },
            'prices': {
                'method': 'minmax',
                'features': ['stock_price', 'price_change'],
                'robust': False
            },
            'volatility': {
                'method': 'standard',
                'features': ['volatility', 'price_volatility'],
                'robust': True
            },
            'counts': {
                'method': 'power',
                'features': ['article_count', 'social_count', 'mention_count'],
                'robust': False
            }
        }
        
        return self.config.get('feature_groups', default_groups)
    
    def fit(self, data: pd.DataFrame) -> 'FeatureScaler':
        """Fit scalers on training data"""
        if data.empty:
            logger.warning("Empty data provided for fitting scalers")
            return self
        
        for group_name, group_config in self.feature_groups.items():
            # Find matching features in data
            group_features = []
            for feature_pattern in group_config['features']:
                matching_features = [col for col in data.columns if feature_pattern in col]
                group_features.extend(matching_features)
            
            if not group_features:
                continue
            
            # Get data for this group
            group_data = data[group_features].select_dtypes(include=[np.number])
            if group_data.empty:
                continue
            
            # Create and fit scaler
            scaler = self._create_scaler(group_config['method'])
            
            try:
                # Handle missing values
                group_data_clean = group_data.fillna(group_data.median())
                scaler.fit(group_data_clean)
                
                self.scalers[group_name] = {
                    'scaler': scaler,
                    'features': list(group_data.columns),
                    'method': group_config['method']
                }
                
                logger.info(f"Fitted {group_config['method']} scaler for {group_name} with {len(group_data.columns)} features")
                
            except Exception as e:
                logger.error(f"Error fitting scaler for {group_name}: {e}")
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scalers"""
        if not self.fitted:
            raise ValueError("Scalers must be fitted before transforming")
        
        if data.empty:
            return data.copy()
        
        transformed_data = data.copy()
        
        for group_name, scaler_info in self.scalers.items():
            scaler = scaler_info['scaler']
            features = scaler_info['features']
            
            # Get features that exist in current data
            available_features = [f for f in features if f in transformed_data.columns]
            if not available_features:
                continue
            
            try:
                # Handle missing values
                feature_data = transformed_data[available_features].fillna(
                    transformed_data[available_features].median()
                )
                
                # Transform features
                scaled_data = scaler.transform(feature_data)
                
                # Update dataframe
                transformed_data[available_features] = scaled_data
                
            except Exception as e:
                logger.error(f"Error transforming features for {group_name}: {e}")
        
        return transformed_data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit scalers and transform data"""
        return self.fit(data).transform(data)
    
    async def scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale features from dictionary format - compatibility method for pipeline"""
        if not features:
            return {}
        
        try:
            # Convert dictionary to DataFrame
            features_df = pd.DataFrame([features])
            
            # If scalers are fitted, use them
            if self.fitted:
                scaled_df = self.transform(features_df)
            else:
                # If not fitted, just return original features
                scaled_df = features_df
            
            # Convert back to dictionary
            scaled_features = scaled_df.iloc[0].to_dict()
            
            # Remove any NaN values
            scaled_features = {k: v for k, v in scaled_features.items() if not pd.isna(v)}
            
            return scaled_features
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return features  # Return original features if scaling fails
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data"""
        if not self.fitted:
            raise ValueError("Scalers must be fitted before inverse transforming")
        
        if data.empty:
            return data.copy()
        
        inverse_data = data.copy()
        
        for group_name, scaler_info in self.scalers.items():
            scaler = scaler_info['scaler']
            features = scaler_info['features']
            
            available_features = [f for f in features if f in inverse_data.columns]
            if not available_features:
                continue
            
            try:
                scaled_data = inverse_data[available_features]
                original_data = scaler.inverse_transform(scaled_data)
                inverse_data[available_features] = original_data
                
            except Exception as e:
                logger.error(f"Error inverse transforming features for {group_name}: {e}")
        
        return inverse_data
    
    def _create_scaler(self, method: str) -> BaseEstimator:
        """Create scaler based on method"""
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'power':
            return PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            logger.warning(f"Unknown scaling method: {method}, using standard scaler")
            return StandardScaler()
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to file"""
        if not self.fitted:
            raise ValueError("No fitted scalers to save")
        
        scaler_data = {
            'scalers': self.scalers,
            'feature_groups': self.feature_groups,
            'fitted': self.fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(scaler_data, filepath)
        logger.info(f"Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        scaler_data = joblib.load(filepath)
        self.scalers = scaler_data['scalers']
        self.feature_groups = scaler_data['feature_groups']
        self.fitted = scaler_data['fitted']
        
        logger.info(f"Scalers loaded from {filepath}")
    
    def get_feature_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about features before and after scaling"""
        if data.empty:
            return {}
        
        stats = {}
        
        # Original data statistics
        numeric_data = data.select_dtypes(include=[np.number])
        stats['original'] = {
            'mean': numeric_data.mean().to_dict(),
            'std': numeric_data.std().to_dict(),
            'min': numeric_data.min().to_dict(),
            'max': numeric_data.max().to_dict()
        }
        
        # Scaled data statistics
        if self.fitted:
            scaled_data = self.transform(data)
            scaled_numeric = scaled_data.select_dtypes(include=[np.number])
            stats['scaled'] = {
                'mean': scaled_numeric.mean().to_dict(),
                'std': scaled_numeric.std().to_dict(),
                'min': scaled_numeric.min().to_dict(),
                'max': scaled_numeric.max().to_dict()
            }
        
        return stats
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """Detect outliers in features"""
        if data.empty:
            return pd.DataFrame()
        
        numeric_data = data.select_dtypes(include=[np.number])
        outlier_mask = pd.DataFrame(False, index=data.index, columns=numeric_data.columns)
        
        for column in numeric_data.columns:
            values = numeric_data[column].dropna()
            if values.empty:
                continue
            
            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask[column] = (numeric_data[column] < lower_bound) | (numeric_data[column] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((values - values.mean()) / values.std())
                outlier_mask[column] = z_scores > threshold
        
        return outlier_mask
    
    def handle_outliers(self, data: pd.DataFrame, method: str = 'clip', outlier_threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers in data"""
        if data.empty:
            return data.copy()
        
        processed_data = data.copy()
        outlier_mask = self.detect_outliers(data, threshold=outlier_threshold)
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column not in outlier_mask.columns:
                continue
            
            outliers = outlier_mask[column]
            if not outliers.any():
                continue
            
            if method == 'clip':
                # Clip to percentiles
                lower_percentile = processed_data[column].quantile(0.01)
                upper_percentile = processed_data[column].quantile(0.99)
                processed_data[column] = processed_data[column].clip(lower_percentile, upper_percentile)
                
            elif method == 'median':
                # Replace with median
                median_value = processed_data[column].median()
                processed_data.loc[outliers, column] = median_value
                
            elif method == 'remove':
                # Set to NaN (will be handled by fillna later)
                processed_data.loc[outliers, column] = np.nan
        
        return processed_data
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of scaling configuration"""
        summary = {
            'fitted': self.fitted,
            'total_groups': len(self.feature_groups),
            'fitted_scalers': len(self.scalers),
            'groups': {}
        }
        
        for group_name, group_config in self.feature_groups.items():
            group_summary = {
                'method': group_config['method'],
                'pattern_count': len(group_config['features']),
                'fitted': group_name in self.scalers
            }
            
            if group_name in self.scalers:
                group_summary['actual_features'] = len(self.scalers[group_name]['features'])
            
            summary['groups'][group_name] = group_summary
        
        return summary
