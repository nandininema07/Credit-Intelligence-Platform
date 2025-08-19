"""
Categorical encoding for Stage 2 feature engineering.
Handles various encoding methods for categorical variables.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

logger = logging.getLogger(__name__)

class CategoricalEncoder:
    """Categorical variable encoder"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.encoders = {}
        self.encoding_methods = {
            'label': LabelEncoder,
            'onehot': OneHotEncoder,
            'ordinal': OrdinalEncoder,
            'target': self._create_target_encoder
        }
        self.fitted = False
        
    def fit(self, data: pd.DataFrame, target: pd.Series = None) -> 'CategoricalEncoder':
        """Fit encoders on training data"""
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            encoding_method = self._get_encoding_method(column)
            
            try:
                if encoding_method == 'target' and target is not None:
                    encoder = self._fit_target_encoder(data[column], target)
                elif encoding_method == 'onehot':
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoder.fit(data[[column]])
                elif encoding_method == 'ordinal':
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    encoder.fit(data[[column]])
                else:  # label encoding
                    encoder = LabelEncoder()
                    encoder.fit(data[column].fillna('missing'))
                
                self.encoders[column] = {
                    'encoder': encoder,
                    'method': encoding_method,
                    'categories': data[column].unique().tolist()
                }
                
                logger.info(f"Fitted {encoding_method} encoder for {column}")
                
            except Exception as e:
                logger.error(f"Error fitting encoder for {column}: {e}")
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables"""
        if not self.fitted:
            raise ValueError("Encoders must be fitted before transforming")
        
        result_data = data.copy()
        
        for column, encoder_info in self.encoders.items():
            if column not in data.columns:
                continue
            
            try:
                encoder = encoder_info['encoder']
                method = encoder_info['method']
                
                if method == 'onehot':
                    # One-hot encoding
                    encoded = encoder.transform(data[[column]])
                    feature_names = [f"{column}_{cat}" for cat in encoder.categories_[0]]
                    
                    # Add encoded columns
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=data.index)
                    result_data = pd.concat([result_data, encoded_df], axis=1)
                    
                    # Remove original column
                    result_data = result_data.drop(columns=[column])
                
                elif method == 'target':
                    # Target encoding
                    result_data[column] = encoder.transform(data[column])
                
                else:
                    # Label or ordinal encoding
                    result_data[column] = encoder.transform(data[column].fillna('missing'))
                
            except Exception as e:
                logger.error(f"Error transforming {column}: {e}")
        
        return result_data
    
    def fit_transform(self, data: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """Fit encoders and transform data"""
        return self.fit(data, target).transform(data)
    
    def _get_encoding_method(self, column: str) -> str:
        """Determine encoding method for column"""
        column_config = self.config.get('encoding_methods', {})
        
        if column in column_config:
            return column_config[column]
        
        # Default rules based on column name
        if 'sector' in column.lower() or 'industry' in column.lower():
            return 'onehot'
        elif 'country' in column.lower() or 'region' in column.lower():
            return 'onehot'
        elif 'rating' in column.lower() or 'grade' in column.lower():
            return 'ordinal'
        else:
            return 'label'
    
    def _create_target_encoder(self):
        """Create target encoder"""
        return TargetEncoder()
    
    def _fit_target_encoder(self, categorical_series: pd.Series, target: pd.Series):
        """Fit target encoder"""
        encoder = TargetEncoder()
        encoder.fit(categorical_series, target)
        return encoder
    
    def save_encoders(self, filepath: str):
        """Save fitted encoders"""
        if not self.fitted:
            raise ValueError("No fitted encoders to save")
        
        encoder_data = {
            'encoders': self.encoders,
            'fitted': self.fitted,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(encoder_data, filepath)
        logger.info(f"Encoders saved to {filepath}")
    
    def load_encoders(self, filepath: str):
        """Load fitted encoders"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Encoder file not found: {filepath}")
        
        encoder_data = joblib.load(filepath)
        self.encoders = encoder_data['encoders']
        self.fitted = encoder_data['fitted']
        
        logger.info(f"Encoders loaded from {filepath}")

class TargetEncoder:
    """Target encoding for categorical variables"""
    
    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing
        self.target_means = {}
        self.global_mean = 0.0
        
    def fit(self, categorical_series: pd.Series, target: pd.Series):
        """Fit target encoder"""
        self.global_mean = target.mean()
        
        # Calculate smoothed target means for each category
        for category in categorical_series.unique():
            if pd.isna(category):
                continue
            
            mask = categorical_series == category
            category_target = target[mask]
            
            if len(category_target) > 0:
                category_mean = category_target.mean()
                category_count = len(category_target)
                
                # Apply smoothing
                smoothed_mean = (
                    (category_count * category_mean + self.smoothing * self.global_mean) /
                    (category_count + self.smoothing)
                )
                
                self.target_means[category] = smoothed_mean
    
    def transform(self, categorical_series: pd.Series) -> pd.Series:
        """Transform categorical series to target encoded values"""
        return categorical_series.map(self.target_means).fillna(self.global_mean)
