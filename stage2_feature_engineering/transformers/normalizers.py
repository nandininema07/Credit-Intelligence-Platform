"""
Feature normalization for Stage 2.
Implements various normalization methods for credit risk modeling.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
import joblib

logger = logging.getLogger(__name__)

class FeatureNormalizer:
    """Feature normalization engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.normalizers = {}
        self.fitted = False
        self.feature_stats = {}
        
    def fit(self, X: pd.DataFrame) -> 'FeatureNormalizer':
        """Fit normalizers on training data"""
        try:
            # Standard normalization (z-score)
            if self.config.get('use_standard', True):
                self.normalizers['standard'] = StandardScaler()
                self.normalizers['standard'].fit(X.select_dtypes(include=[np.number]))
            
            # Min-Max normalization
            if self.config.get('use_minmax', False):
                self.normalizers['minmax'] = MinMaxScaler()
                self.normalizers['minmax'].fit(X.select_dtypes(include=[np.number]))
            
            # Robust normalization (median and IQR)
            if self.config.get('use_robust', False):
                self.normalizers['robust'] = RobustScaler()
                self.normalizers['robust'].fit(X.select_dtypes(include=[np.number]))
            
            # Power transformation (Box-Cox, Yeo-Johnson)
            if self.config.get('use_power', False):
                self.normalizers['power'] = PowerTransformer(method='yeo-johnson')
                self.normalizers['power'].fit(X.select_dtypes(include=[np.number]))
            
            # Store feature statistics
            self.feature_stats = {
                'mean': X.select_dtypes(include=[np.number]).mean().to_dict(),
                'std': X.select_dtypes(include=[np.number]).std().to_dict(),
                'min': X.select_dtypes(include=[np.number]).min().to_dict(),
                'max': X.select_dtypes(include=[np.number]).max().to_dict()
            }
            
            self.fitted = True
            logger.info("Feature normalizers fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting normalizers: {e}")
            raise
            
        return self
    
    def transform(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Transform features using specified normalization method"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if method not in self.normalizers:
            logger.warning(f"Method {method} not available, using standard")
            method = 'standard'
        
        try:
            X_transformed = X.copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                X_transformed[numeric_cols] = self.normalizers[method].transform(X[numeric_cols])
            
            logger.info(f"Features normalized using {method} method")
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            raise
    
    def fit_transform(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Fit normalizer and transform features"""
        return self.fit(X).transform(X, method)
    
    def inverse_transform(self, X: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Inverse transform normalized features"""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")
        
        if method not in self.normalizers:
            logger.warning(f"Method {method} not available, using standard")
            method = 'standard'
        
        try:
            X_inverse = X.copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                X_inverse[numeric_cols] = self.normalizers[method].inverse_transform(X[numeric_cols])
            
            return X_inverse
            
        except Exception as e:
            logger.error(f"Error inverse transforming features: {e}")
            raise
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature statistics"""
        return self.feature_stats
    
    def save(self, filepath: str):
        """Save normalizer to file"""
        try:
            joblib.dump({
                'normalizers': self.normalizers,
                'feature_stats': self.feature_stats,
                'config': self.config,
                'fitted': self.fitted
            }, filepath)
            logger.info(f"Normalizer saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving normalizer: {e}")
            raise
    
    def load(self, filepath: str):
        """Load normalizer from file"""
        try:
            data = joblib.load(filepath)
            self.normalizers = data['normalizers']
            self.feature_stats = data['feature_stats']
            self.config = data['config']
            self.fitted = data['fitted']
            logger.info(f"Normalizer loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading normalizer: {e}")
            raise
