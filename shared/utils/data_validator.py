"""
Data validation utilities for the Credit Intelligence Platform.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and quality checking utilities"""
    
    def __init__(self):
        self.validation_rules = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules"""
        self.validation_rules = {
            'credit_score': {
                'type': 'numeric',
                'min_value': 300,
                'max_value': 850,
                'required': True
            },
            'company_id': {
                'type': 'string',
                'min_length': 1,
                'max_length': 100,
                'required': True
            },
            'risk_category': {
                'type': 'categorical',
                'allowed_values': ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk'],
                'required': True
            },
            'timestamp': {
                'type': 'datetime',
                'required': True
            },
            'probability_default': {
                'type': 'numeric',
                'min_value': 0.0,
                'max_value': 1.0,
                'required': False
            }
        }
    
    def validate_dataframe(self, df: pd.DataFrame, schema: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Validate entire dataframe"""
        if schema is None:
            schema = self.validation_rules
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_data_summary': {},
            'data_quality_score': 0.0
        }
        
        # Check required columns
        required_columns = [col for col, rules in schema.items() if rules.get('required', False)]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Validate each column
        for column, rules in schema.items():
            if column not in df.columns:
                continue
            
            column_validation = self._validate_column(df[column], rules, column)
            
            if not column_validation['is_valid']:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(column_validation['errors'])
            
            validation_results['warnings'].extend(column_validation['warnings'])
            validation_results['missing_data_summary'][column] = column_validation['missing_count']
        
        # Calculate data quality score
        validation_results['data_quality_score'] = self._calculate_quality_score(df, validation_results)
        
        return validation_results
    
    def _validate_column(self, series: pd.Series, rules: Dict[str, Any], column_name: str) -> Dict[str, Any]:
        """Validate individual column"""
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'missing_count': series.isnull().sum()
        }
        
        # Check for missing values
        if rules.get('required', False) and result['missing_count'] > 0:
            result['is_valid'] = False
            result['errors'].append(f"Column '{column_name}' has {result['missing_count']} missing values")
        
        # Skip validation for missing values
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return result
        
        # Type validation
        data_type = rules.get('type', 'string')
        
        if data_type == 'numeric':
            if not pd.api.types.is_numeric_dtype(non_null_series):
                result['is_valid'] = False
                result['errors'].append(f"Column '{column_name}' should be numeric")
            else:
                # Range validation
                if 'min_value' in rules:
                    min_violations = (non_null_series < rules['min_value']).sum()
                    if min_violations > 0:
                        result['is_valid'] = False
                        result['errors'].append(f"Column '{column_name}' has {min_violations} values below minimum {rules['min_value']}")
                
                if 'max_value' in rules:
                    max_violations = (non_null_series > rules['max_value']).sum()
                    if max_violations > 0:
                        result['is_valid'] = False
                        result['errors'].append(f"Column '{column_name}' has {max_violations} values above maximum {rules['max_value']}")
        
        elif data_type == 'string':
            if 'min_length' in rules:
                short_values = non_null_series.astype(str).str.len() < rules['min_length']
                if short_values.sum() > 0:
                    result['warnings'].append(f"Column '{column_name}' has {short_values.sum()} values shorter than {rules['min_length']} characters")
            
            if 'max_length' in rules:
                long_values = non_null_series.astype(str).str.len() > rules['max_length']
                if long_values.sum() > 0:
                    result['warnings'].append(f"Column '{column_name}' has {long_values.sum()} values longer than {rules['max_length']} characters")
        
        elif data_type == 'categorical':
            allowed_values = rules.get('allowed_values', [])
            if allowed_values:
                invalid_values = ~non_null_series.isin(allowed_values)
                if invalid_values.sum() > 0:
                    result['is_valid'] = False
                    unique_invalid = non_null_series[invalid_values].unique()
                    result['errors'].append(f"Column '{column_name}' has invalid values: {list(unique_invalid)}")
        
        elif data_type == 'datetime':
            try:
                pd.to_datetime(non_null_series)
            except:
                result['is_valid'] = False
                result['errors'].append(f"Column '{column_name}' contains invalid datetime values")
        
        return result
    
    def _calculate_quality_score(self, df: pd.DataFrame, validation_results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)"""
        score = 1.0
        
        # Penalize for errors
        error_count = len(validation_results['errors'])
        score -= min(error_count * 0.1, 0.5)  # Max 50% penalty for errors
        
        # Penalize for missing data
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = sum(validation_results['missing_data_summary'].values())
        if total_cells > 0:
            missing_ratio = missing_cells / total_cells
            score -= missing_ratio * 0.3  # Max 30% penalty for missing data
        
        # Penalize for warnings
        warning_count = len(validation_results['warnings'])
        score -= min(warning_count * 0.05, 0.2)  # Max 20% penalty for warnings
        
        return max(0.0, score)
    
    def validate_credit_score_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate credit scoring data"""
        errors = []
        
        # Required fields
        required_fields = ['company_id', 'credit_score', 'risk_category']
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Credit score validation
        if 'credit_score' in data:
            score = data['credit_score']
            if not isinstance(score, (int, float)) or score < 300 or score > 850:
                errors.append("Credit score must be between 300 and 850")
        
        # Risk category validation
        if 'risk_category' in data:
            valid_categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
            if data['risk_category'] not in valid_categories:
                errors.append(f"Invalid risk category. Must be one of: {valid_categories}")
        
        # Probability validation
        if 'probability_default' in data:
            prob = data['probability_default']
            if prob is not None and (not isinstance(prob, (int, float)) or prob < 0 or prob > 1):
                errors.append("Probability of default must be between 0 and 1")
        
        return len(errors) == 0, errors
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate feature data"""
        errors = []
        
        if not isinstance(features, dict):
            errors.append("Features must be a dictionary")
            return False, errors
        
        # Check for numeric values
        for feature_name, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                errors.append(f"Feature '{feature_name}' has invalid value: {value}")
        
        # Check for minimum number of features
        if len(features) < 5:
            errors.append(f"Insufficient features provided. Expected at least 5, got {len(features)}")
        
        return len(errors) == 0, errors
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) == 0:
                continue
            
            if method == 'iqr':
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (series < lower_bound) | (series > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((series - series.mean()) / series.std())
                outlier_mask = z_scores > 3
            
            outlier_count = outlier_mask.sum()
            outlier_percentage = (outlier_count / len(series)) * 100
            
            outliers[column] = {
                'count': int(outlier_count),
                'percentage': float(outlier_percentage),
                'indices': df.index[df[column].isin(series[outlier_mask])].tolist()
            }
        
        return outliers
    
    def check_data_freshness(self, df: pd.DataFrame, timestamp_column: str = 'timestamp', 
                           max_age_hours: int = 24) -> Dict[str, Any]:
        """Check if data is fresh enough"""
        result = {
            'is_fresh': True,
            'oldest_record': None,
            'newest_record': None,
            'stale_records': 0,
            'total_records': len(df)
        }
        
        if timestamp_column not in df.columns:
            result['is_fresh'] = False
            result['error'] = f"Timestamp column '{timestamp_column}' not found"
            return result
        
        try:
            timestamps = pd.to_datetime(df[timestamp_column])
            current_time = datetime.now()
            
            result['oldest_record'] = timestamps.min()
            result['newest_record'] = timestamps.max()
            
            # Check for stale records
            cutoff_time = current_time - pd.Timedelta(hours=max_age_hours)
            stale_mask = timestamps < cutoff_time
            result['stale_records'] = int(stale_mask.sum())
            
            if result['stale_records'] > 0:
                stale_percentage = (result['stale_records'] / len(df)) * 100
                if stale_percentage > 50:  # More than 50% stale
                    result['is_fresh'] = False
            
        except Exception as e:
            result['is_fresh'] = False
            result['error'] = f"Error processing timestamps: {str(e)}"
        
        return result
