"""
Data Quality Validation Module
Ensures training data meets quality standards and detects data leakage.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Validates data quality and detects potential issues"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quality_thresholds = config.get('quality_thresholds', {})
        
    def validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Comprehensive validation of training data"""
        logger.info("Validating training data quality...")
        
        validation_results = {
            'passed': True,
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        # Basic data validation
        basic_validation = self._validate_basic_data(X, y)
        validation_results.update(basic_validation)
        
        # Feature quality validation
        feature_validation = self._validate_features(X, y)
        validation_results.update(feature_validation)
        
        # Target variable validation
        target_validation = self._validate_target(y)
        validation_results.update(target_validation)
        
        # Data leakage detection
        leakage_detection = self._detect_data_leakage(X, y)
        validation_results.update(leakage_detection)
        
        # Overall validation result
        validation_results['passed'] = (
            len(validation_results['errors']) == 0 and
            len(validation_results['warnings']) < 5  # Allow some warnings
        )
        
        if validation_results['passed']:
            logger.info("Data quality validation PASSED")
        else:
            logger.error("Data quality validation FAILED")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
        
        return validation_results
    
    def _validate_basic_data(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate basic data properties"""
        results = {'warnings': [], 'errors': [], 'metrics': {}}
        
        # Check data size
        min_samples = self.config.get('min_samples', 100)
        if len(X) < min_samples:
            results['errors'].append(f"Insufficient samples: {len(X)} < {min_samples}")
        
        # Check feature count
        max_features = self.config.get('max_features', 50)
        if len(X.columns) > max_features:
            results['warnings'].append(f"Too many features: {len(X.columns)} > {max_features}")
        
        # Check for missing values
        missing_pct = X.isnull().sum().sum() / (len(X) * len(X.columns))
        if missing_pct > 0.1:  # More than 10% missing
            results['warnings'].append(f"High missing data: {missing_pct:.1%}")
        
        # Check for infinite values
        inf_count = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            results['warnings'].append(f"Found {inf_count} infinite values")
        
        results['metrics']['sample_count'] = len(X)
        results['metrics']['feature_count'] = len(X.columns)
        results['metrics']['missing_pct'] = missing_pct
        results['metrics']['inf_count'] = inf_count
        
        return results
    
    def _validate_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate feature quality and characteristics"""
        results = {'warnings': [], 'errors': [], 'metrics': {}}
        
        # Check feature variance
        low_variance_features = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                if X[col].std() < 1e-6:  # Very low variance
                    low_variance_features.append(col)
        
        if low_variance_features:
            results['warnings'].append(f"Low variance features: {low_variance_features[:5]}")
        
        # Check feature correlation with target
        try:
            if len(y.unique()) > 1:  # Only if we have multiple classes
                # Calculate mutual information scores
                mi_scores = mutual_info_classif(X.fillna(X.median()), y, random_state=42)
                
                # Check for features with very high MI scores (potential data leakage)
                high_mi_features = []
                for i, score in enumerate(mi_scores):
                    if score > 0.8:  # Very high mutual information
                        high_mi_features.append(X.columns[i])
                
                if high_mi_features:
                    results['warnings'].append(f"Features with very high target correlation: {high_mi_features[:3]}")
                
                results['metrics']['avg_mutual_info'] = float(np.mean(mi_scores))
                results['metrics']['max_mutual_info'] = float(np.max(mi_scores))
                
        except Exception as e:
            results['warnings'].append(f"Could not calculate feature-target correlation: {e}")
        
        # Check feature distributions
        skewed_features = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                skewness = abs(X[col].skew())
                if skewness > 3:  # Highly skewed
                    skewed_features.append(col)
        
        if skewed_features:
            results['warnings'].append(f"Highly skewed features: {skewed_features[:5]}")
        
        results['metrics']['low_variance_count'] = len(low_variance_features)
        results['metrics']['skewed_count'] = len(skewed_features)
        
        return results
    
    def _validate_target(self, y: pd.Series) -> Dict[str, Any]:
        """Validate target variable quality"""
        results = {'warnings': [], 'errors': [], 'metrics': {}}
        
        # Check class distribution
        class_counts = y.value_counts()
        total_samples = len(y)
        
        # Check for class imbalance
        min_class_pct = class_counts.min() / total_samples
        if min_class_pct < 0.1:  # Less than 10% in smallest class
            results['warnings'].append(f"Severe class imbalance: smallest class {min_class_pct:.1%}")
        elif min_class_pct < 0.2:  # Less than 20% in smallest class
            results['warnings'].append(f"Moderate class imbalance: smallest class {min_class_pct:.1%}")
        
        # Check for suspiciously perfect distribution
        if len(class_counts) == 2:  # Binary classification
            if abs(class_counts.iloc[0] - class_counts.iloc[1]) <= 1:
                results['warnings'].append("Suspiciously balanced binary classes - check for data leakage")
        
        # Check target variance
        if y.dtype in ['int64', 'float64']:
            if y.std() < 1e-6:
                results['errors'].append("Target variable has no variance - cannot train model")
        
        results['metrics']['class_distribution'] = class_counts.to_dict()
        results['metrics']['min_class_pct'] = min_class_pct
        results['metrics']['target_variance'] = float(y.var()) if y.dtype in ['int64', 'float64'] else 0
        
        return results
    
    def _detect_data_leakage(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Detect potential data leakage"""
        results = {'warnings': [], 'errors': [], 'metrics': {}}
        
        # Check for target variable in features
        target_columns = [col for col in X.columns if 'target' in col.lower() or 'label' in col.lower()]
        if target_columns:
            results['errors'].append(f"Potential target leakage: found columns {target_columns}")
        
        # Check for perfect correlation with target
        perfect_correlation_features = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                try:
                    correlation = abs(X[col].corr(y))
                    if correlation > 0.95:  # Almost perfect correlation
                        perfect_correlation_features.append(col)
                except:
                    continue
        
        if perfect_correlation_features:
            results['warnings'].append(f"Features with near-perfect target correlation: {perfect_correlation_features[:3]}")
        
        # Check for suspicious feature patterns
        suspicious_patterns = []
        
        # Check if any feature perfectly predicts the target
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                try:
                    # Group by feature value and check target distribution
                    grouped = y.groupby(X[col]).agg(['count', 'nunique'])
                    if len(grouped) > 1:  # Multiple feature values
                        # Check if any feature value perfectly predicts a class
                        for feature_val in grouped.index:
                            target_counts = y[X[col] == feature_val].value_counts()
                            if len(target_counts) == 1:  # Only one target class for this feature value
                                if target_counts.iloc[0] > 1:  # More than one sample
                                    suspicious_patterns.append(f"{col}={feature_val}")
                except:
                    continue
        
        if suspicious_patterns:
            results['warnings'].append(f"Suspicious feature patterns: {suspicious_patterns[:5]}")
        
        # Check for time-based leakage (if timestamp features exist)
        time_columns = [col for col in X.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_columns:
            results['warnings'].append(f"Time-based features detected: {time_columns} - ensure proper temporal splits")
        
        results['metrics']['perfect_correlation_count'] = len(perfect_correlation_features)
        results['metrics']['suspicious_pattern_count'] = len(suspicious_patterns)
        results['metrics']['time_features_count'] = len(time_columns)
        
        return results
    
    def get_data_quality_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive data quality report"""
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY VALIDATION REPORT")
        report.append("=" * 60)
        
        # Overall status
        status = "PASSED" if validation_results['passed'] else "FAILED"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        # Metrics summary
        if validation_results['metrics']:
            report.append("Data Metrics:")
            for key, value in validation_results['metrics'].items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.3f}")
                else:
                    report.append(f"  {key}: {value}")
            report.append("")
        
        # Warnings
        if validation_results['warnings']:
            report.append("Warnings:")
            for warning in validation_results['warnings']:
                report.append(f"  ⚠️  {warning}")
            report.append("")
        
        # Errors
        if validation_results['errors']:
            report.append("Errors:")
            for error in validation_results['errors']:
                report.append(f"  ❌ {error}")
            report.append("")
        
        # Recommendations
        if not validation_results['passed']:
            report.append("Recommendations:")
            report.append("  - Fix all errors before training")
            report.append("  - Address critical warnings")
            report.append("  - Review data preprocessing pipeline")
            report.append("  - Check for data leakage")
        else:
            report.append("Recommendations:")
            report.append("  - Data quality is acceptable for training")
            report.append("  - Monitor model performance for overfitting")
            report.append("  - Consider addressing minor warnings")
        
        return "\n".join(report)
