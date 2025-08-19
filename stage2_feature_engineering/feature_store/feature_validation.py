"""
Feature validation for ensuring data quality and consistency.
Validates feature values, distributions, and business rules.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of feature validation"""
    feature_name: str
    check_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Dict[str, Any]
    timestamp: datetime

class FeatureValidator:
    """Feature validation engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = {}
        self.validation_history = []
        self.setup_default_rules()
        
    def setup_default_rules(self):
        """Setup default validation rules"""
        self.validation_rules = {
            'null_check': self.check_null_values,
            'range_check': self.check_value_range,
            'distribution_check': self.check_distribution,
            'uniqueness_check': self.check_uniqueness,
            'format_check': self.check_format,
            'business_rule_check': self.check_business_rules,
            'drift_check': self.check_data_drift,
            'correlation_check': self.check_correlations
        }
    
    def validate_feature(self, feature_name: str, data: pd.Series, 
                        rules: List[str] = None,
                        reference_data: pd.Series = None) -> List[ValidationResult]:
        """Validate a single feature"""
        if rules is None:
            rules = list(self.validation_rules.keys())
        
        results = []
        
        for rule_name in rules:
            if rule_name in self.validation_rules:
                try:
                    rule_func = self.validation_rules[rule_name]
                    result = rule_func(feature_name, data, reference_data)
                    results.append(result)
                    
                except Exception as e:
                    error_result = ValidationResult(
                        feature_name=feature_name,
                        check_name=rule_name,
                        severity=ValidationSeverity.ERROR,
                        passed=False,
                        message=f"Validation rule failed: {str(e)}",
                        details={'error': str(e)},
                        timestamp=datetime.now()
                    )
                    results.append(error_result)
                    logger.error(f"Validation rule {rule_name} failed for {feature_name}: {e}")
        
        # Store results in history
        self.validation_history.extend(results)
        
        return results
    
    def validate_dataset(self, data: pd.DataFrame, 
                        feature_configs: Dict[str, Dict[str, Any]] = None,
                        reference_data: pd.DataFrame = None) -> Dict[str, List[ValidationResult]]:
        """Validate entire dataset"""
        all_results = {}
        
        for column in data.columns:
            # Get feature-specific configuration
            feature_config = feature_configs.get(column, {}) if feature_configs else {}
            rules = feature_config.get('validation_rules', list(self.validation_rules.keys()))
            
            # Get reference data for this feature
            ref_series = reference_data[column] if reference_data is not None and column in reference_data.columns else None
            
            # Validate feature
            results = self.validate_feature(column, data[column], rules, ref_series)
            all_results[column] = results
        
        return all_results
    
    def check_null_values(self, feature_name: str, data: pd.Series, 
                         reference_data: pd.Series = None) -> ValidationResult:
        """Check for null values"""
        null_count = data.isnull().sum()
        null_percentage = (null_count / len(data)) * 100
        
        # Get threshold from config
        threshold = self.config.get('null_threshold', 10.0)  # 10% default
        
        passed = null_percentage <= threshold
        severity = ValidationSeverity.WARNING if null_percentage > threshold else ValidationSeverity.INFO
        
        if null_percentage > 50:
            severity = ValidationSeverity.CRITICAL
        elif null_percentage > 25:
            severity = ValidationSeverity.ERROR
        
        return ValidationResult(
            feature_name=feature_name,
            check_name='null_check',
            severity=severity,
            passed=passed,
            message=f"Null values: {null_count} ({null_percentage:.2f}%)",
            details={
                'null_count': int(null_count),
                'null_percentage': float(null_percentage),
                'threshold': threshold
            },
            timestamp=datetime.now()
        )
    
    def check_value_range(self, feature_name: str, data: pd.Series, 
                         reference_data: pd.Series = None) -> ValidationResult:
        """Check if values are within expected range"""
        if not pd.api.types.is_numeric_dtype(data):
            return ValidationResult(
                feature_name=feature_name,
                check_name='range_check',
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Range check skipped for non-numeric data",
                details={},
                timestamp=datetime.now()
            )
        
        data_clean = data.dropna()
        if data_clean.empty:
            return ValidationResult(
                feature_name=feature_name,
                check_name='range_check',
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="No valid data for range check",
                details={},
                timestamp=datetime.now()
            )
        
        min_val = float(data_clean.min())
        max_val = float(data_clean.max())
        
        # Check for extreme outliers
        q1 = data_clean.quantile(0.25)
        q3 = data_clean.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
        outlier_percentage = (len(outliers) / len(data_clean)) * 100
        
        passed = outlier_percentage <= 5.0  # 5% threshold
        severity = ValidationSeverity.WARNING if outlier_percentage > 5.0 else ValidationSeverity.INFO
        
        return ValidationResult(
            feature_name=feature_name,
            check_name='range_check',
            severity=severity,
            passed=passed,
            message=f"Range: [{min_val:.2f}, {max_val:.2f}], Outliers: {len(outliers)} ({outlier_percentage:.2f}%)",
            details={
                'min_value': min_val,
                'max_value': max_val,
                'outlier_count': len(outliers),
                'outlier_percentage': float(outlier_percentage),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            },
            timestamp=datetime.now()
        )
    
    def check_distribution(self, feature_name: str, data: pd.Series, 
                          reference_data: pd.Series = None) -> ValidationResult:
        """Check data distribution"""
        data_clean = data.dropna()
        
        if data_clean.empty:
            return ValidationResult(
                feature_name=feature_name,
                check_name='distribution_check',
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="No valid data for distribution check",
                details={},
                timestamp=datetime.now()
            )
        
        details = {
            'count': len(data_clean),
            'unique_values': data_clean.nunique()
        }
        
        if pd.api.types.is_numeric_dtype(data_clean):
            details.update({
                'mean': float(data_clean.mean()),
                'std': float(data_clean.std()),
                'skewness': float(data_clean.skew()),
                'kurtosis': float(data_clean.kurtosis())
            })
            
            # Check for extreme skewness
            skewness = abs(data_clean.skew())
            passed = skewness < 2.0
            severity = ValidationSeverity.WARNING if skewness >= 2.0 else ValidationSeverity.INFO
            message = f"Distribution stats - Mean: {details['mean']:.2f}, Std: {details['std']:.2f}, Skew: {details['skewness']:.2f}"
        
        else:
            # Categorical data
            value_counts = data_clean.value_counts()
            details['top_values'] = value_counts.head(5).to_dict()
            
            # Check for imbalanced categories
            max_frequency = value_counts.max() / len(data_clean)
            passed = max_frequency < 0.95  # No single category dominates 95%
            severity = ValidationSeverity.WARNING if max_frequency >= 0.95 else ValidationSeverity.INFO
            message = f"Categorical distribution - {data_clean.nunique()} unique values, max frequency: {max_frequency:.2%}"
        
        return ValidationResult(
            feature_name=feature_name,
            check_name='distribution_check',
            severity=severity,
            passed=passed,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def check_uniqueness(self, feature_name: str, data: pd.Series, 
                        reference_data: pd.Series = None) -> ValidationResult:
        """Check uniqueness of values"""
        total_count = len(data)
        unique_count = data.nunique()
        duplicate_count = total_count - unique_count
        uniqueness_ratio = unique_count / total_count if total_count > 0 else 0
        
        # Determine if this should be a unique field
        expected_unique = self.config.get('expected_unique_features', [])
        should_be_unique = feature_name in expected_unique
        
        if should_be_unique:
            passed = duplicate_count == 0
            severity = ValidationSeverity.ERROR if duplicate_count > 0 else ValidationSeverity.INFO
            message = f"Uniqueness check - Duplicates: {duplicate_count} (expected 0)"
        else:
            # General uniqueness assessment
            passed = True  # Not a failure, just informational
            severity = ValidationSeverity.INFO
            message = f"Uniqueness ratio: {uniqueness_ratio:.2%} ({unique_count}/{total_count})"
        
        return ValidationResult(
            feature_name=feature_name,
            check_name='uniqueness_check',
            severity=severity,
            passed=passed,
            message=message,
            details={
                'total_count': total_count,
                'unique_count': unique_count,
                'duplicate_count': duplicate_count,
                'uniqueness_ratio': float(uniqueness_ratio)
            },
            timestamp=datetime.now()
        )
    
    def check_format(self, feature_name: str, data: pd.Series, 
                    reference_data: pd.Series = None) -> ValidationResult:
        """Check data format consistency"""
        data_clean = data.dropna().astype(str)
        
        if data_clean.empty:
            return ValidationResult(
                feature_name=feature_name,
                check_name='format_check',
                severity=ValidationSeverity.INFO,
                passed=True,
                message="No data to check format",
                details={},
                timestamp=datetime.now()
            )
        
        # Check for consistent string lengths (for categorical/text data)
        if not pd.api.types.is_numeric_dtype(data):
            lengths = data_clean.str.len()
            length_variance = lengths.var()
            
            # Check for suspicious format patterns
            format_issues = []
            
            # Check for mixed case inconsistency
            if data_clean.dtype == 'object':
                has_mixed_case = any(s != s.lower() and s != s.upper() for s in data_clean.unique()[:100])
                if has_mixed_case:
                    format_issues.append("Mixed case detected")
            
            # Check for leading/trailing whitespace
            has_whitespace = any(s != s.strip() for s in data_clean.unique()[:100])
            if has_whitespace:
                format_issues.append("Leading/trailing whitespace detected")
            
            passed = len(format_issues) == 0
            severity = ValidationSeverity.WARNING if format_issues else ValidationSeverity.INFO
            message = f"Format check - Issues: {', '.join(format_issues) if format_issues else 'None'}"
            
            details = {
                'format_issues': format_issues,
                'avg_length': float(lengths.mean()),
                'length_variance': float(length_variance)
            }
        
        else:
            # Numeric data format check
            passed = True
            severity = ValidationSeverity.INFO
            message = "Numeric data format check passed"
            details = {'data_type': str(data.dtype)}
        
        return ValidationResult(
            feature_name=feature_name,
            check_name='format_check',
            severity=severity,
            passed=passed,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def check_business_rules(self, feature_name: str, data: pd.Series, 
                           reference_data: pd.Series = None) -> ValidationResult:
        """Check business-specific rules"""
        # Get business rules from config
        business_rules = self.config.get('business_rules', {}).get(feature_name, [])
        
        if not business_rules:
            return ValidationResult(
                feature_name=feature_name,
                check_name='business_rule_check',
                severity=ValidationSeverity.INFO,
                passed=True,
                message="No business rules defined",
                details={},
                timestamp=datetime.now()
            )
        
        violations = []
        
        for rule in business_rules:
            rule_name = rule.get('name', 'unnamed_rule')
            rule_condition = rule.get('condition', '')
            
            try:
                # Simple rule evaluation (can be extended)
                if 'min_value' in rule and pd.api.types.is_numeric_dtype(data):
                    min_violations = (data < rule['min_value']).sum()
                    if min_violations > 0:
                        violations.append(f"{rule_name}: {min_violations} values below minimum {rule['min_value']}")
                
                if 'max_value' in rule and pd.api.types.is_numeric_dtype(data):
                    max_violations = (data > rule['max_value']).sum()
                    if max_violations > 0:
                        violations.append(f"{rule_name}: {max_violations} values above maximum {rule['max_value']}")
                
                if 'allowed_values' in rule:
                    invalid_values = ~data.isin(rule['allowed_values'])
                    invalid_count = invalid_values.sum()
                    if invalid_count > 0:
                        violations.append(f"{rule_name}: {invalid_count} invalid values")
                        
            except Exception as e:
                violations.append(f"{rule_name}: Rule evaluation failed - {str(e)}")
        
        passed = len(violations) == 0
        severity = ValidationSeverity.ERROR if violations else ValidationSeverity.INFO
        message = f"Business rules - Violations: {len(violations)}"
        
        return ValidationResult(
            feature_name=feature_name,
            check_name='business_rule_check',
            severity=severity,
            passed=passed,
            message=message,
            details={
                'violations': violations,
                'rules_checked': len(business_rules)
            },
            timestamp=datetime.now()
        )
    
    def check_data_drift(self, feature_name: str, data: pd.Series, 
                        reference_data: pd.Series = None) -> ValidationResult:
        """Check for data drift compared to reference data"""
        if reference_data is None or reference_data.empty:
            return ValidationResult(
                feature_name=feature_name,
                check_name='drift_check',
                severity=ValidationSeverity.INFO,
                passed=True,
                message="No reference data for drift check",
                details={},
                timestamp=datetime.now()
            )
        
        data_clean = data.dropna()
        ref_clean = reference_data.dropna()
        
        if data_clean.empty or ref_clean.empty:
            return ValidationResult(
                feature_name=feature_name,
                check_name='drift_check',
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="Insufficient data for drift check",
                details={},
                timestamp=datetime.now()
            )
        
        drift_detected = False
        drift_score = 0.0
        details = {}
        
        if pd.api.types.is_numeric_dtype(data_clean):
            # Statistical tests for numeric data
            from scipy import stats
            
            # Mean difference
            mean_diff = abs(data_clean.mean() - ref_clean.mean())
            std_pooled = np.sqrt((data_clean.var() + ref_clean.var()) / 2)
            normalized_mean_diff = mean_diff / std_pooled if std_pooled > 0 else 0
            
            # Kolmogorov-Smirnov test
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(data_clean, ref_clean)
                drift_score = ks_stat
                drift_detected = ks_pvalue < 0.05  # 5% significance level
                
                details.update({
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'mean_diff_normalized': float(normalized_mean_diff)
                })
            except Exception:
                drift_detected = normalized_mean_diff > 0.5
                drift_score = normalized_mean_diff
        
        else:
            # Chi-square test for categorical data
            try:
                # Compare value distributions
                data_counts = data_clean.value_counts(normalize=True)
                ref_counts = ref_clean.value_counts(normalize=True)
                
                # Align indices
                all_values = set(data_counts.index) | set(ref_counts.index)
                data_aligned = pd.Series([data_counts.get(v, 0) for v in all_values], index=all_values)
                ref_aligned = pd.Series([ref_counts.get(v, 0) for v in all_values], index=all_values)
                
                # Calculate drift score as total variation distance
                drift_score = 0.5 * abs(data_aligned - ref_aligned).sum()
                drift_detected = drift_score > 0.1  # 10% threshold
                
                details.update({
                    'drift_score': float(drift_score),
                    'new_categories': len(set(data_counts.index) - set(ref_counts.index)),
                    'missing_categories': len(set(ref_counts.index) - set(data_counts.index))
                })
            except Exception:
                drift_detected = False
                drift_score = 0.0
        
        passed = not drift_detected
        severity = ValidationSeverity.WARNING if drift_detected else ValidationSeverity.INFO
        message = f"Data drift - Score: {drift_score:.3f}, Detected: {drift_detected}"
        
        return ValidationResult(
            feature_name=feature_name,
            check_name='drift_check',
            severity=severity,
            passed=passed,
            message=message,
            details=details,
            timestamp=datetime.now()
        )
    
    def check_correlations(self, feature_name: str, data: pd.Series, 
                          reference_data: pd.Series = None) -> ValidationResult:
        """Check for unexpected correlations (requires full dataset context)"""
        # This is a placeholder - would need full dataset context
        return ValidationResult(
            feature_name=feature_name,
            check_name='correlation_check',
            severity=ValidationSeverity.INFO,
            passed=True,
            message="Correlation check requires full dataset context",
            details={},
            timestamp=datetime.now()
        )
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results"""
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        
        by_severity = {}
        by_feature = {}
        
        for result in results:
            # Count by severity
            severity = result.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count by feature
            feature = result.feature_name
            if feature not in by_feature:
                by_feature[feature] = {'total': 0, 'passed': 0, 'failed': 0}
            by_feature[feature]['total'] += 1
            if result.passed:
                by_feature[feature]['passed'] += 1
            else:
                by_feature[feature]['failed'] += 1
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'by_severity': by_severity,
            'by_feature': by_feature,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_validation_report(self, results: List[ValidationResult], filepath: str):
        """Export validation results to file"""
        import json
        
        report_data = {
            'summary': self.get_validation_summary(results),
            'results': []
        }
        
        for result in results:
            result_dict = {
                'feature_name': result.feature_name,
                'check_name': result.check_name,
                'severity': result.severity.value,
                'passed': result.passed,
                'message': result.message,
                'details': result.details,
                'timestamp': result.timestamp.isoformat()
            }
            report_data['results'].append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Validation report exported to {filepath}")
