"""
Model stability testing for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats
import joblib

logger = logging.getLogger(__name__)

class StabilityTester:
    """Model stability testing framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stability_results = {}
        
    def test_temporal_stability(self, model: Any, data: pd.DataFrame,
                              target_column: str, date_column: str,
                              window_size: str = '3M') -> Dict[str, Any]:
        """Test model stability over time"""
        
        # Sort data by date
        data = data.sort_values(date_column).reset_index(drop=True)
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in [target_column, date_column]]
        
        # Create time windows
        start_date = data[date_column].min()
        end_date = data[date_column].max()
        
        window_results = []
        current_date = start_date
        
        while current_date + pd.Timedelta(window_size) <= end_date:
            window_end = current_date + pd.Timedelta(window_size)
            
            # Get window data
            window_mask = (data[date_column] >= current_date) & (data[date_column] < window_end)
            window_data = data[window_mask]
            
            if len(window_data) > 10:  # Minimum samples for evaluation
                X_window = window_data[feature_columns]
                y_window = window_data[target_column]
                
                # Make predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_window)[:, 1]
                else:
                    y_pred_proba = model.predict(X_window)
                
                # Calculate metrics
                if len(np.unique(y_window)) > 1:
                    auc = roc_auc_score(y_window, y_pred_proba)
                else:
                    auc = np.nan
                
                window_results.append({
                    'start_date': current_date,
                    'end_date': window_end,
                    'n_samples': len(window_data),
                    'positive_rate': y_window.mean(),
                    'auc': auc,
                    'mean_score': np.mean(y_pred_proba),
                    'std_score': np.std(y_pred_proba)
                })
            
            current_date += pd.Timedelta('1M')  # Move by 1 month
        
        # Analyze stability
        stability_analysis = self._analyze_temporal_stability(window_results)
        
        return {
            'window_results': window_results,
            'stability_analysis': stability_analysis
        }
    
    def _analyze_temporal_stability(self, window_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal stability from window results"""
        
        aucs = [r['auc'] for r in window_results if not np.isnan(r['auc'])]
        mean_scores = [r['mean_score'] for r in window_results]
        
        if not aucs:
            return {'status': 'insufficient_data'}
        
        # Calculate stability metrics
        auc_cv = np.std(aucs) / np.mean(aucs) if np.mean(aucs) > 0 else np.nan
        score_cv = np.std(mean_scores) / np.mean(mean_scores) if np.mean(mean_scores) > 0 else np.nan
        
        # Trend analysis
        if len(aucs) > 2:
            x = np.arange(len(aucs))
            auc_trend_slope, _, auc_trend_r, auc_trend_p, _ = stats.linregress(x, aucs)
        else:
            auc_trend_slope = auc_trend_r = auc_trend_p = np.nan
        
        # Stability classification
        stability_status = 'stable'
        if auc_cv > 0.1:
            stability_status = 'unstable'
        elif auc_trend_p < 0.05 and auc_trend_slope < -0.01:
            stability_status = 'degrading'
        
        return {
            'auc_coefficient_of_variation': auc_cv,
            'score_coefficient_of_variation': score_cv,
            'auc_trend_slope': auc_trend_slope,
            'auc_trend_correlation': auc_trend_r,
            'auc_trend_p_value': auc_trend_p,
            'stability_status': stability_status,
            'n_windows': len(window_results)
        }
    
    def test_feature_stability(self, model: Any, data: pd.DataFrame,
                             target_column: str, n_bootstrap: int = 100) -> Dict[str, Any]:
        """Test feature importance stability"""
        
        feature_columns = [col for col in data.columns if col != target_column]
        
        # Bootstrap sampling to test feature stability
        feature_importance_samples = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
            bootstrap_data = data.iloc[bootstrap_indices]
            
            X_bootstrap = bootstrap_data[feature_columns]
            y_bootstrap = bootstrap_data[target_column]
            
            # Train model on bootstrap sample
            from sklearn.base import clone
            bootstrap_model = clone(model)
            bootstrap_model.fit(X_bootstrap, y_bootstrap)
            
            # Get feature importance
            if hasattr(bootstrap_model, 'feature_importances_'):
                importance = bootstrap_model.feature_importances_
            elif hasattr(bootstrap_model, 'coef_'):
                importance = np.abs(bootstrap_model.coef_[0])
            else:
                continue
            
            feature_importance_samples.append(importance)
        
        if not feature_importance_samples:
            return {'status': 'no_feature_importance_available'}
        
        # Analyze feature stability
        feature_importance_array = np.array(feature_importance_samples)
        
        stability_results = {}
        for i, feature in enumerate(feature_columns):
            feature_importances = feature_importance_array[:, i]
            
            stability_results[feature] = {
                'mean_importance': np.mean(feature_importances),
                'std_importance': np.std(feature_importances),
                'cv_importance': np.std(feature_importances) / np.mean(feature_importances) if np.mean(feature_importances) > 0 else np.nan,
                'min_importance': np.min(feature_importances),
                'max_importance': np.max(feature_importances)
            }
        
        # Overall feature stability
        cv_values = [r['cv_importance'] for r in stability_results.values() if not np.isnan(r['cv_importance'])]
        overall_stability = np.mean(cv_values) if cv_values else np.nan
        
        return {
            'feature_stability': stability_results,
            'overall_feature_stability': overall_stability,
            'stable_features': [f for f, r in stability_results.items() if r['cv_importance'] < 0.2],
            'unstable_features': [f for f, r in stability_results.items() if r['cv_importance'] > 0.5],
            'n_bootstrap_samples': n_bootstrap
        }
    
    def test_prediction_stability(self, model: Any, X_test: pd.DataFrame,
                                n_perturbations: int = 100, noise_level: float = 0.01) -> Dict[str, Any]:
        """Test prediction stability under input perturbations"""
        
        original_predictions = model.predict_proba(X_test)[:, 1]
        
        perturbed_predictions = []
        
        for i in range(n_perturbations):
            # Add random noise to features
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_perturbed = X_test + noise
            
            # Make predictions
            pred = model.predict_proba(X_perturbed)[:, 1]
            perturbed_predictions.append(pred)
        
        perturbed_predictions = np.array(perturbed_predictions)
        
        # Calculate stability metrics
        prediction_std = np.std(perturbed_predictions, axis=0)
        prediction_cv = prediction_std / np.mean(perturbed_predictions, axis=0)
        
        # Replace inf and nan values
        prediction_cv = np.where(np.isfinite(prediction_cv), prediction_cv, 0)
        
        stability_metrics = {
            'mean_prediction_std': np.mean(prediction_std),
            'mean_prediction_cv': np.mean(prediction_cv),
            'max_prediction_std': np.max(prediction_std),
            'max_prediction_cv': np.max(prediction_cv),
            'stable_predictions_percentage': np.mean(prediction_cv < 0.1) * 100,
            'unstable_predictions_percentage': np.mean(prediction_cv > 0.3) * 100
        }
        
        return {
            'stability_metrics': stability_metrics,
            'prediction_std': prediction_std,
            'prediction_cv': prediction_cv,
            'original_predictions': original_predictions,
            'perturbed_predictions': perturbed_predictions,
            'n_perturbations': n_perturbations,
            'noise_level': noise_level
        }
    
    def test_data_drift_impact(self, model: Any, reference_data: pd.DataFrame,
                             test_data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Test model stability under data drift"""
        
        feature_columns = [col for col in reference_data.columns if col != target_column]
        
        # Calculate feature distributions
        drift_analysis = {}
        
        for feature in feature_columns:
            if reference_data[feature].dtype in ['float64', 'int64']:
                # Kolmogorov-Smirnov test
                ks_stat, ks_p_value = stats.ks_2samp(
                    reference_data[feature].dropna(),
                    test_data[feature].dropna()
                )
                
                drift_analysis[feature] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p_value,
                    'drift_detected': ks_p_value < 0.05,
                    'reference_mean': reference_data[feature].mean(),
                    'test_mean': test_data[feature].mean(),
                    'mean_shift': test_data[feature].mean() - reference_data[feature].mean()
                }
        
        # Model performance on both datasets
        X_ref = reference_data[feature_columns]
        y_ref = reference_data[target_column]
        X_test = test_data[feature_columns]
        y_test = test_data[target_column]
        
        # Predictions
        ref_pred = model.predict_proba(X_ref)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
        
        # Performance comparison
        if len(np.unique(y_ref)) > 1:
            ref_auc = roc_auc_score(y_ref, ref_pred)
        else:
            ref_auc = np.nan
            
        if len(np.unique(y_test)) > 1:
            test_auc = roc_auc_score(y_test, test_pred)
        else:
            test_auc = np.nan
        
        performance_impact = {
            'reference_auc': ref_auc,
            'test_auc': test_auc,
            'auc_difference': test_auc - ref_auc if not (np.isnan(ref_auc) or np.isnan(test_auc)) else np.nan,
            'performance_degradation': (test_auc < ref_auc - 0.05) if not (np.isnan(ref_auc) or np.isnan(test_auc)) else False
        }
        
        return {
            'drift_analysis': drift_analysis,
            'performance_impact': performance_impact,
            'drifted_features': [f for f, r in drift_analysis.items() if r['drift_detected']],
            'n_drifted_features': sum(1 for r in drift_analysis.values() if r['drift_detected'])
        }
    
    def comprehensive_stability_test(self, model: Any, data: pd.DataFrame,
                                   target_column: str, date_column: str = None) -> Dict[str, Any]:
        """Run comprehensive stability tests"""
        
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': type(model).__name__
        }
        
        # Feature stability test
        feature_stability = self.test_feature_stability(model, data, target_column)
        results['feature_stability'] = feature_stability
        
        # Prediction stability test
        feature_columns = [col for col in data.columns if col != target_column and col != date_column]
        X_test = data[feature_columns]
        prediction_stability = self.test_prediction_stability(model, X_test)
        results['prediction_stability'] = prediction_stability
        
        # Temporal stability test (if date column available)
        if date_column and date_column in data.columns:
            temporal_stability = self.test_temporal_stability(model, data, target_column, date_column)
            results['temporal_stability'] = temporal_stability
        
        # Overall stability assessment
        results['overall_assessment'] = self._assess_overall_stability(results)
        
        return results
    
    def _assess_overall_stability(self, stability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall model stability"""
        
        stability_scores = []
        issues = []
        
        # Feature stability assessment
        if 'feature_stability' in stability_results:
            feature_stability = stability_results['feature_stability']['overall_feature_stability']
            if not np.isnan(feature_stability):
                if feature_stability < 0.2:
                    stability_scores.append(1.0)  # Very stable
                elif feature_stability < 0.5:
                    stability_scores.append(0.7)  # Moderately stable
                else:
                    stability_scores.append(0.3)  # Unstable
                    issues.append("High feature importance variability")
        
        # Prediction stability assessment
        if 'prediction_stability' in stability_results:
            pred_stability = stability_results['prediction_stability']['stability_metrics']
            unstable_percentage = pred_stability['unstable_predictions_percentage']
            
            if unstable_percentage < 5:
                stability_scores.append(1.0)  # Very stable
            elif unstable_percentage < 15:
                stability_scores.append(0.7)  # Moderately stable
            else:
                stability_scores.append(0.3)  # Unstable
                issues.append("High prediction variability under noise")
        
        # Temporal stability assessment
        if 'temporal_stability' in stability_results:
            temporal_analysis = stability_results['temporal_stability']['stability_analysis']
            if temporal_analysis.get('stability_status') == 'stable':
                stability_scores.append(1.0)
            elif temporal_analysis.get('stability_status') == 'degrading':
                stability_scores.append(0.5)
                issues.append("Performance degrading over time")
            else:
                stability_scores.append(0.3)
                issues.append("Unstable performance over time")
        
        # Overall score
        overall_score = np.mean(stability_scores) if stability_scores else 0.5
        
        # Classification
        if overall_score >= 0.8:
            classification = 'highly_stable'
        elif overall_score >= 0.6:
            classification = 'moderately_stable'
        else:
            classification = 'unstable'
        
        return {
            'overall_stability_score': overall_score,
            'classification': classification,
            'issues': issues,
            'recommendations': self._get_stability_recommendations(classification, issues)
        }
    
    def _get_stability_recommendations(self, classification: str, issues: List[str]) -> List[str]:
        """Get recommendations based on stability assessment"""
        
        recommendations = []
        
        if classification == 'unstable':
            recommendations.append("Consider model retraining with more recent data")
            recommendations.append("Implement more robust feature engineering")
            recommendations.append("Add regularization to reduce overfitting")
        
        if "High feature importance variability" in issues:
            recommendations.append("Use feature selection to remove unstable features")
            recommendations.append("Consider ensemble methods to improve stability")
        
        if "High prediction variability under noise" in issues:
            recommendations.append("Implement input validation and outlier detection")
            recommendations.append("Add noise regularization during training")
        
        if "Performance degrading over time" in issues:
            recommendations.append("Implement concept drift detection")
            recommendations.append("Set up automated model retraining")
        
        if not recommendations:
            recommendations.append("Model shows good stability - continue monitoring")
        
        return recommendations
    
    def save_stability_results(self, results: Dict[str, Any], filepath: str):
        """Save stability test results"""
        joblib.dump(results, filepath)
        logger.info(f"Stability results saved to {filepath}")
    
    def load_stability_results(self, filepath: str) -> Dict[str, Any]:
        """Load stability test results"""
        results = joblib.load(filepath)
        logger.info(f"Stability results loaded from {filepath}")
        return results
