"""
Model evaluation metrics for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, log_loss, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logger = logging.getLogger(__name__)

class ModelMetrics:
    """Comprehensive model evaluation metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_cache = {}
        
    def calculate_all_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                            y_pred: np.ndarray = None, threshold: float = 0.5) -> Dict[str, Any]:
        """Calculate comprehensive set of metrics"""
        
        if y_pred is None:
            y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._calculate_classification_metrics(y_true, y_pred, y_pred_proba))
        
        # ROC and PR metrics
        metrics.update(self._calculate_roc_pr_metrics(y_true, y_pred_proba))
        
        # Probability calibration metrics
        metrics.update(self._calculate_calibration_metrics(y_true, y_pred_proba))
        
        # Business metrics for credit risk
        metrics.update(self._calculate_business_metrics(y_true, y_pred_proba, threshold))
        
        # Statistical metrics
        metrics.update(self._calculate_statistical_metrics(y_true, y_pred_proba))
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                        y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate basic classification metrics"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba)
        }
    
    def _calculate_roc_pr_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC and Precision-Recall metrics"""
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = roc_thresholds[optimal_idx]
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': optimal_threshold,
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            },
            'pr_curve': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        }
    
    def _calculate_calibration_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate probability calibration metrics"""
        from sklearn.calibration import calibration_curve
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        # Expected Calibration Error
        ece = self._calculate_ece(y_true, y_pred_proba)
        
        # Reliability diagram data
        return {
            'expected_calibration_error': ece,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def _calculate_ece(self, y_true: pd.Series, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_business_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                                  threshold: float) -> Dict[str, Any]:
        """Calculate business-specific metrics for credit risk"""
        
        y_pred = (y_pred_proba > threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Business metrics
        approval_rate = (tn + fp) / len(y_true)  # Percentage approved
        default_rate = y_true.mean()  # Overall default rate
        
        # Risk metrics
        precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Economic metrics (simplified)
        # Assume: profit per good loan = $1000, loss per default = $5000
        profit_per_good = 1000
        loss_per_default = 5000
        
        economic_value = (tn * profit_per_good) - (fn * loss_per_default)
        
        return {
            'approval_rate': approval_rate,
            'default_rate': default_rate,
            'precision_at_threshold': precision_at_threshold,
            'recall_at_threshold': recall_at_threshold,
            'economic_value': economic_value,
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }
        }
    
    def _calculate_statistical_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate statistical metrics"""
        
        # Kolmogorov-Smirnov statistic
        ks_statistic = self._calculate_ks_statistic(y_true, y_pred_proba)
        
        # Gini coefficient
        gini = 2 * roc_auc_score(y_true, y_pred_proba) - 1
        
        # Population Stability Index (simplified)
        psi = self._calculate_psi(y_pred_proba, y_pred_proba)  # Self-comparison for demo
        
        return {
            'ks_statistic': ks_statistic,
            'gini_coefficient': gini,
            'population_stability_index': psi
        }
    
    def _calculate_ks_statistic(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        
        # Separate scores for positive and negative classes
        pos_scores = y_pred_proba[y_true == 1]
        neg_scores = y_pred_proba[y_true == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0
        
        # Calculate empirical CDFs
        all_scores = np.concatenate([pos_scores, neg_scores])
        thresholds = np.sort(np.unique(all_scores))
        
        pos_cdf = np.searchsorted(np.sort(pos_scores), thresholds, side='right') / len(pos_scores)
        neg_cdf = np.searchsorted(np.sort(neg_scores), thresholds, side='right') / len(neg_scores)
        
        # KS statistic is maximum difference between CDFs
        ks_stat = np.max(np.abs(pos_cdf - neg_cdf))
        
        return ks_stat
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                      n_bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        
        # Calculate distributions
        expected_dist, _ = np.histogram(expected, bins=bins)
        actual_dist, _ = np.histogram(actual, bins=bins)
        
        # Normalize to probabilities
        expected_dist = expected_dist / len(expected)
        actual_dist = actual_dist / len(actual)
        
        # Calculate PSI
        psi = 0
        for exp, act in zip(expected_dist, actual_dist):
            if exp > 0 and act > 0:
                psi += (act - exp) * np.log(act / exp)
        
        return psi
    
    def calculate_threshold_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                                  thresholds: List[float] = None) -> pd.DataFrame:
        """Calculate metrics across different thresholds"""
        
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'approval_rate': (tn + fp) / len(y_true),
                'true_positive': int(tp),
                'false_positive': int(fp),
                'true_negative': int(tn),
                'false_negative': int(fn)
            }
            
            threshold_metrics.append(metrics)
        
        return pd.DataFrame(threshold_metrics)
    
    def generate_model_report(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                            model_name: str = "Model") -> Dict[str, Any]:
        """Generate comprehensive model evaluation report"""
        
        # Calculate all metrics
        all_metrics = self.calculate_all_metrics(y_true, y_pred_proba)
        
        # Threshold analysis
        threshold_analysis = self.calculate_threshold_metrics(y_true, y_pred_proba)
        
        # Score distribution analysis
        score_distribution = self._analyze_score_distribution(y_true, y_pred_proba)
        
        report = {
            'model_name': model_name,
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'n_samples': len(y_true),
                'n_positive': int(y_true.sum()),
                'n_negative': int(len(y_true) - y_true.sum()),
                'positive_rate': float(y_true.mean())
            },
            'metrics': all_metrics,
            'threshold_analysis': threshold_analysis.to_dict('records'),
            'score_distribution': score_distribution
        }
        
        return report
    
    def _analyze_score_distribution(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze score distribution by class"""
        
        pos_scores = y_pred_proba[y_true == 1]
        neg_scores = y_pred_proba[y_true == 0]
        
        return {
            'positive_class': {
                'mean': float(np.mean(pos_scores)),
                'std': float(np.std(pos_scores)),
                'min': float(np.min(pos_scores)),
                'max': float(np.max(pos_scores)),
                'percentiles': {
                    '25th': float(np.percentile(pos_scores, 25)),
                    '50th': float(np.percentile(pos_scores, 50)),
                    '75th': float(np.percentile(pos_scores, 75))
                }
            },
            'negative_class': {
                'mean': float(np.mean(neg_scores)),
                'std': float(np.std(neg_scores)),
                'min': float(np.min(neg_scores)),
                'max': float(np.max(neg_scores)),
                'percentiles': {
                    '25th': float(np.percentile(neg_scores, 25)),
                    '50th': float(np.percentile(neg_scores, 50)),
                    '75th': float(np.percentile(neg_scores, 75))
                }
            },
            'separation': {
                'mean_difference': float(np.mean(neg_scores) - np.mean(pos_scores)),
                'overlap_percentage': self._calculate_overlap_percentage(pos_scores, neg_scores)
            }
        }
    
    def _calculate_overlap_percentage(self, pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
        """Calculate percentage of score overlap between classes"""
        
        pos_min, pos_max = np.min(pos_scores), np.max(pos_scores)
        neg_min, neg_max = np.min(neg_scores), np.max(neg_scores)
        
        # Calculate overlap range
        overlap_start = max(pos_min, neg_min)
        overlap_end = min(pos_max, neg_max)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        # Calculate overlap percentage
        total_range = max(pos_max, neg_max) - min(pos_min, neg_min)
        overlap_range = overlap_end - overlap_start
        
        return (overlap_range / total_range) * 100 if total_range > 0 else 0.0
    
    def calculate_lift_chart(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                           n_deciles: int = 10) -> pd.DataFrame:
        """Calculate lift chart data"""
        
        # Create DataFrame with scores and labels
        df = pd.DataFrame({
            'score': y_pred_proba,
            'actual': y_true
        })
        
        # Sort by score (descending)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Create deciles
        df['decile'] = pd.cut(range(len(df)), n_deciles, labels=range(1, n_deciles + 1))
        
        # Calculate lift metrics
        lift_data = []
        cumulative_positives = 0
        cumulative_total = 0
        
        for decile in range(1, n_deciles + 1):
            decile_data = df[df['decile'] == decile]
            
            decile_positives = decile_data['actual'].sum()
            decile_total = len(decile_data)
            
            cumulative_positives += decile_positives
            cumulative_total += decile_total
            
            # Calculate metrics
            decile_rate = decile_positives / decile_total if decile_total > 0 else 0
            cumulative_rate = cumulative_positives / cumulative_total if cumulative_total > 0 else 0
            baseline_rate = y_true.mean()
            
            lift = decile_rate / baseline_rate if baseline_rate > 0 else 0
            cumulative_lift = cumulative_rate / baseline_rate if baseline_rate > 0 else 0
            
            lift_data.append({
                'decile': decile,
                'n_samples': decile_total,
                'n_positives': int(decile_positives),
                'decile_rate': decile_rate,
                'cumulative_rate': cumulative_rate,
                'lift': lift,
                'cumulative_lift': cumulative_lift,
                'cumulative_capture': cumulative_positives / y_true.sum() if y_true.sum() > 0 else 0
            })
        
        return pd.DataFrame(lift_data)
    
    def calculate_gains_chart(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> pd.DataFrame:
        """Calculate gains chart data"""
        
        # Sort by predicted probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        sorted_labels = y_true.iloc[sorted_indices].values
        
        # Calculate cumulative gains
        cumulative_positives = np.cumsum(sorted_labels)
        total_positives = np.sum(sorted_labels)
        
        # Calculate percentage of population and positives captured
        population_percentages = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels) * 100
        gains = cumulative_positives / total_positives * 100 if total_positives > 0 else np.zeros_like(cumulative_positives)
        
        # Sample points for chart (to avoid too many points)
        sample_indices = np.linspace(0, len(population_percentages) - 1, 100).astype(int)
        
        gains_data = pd.DataFrame({
            'population_percentage': population_percentages[sample_indices],
            'gains_percentage': gains[sample_indices]
        })
        
        # Add baseline (random model)
        gains_data['baseline'] = gains_data['population_percentage']
        
        return gains_data
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models"""
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            
            comparison_data.append({
                'Model': model_name,
                'ROC_AUC': metrics.get('roc_auc', np.nan),
                'PR_AUC': metrics.get('pr_auc', np.nan),
                'Accuracy': metrics.get('accuracy', np.nan),
                'Precision': metrics.get('precision', np.nan),
                'Recall': metrics.get('recall', np.nan),
                'F1_Score': metrics.get('f1_score', np.nan),
                'Brier_Score': metrics.get('brier_score', np.nan),
                'Log_Loss': metrics.get('log_loss', np.nan),
                'ECE': metrics.get('expected_calibration_error', np.nan)
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('ROC_AUC', ascending=False)
    
    def save_metrics(self, metrics: Dict[str, Any], filepath: str):
        """Save metrics to file"""
        
        if filepath.endswith('.json'):
            import json
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
        else:
            joblib.dump(metrics, filepath)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """Load metrics from file"""
        
        if filepath.endswith('.json'):
            import json
            with open(filepath, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = joblib.load(filepath)
        
        logger.info(f"Metrics loaded from {filepath}")
        return metrics
