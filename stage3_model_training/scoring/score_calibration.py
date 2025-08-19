"""
Score calibration methods for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import joblib

logger = logging.getLogger(__name__)

class ScoreCalibrator:
    """Score calibration for credit risk models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibrator = None
        self.calibration_method = config.get('calibration_method', 'platt')
        self.is_fitted = False
        
    def fit(self, y_true: pd.Series, y_scores: np.ndarray, 
            method: str = None) -> 'ScoreCalibrator':
        """Fit calibration model"""
        
        if method:
            self.calibration_method = method
            
        if self.calibration_method == 'platt':
            # Platt scaling (logistic regression)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_scores.reshape(-1, 1), y_true)
            
        elif self.calibration_method == 'isotonic':
            # Isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_scores, y_true)
            
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        self.is_fitted = True
        logger.info(f"Score calibrator fitted using {self.calibration_method} method")
        return self
    
    def transform(self, y_scores: np.ndarray) -> np.ndarray:
        """Transform scores using fitted calibrator"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet")
        
        if self.calibration_method == 'platt':
            return self.calibrator.predict_proba(y_scores.reshape(-1, 1))[:, 1]
        elif self.calibration_method == 'isotonic':
            return self.calibrator.transform(y_scores)
    
    def evaluate_calibration(self, y_true: pd.Series, y_scores: np.ndarray,
                           n_bins: int = 10) -> Dict[str, Any]:
        """Evaluate calibration quality"""
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_scores, n_bins=n_bins
        )
        
        # Calculate calibration metrics
        brier_score = brier_score_loss(y_true, y_scores)
        log_loss_score = log_loss(y_true, y_scores)
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(y_true, y_scores, n_bins)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(y_true, y_scores, n_bins)
        
        return {
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            },
            'n_bins': n_bins
        }
    
    def _calculate_ece(self, y_true: pd.Series, y_scores: np.ndarray, 
                      n_bins: int) -> float:
        """Calculate Expected Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_mce(self, y_true: pd.Series, y_scores: np.ndarray,
                      n_bins: int) -> float:
        """Calculate Maximum Calibration Error"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_scores > bin_lower) & (y_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_scores[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def save_calibrator(self, filepath: str):
        """Save calibrator to file"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet")
        
        calibrator_data = {
            'calibrator': self.calibrator,
            'calibration_method': self.calibration_method,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(calibrator_data, filepath)
        logger.info(f"Calibrator saved to {filepath}")
    
    def load_calibrator(self, filepath: str):
        """Load calibrator from file"""
        calibrator_data = joblib.load(filepath)
        
        self.calibrator = calibrator_data['calibrator']
        self.calibration_method = calibrator_data['calibration_method']
        self.config = calibrator_data['config']
        self.is_fitted = calibrator_data['is_fitted']
        
        logger.info(f"Calibrator loaded from {filepath}")

class MultiClassCalibrator:
    """Multi-class calibration for multiple risk categories"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calibrators = {}
        self.classes = None
        self.is_fitted = False
        
    def fit(self, y_true: pd.Series, y_scores: np.ndarray) -> 'MultiClassCalibrator':
        """Fit calibrators for each class"""
        
        self.classes = np.unique(y_true)
        
        # Fit one-vs-rest calibrators
        for class_label in self.classes:
            binary_y = (y_true == class_label).astype(int)
            class_scores = y_scores[:, class_label] if y_scores.ndim > 1 else y_scores
            
            calibrator = ScoreCalibrator(self.config)
            calibrator.fit(binary_y, class_scores)
            self.calibrators[class_label] = calibrator
        
        self.is_fitted = True
        logger.info(f"Multi-class calibrator fitted for {len(self.classes)} classes")
        return self
    
    def transform(self, y_scores: np.ndarray) -> np.ndarray:
        """Transform scores for all classes"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet")
        
        calibrated_scores = np.zeros_like(y_scores)
        
        for i, class_label in enumerate(self.classes):
            class_scores = y_scores[:, i] if y_scores.ndim > 1 else y_scores
            calibrated_scores[:, i] = self.calibrators[class_label].transform(class_scores)
        
        # Normalize to ensure probabilities sum to 1
        calibrated_scores = calibrated_scores / calibrated_scores.sum(axis=1, keepdims=True)
        
        return calibrated_scores

class TemporalCalibrator:
    """Temporal calibration for time-varying score calibration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_windows = config.get('time_windows', ['1M', '3M', '6M', '1Y'])
        self.calibrators = {}
        self.is_fitted = False
        
    def fit(self, y_true: pd.Series, y_scores: np.ndarray, 
            timestamps: pd.Series) -> 'TemporalCalibrator':
        """Fit calibrators for different time windows"""
        
        timestamps = pd.to_datetime(timestamps)
        
        for window in self.time_windows:
            # Get data for this time window
            cutoff_date = timestamps.max() - pd.Timedelta(window)
            window_mask = timestamps >= cutoff_date
            
            if window_mask.sum() > 0:
                window_y_true = y_true[window_mask]
                window_y_scores = y_scores[window_mask]
                
                calibrator = ScoreCalibrator(self.config)
                calibrator.fit(window_y_true, window_y_scores)
                self.calibrators[window] = calibrator
        
        self.is_fitted = True
        logger.info(f"Temporal calibrator fitted for {len(self.calibrators)} time windows")
        return self
    
    def transform(self, y_scores: np.ndarray, time_window: str = '1Y') -> np.ndarray:
        """Transform scores using specified time window calibrator"""
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted yet")
        
        if time_window not in self.calibrators:
            # Fallback to most recent available window
            available_windows = list(self.calibrators.keys())
            time_window = available_windows[-1] if available_windows else None
            
        if time_window is None:
            raise ValueError("No calibrators available")
        
        return self.calibrators[time_window].transform(y_scores)

class CreditScoreMapper:
    """Map probabilities to credit score ranges"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.score_ranges = config.get('score_ranges', {
            'excellent': (800, 1000),
            'good': (600, 799),
            'fair': (400, 599),
            'poor': (0, 399)
        })
        
    def probability_to_score(self, probabilities: np.ndarray) -> np.ndarray:
        """Convert default probabilities to credit scores"""
        # Higher probability of default = lower credit score
        credit_scores = 1000 - (probabilities * 1000)
        return np.clip(credit_scores, 0, 1000).astype(int)
    
    def score_to_category(self, scores: np.ndarray) -> np.ndarray:
        """Convert credit scores to categories"""
        categories = np.full(len(scores), 'unknown', dtype=object)
        
        for category, (min_score, max_score) in self.score_ranges.items():
            mask = (scores >= min_score) & (scores <= max_score)
            categories[mask] = category
        
        return categories
    
    def get_score_distribution(self, scores: np.ndarray) -> Dict[str, int]:
        """Get distribution of scores across categories"""
        categories = self.score_to_category(scores)
        
        distribution = {}
        for category in self.score_ranges.keys():
            distribution[category] = int(np.sum(categories == category))
        
        return distribution
    
    def calibrate_score_mapping(self, y_true: pd.Series, scores: np.ndarray,
                              target_default_rates: Dict[str, float] = None) -> Dict[str, Tuple[int, int]]:
        """Calibrate score ranges based on target default rates"""
        
        if target_default_rates is None:
            target_default_rates = {
                'excellent': 0.01,  # 1% default rate
                'good': 0.05,       # 5% default rate
                'fair': 0.15,       # 15% default rate
                'poor': 0.40        # 40% default rate
            }
        
        # Sort scores and corresponding labels
        sorted_indices = np.argsort(scores)[::-1]  # Descending order
        sorted_scores = scores[sorted_indices]
        sorted_labels = y_true.iloc[sorted_indices].values
        
        # Calculate cumulative default rates
        cumulative_defaults = np.cumsum(sorted_labels)
        cumulative_counts = np.arange(1, len(sorted_labels) + 1)
        cumulative_default_rates = cumulative_defaults / cumulative_counts
        
        # Find score thresholds for target default rates
        new_ranges = {}
        prev_threshold = 1000
        
        for category in ['excellent', 'good', 'fair', 'poor']:
            target_rate = target_default_rates[category]
            
            # Find score where default rate exceeds target
            threshold_idx = np.where(cumulative_default_rates <= target_rate)[0]
            
            if len(threshold_idx) > 0:
                threshold_score = int(sorted_scores[threshold_idx[-1]])
            else:
                threshold_score = 0
            
            new_ranges[category] = (threshold_score, prev_threshold - 1)
            prev_threshold = threshold_score
        
        # Adjust ranges to ensure no gaps
        new_ranges['poor'] = (0, new_ranges['poor'][1])
        
        return new_ranges
