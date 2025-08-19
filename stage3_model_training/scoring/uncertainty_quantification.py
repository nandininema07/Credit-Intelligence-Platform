"""
Uncertainty quantification for Stage 3 model predictions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import joblib

logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """Uncertainty quantification for credit risk predictions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_model = None
        self.uncertainty_models = {}
        self.is_fitted = False
        
    def fit(self, base_model: Any, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'UncertaintyQuantifier':
        """Fit uncertainty quantification models"""
        
        self.base_model = base_model
        
        # Epistemic uncertainty (model uncertainty)
        self._fit_epistemic_uncertainty(X_train, y_train, X_val, y_val)
        
        # Aleatoric uncertainty (data uncertainty)
        self._fit_aleatoric_uncertainty(X_train, y_train)
        
        self.is_fitted = True
        logger.info("Uncertainty quantification models fitted")
        return self
    
    def _fit_epistemic_uncertainty(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Fit model for epistemic uncertainty estimation"""
        
        # Use ensemble of models for epistemic uncertainty
        n_estimators = self.config.get('epistemic_estimators', 10)
        
        models = []
        for i in range(n_estimators):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(len(X_train), len(X_train), replace=True)
            X_bootstrap = X_train.iloc[bootstrap_indices]
            y_bootstrap = y_train.iloc[bootstrap_indices]
            
            # Clone and fit model
            from sklearn.base import clone
            model = clone(self.base_model)
            model.fit(X_bootstrap, y_bootstrap)
            models.append(model)
        
        self.uncertainty_models['epistemic'] = models
    
    def _fit_aleatoric_uncertainty(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fit model for aleatoric uncertainty estimation"""
        
        # Train a model to predict prediction variance
        # Use cross-validation predictions to estimate variance
        cv_predictions = cross_val_predict(
            self.base_model, X_train, y_train, 
            cv=5, method='predict_proba'
        )[:, 1]
        
        # Calculate residuals
        residuals = np.abs(y_train - cv_predictions)
        
        # Train variance model
        variance_model = RandomForestClassifier(
            n_estimators=50, random_state=42
        )
        
        # Convert residuals to binary (high/low variance)
        high_variance = residuals > np.median(residuals)
        variance_model.fit(X_train, high_variance)
        
        self.uncertainty_models['aleatoric'] = variance_model
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict with uncertainty estimates"""
        if not self.is_fitted:
            raise ValueError("Uncertainty quantifier not fitted yet")
        
        # Base prediction
        if hasattr(self.base_model, 'predict_proba'):
            base_prediction = self.base_model.predict_proba(X)[:, 1]
        else:
            base_prediction = self.base_model.predict(X)
        
        # Epistemic uncertainty
        epistemic_uncertainty = self._estimate_epistemic_uncertainty(X)
        
        # Aleatoric uncertainty
        aleatoric_uncertainty = self._estimate_aleatoric_uncertainty(X)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return {
            'predictions': base_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence_intervals': self._calculate_confidence_intervals(
                base_prediction, total_uncertainty
            )
        }
    
    def _estimate_epistemic_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate epistemic uncertainty using model ensemble"""
        
        if 'epistemic' not in self.uncertainty_models:
            return np.zeros(len(X))
        
        predictions = []
        for model in self.uncertainty_models['epistemic']:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Variance across ensemble predictions
        epistemic_uncertainty = np.std(predictions, axis=0)
        
        return epistemic_uncertainty
    
    def _estimate_aleatoric_uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate aleatoric uncertainty using variance model"""
        
        if 'aleatoric' not in self.uncertainty_models:
            return np.zeros(len(X))
        
        variance_model = self.uncertainty_models['aleatoric']
        
        # Predict variance level
        high_variance_prob = variance_model.predict_proba(X)[:, 1]
        
        # Convert to uncertainty estimate
        aleatoric_uncertainty = high_variance_prob * 0.2  # Scale factor
        
        return aleatoric_uncertainty
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, 
                                      uncertainties: np.ndarray,
                                      confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals"""
        
        # Assume normal distribution for confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 99%
        
        lower_bound = predictions - z_score * uncertainties
        upper_bound = predictions + z_score * uncertainties
        
        # Clip to valid probability range
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': upper_bound - lower_bound
        }
    
    def get_uncertainty_statistics(self, uncertainties: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Get statistics about uncertainty estimates"""
        
        stats = {}
        
        for uncertainty_type, values in uncertainties.items():
            if uncertainty_type == 'confidence_intervals':
                continue
                
            stats[uncertainty_type] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentiles': {
                    '25th': float(np.percentile(values, 25)),
                    '50th': float(np.percentile(values, 50)),
                    '75th': float(np.percentile(values, 75)),
                    '95th': float(np.percentile(values, 95))
                }
            }
        
        return stats
    
    def identify_high_uncertainty_samples(self, X: pd.DataFrame, 
                                        threshold_percentile: float = 90) -> Dict[str, Any]:
        """Identify samples with high uncertainty"""
        
        uncertainty_results = self.predict_with_uncertainty(X)
        total_uncertainty = uncertainty_results['total_uncertainty']
        
        # Calculate threshold
        threshold = np.percentile(total_uncertainty, threshold_percentile)
        
        # Find high uncertainty samples
        high_uncertainty_mask = total_uncertainty > threshold
        high_uncertainty_indices = np.where(high_uncertainty_mask)[0]
        
        return {
            'high_uncertainty_indices': high_uncertainty_indices.tolist(),
            'high_uncertainty_samples': len(high_uncertainty_indices),
            'total_samples': len(X),
            'percentage': (len(high_uncertainty_indices) / len(X)) * 100,
            'threshold': threshold,
            'uncertainty_values': total_uncertainty[high_uncertainty_mask].tolist()
        }

class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty estimation in neural networks"""
    
    def __init__(self, model: Any, n_samples: int = 100):
        self.model = model
        self.n_samples = n_samples
        
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict with uncertainty using MC Dropout"""
        
        predictions = []
        
        # Enable dropout during inference
        for _ in range(self.n_samples):
            # This assumes the model has a way to enable dropout during inference
            pred = self._predict_with_dropout(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_prediction = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return {
            'predictions': mean_prediction,
            'uncertainty': uncertainty,
            'all_predictions': predictions
        }
    
    def _predict_with_dropout(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with dropout enabled"""
        # This would need to be implemented based on the specific model type
        # For TensorFlow/Keras models, you would call model(X, training=True)
        # For now, return base prediction
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)

class BayesianUncertainty:
    """Bayesian uncertainty estimation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.posterior_samples = []
        self.is_fitted = False
        
    def fit_bayesian_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'BayesianUncertainty':
        """Fit Bayesian model using variational inference"""
        
        # This is a simplified implementation
        # In practice, you would use libraries like PyMC3, TensorFlow Probability, etc.
        
        n_samples = self.config.get('posterior_samples', 1000)
        
        # Simulate posterior samples (in practice, use MCMC or VI)
        for _ in range(n_samples):
            # Add noise to simulate different posterior samples
            noise_scale = 0.1
            sample_weights = np.random.normal(0, noise_scale, X_train.shape[1])
            self.posterior_samples.append(sample_weights)
        
        self.is_fitted = True
        logger.info(f"Bayesian model fitted with {n_samples} posterior samples")
        return self
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict with Bayesian uncertainty"""
        if not self.is_fitted:
            raise ValueError("Bayesian model not fitted yet")
        
        predictions = []
        
        for weights in self.posterior_samples:
            # Simple linear prediction with sampled weights
            pred = X.values @ weights
            # Apply sigmoid to get probabilities
            pred = 1 / (1 + np.exp(-pred))
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_prediction = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        # Calculate credible intervals
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'predictions': mean_prediction,
            'uncertainty': uncertainty,
            'credible_intervals': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            },
            'all_predictions': predictions
        }

class UncertaintyAggregator:
    """Aggregate uncertainty from multiple sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.uncertainty_sources = []
        
    def add_uncertainty_source(self, source: Any, weight: float = 1.0):
        """Add uncertainty source with weight"""
        self.uncertainty_sources.append({
            'source': source,
            'weight': weight
        })
    
    def aggregate_uncertainties(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Aggregate uncertainties from all sources"""
        
        all_predictions = []
        all_uncertainties = []
        total_weight = 0
        
        for source_info in self.uncertainty_sources:
            source = source_info['source']
            weight = source_info['weight']
            
            result = source.predict_with_uncertainty(X)
            
            all_predictions.append(result['predictions'] * weight)
            all_uncertainties.append(result.get('uncertainty', 
                                               result.get('total_uncertainty', 
                                                        np.zeros(len(X)))) * weight)
            total_weight += weight
        
        # Weighted average
        aggregated_predictions = np.sum(all_predictions, axis=0) / total_weight
        aggregated_uncertainties = np.sum(all_uncertainties, axis=0) / total_weight
        
        return {
            'predictions': aggregated_predictions,
            'uncertainty': aggregated_uncertainties,
            'n_sources': len(self.uncertainty_sources)
        }
