"""
Model ensemble for Stage 3.
Combines multiple models for improved prediction accuracy.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelEnsemble:
    """Model ensemble for combining multiple trained models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.weights = {}
        self.ensemble_method = config.get('ensemble_method', 'voting')
        self.voting_method = config.get('voting_method', 'soft')
        self.fitted = False
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model {name} to ensemble with weight {weight}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelEnsemble':
        """Fit the ensemble"""
        try:
            if len(self.models) == 0:
                raise ValueError("No models added to ensemble")
            
            # Create ensemble based on method
            if self.ensemble_method == 'voting':
                model_list = [(name, model) for name, model in self.models.items()]
                
                # Determine if classification or regression
                is_classification = hasattr(list(self.models.values())[0], 'predict_proba')
                
                if is_classification:
                    self.ensemble = VotingClassifier(
                        estimators=model_list,
                        voting=self.voting_method,
                        weights=list(self.weights.values())
                    )
                else:
                    self.ensemble = VotingRegressor(
                        estimators=model_list,
                        weights=list(self.weights.values())
                    )
                
                self.ensemble.fit(X, y)
            
            elif self.ensemble_method == 'weighted_average':
                # For weighted average, we'll store models and compute predictions manually
                for name, model in self.models.items():
                    if not hasattr(model, 'predict'):
                        raise ValueError(f"Model {name} does not have predict method")
            
            self.fitted = True
            logger.info(f"Ensemble fitted with {len(self.models)} models using {self.ensemble_method}")
            
        except Exception as e:
            logger.error(f"Error fitting ensemble: {e}")
            raise
            
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble"""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        try:
            if self.ensemble_method == 'voting':
                return self.ensemble.predict(X)
            
            elif self.ensemble_method == 'weighted_average':
                predictions = []
                total_weight = sum(self.weights.values())
                
                for name, model in self.models.items():
                    pred = model.predict(X)
                    weight = self.weights[name] / total_weight
                    predictions.append(pred * weight)
                
                return np.sum(predictions, axis=0)
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions using the ensemble"""
        if not self.fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        try:
            if self.ensemble_method == 'voting' and hasattr(self.ensemble, 'predict_proba'):
                return self.ensemble.predict_proba(X)
            
            elif self.ensemble_method == 'weighted_average':
                probabilities = []
                total_weight = sum(self.weights.values())
                
                for name, model in self.models.items():
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X)
                        weight = self.weights[name] / total_weight
                        probabilities.append(prob * weight)
                    else:
                        logger.warning(f"Model {name} does not support probability predictions")
                
                if probabilities:
                    return np.sum(probabilities, axis=0)
                else:
                    raise ValueError("No models support probability predictions")
            
        except Exception as e:
            logger.error(f"Error making ensemble probability predictions: {e}")
            raise
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights"""
        return self.weights.copy()
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update model weights"""
        for name, weight in new_weights.items():
            if name in self.models:
                self.weights[name] = weight
                logger.info(f"Updated weight for model {name} to {weight}")
            else:
                logger.warning(f"Model {name} not found in ensemble")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble"""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            self.fitted = False  # Need to refit after removing a model
            logger.info(f"Removed model {name} from ensemble")
        else:
            logger.warning(f"Model {name} not found in ensemble")
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get individual model performance on test data"""
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        performance = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                
                # Calculate appropriate metric
                if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
                    prob = model.predict_proba(X)[:, 1]
                    performance[name] = roc_auc_score(y, prob)
                else:
                    performance[name] = accuracy_score(y, pred)
                    
            except Exception as e:
                logger.error(f"Error calculating performance for model {name}: {e}")
                performance[name] = 0.0
        
        return performance
    
    def save(self, filepath: str):
        """Save ensemble to file"""
        try:
            ensemble_data = {
                'models': self.models,
                'weights': self.weights,
                'config': self.config,
                'ensemble_method': self.ensemble_method,
                'voting_method': self.voting_method,
                'fitted': self.fitted
            }
            
            if hasattr(self, 'ensemble'):
                ensemble_data['ensemble'] = self.ensemble
            
            joblib.dump(ensemble_data, filepath)
            logger.info(f"Ensemble saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
            raise
    
    def load(self, filepath: str):
        """Load ensemble from file"""
        try:
            ensemble_data = joblib.load(filepath)
            
            self.models = ensemble_data['models']
            self.weights = ensemble_data['weights']
            self.config = ensemble_data['config']
            self.ensemble_method = ensemble_data['ensemble_method']
            self.voting_method = ensemble_data['voting_method']
            self.fitted = ensemble_data['fitted']
            
            if 'ensemble' in ensemble_data:
                self.ensemble = ensemble_data['ensemble']
            
            logger.info(f"Ensemble loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            raise
