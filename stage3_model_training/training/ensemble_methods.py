"""
Ensemble model implementations for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import joblib

logger = logging.getLogger(__name__)

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Stacking ensemble implementation"""
    
    def __init__(self, base_models: List[Any], meta_model: Any, cv_folds: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.fitted_base_models = []
        self.fitted_meta_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit stacking ensemble"""
        from sklearn.model_selection import StratifiedKFold
        
        # Generate meta-features using cross-validation
        meta_features = np.zeros((len(X), len(self.base_models)))
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for i, model in enumerate(self.base_models):
            model_predictions = np.zeros(len(X))
            
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train = y.iloc[train_idx]
                
                # Fit base model on training fold
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)
                
                # Predict on validation fold
                if hasattr(model_copy, 'predict_proba'):
                    val_pred = model_copy.predict_proba(X_val)[:, 1]
                else:
                    val_pred = model_copy.predict(X_val)
                
                model_predictions[val_idx] = val_pred
            
            meta_features[:, i] = model_predictions
        
        # Fit base models on full dataset
        self.fitted_base_models = []
        for model in self.base_models:
            fitted_model = self._clone_model(model)
            fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)
        
        # Fit meta-model
        self.fitted_meta_model = self._clone_model(self.meta_model)
        self.fitted_meta_model.fit(meta_features, y)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using stacking ensemble"""
        if not self.fitted_base_models or not self.fitted_meta_model:
            raise ValueError("Ensemble not fitted yet")
        
        # Get base model predictions
        meta_features = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if hasattr(model, 'predict_proba'):
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)
        
        # Meta-model prediction
        return self.fitted_meta_model.predict_proba(meta_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using stacking ensemble"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def _clone_model(self, model):
        """Clone a model"""
        from sklearn.base import clone
        return clone(model)

class BlendingEnsemble(BaseEstimator, ClassifierMixin):
    """Blending ensemble implementation"""
    
    def __init__(self, base_models: List[Any], blend_ratio: float = 0.2):
        self.base_models = base_models
        self.blend_ratio = blend_ratio
        self.fitted_base_models = []
        self.blend_weights = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit blending ensemble"""
        # Split data for blending
        split_idx = int(len(X) * (1 - self.blend_ratio))
        
        X_train, X_blend = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_blend = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Fit base models on training set
        self.fitted_base_models = []
        blend_predictions = np.zeros((len(X_blend), len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            fitted_model = self._clone_model(model)
            fitted_model.fit(X_train, y_train)
            self.fitted_base_models.append(fitted_model)
            
            # Get predictions on blend set
            if hasattr(fitted_model, 'predict_proba'):
                blend_predictions[:, i] = fitted_model.predict_proba(X_blend)[:, 1]
            else:
                blend_predictions[:, i] = fitted_model.predict(X_blend)
        
        # Optimize blend weights
        self.blend_weights = self._optimize_weights(blend_predictions, y_blend)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using blending ensemble"""
        if not self.fitted_base_models or self.blend_weights is None:
            raise ValueError("Ensemble not fitted yet")
        
        # Get base model predictions
        predictions = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if hasattr(model, 'predict_proba'):
                predictions[:, i] = model.predict_proba(X)[:, 1]
            else:
                predictions[:, i] = model.predict(X)
        
        # Weighted average
        blended_pred = np.average(predictions, weights=self.blend_weights, axis=1)
        
        # Return as probability matrix
        return np.column_stack([1 - blended_pred, blended_pred])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using blending ensemble"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def _optimize_weights(self, predictions: np.ndarray, y_true: pd.Series) -> np.ndarray:
        """Optimize blend weights using grid search"""
        from sklearn.metrics import roc_auc_score
        from itertools import product
        
        best_score = -1
        best_weights = None
        
        # Grid search over weight combinations
        weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for weights in product(weight_options, repeat=len(self.base_models)):
            if sum(weights) == 0:
                continue
            
            # Normalize weights
            normalized_weights = np.array(weights) / sum(weights)
            
            # Calculate blended prediction
            blended_pred = np.average(predictions, weights=normalized_weights, axis=1)
            
            # Calculate score
            score = roc_auc_score(y_true, blended_pred)
            
            if score > best_score:
                best_score = score
                best_weights = normalized_weights
        
        return best_weights if best_weights is not None else np.ones(len(self.base_models)) / len(self.base_models)
    
    def _clone_model(self, model):
        """Clone a model"""
        from sklearn.base import clone
        return clone(model)

class DynamicEnsemble(BaseEstimator, ClassifierMixin):
    """Dynamic ensemble that adapts weights based on local performance"""
    
    def __init__(self, base_models: List[Any], k_neighbors: int = 10):
        self.base_models = base_models
        self.k_neighbors = k_neighbors
        self.fitted_base_models = []
        self.X_train = None
        self.y_train = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit dynamic ensemble"""
        # Store training data for dynamic weighting
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Fit base models
        self.fitted_base_models = []
        for model in self.base_models:
            fitted_model = self._clone_model(model)
            fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using dynamic ensemble"""
        if not self.fitted_base_models:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = []
        
        for idx, row in X.iterrows():
            # Find k nearest neighbors in training set
            distances = self._calculate_distances(row, self.X_train)
            neighbor_indices = np.argsort(distances)[:self.k_neighbors]
            
            # Calculate local model performance
            local_weights = self._calculate_local_weights(neighbor_indices)
            
            # Get model predictions for this instance
            instance_predictions = []
            for model in self.fitted_base_models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(row.to_frame().T)[0, 1]
                else:
                    pred = model.predict(row.to_frame().T)[0]
                instance_predictions.append(pred)
            
            # Weighted prediction
            weighted_pred = np.average(instance_predictions, weights=local_weights)
            predictions.append(weighted_pred)
        
        predictions = np.array(predictions)
        return np.column_stack([1 - predictions, predictions])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using dynamic ensemble"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def _calculate_distances(self, instance: pd.Series, training_data: pd.DataFrame) -> np.ndarray:
        """Calculate distances to training instances"""
        from sklearn.metrics.pairwise import euclidean_distances
        
        instance_array = instance.values.reshape(1, -1)
        training_array = training_data.values
        
        distances = euclidean_distances(instance_array, training_array)[0]
        return distances
    
    def _calculate_local_weights(self, neighbor_indices: np.ndarray) -> np.ndarray:
        """Calculate local model weights based on neighbor performance"""
        from sklearn.metrics import accuracy_score
        
        # Get neighbor data
        X_neighbors = self.X_train.iloc[neighbor_indices]
        y_neighbors = self.y_train.iloc[neighbor_indices]
        
        # Calculate local performance for each model
        local_scores = []
        for model in self.fitted_base_models:
            if hasattr(model, 'predict_proba'):
                neighbor_pred = model.predict_proba(X_neighbors)[:, 1]
                neighbor_pred_binary = (neighbor_pred > 0.5).astype(int)
            else:
                neighbor_pred_binary = model.predict(X_neighbors)
            
            score = accuracy_score(y_neighbors, neighbor_pred_binary)
            local_scores.append(score)
        
        # Convert to weights (higher score = higher weight)
        local_scores = np.array(local_scores)
        if np.sum(local_scores) == 0:
            weights = np.ones(len(local_scores)) / len(local_scores)
        else:
            weights = local_scores / np.sum(local_scores)
        
        return weights
    
    def _clone_model(self, model):
        """Clone a model"""
        from sklearn.base import clone
        return clone(model)

class EnsembleManager:
    """Manager for different ensemble methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ensembles = {}
        
    def create_voting_ensemble(self, models: List[Tuple[str, Any]], 
                             voting: str = 'soft') -> VotingClassifier:
        """Create voting ensemble"""
        ensemble = VotingClassifier(estimators=models, voting=voting)
        self.ensembles['voting'] = ensemble
        return ensemble
    
    def create_bagging_ensemble(self, base_estimator: Any, 
                              n_estimators: int = 10) -> BaggingClassifier:
        """Create bagging ensemble"""
        ensemble = BaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=42
        )
        self.ensembles['bagging'] = ensemble
        return ensemble
    
    def create_boosting_ensemble(self, base_estimator: Any = None,
                               n_estimators: int = 50) -> AdaBoostClassifier:
        """Create AdaBoost ensemble"""
        ensemble = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=42
        )
        self.ensembles['boosting'] = ensemble
        return ensemble
    
    def create_stacking_ensemble(self, base_models: List[Any], 
                               meta_model: Any) -> StackingEnsemble:
        """Create stacking ensemble"""
        ensemble = StackingEnsemble(base_models, meta_model)
        self.ensembles['stacking'] = ensemble
        return ensemble
    
    def create_blending_ensemble(self, base_models: List[Any],
                               blend_ratio: float = 0.2) -> BlendingEnsemble:
        """Create blending ensemble"""
        ensemble = BlendingEnsemble(base_models, blend_ratio)
        self.ensembles['blending'] = ensemble
        return ensemble
    
    def create_dynamic_ensemble(self, base_models: List[Any],
                              k_neighbors: int = 10) -> DynamicEnsemble:
        """Create dynamic ensemble"""
        ensemble = DynamicEnsemble(base_models, k_neighbors)
        self.ensembles['dynamic'] = ensemble
        return ensemble
    
    async def compare_ensembles(self, X: pd.DataFrame, y: pd.Series,
                              cv_folds: int = 5) -> pd.DataFrame:
        """Compare different ensemble methods"""
        from sklearn.model_selection import cross_val_score
        
        results = []
        
        for name, ensemble in self.ensembles.items():
            try:
                scores = cross_val_score(ensemble, X, y, cv=cv_folds, scoring='roc_auc')
                results.append({
                    'Ensemble': name,
                    'Mean_Score': scores.mean(),
                    'Std_Score': scores.std(),
                    'Min_Score': scores.min(),
                    'Max_Score': scores.max()
                })
            except Exception as e:
                logger.warning(f"Failed to evaluate {name} ensemble: {e}")
        
        return pd.DataFrame(results).sort_values('Mean_Score', ascending=False)
    
    def get_ensemble_diversity(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate diversity metrics for ensembles"""
        diversity_metrics = {}
        
        for name, ensemble in self.ensembles.items():
            if hasattr(ensemble, 'fitted_base_models'):
                # For custom ensembles
                base_models = ensemble.fitted_base_models
            elif hasattr(ensemble, 'estimators_'):
                # For sklearn ensembles
                base_models = ensemble.estimators_
            else:
                continue
            
            # Calculate pairwise correlation of predictions
            predictions = []
            for model in base_models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            
            if len(predictions) > 1:
                pred_df = pd.DataFrame(predictions).T
                correlation_matrix = pred_df.corr()
                
                # Average pairwise correlation (excluding diagonal)
                mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
                avg_correlation = correlation_matrix.values[mask].mean()
                
                diversity_metrics[name] = 1 - avg_correlation  # Diversity = 1 - correlation
        
        return diversity_metrics
    
    def save_ensemble(self, name: str, filepath: str):
        """Save ensemble model"""
        if name not in self.ensembles:
            raise ValueError(f"Ensemble {name} not found")
        
        joblib.dump(self.ensembles[name], filepath)
        logger.info(f"Ensemble {name} saved to {filepath}")
    
    def load_ensemble(self, name: str, filepath: str):
        """Load ensemble model"""
        ensemble = joblib.load(filepath)
        self.ensembles[name] = ensemble
        logger.info(f"Ensemble {name} loaded from {filepath}")
        return ensemble
