"""
Ensemble model implementations for credit risk scoring.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, VotingClassifier, ExtraTreesClassifier
)
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import joblib

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model wrapper for credit risk assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        # Model type selection
        self.model_type = config.get('model_type', 'random_forest')
        
        # Default parameters
        self.default_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'random_state': 42
            },
            'ada_boost': {
                'n_estimators': 50,
                'learning_rate': 1.0,
                'random_state': 42
            },
            'extra_trees': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            },
            'bagging': {
                'n_estimators': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Update with config parameters
        self.params = {**self.default_params.get(self.model_type, {}), 
                      **config.get('model_params', {})}
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**self.params)
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**self.params)
        elif self.model_type == 'ada_boost':
            return AdaBoostClassifier(**self.params)
        elif self.model_type == 'extra_trees':
            return ExtraTreesClassifier(**self.params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**self.params)
        elif self.model_type == 'bagging':
            return BaggingClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'EnsembleModel':
        """Train ensemble model"""
        
        self.feature_names = X_train.columns.tolist()
        
        # Create model
        self.model = self._create_model()
        
        # Special handling for LightGBM with validation
        if self.model_type == 'lightgbm' and X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        logger.info(f"{self.model_type} ensemble model training completed")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        return self.model.predict(X)
    
    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict credit scores (0-1000 scale)"""
        proba = self.predict_proba(X)[:, 1]
        
        # Convert to credit score scale (higher score = lower risk)
        credit_scores = 1000 - (proba * 1000)
        return credit_scores.astype(int)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            # For models without direct feature importance
            return {}
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    async def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20],
                'max_features': ['sqrt', 'log2']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ada_boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            },
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [15, 31, 63],
                'subsample': [0.8, 0.9, 1.0]
            },
            'bagging': {
                'n_estimators': [10, 20, 50],
                'max_samples': [0.5, 0.7, 1.0],
                'max_features': [0.5, 0.7, 1.0]
            }
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid defined for {self.model_type}")
            return {}
        
        # Create base model
        base_model = self._create_model()
        
        # Grid search
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv_folds,
            scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.params.update(grid_search.best_params_)
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        logger.info(f"Hyperparameter tuning completed - Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Predictions
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = self.predict(X_test)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_importance = self.get_feature_importance()
        top_features = dict(list(feature_importance.items())[:10])
        
        return {
            'auc_score': auc_score,
            'classification_report': class_report,
            'top_features': top_features,
            'n_samples': len(X_test),
            'positive_rate': y_test.mean()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        info = {
            'model_type': f'Ensemble ({self.model_type})',
            'parameters': self.params,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'estimators_'):
            info['n_fitted_estimators'] = len(self.model.estimators_)
        
        return info
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'params': self.params,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Ensemble model loaded from {filepath}")
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual estimators"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if not hasattr(self.model, 'estimators_'):
            return {}
        
        predictions = {}
        for i, estimator in enumerate(self.model.estimators_):
            if hasattr(estimator, 'predict_proba'):
                pred = estimator.predict_proba(X)[:, 1]
            else:
                pred = estimator.predict(X)
            predictions[f'estimator_{i}'] = pred
        
        return predictions
    
    def get_ensemble_diversity(self, X: pd.DataFrame) -> Dict[str, float]:
        """Calculate ensemble diversity metrics"""
        individual_preds = self.get_individual_predictions(X)
        
        if len(individual_preds) < 2:
            return {}
        
        # Convert to DataFrame for correlation calculation
        pred_df = pd.DataFrame(individual_preds)
        
        # Calculate pairwise correlations
        correlation_matrix = pred_df.corr()
        
        # Remove diagonal (self-correlations)
        mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        correlations = correlation_matrix.values[mask]
        
        return {
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'min_correlation': float(np.min(correlations)),
            'max_correlation': float(np.max(correlations)),
            'diversity_score': float(1 - np.mean(correlations))  # Higher = more diverse
        }

class VotingEnsemble:
    """Custom voting ensemble for multiple models"""
    
    def __init__(self, models: List[Tuple[str, Any]], voting: str = 'soft'):
        self.models = models
        self.voting = voting
        self.ensemble = None
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'VotingEnsemble':
        """Fit voting ensemble"""
        
        # Fit individual models first
        fitted_models = []
        for name, model in self.models:
            model.fit(X_train, y_train)
            fitted_models.append((name, model))
        
        # Create voting classifier
        self.ensemble = VotingClassifier(
            estimators=fitted_models,
            voting=self.voting
        )
        
        # Fit ensemble (this just sets up the voting mechanism)
        self.ensemble.fit(X_train, y_train)
        
        self.is_fitted = True
        logger.info(f"Voting ensemble with {len(self.models)} models fitted")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        return self.ensemble.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        return self.ensemble.predict(X)
    
    def get_individual_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        predictions = {}
        for name, model in self.models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions[name] = pred
        
        return predictions

class StackedEnsemble:
    """Custom stacked ensemble implementation"""
    
    def __init__(self, base_models: List[Tuple[str, Any]], meta_model: Any, cv_folds: int = 5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.fitted_base_models = []
        self.fitted_meta_model = None
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'StackedEnsemble':
        """Fit stacked ensemble"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        
        # Generate meta-features using cross-validation
        meta_features = np.zeros((len(X_train), len(self.base_models)))
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models):
            model_predictions = np.zeros(len(X_train))
            
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_fold_train = X_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                
                # Clone and fit model on fold
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict on validation fold
                if hasattr(fold_model, 'predict_proba'):
                    val_pred = fold_model.predict_proba(X_fold_val)[:, 1]
                else:
                    val_pred = fold_model.predict(X_fold_val)
                
                model_predictions[val_idx] = val_pred
            
            meta_features[:, i] = model_predictions
        
        # Fit base models on full training set
        self.fitted_base_models = []
        for name, model in self.base_models:
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)
            self.fitted_base_models.append((name, fitted_model))
        
        # Fit meta-model
        self.fitted_meta_model = clone(self.meta_model)
        self.fitted_meta_model.fit(meta_features, y_train)
        
        self.is_fitted = True
        logger.info(f"Stacked ensemble with {len(self.base_models)} base models fitted")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet")
        
        # Get base model predictions
        meta_features = np.zeros((len(X), len(self.fitted_base_models)))
        
        for i, (name, model) in enumerate(self.fitted_base_models):
            if hasattr(model, 'predict_proba'):
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            else:
                meta_features[:, i] = model.predict(X)
        
        # Meta-model prediction
        return self.fitted_meta_model.predict_proba(meta_features)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
