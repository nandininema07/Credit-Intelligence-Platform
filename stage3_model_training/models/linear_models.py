"""
Linear model implementations for credit risk scoring.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet,
    SGDClassifier, PassiveAggressiveClassifier
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib

logger = logging.getLogger(__name__)

class LinearModel:
    """Linear model wrapper for credit risk assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.poly_features = None
        self.feature_names = None
        self.is_fitted = False
        
        # Model type selection
        self.model_type = config.get('model_type', 'logistic')
        self.use_polynomial = config.get('use_polynomial', False)
        self.poly_degree = config.get('polynomial_degree', 2)
        
        # Default parameters for different models
        self.default_params = {
            'logistic': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'random_state': 42,
                'max_iter': 1000
            },
            'ridge': {
                'alpha': 1.0,
                'random_state': 42
            },
            'lasso': {
                'alpha': 1.0,
                'random_state': 42,
                'max_iter': 1000
            },
            'elastic_net': {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'random_state': 42,
                'max_iter': 1000
            },
            'sgd': {
                'loss': 'log_loss',
                'penalty': 'l2',
                'alpha': 0.0001,
                'random_state': 42,
                'max_iter': 1000
            }
        }
        
        # Update with config parameters
        self.params = {**self.default_params.get(self.model_type, {}), 
                      **config.get('model_params', {})}
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == 'logistic':
            return LogisticRegression(**self.params)
        elif self.model_type == 'ridge':
            return Ridge(**self.params)
        elif self.model_type == 'lasso':
            return Lasso(**self.params)
        elif self.model_type == 'elastic_net':
            return ElasticNet(**self.params)
        elif self.model_type == 'sgd':
            return SGDClassifier(**self.params)
        elif self.model_type == 'passive_aggressive':
            return PassiveAggressiveClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LinearModel':
        """Train linear model"""
        
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Add polynomial features if requested
        if self.use_polynomial:
            self.poly_features = PolynomialFeatures(
                degree=self.poly_degree, 
                include_bias=False,
                interaction_only=False
            )
            X_train_scaled = self.poly_features.fit_transform(X_train_scaled)
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        logger.info(f"{self.model_type} model training completed")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self._transform_features(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For regression models, convert to probabilities
            predictions = self.model.predict(X_scaled)
            # Apply sigmoid to convert to probabilities
            proba_pos = 1 / (1 + np.exp(-predictions))
            proba_neg = 1 - proba_pos
            return np.column_stack([proba_neg, proba_pos])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if hasattr(self.model, 'predict'):
            X_scaled = self._transform_features(X)
            return self.model.predict(X_scaled)
        else:
            proba = self.predict_proba(X)
            return (proba[:, 1] > 0.5).astype(int)
    
    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict credit scores (0-1000 scale)"""
        proba = self.predict_proba(X)[:, 1]
        
        # Convert to credit score scale (higher score = lower risk)
        credit_scores = 1000 - (proba * 1000)
        return credit_scores.astype(int)
    
    def _transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features with scaling and polynomial features"""
        X_scaled = self.scaler.transform(X)
        
        if self.use_polynomial and self.poly_features is not None:
            X_scaled = self.poly_features.transform(X_scaled)
        
        return X_scaled
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (coefficients)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]  # For binary classification
            
            # Handle polynomial features
            if self.use_polynomial and self.poly_features is not None:
                feature_names = self.poly_features.get_feature_names_out(self.feature_names)
            else:
                feature_names = self.feature_names
            
            # Create importance dictionary
            importance = dict(zip(feature_names, np.abs(coefficients)))
            
            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
            
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def get_coefficients(self) -> Dict[str, float]:
        """Get raw model coefficients"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_
            if coefficients.ndim > 1:
                coefficients = coefficients[0]
            
            # Handle polynomial features
            if self.use_polynomial and self.poly_features is not None:
                feature_names = self.poly_features.get_feature_names_out(self.feature_names)
            else:
                feature_names = self.feature_names
            
            return dict(zip(feature_names, coefficients))
        
        return {}
    
    async def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.use_polynomial:
            self.poly_features = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_train_scaled = self.poly_features.fit_transform(X_train_scaled)
        
        # Define parameter grids for different models
        param_grids = {
            'logistic': {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'ridge': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'sgd': {
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'optimal', 'adaptive'],
                'eta0': [0.01, 0.1, 1.0]
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
        
        grid_search.fit(X_train_scaled, y_train)
        
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
            'coefficients': self.get_coefficients(),
            'n_samples': len(X_test),
            'positive_rate': y_test.mean()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        info = {
            'model_type': f'Linear ({self.model_type})',
            'parameters': self.params,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'use_polynomial': self.use_polynomial
        }
        
        if self.use_polynomial:
            info['polynomial_degree'] = self.poly_degree
            if self.poly_features is not None:
                info['n_polynomial_features'] = self.poly_features.n_output_features_
        
        return info
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'poly_features': self.poly_features,
            'config': self.config,
            'params': self.params,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'use_polynomial': self.use_polynomial,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Linear model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.poly_features = model_data['poly_features']
        self.config = model_data['config']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.use_polynomial = model_data['use_polynomial']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Linear model loaded from {filepath}")

class RegularizedLinearModel(LinearModel):
    """Regularized linear model with automatic regularization selection"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.regularization_path = None
        
    def fit_with_regularization_path(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   alphas: List[float] = None) -> 'RegularizedLinearModel':
        """Fit model with regularization path analysis"""
        
        if alphas is None:
            alphas = np.logspace(-4, 2, 50)
        
        self.feature_names = X_train.columns.tolist()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if self.use_polynomial:
            self.poly_features = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_train_scaled = self.poly_features.fit_transform(X_train_scaled)
        
        # Calculate regularization path
        if self.model_type == 'lasso':
            from sklearn.linear_model import lasso_path
            alphas, coefs, _ = lasso_path(X_train_scaled, y_train, alphas=alphas)
        elif self.model_type == 'elastic_net':
            from sklearn.linear_model import enet_path
            alphas, coefs, _ = enet_path(X_train_scaled, y_train, alphas=alphas, 
                                       l1_ratio=self.params.get('l1_ratio', 0.5))
        else:
            # For other models, use cross-validation
            from sklearn.model_selection import validation_curve
            train_scores, val_scores = validation_curve(
                self._create_model(), X_train_scaled, y_train,
                param_name='alpha' if 'alpha' in self.params else 'C',
                param_range=alphas, cv=5, scoring='roc_auc'
            )
            coefs = None
        
        self.regularization_path = {
            'alphas': alphas,
            'coefficients': coefs,
            'train_scores': train_scores if 'train_scores' in locals() else None,
            'val_scores': val_scores if 'val_scores' in locals() else None
        }
        
        # Fit final model with best alpha
        best_alpha = self._select_best_alpha(X_train_scaled, y_train, alphas)
        self.params['alpha'] = best_alpha
        
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        logger.info(f"Regularized model fitted with alpha={best_alpha:.6f}")
        return self
    
    def _select_best_alpha(self, X_scaled: np.ndarray, y: pd.Series, 
                          alphas: np.ndarray) -> float:
        """Select best regularization parameter using cross-validation"""
        from sklearn.model_selection import cross_val_score
        
        best_score = -np.inf
        best_alpha = alphas[0]
        
        for alpha in alphas:
            temp_params = self.params.copy()
            temp_params['alpha'] = alpha
            
            if self.model_type == 'logistic':
                temp_params['C'] = 1.0 / alpha  # C is inverse of alpha for LogisticRegression
                temp_model = LogisticRegression(**temp_params)
            else:
                temp_model = self._create_model()
                temp_model.set_params(alpha=alpha)
            
            scores = cross_val_score(temp_model, X_scaled, y, cv=5, scoring='roc_auc')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_alpha = alpha
        
        return best_alpha
    
    def get_regularization_path(self) -> Dict[str, Any]:
        """Get regularization path results"""
        return self.regularization_path or {}

class MultiTaskLinearModel:
    """Multi-task linear model for multiple credit risk targets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> 'MultiTaskLinearModel':
        """Fit multi-task model"""
        
        self.feature_names = X_train.columns.tolist()
        self.target_names = y_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit separate model for each target
        for target in self.target_names:
            model_config = self.config.copy()
            model = LinearModel(model_config)
            
            # Fit on scaled data
            X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
            model.scaler = StandardScaler()  # Already scaled
            model.scaler.mean_ = np.zeros(X_train_scaled.shape[1])
            model.scaler.scale_ = np.ones(X_train_scaled.shape[1])
            
            model.model = model._create_model()
            model.model.fit(X_train_scaled, y_train[target])
            model.feature_names = self.feature_names
            model.is_fitted = True
            
            self.models[target] = model
        
        self.is_fitted = True
        logger.info(f"Multi-task model fitted for {len(self.target_names)} targets")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict probabilities for all targets"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict_proba(X_scaled_df)
        
        return predictions
    
    def predict(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict classes for all targets"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        predictions = {}
        for target, model in self.models.items():
            predictions[target] = model.predict(X_scaled_df)
        
        return predictions
    
    def get_feature_importance(self, target: str = None) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all or specific target"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if target:
            if target not in self.models:
                raise ValueError(f"Target {target} not found")
            return {target: self.models[target].get_feature_importance()}
        
        importance = {}
        for target, model in self.models.items():
            importance[target] = model.get_feature_importance()
        
        return importance
    
    def save_model(self, filepath: str):
        """Save multi-task model"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Multi-task model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load multi-task model"""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Multi-task model loaded from {filepath}")
