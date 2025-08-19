"""
XGBoost model implementation for credit risk scoring.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import shap

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost classifier for credit risk modeling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        self.feature_importance = None
        self.shap_explainer = None
        
        # Default parameters
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with config parameters
        self.params = {**self.default_params, **config.get('xgboost_params', {})}
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            early_stopping_rounds: int = 50) -> 'XGBoostModel':
        """Train XGBoost model"""
        
        self.feature_names = X_train.columns.tolist()
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare validation data for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Fit model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        self.is_fitted = True
        
        # Calculate feature importance
        self.feature_importance = dict(zip(
            self.feature_names, 
            self.model.feature_importances_
        ))
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer(X_train.sample(min(1000, len(X_train))))
        
        logger.info("XGBoost model training completed")
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
    
    async def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  method: str = 'random', cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300, 500],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        
        if method == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=cv_folds, 
                scoring='roc_auc', n_jobs=-1, verbose=1
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv_folds, 
                scoring='roc_auc', n_jobs=-1, verbose=1,
                n_iter=50, random_state=42
            )
        
        search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.params.update(search.best_params_)
        self.model = search.best_estimator_
        self.is_fitted = True
        
        # Update feature importance
        self.feature_importance = dict(zip(
            X_train.columns.tolist(), 
            self.model.feature_importances_
        ))
        
        logger.info(f"Hyperparameter tuning completed - Best score: {search.best_score_:.4f}")
        
        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'cv_results': search.cv_results_
        }
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if importance_type == 'gain':
            importance = self.model.get_booster().get_score(importance_type='gain')
        elif importance_type == 'weight':
            importance = self.model.get_booster().get_score(importance_type='weight')
        elif importance_type == 'cover':
            importance = self.model.get_booster().get_score(importance_type='cover')
        else:
            importance = self.feature_importance
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def _initialize_shap_explainer(self, X_sample: pd.DataFrame):
        """Initialize SHAP explainer"""
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
    
    def explain_prediction(self, X: pd.DataFrame, max_samples: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanations for predictions"""
        if self.shap_explainer is None:
            return {"error": "SHAP explainer not available"}
        
        # Limit samples for performance
        X_explain = X.head(max_samples)
        
        try:
            shap_values = self.shap_explainer.shap_values(X_explain)
            
            # For binary classification, take positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate feature contributions
            feature_contributions = {}
            for i, feature in enumerate(self.feature_names):
                feature_contributions[feature] = {
                    'mean_impact': float(np.mean(np.abs(shap_values[:, i]))),
                    'values': shap_values[:, i].tolist()
                }
            
            return {
                'shap_values': shap_values.tolist(),
                'feature_contributions': feature_contributions,
                'base_value': float(self.shap_explainer.expected_value),
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {e}")
            return {"error": str(e)}
    
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
        top_features = dict(list(self.get_feature_importance().items())[:10])
        
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
        
        return {
            'model_type': 'XGBoost',
            'parameters': self.params,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'n_estimators': self.model.n_estimators,
            'best_iteration': getattr(self.model, 'best_iteration', None)
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"XGBoost model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.is_fitted = model_data['is_fitted']
        
        # Reinitialize SHAP explainer
        if self.is_fitted:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
            except:
                pass
        
        logger.info(f"XGBoost model loaded from {filepath}")
    
    def get_learning_curve(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, List[float]]:
        """Get learning curve data"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Get evaluation results
        evals_result = self.model.evals_result()
        
        learning_curve = {}
        
        if 'validation_0' in evals_result:
            learning_curve['train_auc'] = evals_result['validation_0']['auc']
        
        if 'validation_1' in evals_result:
            learning_curve['val_auc'] = evals_result['validation_1']['auc']
        
        return learning_curve
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_estimators_list: List[int] = None) -> Dict[str, np.ndarray]:
        """Predict with uncertainty estimation using different n_estimators"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if n_estimators_list is None:
            n_estimators_list = [10, 25, 50, 75, 100]
        
        predictions = []
        
        for n_est in n_estimators_list:
            if n_est <= self.model.n_estimators:
                # Create temporary model with fewer estimators
                temp_model = xgb.XGBClassifier(**self.params)
                temp_model.n_estimators = n_est
                temp_model._Booster = self.model.get_booster()
                
                pred = temp_model.predict_proba(X)[:, 1]
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            'mean_prediction': np.mean(predictions, axis=0),
            'std_prediction': np.std(predictions, axis=0),
            'predictions': predictions
        }
