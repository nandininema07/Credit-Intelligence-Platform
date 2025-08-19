"""
Tree-based model implementations for credit risk scoring.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
import joblib

logger = logging.getLogger(__name__)

class TreeModel:
    """Tree-based model wrapper for credit risk assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        # Model type selection
        self.model_type = config.get('model_type', 'decision_tree')
        
        # Default parameters
        self.default_params = {
            'decision_tree': {
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42,
                'n_jobs': -1
            },
            'extra_trees': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Update with config parameters
        self.params = {**self.default_params.get(self.model_type, {}), 
                      **config.get('model_params', {})}
    
    def _create_model(self):
        """Create model based on type"""
        if self.model_type == 'decision_tree':
            return DecisionTreeClassifier(**self.params)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**self.params)
        elif self.model_type == 'extra_trees':
            return ExtraTreesClassifier(**self.params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'TreeModel':
        """Train tree model"""
        
        self.feature_names = X_train.columns.tolist()
        
        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        logger.info(f"{self.model_type} model training completed")
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
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_tree_structure(self, tree_index: int = 0) -> Dict[str, Any]:
        """Get tree structure for interpretation"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if self.model_type == 'decision_tree':
            tree = self.model.tree_
        elif hasattr(self.model, 'estimators_'):
            if tree_index >= len(self.model.estimators_):
                raise ValueError(f"Tree index {tree_index} out of range")
            tree = self.model.estimators_[tree_index].tree_
        else:
            return {}
        
        def recurse(node_id):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                return {
                    'type': 'leaf',
                    'value': tree.value[node_id].tolist(),
                    'samples': int(tree.n_node_samples[node_id])
                }
            else:
                # Internal node
                return {
                    'type': 'split',
                    'feature': self.feature_names[tree.feature[node_id]],
                    'threshold': float(tree.threshold[node_id]),
                    'samples': int(tree.n_node_samples[node_id]),
                    'left': recurse(tree.children_left[node_id]),
                    'right': recurse(tree.children_right[node_id])
                }
        
        return recurse(0)
    
    async def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  cv_folds: int = 5) -> Dict[str, Any]:
        """Perform hyperparameter tuning"""
        
        # Define parameter grids
        param_grids = {
            'decision_tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20],
                'max_features': ['sqrt', 'log2', None]
            },
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20],
                'max_features': ['sqrt', 'log2', None]
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
            'model_type': f'Tree ({self.model_type})',
            'parameters': self.params,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        
        if hasattr(self.model, 'tree_'):
            info['tree_depth'] = self.model.tree_.max_depth
            info['n_nodes'] = self.model.tree_.node_count
            info['n_leaves'] = self.model.tree_.n_leaves
        
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
        logger.info(f"Tree model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Tree model loaded from {filepath}")
    
    def visualize_tree(self, tree_index: int = 0, max_depth: int = 3) -> str:
        """Generate tree visualization"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        try:
            from sklearn.tree import export_text
            
            if self.model_type == 'decision_tree':
                tree_rules = export_text(
                    self.model, 
                    feature_names=self.feature_names,
                    max_depth=max_depth
                )
            elif hasattr(self.model, 'estimators_'):
                if tree_index >= len(self.model.estimators_):
                    raise ValueError(f"Tree index {tree_index} out of range")
                tree_rules = export_text(
                    self.model.estimators_[tree_index],
                    feature_names=self.feature_names,
                    max_depth=max_depth
                )
            else:
                return "Tree visualization not available for this model type"
            
            return tree_rules
            
        except Exception as e:
            logger.error(f"Error generating tree visualization: {e}")
            return f"Error: {str(e)}"
