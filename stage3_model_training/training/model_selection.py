"""
Model selection and hyperparameter tuning for Stage 3.
Handles automated model selection and optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import optuna
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration for selection"""
    name: str
    model_class: Any
    param_space: Dict[str, Any]
    search_method: str = 'grid'  # 'grid', 'random', 'bayesian'
    cv_folds: int = 5
    scoring: str = 'roc_auc'

class ModelSelection:
    """Automated model selection and hyperparameter tuning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_configs = []
        self.results = {}
        self.best_model = None
        self.best_score = -np.inf
        
    def add_model_config(self, model_config: ModelConfig):
        """Add model configuration for selection"""
        self.model_configs.append(model_config)
        logger.info(f"Added model config: {model_config.name}")
    
    async def select_best_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Select best model through hyperparameter optimization"""
        
        for model_config in self.model_configs:
            logger.info(f"Optimizing {model_config.name}...")
            
            try:
                if model_config.search_method == 'bayesian':
                    result = await self._bayesian_optimization(
                        model_config, X_train, y_train, X_val, y_val
                    )
                elif model_config.search_method == 'random':
                    result = await self._random_search(
                        model_config, X_train, y_train, X_val, y_val
                    )
                else:  # grid search
                    result = await self._grid_search(
                        model_config, X_train, y_train, X_val, y_val
                    )
                
                self.results[model_config.name] = result
                
                if result['score'] > self.best_score:
                    self.best_score = result['score']
                    self.best_model = result
                    
                logger.info(f"Completed {model_config.name} - Score: {result['score']:.4f}")
                
            except Exception as e:
                logger.error(f"Error optimizing {model_config.name}: {e}")
        
        return self.best_model
    
    async def _grid_search(self, model_config: ModelConfig, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame = None, 
                          y_val: pd.Series = None) -> Dict[str, Any]:
        """Perform grid search optimization"""
        
        model = model_config.model_class()
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=model_config.param_space,
            cv=model_config.cv_folds,
            scoring=model_config.scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        val_score = None
        if X_val is not None and y_val is not None:
            val_predictions = grid_search.best_estimator_.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_predictions)
        
        return {
            'model_name': model_config.name,
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'score': grid_search.best_score_,
            'val_score': val_score,
            'cv_results': grid_search.cv_results_,
            'search_method': 'grid'
        }
    
    async def _random_search(self, model_config: ModelConfig, X_train: pd.DataFrame,
                           y_train: pd.Series, X_val: pd.DataFrame = None,
                           y_val: pd.Series = None) -> Dict[str, Any]:
        """Perform random search optimization"""
        
        model = model_config.model_class()
        n_iter = self.config.get('random_search_iterations', 100)
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=model_config.param_space,
            n_iter=n_iter,
            cv=model_config.cv_folds,
            scoring=model_config.scoring,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        val_score = None
        if X_val is not None and y_val is not None:
            val_predictions = random_search.best_estimator_.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_predictions)
        
        return {
            'model_name': model_config.name,
            'best_estimator': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'score': random_search.best_score_,
            'val_score': val_score,
            'cv_results': random_search.cv_results_,
            'search_method': 'random'
        }
    
    async def _bayesian_optimization(self, model_config: ModelConfig, X_train: pd.DataFrame,
                                   y_train: pd.Series, X_val: pd.DataFrame = None,
                                   y_val: pd.Series = None) -> Dict[str, Any]:
        """Perform Bayesian optimization using Optuna"""
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in model_config.param_space.items():
                if isinstance(param_config, dict):
                    if param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name, param_config['choices']
                        )
                else:
                    # Handle simple list format
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
            
            # Create and train model
            model = model_config.model_class(**params)
            
            # Cross-validation score
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=model_config.cv_folds, 
                scoring=model_config.scoring
            )
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize')
        n_trials = self.config.get('bayesian_trials', 100)
        study.optimize(objective, n_trials=n_trials)
        
        # Train best model
        best_model = model_config.model_class(**study.best_params)
        best_model.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        val_score = None
        if X_val is not None and y_val is not None:
            val_predictions = best_model.predict_proba(X_val)[:, 1]
            val_score = roc_auc_score(y_val, val_predictions)
        
        return {
            'model_name': model_config.name,
            'best_estimator': best_model,
            'best_params': study.best_params,
            'score': study.best_value,
            'val_score': val_score,
            'study': study,
            'search_method': 'bayesian'
        }
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of model selection results"""
        if not self.results:
            return {}
        
        summary = {
            'total_models': len(self.results),
            'best_model': self.best_model['model_name'] if self.best_model else None,
            'best_score': self.best_score,
            'model_scores': {}
        }
        
        for model_name, result in self.results.items():
            summary['model_scores'][model_name] = {
                'cv_score': result['score'],
                'val_score': result.get('val_score'),
                'search_method': result['search_method']
            }
        
        return summary
    
    def compare_models(self) -> pd.DataFrame:
        """Compare model performance"""
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'CV_Score': result['score'],
                'Val_Score': result.get('val_score', np.nan),
                'Search_Method': result['search_method'],
                'Best_Params': str(result['best_params'])
            })
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('CV_Score', ascending=False)
    
    def get_feature_importance(self, model_name: str = None) -> Dict[str, float]:
        """Get feature importance from best model or specified model"""
        if model_name:
            if model_name not in self.results:
                return {}
            model = self.results[model_name]['best_estimator']
        else:
            if not self.best_model:
                return {}
            model = self.best_model['best_estimator']
        
        # Extract feature importance
        if hasattr(model, 'feature_importances_'):
            return dict(enumerate(model.feature_importances_))
        elif hasattr(model, 'coef_'):
            return dict(enumerate(np.abs(model.coef_[0])))
        else:
            return {}
    
    def save_results(self, filepath: str):
        """Save model selection results"""
        import joblib
        
        save_data = {
            'results': self.results,
            'best_model': self.best_model,
            'best_score': self.best_score,
            'config': self.config
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Model selection results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load model selection results"""
        import joblib
        
        save_data = joblib.load(filepath)
        self.results = save_data['results']
        self.best_model = save_data['best_model']
        self.best_score = save_data['best_score']
        
        logger.info(f"Model selection results loaded from {filepath}")

class AutoMLSelector:
    """Automated machine learning model selection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selector = ModelSelector(config)
        self._setup_default_models()
    
    def _setup_default_models(self):
        """Setup default model configurations"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        import xgboost as xgb
        import lightgbm as lgb
        
        # Random Forest
        rf_config = ModelConfig(
            name='RandomForest',
            model_class=RandomForestClassifier,
            param_space={
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            search_method='random'
        )
        self.selector.add_model_config(rf_config)
        
        # XGBoost
        xgb_config = ModelConfig(
            name='XGBoost',
            model_class=xgb.XGBClassifier,
            param_space={
                'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
            },
            search_method='bayesian'
        )
        self.selector.add_model_config(xgb_config)
        
        # LightGBM
        lgb_config = ModelConfig(
            name='LightGBM',
            model_class=lgb.LGBMClassifier,
            param_space={
                'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'num_leaves': {'type': 'int', 'low': 10, 'high': 100}
            },
            search_method='bayesian'
        )
        self.selector.add_model_config(lgb_config)
        
        # Logistic Regression
        lr_config = ModelConfig(
            name='LogisticRegression',
            model_class=LogisticRegression,
            param_space={
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            search_method='grid'
        )
        self.selector.add_model_config(lr_config)
    
    async def auto_select(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Automatically select best model"""
        return await self.selector.select_best_model(X_train, y_train, X_val, y_val)
    
    def get_results(self) -> Dict[str, Any]:
        """Get selection results"""
        return self.selector.get_selection_summary()
