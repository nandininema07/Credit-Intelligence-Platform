"""
Model training orchestrator for credit risk models.
Handles training pipeline, hyperparameter tuning, and model validation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    cv_folds: int = 5
    scoring_metric: str = 'roc_auc'

@dataclass
class TrainingResult:
    """Training result"""
    model_name: str
    model: Any
    train_score: float
    val_score: float
    test_score: float
    feature_importance: Dict[str, float]
    training_time: float
    hyperparameters: Dict[str, Any]

class ModelTrainer:
    """Credit risk model trainer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.training_results = {}
        self.feature_columns = []
        self.target_column = config.get('target_column', 'default_risk')
        
        # Setup model configurations
        self.model_configs = self._setup_model_configs()
        
    def _setup_model_configs(self) -> List[ModelConfig]:
        """Setup default model configurations"""
        default_configs = [
            ModelConfig(
                name='logistic_regression',
                model_type='sklearn',
                hyperparameters={
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'max_iter': [1000]
                }
            ),
            ModelConfig(
                name='random_forest',
                model_type='sklearn',
                hyperparameters={
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            ),
            ModelConfig(
                name='gradient_boosting',
                model_type='sklearn',
                hyperparameters={
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            ),
            ModelConfig(
                name='xgboost',
                model_type='xgboost',
                hyperparameters={
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            ),
            ModelConfig(
                name='lightgbm',
                model_type='lightgbm',
                hyperparameters={
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5]
                }
            )
        ]
        
        return self.config.get('model_configs', default_configs)
    
    async def train_models(self, training_data: pd.DataFrame,
                          feature_columns: List[str] = None,
                          test_size: float = 0.2,
                          val_size: float = 0.2) -> Dict[str, TrainingResult]:
        """Train all configured models"""
        if training_data.empty:
            raise ValueError("Training data is empty")
        
        # Setup features and target
        if feature_columns is None:
            feature_columns = [col for col in training_data.columns if col != self.target_column]
        
        self.feature_columns = feature_columns
        
        # Prepare data
        X = training_data[feature_columns].fillna(0)
        y = training_data[self.target_column]
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Train each model
        training_results = {}
        
        for model_config in self.model_configs:
            try:
                logger.info(f"Training {model_config.name}...")
                start_time = datetime.now()
                
                result = await self._train_single_model(
                    model_config, X_train, y_train, X_val, y_val, X_test, y_test
                )
                
                training_time = (datetime.now() - start_time).total_seconds()
                result.training_time = training_time
                
                training_results[model_config.name] = result
                self.training_results[model_config.name] = result
                
                logger.info(f"Completed {model_config.name} - Val AUC: {result.val_score:.4f}, Test AUC: {result.test_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_config.name}: {e}")
        
        return training_results
    
    async def _train_single_model(self, model_config: ModelConfig,
                                X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series) -> TrainingResult:
        """Train a single model with hyperparameter tuning"""
        
        # Create base model
        base_model = self._create_base_model(model_config)
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            base_model,
            model_config.hyperparameters,
            cv=model_config.cv_folds,
            scoring=model_config.scoring_metric,
            n_jobs=-1,
            verbose=0
        )
        
        # Fit model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Evaluate
        train_score = roc_auc_score(y_train, best_model.predict_proba(X_train)[:, 1])
        val_score = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
        test_score = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        
        # Feature importance
        feature_importance = self._get_feature_importance(best_model, X_train.columns)
        
        # Store model
        self.models[model_config.name] = best_model
        
        return TrainingResult(
            model_name=model_config.name,
            model=best_model,
            train_score=train_score,
            val_score=val_score,
            test_score=test_score,
            feature_importance=feature_importance,
            training_time=0.0,  # Will be set by caller
            hyperparameters=grid_search.best_params_
        )
    
    def _create_base_model(self, model_config: ModelConfig):
        """Create base model instance"""
        if model_config.model_type == 'sklearn':
            if model_config.name == 'logistic_regression':
                return LogisticRegression(random_state=42)
            elif model_config.name == 'random_forest':
                return RandomForestClassifier(random_state=42)
            elif model_config.name == 'gradient_boosting':
                return GradientBoostingClassifier(random_state=42)
        
        elif model_config.model_type == 'xgboost':
            return xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        elif model_config.model_type == 'lightgbm':
            return lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        else:
            raise ValueError(f"Unknown model type: {model_config.model_type}")
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                return {}
            
            return dict(zip(feature_names, importances.tolist()))
        
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    async def evaluate_model(self, model_name: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate a trained model on test data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        X_test = test_data[self.feature_columns].fillna(0)
        y_test = test_data[self.target_column]
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'model_name': model_name,
            'auc_score': auc_score,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred_proba.tolist(),
            'actual': y_test.tolist()
        }
    
    def get_best_model(self, metric: str = 'test_score') -> Tuple[str, TrainingResult]:
        """Get best performing model"""
        if not self.training_results:
            raise ValueError("No models have been trained")
        
        best_model_name = max(
            self.training_results.keys(),
            key=lambda name: getattr(self.training_results[name], metric)
        )
        
        return best_model_name, self.training_results[best_model_name]
    
    async def predict(self, model_name: str, features: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        X = features[self.feature_columns].fillna(0)
        
        return model.predict_proba(X)[:, 1]
    
    def save_models(self, directory: str):
        """Save all trained models"""
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(directory, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'training_results': {
                name: {
                    'train_score': result.train_score,
                    'val_score': result.val_score,
                    'test_score': result.test_score,
                    'training_time': result.training_time,
                    'hyperparameters': result.hyperparameters
                }
                for name, result in self.training_results.items()
            }
        }
        
        metadata_path = os.path.join(directory, 'model_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load trained models"""
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Model directory not found: {directory}")
        
        # Load metadata
        metadata_path = os.path.join(directory, 'model_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.target_column = metadata['target_column']
        
        # Load models
        for model_file in os.listdir(directory):
            if model_file.endswith('.joblib'):
                model_name = model_file.replace('.joblib', '')
                model_path = os.path.join(directory, model_file)
                self.models[model_name] = joblib.load(model_path)
        
        logger.info(f"Loaded {len(self.models)} models from {directory}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results"""
        if not self.training_results:
            return {'trained_models': 0}
        
        summary = {
            'trained_models': len(self.training_results),
            'best_model': None,
            'model_performance': {},
            'feature_count': len(self.feature_columns)
        }
        
        # Find best model
        best_name, best_result = self.get_best_model()
        summary['best_model'] = {
            'name': best_name,
            'test_score': best_result.test_score,
            'val_score': best_result.val_score
        }
        
        # Performance summary
        for name, result in self.training_results.items():
            summary['model_performance'][name] = {
                'train_score': result.train_score,
                'val_score': result.val_score,
                'test_score': result.test_score,
                'training_time': result.training_time
            }
        
        return summary
