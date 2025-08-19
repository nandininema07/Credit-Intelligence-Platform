"""
Training pipeline for credit risk models.
Handles data preparation, model training, validation, and persistence.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import joblib
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    models_to_train: List[str] = None
    hyperparameter_tuning: bool = True
    feature_selection: bool = True
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']

@dataclass
class ModelResults:
    """Model training results"""
    model_name: str
    model: Any
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    training_time: float
    hyperparameters: Dict[str, Any]

class TrainingPipeline:
    """ML training pipeline for credit risk models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.trained_models = {}
        
    async def prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        logger.info(f"Preparing data with {len(df)} samples")
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(df.mode().iloc[0])
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"Data prepared: {X.shape[1]} features, {len(y)} samples")
        return X, y
    
    def get_model(self, model_name: str) -> Any:
        """Get model instance by name"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state,
                n_jobs=-1,
                verbose=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=self.config.random_state
            )
        }
        return models.get(model_name)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        """Calculate model performance metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        
        return metrics
    
    def get_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return {}
        
        for feature, importance in zip(feature_names, importances):
            importance_dict[feature] = float(importance)
        
        # Sort by importance
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    async def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> ModelResults:
        """Train a single model"""
        start_time = datetime.now()
        logger.info(f"Training {model_name} model")
        
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Scale features for certain models
        if model_name in ['logistic_regression', 'svm']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_val_scaled = X_val
            X_test_scaled = X_test
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        test_pred = model.predict(X_test_scaled)
        
        # Get probabilities if available
        train_prob = model.predict_proba(X_train_scaled) if hasattr(model, 'predict_proba') else None
        val_prob = model.predict_proba(X_val_scaled) if hasattr(model, 'predict_proba') else None
        test_prob = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, train_pred, train_prob)
        val_metrics = self.calculate_metrics(y_val, val_pred, val_prob)
        test_metrics = self.calculate_metrics(y_test, test_pred, test_prob)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(model, X_train.columns.tolist())
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get hyperparameters
        hyperparameters = model.get_params()
        
        logger.info(f"Completed training {model_name} in {training_time:.2f} seconds")
        logger.info(f"Validation F1 Score: {val_metrics.get('f1_score', 0):.4f}")
        
        return ModelResults(
            model_name=model_name,
            model=model,
            train_metrics=train_metrics,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            training_time=training_time,
            hyperparameters=hyperparameters
        )
    
    async def train_models(self, df: pd.DataFrame, target_column: str) -> List[ModelResults]:
        """Train multiple models"""
        logger.info("Starting model training pipeline")
        
        # Prepare data
        X, y = await self.prepare_data(df, target_column)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size, 
            random_state=self.config.random_state, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=self.config.validation_size,
            random_state=self.config.random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train models
        results = []
        for model_name in self.config.models_to_train:
            try:
                result = await self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, X_test, y_test
                )
                results.append(result)
                self.trained_models[model_name] = result.model
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Sort by validation F1 score
        results.sort(key=lambda x: x.validation_metrics.get('f1_score', 0), reverse=True)
        
        logger.info(f"Training completed. Best model: {results[0].model_name}")
        return results
    
    def save_models(self, results: List[ModelResults], save_path: str = './models/'):
        """Save trained models"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for result in results:
            model_path = os.path.join(save_path, f"{result.model_name}.pkl")
            joblib.dump(result.model, model_path)
            
            # Save scaler if used
            if result.model_name in ['logistic_regression', 'svm']:
                scaler_path = os.path.join(save_path, f"{result.model_name}_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Saved {result.model_name} model to {model_path}")
    
    def load_model(self, model_name: str, model_path: str = './models/') -> Any:
        """Load a trained model"""
        model_file = os.path.join(model_path, f"{model_name}.pkl")
        model = joblib.load(model_file)
        
        # Load scaler if exists
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        if os.path.exists(scaler_file):
            self.scaler = joblib.load(scaler_file)
        
        logger.info(f"Loaded {model_name} model from {model_file}")
        return model
