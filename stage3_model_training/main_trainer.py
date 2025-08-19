"""
Main model training pipeline for Stage 3.
Handles ML model training, evaluation, and deployment.
"""

import asyncio
import logging
import pickle
import joblib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main model training and management class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_path = Path(config.get('model_path', './models/'))
        self.feature_store_path = Path(config.get('feature_store_path', './feature_store/'))
        self.models = {}
        self.model_metadata = {}
        
        # Create directories
        self.model_path.mkdir(exist_ok=True)
        self.feature_store_path.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize the model trainer"""
        try:
            # Load existing models
            await self._load_existing_models()
            logger.info("Stage 3 Model Training initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Stage 3: {e}")
            raise
    
    async def _load_existing_models(self):
        """Load existing trained models"""
        try:
            for model_file in self.model_path.glob("*.pkl"):
                model_name = model_file.stem
                try:
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    
                    # Load metadata if exists
                    metadata_file = self.model_path / f"{model_name}_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            self.model_metadata[model_name] = json.load(f)
                    
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
    
    async def should_retrain(self) -> bool:
        """Check if models should be retrained"""
        try:
            # Check if we have any models
            if not self.models:
                logger.info("No models found, retraining needed")
                return True
            
            # Check model age
            retrain_threshold_hours = self.config.get('retrain_threshold_hours', 24)
            
            for model_name, metadata in self.model_metadata.items():
                last_trained = datetime.fromisoformat(metadata.get('last_trained', '2020-01-01'))
                hours_since_training = (datetime.now() - last_trained).total_seconds() / 3600
                
                if hours_since_training > retrain_threshold_hours:
                    logger.info(f"Model {model_name} needs retraining (age: {hours_since_training:.1f}h)")
                    return True
            
            # Check performance degradation
            performance_threshold = self.config.get('retrain_threshold', 0.05)
            current_performance = await self._evaluate_current_performance()
            
            for model_name, perf in current_performance.items():
                if model_name in self.model_metadata:
                    baseline_perf = self.model_metadata[model_name].get('performance', {}).get('f1_score', 0)
                    if baseline_perf - perf.get('f1_score', 0) > performance_threshold:
                        logger.info(f"Model {model_name} performance degraded, retraining needed")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking retrain status: {e}")
            return True
    
    async def train_models(self) -> Dict[str, Any]:
        """Train all configured models"""
        try:
            logger.info("Starting model training...")
            
            # Get training data
            X, y = await self._prepare_training_data()
            
            if X is None or len(X) == 0:
                logger.warning("No training data available")
                return {'success': False, 'error': 'No training data'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.get('validation_split', 0.2),
                random_state=42,
                stratify=y
            )
            
            results = {}
            model_types = self.config.get('model_types', ['xgboost', 'lightgbm', 'random_forest'])
            
            for model_type in model_types:
                try:
                    logger.info(f"Training {model_type} model...")
                    
                    # Train model
                    model, performance = await self._train_single_model(
                        model_type, X_train, y_train, X_test, y_test
                    )
                    
                    if model:
                        # Save model
                        model_name = f"{model_type}_v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        await self._save_model(model_name, model, performance, X.columns.tolist())
                        
                        self.models[model_name] = model
                        results[model_type] = {
                            'success': True,
                            'model_name': model_name,
                            'performance': performance
                        }
                        
                        logger.info(f"Successfully trained {model_type}: F1={performance.get('f1_score', 0):.3f}")
                    else:
                        results[model_type] = {'success': False, 'error': 'Training failed'}
                        
                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
                    results[model_type] = {'success': False, 'error': str(e)}
            
            # Select best model as default
            await self._select_best_model(results)
            
            logger.info("Model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _prepare_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare training data from feature store"""
        try:
            # This would typically load from the feature store
            # For now, generate mock training data
            logger.info("Preparing training data...")
            
            # Mock data generation
            n_samples = 1000
            n_features = 50
            
            # Generate features
            feature_names = [f'feature_{i}' for i in range(n_features)]
            X = pd.DataFrame(
                np.random.randn(n_samples, n_features),
                columns=feature_names
            )
            
            # Generate target (credit scores converted to risk categories)
            # 0: Low risk (700+), 1: Medium risk (600-699), 2: High risk (<600)
            scores = np.random.normal(650, 100, n_samples)
            y = pd.Series(np.where(scores >= 700, 0, np.where(scores >= 600, 1, 2)))
            
            logger.info(f"Prepared training data: {len(X)} samples, {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    async def _train_single_model(self, model_type: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_test: pd.DataFrame, 
                                y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train a single model"""
        try:
            model = None
            
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            elif model_type == 'lightgbm':
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            elif model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Hyperparameter tuning if enabled
            if self.config.get('hyperparameter_tuning', False):
                model = await self._tune_hyperparameters(model, model_type, X_train, y_train)
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            performance = await self._evaluate_model(model, X_test, y_test)
            
            return model, performance
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            return None, {}
    
    async def _tune_hyperparameters(self, model: Any, model_type: str, 
                                  X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Tune hyperparameters using GridSearchCV"""
        try:
            param_grids = {
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'lightgbm': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2]
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            }
            
            param_grid = param_grids.get(model_type, {})
            if not param_grid:
                return model
            
            grid_search = GridSearchCV(
                model, param_grid, 
                cv=3, scoring='f1_macro', 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            logger.info(f"Best parameters for {model_type}: {grid_search.best_params_}")
            
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error tuning hyperparameters for {model_type}: {e}")
            return model
    
    async def _evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                            y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            performance = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro'),
                'f1_score': f1_score(y_test, y_pred, average='macro')
            }
            
            # Add AUC if binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                performance['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    async def _save_model(self, model_name: str, model: Any, 
                         performance: Dict[str, float], feature_names: List[str]):
        """Save model and metadata"""
        try:
            # Save model
            model_file = self.model_path / f"{model_name}.pkl"
            joblib.dump(model, model_file)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'performance': performance,
                'feature_names': feature_names,
                'training_date': datetime.now().isoformat(),
                'last_trained': datetime.now().isoformat(),
                'feature_count': len(feature_names)
            }
            
            metadata_file = self.model_path / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.model_metadata[model_name] = metadata
            logger.info(f"Saved model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
    
    async def _select_best_model(self, results: Dict[str, Any]):
        """Select the best performing model as default"""
        try:
            best_model = None
            best_f1 = 0
            
            for model_type, result in results.items():
                if result.get('success') and result.get('performance', {}).get('f1_score', 0) > best_f1:
                    best_f1 = result['performance']['f1_score']
                    best_model = result['model_name']
            
            if best_model:
                # Update config to use best model as default
                self.config['default_model'] = best_model
                logger.info(f"Selected best model: {best_model} (F1: {best_f1:.3f})")
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
    
    async def score_company(self, company_name: str) -> Dict[str, Any]:
        """Generate credit score for a company"""
        try:
            # Get features for the company
            features = await self._get_company_features(company_name)
            
            if not features:
                logger.warning(f"No features found for {company_name}")
                return {'error': 'No features available'}
            
            # Get default model
            default_model_name = self.config.get('default_model')
            if not default_model_name or default_model_name not in self.models:
                logger.error("No default model available")
                return {'error': 'No model available'}
            
            model = self.models[default_model_name]
            
            # Prepare features for prediction
            feature_names = self.model_metadata[default_model_name]['feature_names']
            X = pd.DataFrame([features], columns=feature_names)
            
            # Make prediction
            risk_category = model.predict(X)[0]
            confidence = np.max(model.predict_proba(X)[0]) if hasattr(model, 'predict_proba') else 0.5
            
            # Convert risk category to credit score
            score_mapping = {0: 750, 1: 650, 2: 550}  # Low, Medium, High risk
            credit_score = score_mapping.get(risk_category, 600)
            
            result = {
                'company': company_name,
                'score': credit_score,
                'risk_category': ['Low', 'Medium', 'High'][risk_category],
                'confidence': float(confidence),
                'model_used': default_model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Generated score for {company_name}: {credit_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error scoring company {company_name}: {e}")
            return {'error': str(e)}
    
    async def _get_company_features(self, company_name: str) -> Dict[str, float]:
        """Get features for a company from feature store"""
        # This would typically load from the feature store
        # For now, return mock features
        return {f'feature_{i}': np.random.randn() for i in range(50)}
    
    async def _evaluate_current_performance(self) -> Dict[str, Dict[str, float]]:
        """Evaluate current model performance on recent data"""
        # This would evaluate models on recent data
        # For now, return mock performance
        performance = {}
        for model_name in self.models:
            performance[model_name] = {
                'f1_score': np.random.uniform(0.7, 0.9),
                'accuracy': np.random.uniform(0.75, 0.95)
            }
        return performance
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'healthy': True,
            'models_loaded': len(self.models),
            'default_model': self.config.get('default_model'),
            'last_training_time': max([
                datetime.fromisoformat(meta.get('last_trained', '2020-01-01'))
                for meta in self.model_metadata.values()
            ], default=datetime(2020, 1, 1)),
            'model_performance': {
                name: meta.get('performance', {})
                for name, meta in self.model_metadata.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.models.clear()
            self.model_metadata.clear()
            logger.info("Stage 3 cleanup completed")
        except Exception as e:
            logger.error(f"Error during Stage 3 cleanup: {e}")
