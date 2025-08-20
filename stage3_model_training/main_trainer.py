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
        """Prepare training data from real financial sources"""
        try:
            logger.info("Preparing training data from real financial sources...")
            
            # Try to load from feature store first
            try:
                X, y = await self._load_from_feature_store()
                if X is not None and len(X) > 0:
                    logger.info(f"Loaded {len(X)} samples from feature store")
                    return X, y
            except Exception as e:
                logger.warning(f"Could not load from feature store: {e}")
            
            # PRIORITY 1: Try to get Kaggle datasets (real credit data)
            try:
                from stage2_feature_engineering.kaggle_data_integration import KaggleDataIntegration
                
                # Initialize Kaggle data integration
                kaggle_config = {
                    'kaggle_username': self.config.get('kaggle_username'),
                    'kaggle_key': self.config.get('kaggle_key'),
                    'data_cache_path': './kaggle_data/'
                }
                
                kaggle_data = KaggleDataIntegration(kaggle_config)
                await kaggle_data.initialize()
                
                # Try to get credit score classification dataset (most relevant)
                preferred_datasets = ['credit_score_classification', 'german_credit', 'loan_prediction']
                
                for dataset_name in preferred_datasets:
                    try:
                        logger.info(f"Attempting to load Kaggle dataset: {dataset_name}")
                        X, y = await kaggle_data.get_training_dataset(dataset_name, sample_size=2000)
                        
                        if X is not None and len(X) > 0:
                            logger.info(f"Successfully loaded Kaggle dataset: {dataset_name}")
                            logger.info(f"Real credit data: {len(X)} samples, {len(X.columns)} features")
                            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
                            return X, y
                            
                    except Exception as e:
                        logger.warning(f"Could not load Kaggle dataset {dataset_name}: {e}")
                        continue
                
                logger.warning("No Kaggle datasets could be loaded")
                
            except Exception as e:
                logger.warning(f"Could not initialize Kaggle integration: {e}")
            
            # PRIORITY 2: Try to get real financial data from APIs
            try:
                from stage2_feature_engineering.real_data_integration import RealDataIntegration
                
                # Initialize real data integration
                real_data_config = {
                    'alpha_vantage_key': self.config.get('alpha_vantage_key'),
                    'news_api_key': self.config.get('news_api_key'),
                    'yahoo_finance_enabled': self.config.get('yahoo_finance_enabled', True)
                }
                
                real_data = RealDataIntegration(real_data_config)
                await real_data.initialize()
                
                # Get real training dataset
                X, y = await real_data.get_training_dataset()
                
                if X is not None and len(X) > 0:
                    logger.info(f"Successfully loaded real financial data: {len(X)} samples, {len(X.columns)} features")
                    return X, y
                    
            except Exception as e:
                logger.warning(f"Could not load real financial data: {e}")
            
            # Fallback to realistic synthetic data if real data is unavailable
            logger.warning("Falling back to realistic synthetic data...")
            X, y = await self._generate_realistic_synthetic_data()
            
            logger.info(f"Prepared training data: {len(X)} samples, {len(X.columns)} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    async def _load_from_feature_store(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load training data from feature store"""
        try:
            # This would connect to the actual feature store
            # For now, return None to trigger synthetic data generation
            return None, None
        except Exception as e:
            logger.error(f"Error loading from feature store: {e}")
            return None, None
    
    async def _generate_synthetic_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic training data with meaningful financial features"""
        n_samples = 1000
        
        # Generate realistic financial features with noise and outliers
        features = {}
        
        # Financial ratios (liquidity, profitability, leverage, efficiency)
        features['liquidity_current_ratio'] = np.random.normal(1.5, 0.5, n_samples)
        features['liquidity_quick_ratio'] = np.random.normal(1.2, 0.4, n_samples)
        features['liquidity_cash_ratio'] = np.random.normal(0.3, 0.2, n_samples)
        
        features['profitability_gross_margin'] = np.random.normal(0.25, 0.1, n_samples)
        features['profitability_net_margin'] = np.random.normal(0.08, 0.05, n_samples)
        features['profitability_roa'] = np.random.normal(0.06, 0.03, n_samples)
        features['profitability_roe'] = np.random.normal(0.12, 0.06, n_samples)
        
        features['leverage_debt_to_equity'] = np.random.normal(0.8, 0.4, n_samples)
        features['leverage_debt_to_assets'] = np.random.normal(0.4, 0.2, n_samples)
        features['leverage_interest_coverage'] = np.random.normal(4.0, 2.0, n_samples)
        
        features['efficiency_asset_turnover'] = np.random.normal(0.8, 0.3, n_samples)
        features['efficiency_inventory_turnover'] = np.random.normal(6.0, 2.0, n_samples)
        features['efficiency_receivables_turnover'] = np.random.normal(8.0, 3.0, n_samples)
        
        # Credit-specific metrics
        features['credit_cash_flow_to_debt'] = np.random.normal(0.15, 0.08, n_samples)
        features['credit_operating_cash_flow_to_debt'] = np.random.normal(0.20, 0.10, n_samples)
        features['credit_free_cash_flow_to_debt'] = np.random.normal(0.12, 0.07, n_samples)
        features['credit_net_working_capital_to_assets'] = np.random.normal(0.15, 0.10, n_samples)
        
        # Market indicators
        features['market_pe_ratio'] = np.random.normal(15.0, 5.0, n_samples)
        features['market_pb_ratio'] = np.random.normal(1.5, 0.5, n_samples)
        features['market_ev_to_ebitda'] = np.random.normal(12.0, 4.0, n_samples)
        features['market_beta'] = np.random.normal(1.0, 0.3, n_samples)
        
        # Sentiment and news features
        features['sentiment_avg'] = np.random.normal(0.0, 0.3, n_samples)
        features['sentiment_std'] = np.random.uniform(0.1, 0.5, n_samples)
        features['positive_sentiment_ratio'] = np.random.uniform(0.2, 0.6, n_samples)
        features['negative_sentiment_ratio'] = np.random.uniform(0.1, 0.4, n_samples)
        features['event_count'] = np.random.poisson(3, n_samples)
        features['critical_event_count'] = np.random.poisson(0.5, n_samples)
        
        # Industry and sector features (encoded as continuous for now)
        features['industry_risk_score'] = np.random.uniform(0.0, 1.0, n_samples)
        features['sector_volatility'] = np.random.uniform(0.1, 0.5, n_samples)
        
        # Size and growth features
        features['company_size_log'] = np.random.normal(8.0, 1.5, n_samples)  # log of market cap
        features['revenue_growth_rate'] = np.random.normal(0.08, 0.15, n_samples)
        features['earnings_growth_rate'] = np.random.normal(0.10, 0.20, n_samples)
        
        # Macroeconomic factors
        features['interest_rate_environment'] = np.random.normal(0.04, 0.02, n_samples)
        features['gdp_growth_rate'] = np.random.normal(0.025, 0.01, n_samples)
        features['inflation_rate'] = np.random.normal(0.02, 0.01, n_samples)
        
        # Create DataFrame
        X = pd.DataFrame(features)
        
        # Add realistic noise and outliers to make data more realistic
        for col in X.columns:
            # Add some outliers (5% of data)
            outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
            X.loc[outlier_indices, col] = X[col].mean() + np.random.normal(0, 3 * X[col].std(), len(outlier_indices))
            
            # Add some missing values (2% of data)
            missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
            X.loc[missing_indices, col] = np.nan
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Generate target labels based on credit risk - make it more realistic
        # Use a more complex, realistic scoring model with noise
        credit_scores = self._calculate_realistic_credit_scores(X)
        
        # Convert to risk categories: 0=Low, 1=Medium, 2=High
        y = pd.Series(np.where(credit_scores >= 700, 0, np.where(credit_scores >= 600, 1, 2)))
        
        # Add some noise to make classification more challenging
        noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
        for idx in noise_indices:
            y.iloc[idx] = np.random.choice([0, 1, 2])
        
        return X, y
    
    def _calculate_realistic_credit_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate realistic credit scores with noise and uncertainty"""
        scores = np.zeros(len(X))
        
        # Base score with realistic variation
        base_score = 650
        
        # Feature weights (positive = good for credit, negative = bad for credit)
        weights = {
            'liquidity_current_ratio': 30,      # Higher is better
            'liquidity_quick_ratio': 25,        # Higher is better
            'profitability_gross_margin': 20,   # Higher is better
            'profitability_net_margin': 25,     # Higher is better
            'leverage_debt_to_equity': -20,     # Lower is better
            'leverage_debt_to_assets': -15,     # Lower is better
            'leverage_interest_coverage': 15,   # Higher is better
            'efficiency_asset_turnover': 10,    # Higher is better
            'credit_cash_flow_to_debt': 20,    # Higher is better
            'market_pe_ratio': -8,              # Lower is better (value investing)
            'sentiment_avg': 15,                # Higher is better
            'event_count': -3,                  # Lower is better
            'critical_event_count': -15,        # Lower is better
            'company_size_log': 10,             # Larger companies are safer
            'revenue_growth_rate': 15,          # Growth is good
            'earnings_growth_rate': 18,         # Earnings growth is very good
        }
        
        # Calculate weighted score with realistic noise
        for feature, weight in weights.items():
            if feature in X.columns:
                # Normalize feature to 0-1 range for scoring
                feature_values = X[feature].values
                if feature_values.max() != feature_values.min():
                    normalized_values = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                else:
                    normalized_values = np.ones_like(feature_values) * 0.5
                
                # Apply weight (positive weights increase score, negative decrease)
                scores += weight * normalized_values
        
        # Add base score and ensure range
        scores = base_score + scores
        
        # Add realistic noise to scores
        noise = np.random.normal(0, 50, len(scores))  # ±50 point noise
        scores = scores + noise
        
        # Ensure realistic range and add some extreme cases
        scores = np.clip(scores, 300, 850)  # Credit score range
        
        # Add some extreme cases (very good and very bad) to make it realistic
        extreme_good = np.random.choice(len(scores), size=int(len(scores) * 0.05), replace=False)
        extreme_bad = np.random.choice(len(scores), size=int(len(scores) * 0.05), replace=False)
        
        scores[extreme_good] = np.random.uniform(750, 850, len(extreme_good))
        scores[extreme_bad] = np.random.uniform(300, 450, len(extreme_bad))
        
        return scores
    
    async def _train_single_model(self, model_type: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_test: pd.DataFrame, 
                                y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
        """Train a single model"""
        try:
            model = None
            
            if model_type == 'xgboost':
                # Get the number of unique classes
                n_classes = len(np.unique(y_train))
                
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
                    num_class=n_classes if n_classes > 2 else None,
                    eval_metric='mlogloss' if n_classes > 2 else 'logloss',
                    base_score=0.5
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
            
            # Basic performance metrics
            performance = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='macro'),
                'recall': recall_score(y_test, y_pred, average='macro'),
                'f1_score': f1_score(y_test, y_pred, average='macro')
            }
            
            # Add AUC if binary classification
            if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                performance['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Add cross-validation score for better validation
            try:
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='f1_macro')
                performance['cv_f1_mean'] = cv_scores.mean()
                performance['cv_f1_std'] = cv_scores.std()
                
                # Check for overfitting: if CV score is much lower than test score
                if performance['cv_f1_mean'] < performance['f1_score'] - 0.1:
                    performance['overfitting_warning'] = True
                    performance['overfitting_gap'] = performance['f1_score'] - performance['cv_f1_mean']
                else:
                    performance['overfitting_warning'] = False
                    
            except Exception as e:
                logger.warning(f"Could not perform cross-validation: {e}")
                performance['cv_f1_mean'] = None
                performance['cv_f1_std'] = None
                performance['overfitting_warning'] = False
            
            # Validate performance meets minimum thresholds
            validation_result = self._validate_model_performance(performance)
            performance['validation_passed'] = validation_result['passed']
            performance['validation_warnings'] = validation_result['warnings']
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def _validate_model_performance(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Validate that model performance meets minimum thresholds for production use"""
        thresholds = {
            'accuracy': 0.70,      # Minimum 70% accuracy
            'precision': 0.65,     # Minimum 65% precision
            'recall': 0.60,        # Minimum 60% recall
            'f1_score': 0.62       # Minimum 62% F1-score
        }
        
        warnings = []
        passed = True
        
        # CRITICAL: Check for suspiciously high performance (overfitting/data leakage)
        suspicious_thresholds = {
            'accuracy': 0.95,      # Above 95% is suspicious
            'precision': 0.95,     # Above 95% is suspicious
            'recall': 0.95,        # Above 95% is suspicious
            'f1_score': 0.95       # Above 95% is suspicious
        }
        
        for metric, threshold in thresholds.items():
            if metric in performance:
                if performance[metric] < threshold:
                    warnings.append(f"{metric}: {performance[metric]:.3f} < {threshold} (threshold)")
                    passed = False
                elif performance[metric] < threshold + 0.05:  # Warning zone
                    warnings.append(f"{metric}: {performance[metric]:.3f} close to threshold {threshold}")
                
                # Check for suspiciously high performance
                if performance[metric] > suspicious_thresholds[metric]:
                    warnings.append(f"SUSPICIOUS: {metric} too high ({performance[metric]:.3f}) - likely overfitting or data leakage!")
                    passed = False
        
        # Additional validation rules
        if performance.get('f1_score', 0) < 0.50:
            warnings.append("F1-score too low for production use")
            passed = False
        
        if performance.get('accuracy', 0) < 0.60:
            warnings.append("Accuracy too low for production use")
            passed = False
        
        # CRITICAL: If all metrics are suspiciously high, force failure
        high_metrics = 0
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in performance and performance[metric] > 0.95:
                high_metrics += 1
        
        if high_metrics >= 3:
            warnings.append("CRITICAL: Multiple metrics >95% - likely overfitting or data leakage!")
            passed = False
        
        return {
            'passed': passed,
            'warnings': warnings,
            'thresholds': thresholds,
            'suspicious_thresholds': suspicious_thresholds
        }
    
    async def _save_model(self, model_name: str, model: Any, 
                         performance: Dict[str, float], feature_names: List[str]):
        """Save model and metadata"""
        try:
            # Check if model meets production thresholds
            validation_passed = performance.get('validation_passed', False)
            warnings = performance.get('validation_warnings', [])
            
            if not validation_passed:
                logger.warning(f"Model {model_name} does not meet production thresholds:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
                
                # Add fallback indicator to model name
                if not model_name.endswith('_fallback'):
                    model_name = f"{model_name}_fallback"
                    logger.info(f"Renamed model to {model_name} to indicate fallback status")
            
            # Save model
            model_file = self.model_path / f"{model_name}.pkl"
            joblib.dump(model, model_file)
            
            # Save metadata with validation information
            metadata = {
                'model_name': model_name,
                'model_type': type(model).__name__,
                'performance': performance,
                'feature_names': feature_names,
                'training_date': datetime.now().isoformat(),
                'last_trained': datetime.now().isoformat(),
                'feature_count': len(feature_names),
                'production_ready': validation_passed,
                'validation_warnings': warnings,
                'fallback_model': not validation_passed
            }
            
            metadata_file = self.model_path / f"{model_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.model_metadata[model_name] = metadata
            logger.info(f"Saved model: {model_name} (Production ready: {validation_passed})")
            
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
    
    def get_production_ready_models(self) -> List[str]:
        """Get list of models that meet production thresholds"""
        production_models = []
        
        for model_name, metadata in self.model_metadata.items():
            if metadata.get('production_ready', False):
                production_models.append(model_name)
        
        return production_models
    
    def get_fallback_models(self) -> List[str]:
        """Get list of fallback models (don't meet production thresholds)"""
        fallback_models = []
        
        for model_name, metadata in self.model_metadata.items():
            if metadata.get('fallback_model', False):
                fallback_models.append(model_name)
        
        return fallback_models
