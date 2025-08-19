"""
Incremental learning algorithms for online model updates.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class IncrementalModel(BaseEstimator, ClassifierMixin):
    """Base class for incremental learning models"""
    
    def __init__(self, base_model: Any, buffer_size: int = 1000):
        self.base_model = base_model
        self.buffer_size = buffer_size
        self.data_buffer = []
        self.label_buffer = []
        self.update_count = 0
        self.last_update = None
        
    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """Incrementally fit the model"""
        # Add to buffer
        self.data_buffer.extend(X.values.tolist())
        self.label_buffer.extend(y.values.tolist())
        
        # Update if buffer is full
        if len(self.data_buffer) >= self.buffer_size:
            self._update_model()
        
        return self
    
    def _update_model(self):
        """Update model with buffered data"""
        if not self.data_buffer:
            return
        
        X_buffer = pd.DataFrame(self.data_buffer)
        y_buffer = pd.Series(self.label_buffer)
        
        if hasattr(self.base_model, 'partial_fit'):
            self.base_model.partial_fit(X_buffer, y_buffer)
        else:
            # Retrain from scratch for models without partial_fit
            self.base_model.fit(X_buffer, y_buffer)
        
        # Clear buffer
        self.data_buffer = []
        self.label_buffer = []
        self.update_count += 1
        self.last_update = datetime.now()
        
        logger.info(f"Model updated - Update #{self.update_count}")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        return self.base_model.predict_proba(X)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        return self.base_model.predict(X)
    
    def force_update(self):
        """Force model update with current buffer"""
        if self.data_buffer:
            self._update_model()

class OnlineGradientBoosting:
    """Online gradient boosting implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_estimators = config.get('n_estimators', 100)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.max_depth = config.get('max_depth', 3)
        
        self.estimators = []
        self.estimator_weights = []
        self.feature_names = None
        self.n_classes = 2
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Initial fit"""
        self.feature_names = X.columns.tolist()
        
        # Initialize with first weak learner
        from sklearn.tree import DecisionTreeClassifier
        
        weak_learner = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        weak_learner.fit(X, y)
        
        self.estimators = [weak_learner]
        self.estimator_weights = [1.0]
        
        return self
    
    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """Add new weak learner based on residuals"""
        if not self.estimators:
            return self.fit(X, y)
        
        # Calculate current predictions
        current_pred = self._predict_raw(X)
        
        # Calculate residuals (simplified for binary classification)
        residuals = y - current_pred
        
        # Train new weak learner on residuals
        from sklearn.tree import DecisionTreeClassifier
        
        weak_learner = DecisionTreeClassifier(max_depth=self.max_depth, random_state=len(self.estimators))
        weak_learner.fit(X, residuals)
        
        # Add to ensemble
        self.estimators.append(weak_learner)
        self.estimator_weights.append(self.learning_rate)
        
        # Limit ensemble size
        if len(self.estimators) > self.n_estimators:
            self.estimators.pop(0)
            self.estimator_weights.pop(0)
        
        return self
    
    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Raw prediction (before sigmoid)"""
        if not self.estimators:
            return np.zeros(len(X))
        
        predictions = np.zeros(len(X))
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            pred = estimator.predict(X)
            predictions += weight * pred
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        raw_pred = self._predict_raw(X)
        
        # Apply sigmoid
        proba_pos = 1 / (1 + np.exp(-raw_pred))
        proba_neg = 1 - proba_pos
        
        return np.column_stack([proba_neg, proba_pos])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

class ConceptDriftDetector:
    """Detect concept drift in streaming data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('drift_window_size', 1000)
        self.threshold = config.get('drift_threshold', 0.05)
        
        self.reference_data = None
        self.reference_labels = None
        self.current_window = []
        self.current_labels = []
        self.drift_detected = False
        
    def set_reference(self, X: pd.DataFrame, y: pd.Series):
        """Set reference distribution"""
        self.reference_data = X.copy()
        self.reference_labels = y.copy()
        
    def add_sample(self, x: pd.Series, y: int) -> bool:
        """Add new sample and check for drift"""
        self.current_window.append(x.values)
        self.current_labels.append(y)
        
        # Maintain window size
        if len(self.current_window) > self.window_size:
            self.current_window.pop(0)
            self.current_labels.pop(0)
        
        # Check for drift if window is full
        if len(self.current_window) >= self.window_size:
            return self._detect_drift()
        
        return False
    
    def _detect_drift(self) -> bool:
        """Detect concept drift using statistical tests"""
        if self.reference_data is None:
            return False
        
        current_df = pd.DataFrame(self.current_window, columns=self.reference_data.columns)
        
        # Kolmogorov-Smirnov test for each feature
        from scipy.stats import ks_2samp
        
        drift_scores = []
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['float64', 'int64']:
                statistic, p_value = ks_2samp(
                    self.reference_data[column].values,
                    current_df[column].values
                )
                drift_scores.append(p_value)
        
        # Check if any feature shows significant drift
        min_p_value = min(drift_scores) if drift_scores else 1.0
        self.drift_detected = min_p_value < self.threshold
        
        if self.drift_detected:
            logger.warning(f"Concept drift detected! Min p-value: {min_p_value:.6f}")
        
        return self.drift_detected
    
    def reset_drift_status(self):
        """Reset drift detection status"""
        self.drift_detected = False
        self.current_window = []
        self.current_labels = []

class IncrementalLearningPipeline:
    """Complete incremental learning pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.drift_detector = ConceptDriftDetector(config)
        self.performance_tracker = []
        self.retrain_threshold = config.get('retrain_threshold', 0.05)
        
    def initialize_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'sgd'):
        """Initialize incremental model"""
        
        if model_type == 'sgd':
            base_model = SGDClassifier(
                loss='log_loss',
                learning_rate='adaptive',
                eta0=0.01,
                random_state=42
            )
        elif model_type == 'passive_aggressive':
            base_model = PassiveAggressiveClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Initial fit
        base_model.fit(X, y)
        
        # Wrap in incremental model
        self.model = IncrementalModel(base_model, self.config.get('buffer_size', 1000))
        
        # Set reference for drift detection
        self.drift_detector.set_reference(X, y)
        
        logger.info(f"Initialized {model_type} incremental model")
        return self.model
    
    async def update_with_new_data(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Update model with new data"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Check for concept drift
        drift_detected = False
        for idx, (_, row) in enumerate(X_new.iterrows()):
            label = y_new.iloc[idx]
            if self.drift_detector.add_sample(row, label):
                drift_detected = True
                break
        
        if drift_detected:
            logger.warning("Concept drift detected - considering model retrain")
            return await self._handle_concept_drift(X_new, y_new)
        
        # Normal incremental update
        self.model.partial_fit(X_new, y_new)
        
        # Track performance
        await self._track_performance(X_new, y_new)
        
        return {'status': 'updated', 'drift_detected': False}
    
    async def _handle_concept_drift(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Handle concept drift detection"""
        
        # Evaluate current performance
        if hasattr(self.model, 'predict_proba'):
            y_pred = self.model.predict_proba(X_new)[:, 1]
        else:
            y_pred = self.model.predict(X_new)
        
        from sklearn.metrics import roc_auc_score
        current_performance = roc_auc_score(y_new, y_pred)
        
        # Check if performance degraded significantly
        if self.performance_tracker:
            avg_performance = np.mean([p['score'] for p in self.performance_tracker[-10:]])
            performance_drop = avg_performance - current_performance
            
            if performance_drop > self.retrain_threshold:
                logger.info("Performance degradation detected - triggering retrain")
                return await self._retrain_model(X_new, y_new)
        
        # Continue with incremental update
        self.model.partial_fit(X_new, y_new)
        self.drift_detector.reset_drift_status()
        
        return {'status': 'drift_handled', 'drift_detected': True, 'retrained': False}
    
    async def _retrain_model(self, X_new: pd.DataFrame, y_new: pd.Series):
        """Retrain model from scratch"""
        
        # Combine with recent data from buffer
        if hasattr(self.model, 'data_buffer') and self.model.data_buffer:
            buffer_df = pd.DataFrame(self.model.data_buffer)
            buffer_labels = pd.Series(self.model.label_buffer)
            
            X_combined = pd.concat([buffer_df, X_new], ignore_index=True)
            y_combined = pd.concat([buffer_labels, y_new], ignore_index=True)
        else:
            X_combined = X_new
            y_combined = y_new
        
        # Retrain model
        self.model.base_model.fit(X_combined, y_combined)
        
        # Reset buffers and drift detector
        self.model.data_buffer = []
        self.model.label_buffer = []
        self.drift_detector.reset_drift_status()
        self.drift_detector.set_reference(X_combined, y_combined)
        
        logger.info("Model retrained due to concept drift")
        return {'status': 'retrained', 'drift_detected': True, 'retrained': True}
    
    async def _track_performance(self, X: pd.DataFrame, y: pd.Series):
        """Track model performance over time"""
        if hasattr(self.model, 'predict_proba'):
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict(X)
        
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        performance = {
            'timestamp': datetime.now(),
            'score': roc_auc_score(y, y_pred),
            'accuracy': accuracy_score(y, (y_pred > 0.5).astype(int)),
            'sample_size': len(X),
            'positive_rate': y.mean()
        }
        
        self.performance_tracker.append(performance)
        
        # Keep only recent performance history
        max_history = self.config.get('performance_history_size', 100)
        if len(self.performance_tracker) > max_history:
            self.performance_tracker = self.performance_tracker[-max_history:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.performance_tracker:
            return {}
        
        recent_scores = [p['score'] for p in self.performance_tracker[-10:]]
        
        return {
            'total_updates': len(self.performance_tracker),
            'recent_mean_score': np.mean(recent_scores),
            'recent_std_score': np.std(recent_scores),
            'last_update': self.performance_tracker[-1]['timestamp'],
            'drift_detected': self.drift_detector.drift_detected,
            'model_update_count': self.model.update_count if self.model else 0
        }
    
    def save_model(self, filepath: str):
        """Save incremental model"""
        save_data = {
            'model': self.model,
            'drift_detector': self.drift_detector,
            'performance_tracker': self.performance_tracker,
            'config': self.config
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Incremental model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load incremental model"""
        save_data = joblib.load(filepath)
        
        self.model = save_data['model']
        self.drift_detector = save_data['drift_detector']
        self.performance_tracker = save_data['performance_tracker']
        
        logger.info(f"Incremental model loaded from {filepath}")

class OnlineEnsemble:
    """Online ensemble learning"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_models = []
        self.model_weights = []
        self.performance_history = []
        
    def add_base_model(self, model: Any, initial_weight: float = 1.0):
        """Add base model to ensemble"""
        incremental_model = IncrementalModel(model, self.config.get('buffer_size', 1000))
        self.base_models.append(incremental_model)
        self.model_weights.append(initial_weight)
        
    def partial_fit(self, X: pd.DataFrame, y: pd.Series):
        """Update all base models"""
        for model in self.base_models:
            model.partial_fit(X, y)
        
        # Update model weights based on recent performance
        self._update_weights(X, y)
        
        return self
    
    def _update_weights(self, X: pd.DataFrame, y: pd.Series):
        """Update model weights based on performance"""
        if len(self.base_models) <= 1:
            return
        
        # Calculate performance for each model
        performances = []
        for model in self.base_models:
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X)[:, 1]
                else:
                    y_pred = model.predict(X)
                
                from sklearn.metrics import roc_auc_score
                score = roc_auc_score(y, y_pred)
                performances.append(score)
            except:
                performances.append(0.0)
        
        # Update weights (softmax of performances)
        performances = np.array(performances)
        exp_perf = np.exp(performances - np.max(performances))
        self.model_weights = exp_perf / np.sum(exp_perf)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities using weighted ensemble"""
        if not self.base_models:
            raise ValueError("No base models in ensemble")
        
        predictions = np.zeros((len(X), 2))
        total_weight = 0
        
        for model, weight in zip(self.base_models, self.model_weights):
            if hasattr(model, 'predict_proba'):
                model_pred = model.predict_proba(X)
                predictions += weight * model_pred
                total_weight += weight
        
        if total_weight > 0:
            predictions /= total_weight
        
        return predictions
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
