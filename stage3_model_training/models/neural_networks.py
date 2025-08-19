"""
Neural network models for credit risk scoring.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
import joblib

logger = logging.getLogger(__name__)

class NeuralNetworkModel:
    """Deep learning model for credit risk assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.history = None
        
        # Default architecture parameters
        self.default_params = {
            'hidden_layers': [128, 64, 32],
            'dropout_rate': 0.3,
            'activation': 'relu',
            'output_activation': 'sigmoid',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2
        }
        
        # Update with config parameters
        self.params = {**self.default_params, **config.get('neural_network_params', {})}
        
    def _build_model(self, input_dim: int) -> keras.Model:
        """Build neural network architecture"""
        
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(
            self.params['hidden_layers'][0],
            activation=self.params['activation'],
            input_shape=(input_dim,),
            name='input_layer'
        ))
        model.add(layers.Dropout(self.params['dropout_rate']))
        
        # Hidden layers
        for i, units in enumerate(self.params['hidden_layers'][1:], 1):
            model.add(layers.Dense(
                units,
                activation=self.params['activation'],
                name=f'hidden_layer_{i}'
            ))
            model.add(layers.Dropout(self.params['dropout_rate']))
        
        # Output layer
        model.add(layers.Dense(
            1,
            activation=self.params['output_activation'],
            name='output_layer'
        ))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.params['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> 'NeuralNetworkModel':
        """Train neural network model"""
        
        self.feature_names = X_train.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Build model
        self.model = self._build_model(X_train_scaled.shape[1])
        
        # Prepare callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.params['early_stopping_patience'],
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
            validation_split = None
        else:
            validation_split = self.params['validation_split']
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info("Neural network model training completed")
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled, verbose=0)
        
        # Convert to probability format [prob_class_0, prob_class_1]
        prob_class_1 = predictions.flatten()
        prob_class_0 = 1 - prob_class_1
        
        return np.column_stack([prob_class_0, prob_class_1])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
    
    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Predict credit scores (0-1000 scale)"""
        proba = self.predict_proba(X)[:, 1]
        
        # Convert to credit score scale (higher score = lower risk)
        credit_scores = 1000 - (proba * 1000)
        return credit_scores.astype(int)
    
    def get_feature_importance(self, X_sample: pd.DataFrame, method: str = 'permutation') -> Dict[str, float]:
        """Calculate feature importance using permutation or gradient-based methods"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        if method == 'permutation':
            return self._permutation_importance(X_sample)
        elif method == 'gradient':
            return self._gradient_importance(X_sample)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _permutation_importance(self, X_sample: pd.DataFrame) -> Dict[str, float]:
        """Calculate permutation-based feature importance"""
        X_scaled = self.scaler.transform(X_sample)
        baseline_score = self.model.predict(X_scaled, verbose=0).flatten()
        
        importance_scores = {}
        
        for i, feature in enumerate(self.feature_names):
            # Create permuted version
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate score drop
            permuted_score = self.model.predict(X_permuted, verbose=0).flatten()
            importance = np.mean(np.abs(baseline_score - permuted_score))
            importance_scores[feature] = importance
        
        # Normalize
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def _gradient_importance(self, X_sample: pd.DataFrame) -> Dict[str, float]:
        """Calculate gradient-based feature importance"""
        X_scaled = self.scaler.transform(X_sample)
        X_tensor = tf.constant(X_scaled, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)
        
        gradients = tape.gradient(predictions, X_tensor)
        
        # Calculate importance as mean absolute gradient
        importance_scores = {}
        for i, feature in enumerate(self.feature_names):
            importance = np.mean(np.abs(gradients[:, i].numpy()))
            importance_scores[feature] = importance
        
        # Normalize
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    def explain_prediction(self, X: pd.DataFrame, max_samples: int = 100) -> Dict[str, Any]:
        """Generate explanations for predictions using integrated gradients"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_explain = X.head(max_samples)
        X_scaled = self.scaler.transform(X_explain)
        
        try:
            # Calculate integrated gradients
            baseline = np.zeros_like(X_scaled[0])
            explanations = []
            
            for sample in X_scaled:
                integrated_grad = self._integrated_gradients(baseline, sample)
                explanations.append(integrated_grad)
            
            explanations = np.array(explanations)
            
            # Calculate feature contributions
            feature_contributions = {}
            for i, feature in enumerate(self.feature_names):
                feature_contributions[feature] = {
                    'mean_impact': float(np.mean(np.abs(explanations[:, i]))),
                    'values': explanations[:, i].tolist()
                }
            
            return {
                'integrated_gradients': explanations.tolist(),
                'feature_contributions': feature_contributions,
                'feature_names': self.feature_names
            }
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            return {"error": str(e)}
    
    def _integrated_gradients(self, baseline: np.ndarray, sample: np.ndarray, 
                            steps: int = 50) -> np.ndarray:
        """Calculate integrated gradients for a single sample"""
        
        # Generate interpolated samples
        alphas = np.linspace(0, 1, steps)
        interpolated = np.array([baseline + alpha * (sample - baseline) for alpha in alphas])
        
        # Calculate gradients
        gradients = []
        for interpolated_sample in interpolated:
            X_tensor = tf.constant(interpolated_sample.reshape(1, -1), dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                prediction = self.model(X_tensor)
            
            gradient = tape.gradient(prediction, X_tensor)
            gradients.append(gradient.numpy().flatten())
        
        # Integrate gradients
        gradients = np.array(gradients)
        integrated_grad = np.mean(gradients, axis=0) * (sample - baseline)
        
        return integrated_grad
    
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
        
        # Model evaluation
        X_test_scaled = self.scaler.transform(X_test)
        test_loss, test_accuracy, test_auc = self.model.evaluate(X_test_scaled, y_test, verbose=0)
        
        return {
            'auc_score': auc_score,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'classification_report': class_report,
            'n_samples': len(X_test),
            'positive_rate': y_test.mean()
        }
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        if self.history is None:
            return {}
        
        return {
            'loss': self.history.history.get('loss', []),
            'val_loss': self.history.history.get('val_loss', []),
            'accuracy': self.history.history.get('accuracy', []),
            'val_accuracy': self.history.history.get('val_accuracy', []),
            'auc': self.history.history.get('auc', []),
            'val_auc': self.history.history.get('val_auc', [])
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            'model_type': 'Neural Network',
            'parameters': self.params,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'model_summary': self._get_model_summary()
        }
    
    def _get_model_summary(self) -> Dict[str, Any]:
        """Get model architecture summary"""
        if self.model is None:
            return {}
        
        return {
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'layers': len(self.model.layers),
            'architecture': [layer.output_shape for layer in self.model.layers]
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Save Keras model
        model_path = filepath.replace('.pkl', '_model.h5')
        self.model.save(model_path)
        
        # Save other components
        model_data = {
            'config': self.config,
            'params': self.params,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'model_path': model_path
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Neural network model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        
        # Load Keras model
        model_path = model_data['model_path']
        self.model = keras.models.load_model(model_path)
        
        logger.info(f"Neural network model loaded from {filepath}")
    
    async def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Perform hyperparameter tuning using Keras Tuner"""
        try:
            import keras_tuner as kt
        except ImportError:
            logger.warning("keras_tuner not available, skipping hyperparameter tuning")
            return {}
        
        def build_model(hp):
            model = keras.Sequential()
            
            # Tune number of layers and units
            for i in range(hp.Int('num_layers', 2, 5)):
                model.add(layers.Dense(
                    units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                    activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
                ))
                model.add(layers.Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
            
            model.add(layers.Dense(1, activation='sigmoid'))
            
            model.compile(
                optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                loss='binary_crossentropy',
                metrics=['AUC']
            )
            
            return model
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create tuner
        tuner = kt.RandomSearch(
            build_model,
            objective='val_auc',
            max_trials=20,
            directory='keras_tuner',
            project_name='credit_risk_nn'
        )
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Search for best hyperparameters
        tuner.search(
            X_train_scaled, y_train,
            epochs=50,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else None,
            callbacks=[callbacks.EarlyStopping(patience=5)],
            verbose=0
        )
        
        # Get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Update model
        self.model = best_model
        self.is_fitted = True
        
        logger.info("Hyperparameter tuning completed")
        
        return {
            'best_params': best_params.values,
            'best_score': tuner.oracle.get_best_trials(1)[0].score
        }
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """Predict with uncertainty estimation using Monte Carlo dropout"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        # Enable dropout during inference
        predictions = []
        for _ in range(n_samples):
            # Use model in training mode to enable dropout
            pred = self.model(X_scaled, training=True).numpy().flatten()
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            'mean_prediction': np.mean(predictions, axis=0),
            'std_prediction': np.std(predictions, axis=0),
            'predictions': predictions
        }
