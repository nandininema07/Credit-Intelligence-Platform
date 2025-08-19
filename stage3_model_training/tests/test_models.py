"""
Tests for Stage 3 model components.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import tempfile
import os

from ..models.xgboost_model import XGBoostModel
from ..models.neural_networks import NeuralNetworkModel
from ..models.linear_models import LinearModel
from ..models.tree_models import TreeModel
from ..models.ensemble_models import EnsembleModel

class TestXGBoostModel(unittest.TestCase):
    """Test XGBoost model wrapper"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        self.config = {
            'random_state': 42,
            'n_estimators': 10,
            'max_depth': 3
        }
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = XGBoostModel(self.config)
        self.assertIsInstance(model, XGBoostModel)
    
    def test_model_training(self):
        """Test model training"""
        model = XGBoostModel(self.config)
        model.train(self.X_df, self.y_series)
        
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
    
    def test_model_prediction(self):
        """Test model prediction"""
        model = XGBoostModel(self.config)
        model.train(self.X_df, self.y_series)
        
        predictions = model.predict(self.X_df)
        probabilities = model.predict_proba(self.X_df)
        
        self.assertEqual(len(predictions), len(self.y_series))
        self.assertEqual(len(probabilities), len(self.y_series))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        model = XGBoostModel(self.config)
        model.train(self.X_df, self.y_series)
        
        metrics = model.evaluate(self.X_df, self.y_series)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('roc_auc', metrics)
        self.assertIn('accuracy', metrics)
    
    def test_model_save_load(self):
        """Test model save and load"""
        model = XGBoostModel(self.config)
        model.train(self.X_df, self.y_series)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            model.save_model(model_path)
            
            # Load model
            loaded_model = XGBoostModel(self.config)
            loaded_model.load_model(model_path)
            
            # Test predictions are same
            original_pred = model.predict(self.X_df)
            loaded_pred = loaded_model.predict(self.X_df)
            
            np.testing.assert_array_equal(original_pred, loaded_pred)

class TestNeuralNetworkModel(unittest.TestCase):
    """Test Neural Network model wrapper"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        self.config = {
            'random_state': 42,
            'epochs': 5,
            'batch_size': 32,
            'hidden_layers': [16, 8],
            'dropout_rate': 0.2
        }
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = NeuralNetworkModel(self.config)
        self.assertIsInstance(model, NeuralNetworkModel)
    
    def test_model_training(self):
        """Test model training"""
        model = NeuralNetworkModel(self.config)
        model.train(self.X_df, self.y_series)
        
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
    
    def test_model_prediction(self):
        """Test model prediction"""
        model = NeuralNetworkModel(self.config)
        model.train(self.X_df, self.y_series)
        
        predictions = model.predict(self.X_df)
        probabilities = model.predict_proba(self.X_df)
        
        self.assertEqual(len(predictions), len(self.y_series))
        self.assertEqual(len(probabilities), len(self.y_series))

class TestLinearModel(unittest.TestCase):
    """Test Linear model wrapper"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        self.config = {
            'random_state': 42,
            'model_type': 'logistic_regression',
            'max_iter': 1000
        }
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = LinearModel(self.config)
        self.assertIsInstance(model, LinearModel)
    
    def test_model_training(self):
        """Test model training"""
        model = LinearModel(self.config)
        model.train(self.X_df, self.y_series)
        
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
    
    def test_different_model_types(self):
        """Test different linear model types"""
        model_types = ['logistic_regression', 'ridge', 'lasso', 'elastic_net']
        
        for model_type in model_types:
            config = {**self.config, 'model_type': model_type}
            model = LinearModel(config)
            model.train(self.X_df, self.y_series)
            
            predictions = model.predict(self.X_df)
            self.assertEqual(len(predictions), len(self.y_series))

class TestTreeModel(unittest.TestCase):
    """Test Tree model wrapper"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        self.config = {
            'random_state': 42,
            'model_type': 'random_forest',
            'n_estimators': 10,
            'max_depth': 5
        }
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = TreeModel(self.config)
        self.assertIsInstance(model, TreeModel)
    
    def test_model_training(self):
        """Test model training"""
        model = TreeModel(self.config)
        model.train(self.X_df, self.y_series)
        
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
    
    def test_different_tree_types(self):
        """Test different tree model types"""
        model_types = ['decision_tree', 'random_forest', 'extra_trees']
        
        for model_type in model_types:
            config = {**self.config, 'model_type': model_type}
            model = TreeModel(config)
            model.train(self.X_df, self.y_series)
            
            predictions = model.predict(self.X_df)
            self.assertEqual(len(predictions), len(self.y_series))
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        model = TreeModel(self.config)
        model.train(self.X_df, self.y_series)
        
        importance = model.get_feature_importance()
        
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(self.X_df.columns))

class TestEnsembleModel(unittest.TestCase):
    """Test Ensemble model wrapper"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        self.config = {
            'random_state': 42,
            'model_type': 'gradient_boosting',
            'n_estimators': 10,
            'learning_rate': 0.1
        }
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = EnsembleModel(self.config)
        self.assertIsInstance(model, EnsembleModel)
    
    def test_model_training(self):
        """Test model training"""
        model = EnsembleModel(self.config)
        model.train(self.X_df, self.y_series)
        
        self.assertTrue(model.is_trained)
        self.assertIsNotNone(model.model)
    
    def test_different_ensemble_types(self):
        """Test different ensemble model types"""
        model_types = ['random_forest', 'gradient_boosting', 'ada_boost', 'lightgbm']
        
        for model_type in model_types:
            config = {**self.config, 'model_type': model_type}
            model = EnsembleModel(config)
            
            try:
                model.train(self.X_df, self.y_series)
                predictions = model.predict(self.X_df)
                self.assertEqual(len(predictions), len(self.y_series))
            except ImportError:
                # Skip if optional dependency not available
                continue
    
    def test_individual_predictions(self):
        """Test individual tree predictions"""
        model = EnsembleModel(self.config)
        model.train(self.X_df, self.y_series)
        
        individual_preds = model.get_individual_predictions(self.X_df)
        
        if individual_preds is not None:
            self.assertIsInstance(individual_preds, np.ndarray)
            self.assertEqual(individual_preds.shape[0], len(self.X_df))

if __name__ == '__main__':
    unittest.main()
