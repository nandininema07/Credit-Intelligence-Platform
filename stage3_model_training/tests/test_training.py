"""
Tests for Stage 3 training components.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..training.model_selection import ModelSelector, AutoMLSelector
from ..training.cross_validation import CrossValidator, NestedCrossValidator
from ..training.ensemble_methods import StackingEnsemble, BlendingEnsemble, DynamicEnsemble
from ..training.incremental_learning import IncrementalLearningPipeline

class TestModelSelection(unittest.TestCase):
    """Test model selection functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=1000, n_features=20, n_informative=10,
            n_redundant=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(20)])
        self.y_series = pd.Series(self.y)
        
        self.config = {
            'random_state': 42,
            'cv_folds': 3,
            'scoring': 'roc_auc'
        }
    
    def test_model_selector_initialization(self):
        """Test ModelSelector initialization"""
        selector = ModelSelector(self.config)
        self.assertIsInstance(selector, ModelSelector)
        self.assertEqual(selector.config['random_state'], 42)
    
    def test_grid_search(self):
        """Test grid search functionality"""
        selector = ModelSelector(self.config)
        
        models = {
            'rf': RandomForestClassifier(random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        param_grids = {
            'rf': {'n_estimators': [10, 50], 'max_depth': [3, 5]},
            'lr': {'C': [0.1, 1.0]}
        }
        
        results = selector.grid_search(
            models, param_grids, self.X_df, self.y_series
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('best_model', results)
        self.assertIn('best_score', results)
        self.assertIn('results', results)

class TestCrossValidation(unittest.TestCase):
    """Test cross-validation functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        # Add date column for time series CV
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        self.X_df['date'] = dates
        
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.config = {'random_state': 42}
    
    def test_cross_validator_initialization(self):
        """Test CrossValidator initialization"""
        cv = CrossValidator(self.config)
        self.assertIsInstance(cv, CrossValidator)
    
    def test_kfold_cv(self):
        """Test K-fold cross-validation"""
        cv = CrossValidator(self.config)
        
        results = cv.cross_validate(
            self.model, self.X_df.drop('date', axis=1), self.y_series,
            cv_type='kfold', n_splits=3
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('scores', results)
        self.assertIn('mean_score', results)
        self.assertEqual(len(results['scores']), 3)
    
    def test_time_series_cv(self):
        """Test time series cross-validation"""
        cv = CrossValidator(self.config)
        
        results = cv.cross_validate(
            self.model, self.X_df, self.y_series,
            cv_type='time_series', date_column='date', n_splits=3
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('scores', results)

class TestEnsembleMethods(unittest.TestCase):
    """Test ensemble methods"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        self.base_models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000)
        ]
        
        self.config = {'random_state': 42}
    
    def test_stacking_ensemble(self):
        """Test stacking ensemble"""
        ensemble = StackingEnsemble(self.config)
        
        ensemble.fit(self.base_models, self.X_df, self.y_series)
        predictions = ensemble.predict(self.X_df)
        
        self.assertEqual(len(predictions), len(self.y_series))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_blending_ensemble(self):
        """Test blending ensemble"""
        ensemble = BlendingEnsemble(self.config)
        
        ensemble.fit(self.base_models, self.X_df, self.y_series)
        predictions = ensemble.predict(self.X_df)
        
        self.assertEqual(len(predictions), len(self.y_series))

class TestIncrementalLearning(unittest.TestCase):
    """Test incremental learning"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=1000, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        self.config = {
            'buffer_size': 100,
            'drift_threshold': 0.1,
            'random_state': 42
        }
    
    def test_incremental_pipeline_initialization(self):
        """Test incremental learning pipeline initialization"""
        pipeline = IncrementalLearningPipeline(self.config)
        self.assertIsInstance(pipeline, IncrementalLearningPipeline)
    
    def test_partial_fit(self):
        """Test partial fit functionality"""
        from sklearn.linear_model import SGDClassifier
        
        pipeline = IncrementalLearningPipeline(self.config)
        model = SGDClassifier(random_state=42)
        
        # Initial training
        initial_X = self.X_df[:500]
        initial_y = self.y_series[:500]
        
        pipeline.initialize_model(model, initial_X, initial_y)
        
        # Incremental update
        new_X = self.X_df[500:600]
        new_y = self.y_series[500:600]
        
        pipeline.update_model(new_X, new_y)
        
        # Make predictions
        predictions = pipeline.predict(self.X_df[600:700])
        self.assertEqual(len(predictions), 100)

if __name__ == '__main__':
    unittest.main()
