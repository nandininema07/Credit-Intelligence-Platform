"""
Tests for Stage 3 scoring components.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import asyncio

from ..scoring.real_time_scorer import RealTimeScorer
from ..scoring.batch_scorer import BatchScorer
from ..scoring.score_calibration import ScoreCalibrator, PlattCalibrator, IsotonicCalibrator
from ..scoring.uncertainty_quantification import UncertaintyQuantifier

class TestRealTimeScorer(unittest.TestCase):
    """Test real-time scoring functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_df, self.y_series)
        
        self.config = {
            'cache_enabled': True,
            'cache_ttl': 300,
            'max_batch_size': 100
        }
    
    def test_scorer_initialization(self):
        """Test scorer initialization"""
        scorer = RealTimeScorer(self.config)
        self.assertIsInstance(scorer, RealTimeScorer)
    
    def test_single_prediction(self):
        """Test single prediction scoring"""
        scorer = RealTimeScorer(self.config)
        scorer.load_model(self.model, 'test_model')
        
        # Test single prediction
        features = self.X_df.iloc[0].to_dict()
        result = asyncio.run(scorer.score_single(features, 'test_model'))
        
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('probability', result)
        self.assertIn('model_id', result)
    
    def test_batch_prediction(self):
        """Test batch prediction scoring"""
        scorer = RealTimeScorer(self.config)
        scorer.load_model(self.model, 'test_model')
        
        # Test batch prediction
        features_list = [self.X_df.iloc[i].to_dict() for i in range(5)]
        results = asyncio.run(scorer.score_batch(features_list, 'test_model'))
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 5)
        
        for result in results:
            self.assertIn('prediction', result)
            self.assertIn('probability', result)

class TestBatchScorer(unittest.TestCase):
    """Test batch scoring functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=1000, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_df, self.y_series)
        
        self.config = {
            'batch_size': 100,
            'n_workers': 2,
            'output_format': 'dataframe'
        }
    
    def test_batch_scorer_initialization(self):
        """Test batch scorer initialization"""
        scorer = BatchScorer(self.config)
        self.assertIsInstance(scorer, BatchScorer)
    
    def test_batch_scoring(self):
        """Test batch scoring"""
        scorer = BatchScorer(self.config)
        
        results = asyncio.run(scorer.score_batch(self.model, self.X_df))
        
        self.assertIsInstance(results, dict)
        self.assertIn('predictions', results)
        self.assertIn('probabilities', results)
        self.assertEqual(len(results['predictions']), len(self.X_df))
    
    def test_distributed_scoring(self):
        """Test distributed scoring"""
        scorer = BatchScorer(self.config)
        
        results = asyncio.run(scorer.score_distributed(self.model, self.X_df))
        
        self.assertIsInstance(results, dict)
        self.assertIn('predictions', results)
        self.assertEqual(len(results['predictions']), len(self.X_df))

class TestScoreCalibration(unittest.TestCase):
    """Test score calibration functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_df, self.y_series)
        
        # Get uncalibrated predictions
        self.predictions = self.model.predict_proba(self.X_df)[:, 1]
        
        self.config = {'method': 'platt'}
    
    def test_score_calibrator_initialization(self):
        """Test score calibrator initialization"""
        calibrator = ScoreCalibrator(self.config)
        self.assertIsInstance(calibrator, ScoreCalibrator)
    
    def test_platt_calibration(self):
        """Test Platt scaling calibration"""
        calibrator = PlattCalibrator()
        calibrator.fit(self.predictions, self.y_series)
        
        calibrated_scores = calibrator.calibrate(self.predictions)
        
        self.assertEqual(len(calibrated_scores), len(self.predictions))
        self.assertTrue(all(0 <= score <= 1 for score in calibrated_scores))
    
    def test_isotonic_calibration(self):
        """Test isotonic regression calibration"""
        calibrator = IsotonicCalibrator()
        calibrator.fit(self.predictions, self.y_series)
        
        calibrated_scores = calibrator.calibrate(self.predictions)
        
        self.assertEqual(len(calibrated_scores), len(self.predictions))
        self.assertTrue(all(0 <= score <= 1 for score in calibrated_scores))
    
    def test_calibration_evaluation(self):
        """Test calibration evaluation"""
        calibrator = PlattCalibrator()
        calibrator.fit(self.predictions, self.y_series)
        
        calibrated_scores = calibrator.calibrate(self.predictions)
        evaluation = calibrator.evaluate_calibration(calibrated_scores, self.y_series)
        
        self.assertIsInstance(evaluation, dict)
        self.assertIn('brier_score', evaluation)
        self.assertIn('reliability', evaluation)

class TestUncertaintyQuantification(unittest.TestCase):
    """Test uncertainty quantification functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.X, self.y = make_classification(
            n_samples=500, n_features=10, n_classes=2, random_state=42
        )
        self.X_df = pd.DataFrame(self.X, columns=[f'feature_{i}' for i in range(10)])
        self.y_series = pd.Series(self.y)
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_df, self.y_series)
        
        self.config = {
            'n_bootstrap_samples': 10,
            'confidence_level': 0.95
        }
    
    def test_uncertainty_quantifier_initialization(self):
        """Test uncertainty quantifier initialization"""
        quantifier = UncertaintyQuantifier(self.config)
        self.assertIsInstance(quantifier, UncertaintyQuantifier)
    
    def test_epistemic_uncertainty(self):
        """Test epistemic uncertainty estimation"""
        quantifier = UncertaintyQuantifier(self.config)
        
        uncertainty = quantifier.estimate_epistemic_uncertainty(
            self.model, self.X_df, self.y_series
        )
        
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('mean_uncertainty', uncertainty)
        self.assertIn('uncertainty_scores', uncertainty)
        self.assertEqual(len(uncertainty['uncertainty_scores']), len(self.X_df))
    
    def test_aleatoric_uncertainty(self):
        """Test aleatoric uncertainty estimation"""
        quantifier = UncertaintyQuantifier(self.config)
        
        uncertainty = quantifier.estimate_aleatoric_uncertainty(
            self.model, self.X_df
        )
        
        self.assertIsInstance(uncertainty, dict)
        self.assertIn('uncertainty_scores', uncertainty)
        self.assertEqual(len(uncertainty['uncertainty_scores']), len(self.X_df))
    
    def test_prediction_intervals(self):
        """Test prediction interval estimation"""
        quantifier = UncertaintyQuantifier(self.config)
        
        intervals = quantifier.get_prediction_intervals(
            self.model, self.X_df, self.y_series
        )
        
        self.assertIsInstance(intervals, dict)
        self.assertIn('lower_bounds', intervals)
        self.assertIn('upper_bounds', intervals)
        self.assertEqual(len(intervals['lower_bounds']), len(self.X_df))
        self.assertEqual(len(intervals['upper_bounds']), len(self.X_df))

if __name__ == '__main__':
    unittest.main()
