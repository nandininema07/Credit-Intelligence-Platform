"""
Backtesting framework for Stage 3 model validation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

logger = logging.getLogger(__name__)

class BacktestingFramework:
    """Comprehensive backtesting framework for credit risk models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backtest_results = {}
        
    def time_series_backtest(self, data: pd.DataFrame, model: Any,
                           target_column: str, date_column: str,
                           train_window: str = '12M', test_window: str = '3M',
                           step_size: str = '1M') -> Dict[str, Any]:
        """Perform time series backtesting"""
        
        # Sort data by date
        data = data.sort_values(date_column).reset_index(drop=True)
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in [target_column, date_column]]
        
        backtest_results = []
        
        # Define time windows
        start_date = data[date_column].min() + pd.Timedelta(train_window)
        end_date = data[date_column].max() - pd.Timedelta(test_window)
        
        current_date = start_date
        fold = 0
        
        while current_date <= end_date:
            fold += 1
            
            # Define train and test periods
            train_start = current_date - pd.Timedelta(train_window)
            train_end = current_date
            test_start = current_date
            test_end = current_date + pd.Timedelta(test_window)
            
            # Get train and test data
            train_mask = (data[date_column] >= train_start) & (data[date_column] < train_end)
            test_mask = (data[date_column] >= test_start) & (data[date_column] < test_end)
            
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            if len(train_data) == 0 or len(test_data) == 0:
                current_date += pd.Timedelta(step_size)
                continue
            
            # Prepare features and targets
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Train model
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = fold_model.predict(X_test)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            fold_result = {
                'fold': fold,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'train_positive_rate': y_train.mean(),
                'test_positive_rate': y_test.mean(),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
                'accuracy': accuracy_score(y_test, y_pred),
                'predictions': y_pred_proba.tolist(),
                'actuals': y_test.tolist()
            }
            
            backtest_results.append(fold_result)
            
            # Move to next period
            current_date += pd.Timedelta(step_size)
        
        # Aggregate results
        aggregated_results = self._aggregate_backtest_results(backtest_results)
        
        return {
            'method': 'time_series',
            'fold_results': backtest_results,
            'aggregated_results': aggregated_results,
            'config': {
                'train_window': train_window,
                'test_window': test_window,
                'step_size': step_size
            }
        }
    
    def walk_forward_backtest(self, data: pd.DataFrame, model: Any,
                            target_column: str, date_column: str,
                            initial_train_size: int = 1000,
                            step_size: int = 100) -> Dict[str, Any]:
        """Perform walk-forward backtesting"""
        
        # Sort data by date
        data = data.sort_values(date_column).reset_index(drop=True)
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in [target_column, date_column]]
        
        backtest_results = []
        
        # Start with initial training set
        train_end = initial_train_size
        fold = 0
        
        while train_end + step_size < len(data):
            fold += 1
            
            # Define train and test indices
            train_indices = list(range(0, train_end))
            test_indices = list(range(train_end, min(train_end + step_size, len(data))))
            
            # Get train and test data
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            # Prepare features and targets
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Train model
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = fold_model.predict(X_test)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            fold_result = {
                'fold': fold,
                'train_size': len(train_data),
                'test_size': len(test_data),
                'train_positive_rate': y_train.mean(),
                'test_positive_rate': y_test.mean(),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
                'accuracy': accuracy_score(y_test, y_pred),
                'predictions': y_pred_proba.tolist(),
                'actuals': y_test.tolist()
            }
            
            backtest_results.append(fold_result)
            
            # Move window forward
            train_end += step_size
        
        # Aggregate results
        aggregated_results = self._aggregate_backtest_results(backtest_results)
        
        return {
            'method': 'walk_forward',
            'fold_results': backtest_results,
            'aggregated_results': aggregated_results,
            'config': {
                'initial_train_size': initial_train_size,
                'step_size': step_size
            }
        }
    
    def purged_cross_validation(self, data: pd.DataFrame, model: Any,
                              target_column: str, date_column: str,
                              n_splits: int = 5, purge_gap: str = '1M') -> Dict[str, Any]:
        """Perform purged cross-validation for time series data"""
        
        # Sort data by date
        data = data.sort_values(date_column).reset_index(drop=True)
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col not in [target_column, date_column]]
        
        # Create time-based splits
        date_range = data[date_column].max() - data[date_column].min()
        split_size = date_range / n_splits
        
        backtest_results = []
        
        for fold in range(n_splits):
            # Define test period
            test_start = data[date_column].min() + fold * split_size
            test_end = test_start + split_size
            
            # Define purge periods
            purge_before = test_start - pd.Timedelta(purge_gap)
            purge_after = test_end + pd.Timedelta(purge_gap)
            
            # Create masks
            test_mask = (data[date_column] >= test_start) & (data[date_column] < test_end)
            purge_mask = (data[date_column] >= purge_before) & (data[date_column] <= purge_after)
            train_mask = ~purge_mask
            
            # Get train and test data
            train_data = data[train_mask]
            test_data = data[test_mask]
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # Prepare features and targets
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Train model
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = fold_model.predict(X_test)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            fold_result = {
                'fold': fold + 1,
                'test_start': test_start,
                'test_end': test_end,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'purged_samples': int(purge_mask.sum()) - len(test_data),
                'train_positive_rate': y_train.mean(),
                'test_positive_rate': y_test.mean(),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
                'accuracy': accuracy_score(y_test, y_pred),
                'predictions': y_pred_proba.tolist(),
                'actuals': y_test.tolist()
            }
            
            backtest_results.append(fold_result)
        
        # Aggregate results
        aggregated_results = self._aggregate_backtest_results(backtest_results)
        
        return {
            'method': 'purged_cv',
            'fold_results': backtest_results,
            'aggregated_results': aggregated_results,
            'config': {
                'n_splits': n_splits,
                'purge_gap': purge_gap
            }
        }
    
    def _aggregate_backtest_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across folds"""
        
        # Extract metrics
        roc_aucs = [r['roc_auc'] for r in fold_results if not np.isnan(r['roc_auc'])]
        accuracies = [r['accuracy'] for r in fold_results]
        
        # Combine all predictions and actuals
        all_predictions = []
        all_actuals = []
        
        for result in fold_results:
            all_predictions.extend(result['predictions'])
            all_actuals.extend(result['actuals'])
        
        # Calculate overall metrics
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        overall_roc_auc = roc_auc_score(all_actuals, all_predictions) if len(np.unique(all_actuals)) > 1 else np.nan
        overall_accuracy = accuracy_score(all_actuals, (np.array(all_predictions) > 0.5).astype(int))
        
        return {
            'n_folds': len(fold_results),
            'mean_roc_auc': np.mean(roc_aucs) if roc_aucs else np.nan,
            'std_roc_auc': np.std(roc_aucs) if roc_aucs else np.nan,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'overall_roc_auc': overall_roc_auc,
            'overall_accuracy': overall_accuracy,
            'total_samples': len(all_actuals),
            'overall_positive_rate': np.mean(all_actuals)
        }
    
    def monte_carlo_backtest(self, data: pd.DataFrame, model: Any,
                           target_column: str, n_iterations: int = 100,
                           train_ratio: float = 0.7) -> Dict[str, Any]:
        """Perform Monte Carlo backtesting with random splits"""
        
        # Get feature columns
        feature_columns = [col for col in data.columns if col != target_column]
        
        backtest_results = []
        
        for iteration in range(n_iterations):
            # Random split
            train_indices = np.random.choice(
                len(data), 
                size=int(len(data) * train_ratio), 
                replace=False
            )
            test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
            
            # Get train and test data
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            
            # Prepare features and targets
            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]
            
            # Train model
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = fold_model.predict(X_test)
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import roc_auc_score, accuracy_score
            
            iteration_result = {
                'iteration': iteration + 1,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'train_positive_rate': y_train.mean(),
                'test_positive_rate': y_test.mean(),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else np.nan,
                'accuracy': accuracy_score(y_test, y_pred)
            }
            
            backtest_results.append(iteration_result)
        
        # Aggregate results
        roc_aucs = [r['roc_auc'] for r in backtest_results if not np.isnan(r['roc_auc'])]
        accuracies = [r['accuracy'] for r in backtest_results]
        
        aggregated_results = {
            'n_iterations': n_iterations,
            'mean_roc_auc': np.mean(roc_aucs) if roc_aucs else np.nan,
            'std_roc_auc': np.std(roc_aucs) if roc_aucs else np.nan,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'roc_auc_confidence_interval': {
                '5th': np.percentile(roc_aucs, 5) if roc_aucs else np.nan,
                '95th': np.percentile(roc_aucs, 95) if roc_aucs else np.nan
            },
            'accuracy_confidence_interval': {
                '5th': np.percentile(accuracies, 5),
                '95th': np.percentile(accuracies, 95)
            }
        }
        
        return {
            'method': 'monte_carlo',
            'iteration_results': backtest_results,
            'aggregated_results': aggregated_results,
            'config': {
                'n_iterations': n_iterations,
                'train_ratio': train_ratio
            }
        }
    
    def performance_degradation_analysis(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance degradation over time"""
        
        if 'fold_results' not in backtest_results:
            return {}
        
        fold_results = backtest_results['fold_results']
        
        # Extract time series of performance
        if backtest_results['method'] in ['time_series', 'walk_forward']:
            performance_series = []
            
            for result in fold_results:
                performance_series.append({
                    'fold': result['fold'],
                    'roc_auc': result['roc_auc'],
                    'accuracy': result['accuracy']
                })
            
            # Calculate trends
            roc_aucs = [p['roc_auc'] for p in performance_series if not np.isnan(p['roc_auc'])]
            accuracies = [p['accuracy'] for p in performance_series]
            
            # Linear trend analysis
            if len(roc_aucs) > 1:
                x = np.arange(len(roc_aucs))
                roc_auc_trend = np.polyfit(x, roc_aucs, 1)[0]  # Slope
            else:
                roc_auc_trend = 0
            
            if len(accuracies) > 1:
                x = np.arange(len(accuracies))
                accuracy_trend = np.polyfit(x, accuracies, 1)[0]  # Slope
            else:
                accuracy_trend = 0
            
            return {
                'performance_series': performance_series,
                'roc_auc_trend': roc_auc_trend,
                'accuracy_trend': accuracy_trend,
                'performance_degradation': {
                    'roc_auc': roc_auc_trend < -0.001,  # Threshold for degradation
                    'accuracy': accuracy_trend < -0.001
                }
            }
        
        return {}
    
    def stability_analysis(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model stability across folds"""
        
        if 'fold_results' not in backtest_results:
            return {}
        
        fold_results = backtest_results['fold_results']
        
        # Extract performance metrics
        roc_aucs = [r['roc_auc'] for r in fold_results if not np.isnan(r['roc_auc'])]
        accuracies = [r['accuracy'] for r in fold_results]
        
        # Calculate stability metrics
        roc_auc_cv = np.std(roc_aucs) / np.mean(roc_aucs) if roc_aucs and np.mean(roc_aucs) > 0 else np.nan
        accuracy_cv = np.std(accuracies) / np.mean(accuracies) if accuracies and np.mean(accuracies) > 0 else np.nan
        
        return {
            'roc_auc_stability': {
                'coefficient_of_variation': roc_auc_cv,
                'min': np.min(roc_aucs) if roc_aucs else np.nan,
                'max': np.max(roc_aucs) if roc_aucs else np.nan,
                'range': np.max(roc_aucs) - np.min(roc_aucs) if roc_aucs else np.nan
            },
            'accuracy_stability': {
                'coefficient_of_variation': accuracy_cv,
                'min': np.min(accuracies),
                'max': np.max(accuracies),
                'range': np.max(accuracies) - np.min(accuracies)
            },
            'overall_stability': 'stable' if (roc_auc_cv < 0.1 and accuracy_cv < 0.1) else 'unstable'
        }
    
    def generate_backtest_report(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive backtest report"""
        
        report = {
            'backtest_method': backtest_results['method'],
            'config': backtest_results['config'],
            'aggregated_results': backtest_results['aggregated_results'],
            'performance_degradation': self.performance_degradation_analysis(backtest_results),
            'stability_analysis': self.stability_analysis(backtest_results),
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def save_backtest_results(self, results: Dict[str, Any], filepath: str):
        """Save backtest results to file"""
        joblib.dump(results, filepath)
        logger.info(f"Backtest results saved to {filepath}")
    
    def load_backtest_results(self, filepath: str) -> Dict[str, Any]:
        """Load backtest results from file"""
        results = joblib.load(filepath)
        logger.info(f"Backtest results loaded from {filepath}")
        return results
