"""
Cross-validation strategies for model training and evaluation.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Iterator
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate
)
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class CVResult:
    """Cross-validation result container"""
    scores: List[float]
    mean_score: float
    std_score: float
    fold_details: List[Dict[str, Any]]
    strategy: str
    
class CrossValidation:
    """Advanced cross-validation strategies for credit risk models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cv_strategies = {
            'kfold': self._kfold_cv,
            'stratified': self._stratified_cv,
            'timeseries': self._timeseries_cv,
            'group': self._group_cv,
            'purged_group': self._purged_group_cv,
            'walk_forward': self._walk_forward_cv
        }
    
    async def cross_validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                 strategy: str = 'stratified', 
                                 scoring: str = 'roc_auc',
                                 **kwargs) -> CVResult:
        """Perform cross-validation with specified strategy"""
        
        if strategy not in self.cv_strategies:
            raise ValueError(f"Unknown CV strategy: {strategy}")
        
        logger.info(f"Starting {strategy} cross-validation with {scoring} scoring")
        
        cv_func = self.cv_strategies[strategy]
        result = await cv_func(model, X, y, scoring, **kwargs)
        
        logger.info(f"CV completed - Mean: {result.mean_score:.4f} ± {result.std_score:.4f}")
        return result
    
    async def _kfold_cv(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       scoring: str, **kwargs) -> CVResult:
        """Standard K-Fold cross-validation"""
        
        n_splits = kwargs.get('n_splits', self.config.get('cv_folds', 5))
        random_state = kwargs.get('random_state', 42)
        
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        scores = []
        fold_details = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and score
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            score = self._calculate_score(y_val, y_pred, scoring)
            scores.append(score)
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_positive_rate': y_train.mean(),
                'val_positive_rate': y_val.mean()
            })
        
        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            fold_details=fold_details,
            strategy='kfold'
        )
    
    async def _stratified_cv(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           scoring: str, **kwargs) -> CVResult:
        """Stratified K-Fold cross-validation"""
        
        n_splits = kwargs.get('n_splits', self.config.get('cv_folds', 5))
        random_state = kwargs.get('random_state', 42)
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        scores = []
        fold_details = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and score
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            score = self._calculate_score(y_val, y_pred, scoring)
            scores.append(score)
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_positive_rate': y_train.mean(),
                'val_positive_rate': y_val.mean()
            })
        
        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            fold_details=fold_details,
            strategy='stratified'
        )
    
    async def _timeseries_cv(self, model: Any, X: pd.DataFrame, y: pd.Series,
                           scoring: str, **kwargs) -> CVResult:
        """Time series cross-validation"""
        
        n_splits = kwargs.get('n_splits', self.config.get('cv_folds', 5))
        max_train_size = kwargs.get('max_train_size', None)
        
        cv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
        
        scores = []
        fold_details = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and score
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            score = self._calculate_score(y_val, y_pred, scoring)
            scores.append(score)
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_start_idx': train_idx[0],
                'train_end_idx': train_idx[-1],
                'val_start_idx': val_idx[0],
                'val_end_idx': val_idx[-1]
            })
        
        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            fold_details=fold_details,
            strategy='timeseries'
        )
    
    async def _group_cv(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       scoring: str, **kwargs) -> CVResult:
        """Group-based cross-validation"""
        
        groups = kwargs.get('groups')
        if groups is None:
            raise ValueError("Groups must be provided for group CV")
        
        n_splits = kwargs.get('n_splits', self.config.get('cv_folds', 5))
        
        cv = GroupKFold(n_splits=n_splits)
        
        scores = []
        fold_details = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and score
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            score = self._calculate_score(y_val, y_pred, scoring)
            scores.append(score)
            
            train_groups = set(groups.iloc[train_idx])
            val_groups = set(groups.iloc[val_idx])
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_groups': len(train_groups),
                'val_groups': len(val_groups),
                'group_overlap': len(train_groups.intersection(val_groups))
            })
        
        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            fold_details=fold_details,
            strategy='group'
        )
    
    async def _purged_group_cv(self, model: Any, X: pd.DataFrame, y: pd.Series,
                             scoring: str, **kwargs) -> CVResult:
        """Purged group cross-validation for financial data"""
        
        groups = kwargs.get('groups')
        if groups is None:
            raise ValueError("Groups must be provided for purged group CV")
        
        purge_buffer = kwargs.get('purge_buffer', 1)  # Number of groups to purge
        n_splits = kwargs.get('n_splits', self.config.get('cv_folds', 5))
        
        unique_groups = sorted(groups.unique())
        group_size = len(unique_groups) // n_splits
        
        scores = []
        fold_details = []
        
        for fold in range(n_splits):
            # Define validation groups
            val_start = fold * group_size
            val_end = (fold + 1) * group_size if fold < n_splits - 1 else len(unique_groups)
            val_groups = set(unique_groups[val_start:val_end])
            
            # Define purged groups (buffer around validation)
            purge_start = max(0, val_start - purge_buffer)
            purge_end = min(len(unique_groups), val_end + purge_buffer)
            purged_groups = set(unique_groups[purge_start:purge_end])
            
            # Training groups (excluding validation and purged)
            train_groups = set(unique_groups) - purged_groups
            
            # Get indices
            train_idx = groups[groups.isin(train_groups)].index
            val_idx = groups[groups.isin(val_groups)].index
            
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            
            X_train, X_val = X.loc[train_idx], X.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and score
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            score = self._calculate_score(y_val, y_pred, scoring)
            scores.append(score)
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_groups': len(train_groups),
                'val_groups': len(val_groups),
                'purged_groups': len(purged_groups) - len(val_groups)
            })
        
        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            fold_details=fold_details,
            strategy='purged_group'
        )
    
    async def _walk_forward_cv(self, model: Any, X: pd.DataFrame, y: pd.Series,
                             scoring: str, **kwargs) -> CVResult:
        """Walk-forward cross-validation for time series"""
        
        min_train_size = kwargs.get('min_train_size', len(X) // 4)
        step_size = kwargs.get('step_size', len(X) // 10)
        window_size = kwargs.get('window_size', len(X) // 5)
        
        scores = []
        fold_details = []
        fold = 0
        
        for train_end in range(min_train_size, len(X) - window_size, step_size):
            train_start = 0
            val_start = train_end
            val_end = min(train_end + window_size, len(X))
            
            train_idx = list(range(train_start, train_end))
            val_idx = list(range(val_start, val_end))
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict and score
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            score = self._calculate_score(y_val, y_pred, scoring)
            scores.append(score)
            
            fold_details.append({
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end
            })
            
            fold += 1
        
        return CVResult(
            scores=scores,
            mean_score=np.mean(scores),
            std_score=np.std(scores),
            fold_details=fold_details,
            strategy='walk_forward'
        )
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray, scoring: str) -> float:
        """Calculate score based on scoring metric"""
        
        if scoring == 'roc_auc':
            return roc_auc_score(y_true, y_pred)
        elif scoring == 'accuracy':
            y_pred_binary = (y_pred > 0.5).astype(int)
            return accuracy_score(y_true, y_pred_binary)
        elif scoring == 'precision':
            y_pred_binary = (y_pred > 0.5).astype(int)
            return precision_score(y_true, y_pred_binary)
        elif scoring == 'recall':
            y_pred_binary = (y_pred > 0.5).astype(int)
            return recall_score(y_true, y_pred_binary)
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")
    
    def compare_cv_strategies(self, model: Any, X: pd.DataFrame, y: pd.Series,
                            strategies: List[str] = None, **kwargs) -> pd.DataFrame:
        """Compare different CV strategies"""
        
        if strategies is None:
            strategies = ['kfold', 'stratified', 'timeseries']
        
        results = []
        
        for strategy in strategies:
            try:
                result = self.cross_validate_model(model, X, y, strategy, **kwargs)
                results.append({
                    'Strategy': strategy,
                    'Mean_Score': result.mean_score,
                    'Std_Score': result.std_score,
                    'Min_Score': min(result.scores),
                    'Max_Score': max(result.scores),
                    'N_Folds': len(result.scores)
                })
            except Exception as e:
                logger.warning(f"Failed to run {strategy} CV: {e}")
        
        return pd.DataFrame(results)
    
    def get_cv_summary(self, result: CVResult) -> Dict[str, Any]:
        """Get detailed summary of CV results"""
        
        return {
            'strategy': result.strategy,
            'mean_score': result.mean_score,
            'std_score': result.std_score,
            'min_score': min(result.scores),
            'max_score': max(result.scores),
            'score_range': max(result.scores) - min(result.scores),
            'n_folds': len(result.scores),
            'fold_scores': result.scores,
            'fold_details': result.fold_details
        }

class NestedCrossValidator:
    """Nested cross-validation for unbiased model evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.outer_cv = config.get('outer_cv', 'stratified')
        self.inner_cv = config.get('inner_cv', 'stratified')
        self.outer_folds = config.get('outer_folds', 5)
        self.inner_folds = config.get('inner_folds', 3)
    
    async def nested_cross_validate(self, models: List[Any], X: pd.DataFrame, 
                                  y: pd.Series, scoring: str = 'roc_auc') -> Dict[str, Any]:
        """Perform nested cross-validation"""
        
        from sklearn.model_selection import StratifiedKFold
        
        outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=True, random_state=42)
        
        nested_scores = {f'model_{i}': [] for i in range(len(models))}
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            fold_result = {'fold': fold, 'model_scores': {}}
            
            # Inner CV for each model
            for model_idx, model in enumerate(models):
                inner_cv = CrossValidator(self.config)
                cv_result = await inner_cv.cross_validate_model(
                    model, X_train, y_train, self.inner_cv, scoring
                )
                
                # Train on full training set and evaluate on test set
                model.fit(X_train, y_train)
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred = model.predict(X_test)
                
                test_score = self._calculate_score(y_test, y_pred, scoring)
                nested_scores[f'model_{model_idx}'].append(test_score)
                fold_result['model_scores'][f'model_{model_idx}'] = {
                    'inner_cv_score': cv_result.mean_score,
                    'test_score': test_score
                }
            
            fold_results.append(fold_result)
        
        # Calculate final statistics
        final_results = {}
        for model_name, scores in nested_scores.items():
            final_results[model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores
            }
        
        return {
            'nested_scores': final_results,
            'fold_results': fold_results,
            'best_model': max(final_results.keys(), 
                            key=lambda k: final_results[k]['mean_score'])
        }
    
    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray, scoring: str) -> float:
        """Calculate score based on scoring metric"""
        if scoring == 'roc_auc':
            return roc_auc_score(y_true, y_pred)
        elif scoring == 'accuracy':
            y_pred_binary = (y_pred > 0.5).astype(int)
            return accuracy_score(y_true, y_pred_binary)
        else:
            raise ValueError(f"Unsupported scoring metric: {scoring}")
