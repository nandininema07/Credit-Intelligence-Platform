"""
Feature selection algorithms for Stage 2.
Implements various feature selection methods for credit risk modeling.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import joblib

logger = logging.getLogger(__name__)

class FeatureSelector:
    """Feature selection engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selectors = {}
        self.selected_features = {}
        self.feature_scores = {}
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fit feature selectors"""
        try:
            # Statistical feature selection
            self._fit_statistical_selector(X, y)
            
            # Model-based feature selection
            self._fit_model_based_selector(X, y)
            
            # Recursive feature elimination
            self._fit_rfe_selector(X, y)
            
            # Correlation-based selection
            self._fit_correlation_selector(X, y)
            
            self.fitted = True
            logger.info("Feature selectors fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting feature selectors: {e}")
            
        return self
    
    def _fit_statistical_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit statistical feature selector"""
        k_best = self.config.get('k_best_features', 50)
        
        # F-test selector
        f_selector = SelectKBest(score_func=f_classif, k=min(k_best, X.shape[1]))
        f_selector.fit(X, y)
        
        # Mutual information selector
        mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(k_best, X.shape[1]))
        mi_selector.fit(X, y)
        
        self.selectors['f_test'] = f_selector
        self.selectors['mutual_info'] = mi_selector
        
        # Store feature scores
        self.feature_scores['f_test'] = dict(zip(X.columns, f_selector.scores_))
        self.feature_scores['mutual_info'] = dict(zip(X.columns, mi_selector.scores_))
        
        # Store selected features
        self.selected_features['f_test'] = X.columns[f_selector.get_support()].tolist()
        self.selected_features['mutual_info'] = X.columns[mi_selector.get_support()].tolist()
    
    def _fit_model_based_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit model-based feature selector"""
        # Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_selector = SelectFromModel(rf, threshold='median')
        rf_selector.fit(X, y)
        
        # LASSO feature selection
        lasso = LassoCV(cv=5, random_state=42)
        lasso_selector = SelectFromModel(lasso, threshold='median')
        lasso_selector.fit(X, y)
        
        self.selectors['random_forest'] = rf_selector
        self.selectors['lasso'] = lasso_selector
        
        # Store feature importance
        self.feature_scores['random_forest'] = dict(zip(X.columns, rf.feature_importances_))
        self.feature_scores['lasso'] = dict(zip(X.columns, np.abs(lasso.coef_)))
        
        # Store selected features
        self.selected_features['random_forest'] = X.columns[rf_selector.get_support()].tolist()
        self.selected_features['lasso'] = X.columns[lasso_selector.get_support()].tolist()
    
    def _fit_rfe_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit recursive feature elimination selector"""
        n_features = self.config.get('rfe_features', 30)
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe_selector = RFE(estimator=rf, n_features_to_select=min(n_features, X.shape[1]))
        rfe_selector.fit(X, y)
        
        self.selectors['rfe'] = rfe_selector
        self.selected_features['rfe'] = X.columns[rfe_selector.get_support()].tolist()
        
        # Store ranking as scores (lower rank = higher importance)
        self.feature_scores['rfe'] = dict(zip(X.columns, 1.0 / rfe_selector.ranking_))
    
    def _fit_correlation_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit correlation-based feature selector"""
        correlation_threshold = self.config.get('correlation_threshold', 0.95)
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Features to drop (keep first occurrence)
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
        
        # Selected features (after removing highly correlated ones)
        selected_features = [col for col in X.columns if col not in to_drop]
        
        self.selected_features['correlation'] = selected_features
        
        # Calculate correlation with target as scores
        target_corr = X.corrwith(y).abs()
        self.feature_scores['correlation'] = target_corr.to_dict()
    
    def get_selected_features(self, method: str = 'ensemble') -> List[str]:
        """Get selected features using specified method"""
        if not self.fitted:
            raise ValueError("Feature selector must be fitted first")
        
        if method == 'ensemble':
            return self._get_ensemble_features()
        elif method in self.selected_features:
            return self.selected_features[method]
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _get_ensemble_features(self) -> List[str]:
        """Get ensemble of selected features"""
        # Count how many methods selected each feature
        feature_counts = {}
        
        for method, features in self.selected_features.items():
            for feature in features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Select features chosen by majority of methods
        min_votes = len(self.selected_features) // 2 + 1
        ensemble_features = [
            feature for feature, count in feature_counts.items()
            if count >= min_votes
        ]
        
        return ensemble_features
    
    def get_feature_importance(self, method: str = 'ensemble') -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.fitted:
            raise ValueError("Feature selector must be fitted first")
        
        if method == 'ensemble':
            return self._get_ensemble_importance()
        elif method in self.feature_scores:
            return self.feature_scores[method]
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _get_ensemble_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance"""
        # Normalize scores for each method
        normalized_scores = {}
        
        for method, scores in self.feature_scores.items():
            if not scores:
                continue
            
            # Normalize to 0-1 range
            score_values = np.array(list(scores.values()))
            min_score = score_values.min()
            max_score = score_values.max()
            
            if max_score > min_score:
                normalized = {
                    feature: (score - min_score) / (max_score - min_score)
                    for feature, score in scores.items()
                }
            else:
                normalized = {feature: 0.5 for feature in scores.keys()}
            
            normalized_scores[method] = normalized
        
        # Average normalized scores
        all_features = set()
        for scores in normalized_scores.values():
            all_features.update(scores.keys())
        
        ensemble_scores = {}
        for feature in all_features:
            scores = [
                method_scores.get(feature, 0)
                for method_scores in normalized_scores.values()
            ]
            ensemble_scores[feature] = np.mean(scores)
        
        return ensemble_scores
    
    def transform(self, X: pd.DataFrame, method: str = 'ensemble') -> pd.DataFrame:
        """Transform data using selected features"""
        selected_features = self.get_selected_features(method)
        available_features = [f for f in selected_features if f in X.columns]
        
        if not available_features:
            logger.warning("No selected features found in data")
            return pd.DataFrame(index=X.index)
        
        return X[available_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, method: str = 'ensemble') -> pd.DataFrame:
        """Fit selector and transform data"""
        return self.fit(X, y).transform(X, method)
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of feature selection results"""
        if not self.fitted:
            return {}
        
        summary = {
            'total_features': 0,
            'methods': {},
            'ensemble_features': len(self.get_selected_features('ensemble')),
            'top_features': []
        }
        
        # Get total features from first method
        if self.feature_scores:
            first_method = list(self.feature_scores.keys())[0]
            summary['total_features'] = len(self.feature_scores[first_method])
        
        # Summary by method
        for method in self.selected_features:
            summary['methods'][method] = {
                'selected_count': len(self.selected_features[method]),
                'selection_rate': len(self.selected_features[method]) / summary['total_features'] if summary['total_features'] > 0 else 0
            }
        
        # Top features by ensemble importance
        ensemble_importance = self.get_feature_importance('ensemble')
        top_features = sorted(
            ensemble_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        summary['top_features'] = [
            {'feature': feature, 'importance': importance}
            for feature, importance in top_features
        ]
        
        return summary
    
    def save_selector(self, filepath: str):
        """Save fitted feature selector"""
        if not self.fitted:
            raise ValueError("No fitted selector to save")
        
        selector_data = {
            'selectors': self.selectors,
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'fitted': self.fitted,
            'config': self.config
        }
        
        joblib.dump(selector_data, filepath)
        logger.info(f"Feature selector saved to {filepath}")
    
    def load_selector(self, filepath: str):
        """Load fitted feature selector"""
        selector_data = joblib.load(filepath)
        
        self.selectors = selector_data['selectors']
        self.selected_features = selector_data['selected_features']
        self.feature_scores = selector_data['feature_scores']
        self.fitted = selector_data['fitted']
        
        logger.info(f"Feature selector loaded from {filepath}")
