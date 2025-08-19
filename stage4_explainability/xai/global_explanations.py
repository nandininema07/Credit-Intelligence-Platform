"""
Global explanations for Stage 4.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class GlobalExplainer:
    """Generate global model explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.training_data = None
        self.feature_names = None
        self.target_name = None
        self.is_classifier = None
        self.feature_importance_cache = {}
        
    def initialize(self, model: Any, training_data: pd.DataFrame,
                  target_column: str, is_classifier: bool = True):
        """Initialize global explainer"""
        
        self.model = model
        self.training_data = training_data.copy()
        self.target_name = target_column
        self.is_classifier = is_classifier
        
        # Separate features and target
        if target_column in training_data.columns:
            self.feature_names = [col for col in training_data.columns if col != target_column]
            self.X = training_data[self.feature_names]
            self.y = training_data[target_column]
        else:
            self.feature_names = list(training_data.columns)
            self.X = training_data
            self.y = None
        
        logger.info("Global explainer initialized")
    
    def generate_global_feature_importance(self, method: str = 'all') -> Dict[str, Any]:
        """Generate global feature importance"""
        
        if self.model is None:
            raise ValueError("Explainer not initialized. Call initialize first.")
        
        importance_results = {}
        
        if method in ['all', 'model_based']:
            importance_results['model_based'] = self._get_model_based_importance()
        
        if method in ['all', 'permutation']:
            importance_results['permutation'] = self._get_permutation_importance()
        
        if method in ['all', 'mutual_info']:
            importance_results['mutual_info'] = self._get_mutual_info_importance()
        
        if method in ['all', 'correlation']:
            importance_results['correlation'] = self._get_correlation_importance()
        
        # Create consensus ranking
        if len(importance_results) > 1:
            importance_results['consensus'] = self._create_consensus_ranking(importance_results)
        
        return {
            'importance_methods': importance_results,
            'feature_names': self.feature_names,
            'generation_timestamp': datetime.now().isoformat()
        }
    
    def _get_model_based_importance(self) -> Dict[str, float]:
        """Get model-based feature importance"""
        
        try:
            importance = {}
            
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models
                importances = self.model.feature_importances_
                for i, feature in enumerate(self.feature_names):
                    importance[feature] = float(importances[i])
            
            elif hasattr(self.model, 'coef_'):
                # Linear models
                coefs = self.model.coef_
                if len(coefs.shape) > 1:
                    coefs = coefs[0]  # Binary classification
                
                for i, feature in enumerate(self.feature_names):
                    importance[feature] = float(abs(coefs[i]))
            
            else:
                # Try to get feature importance from wrapped models
                if hasattr(self.model, 'get_feature_importance'):
                    importances = self.model.get_feature_importance()
                    for i, feature in enumerate(self.feature_names):
                        importance[feature] = float(importances[i])
                else:
                    logger.warning("Model does not support feature importance extraction")
                    return {}
            
            # Normalize importance scores
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error getting model-based importance: {e}")
            return {}
    
    def _get_permutation_importance(self, n_repeats: int = 10) -> Dict[str, float]:
        """Get permutation-based feature importance"""
        
        try:
            from sklearn.inspection import permutation_importance
            
            # Get baseline score
            if self.is_classifier:
                baseline_score = self.model.score(self.X, self.y)
            else:
                baseline_score = self.model.score(self.X, self.y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, self.X, self.y, n_repeats=n_repeats, random_state=42
            )
            
            importance = {}
            for i, feature in enumerate(self.feature_names):
                importance[feature] = float(perm_importance.importances_mean[i])
            
            # Normalize
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return {}
    
    def _get_mutual_info_importance(self) -> Dict[str, float]:
        """Get mutual information-based importance"""
        
        try:
            if self.y is None:
                return {}
            
            if self.is_classifier:
                mi_scores = mutual_info_classif(self.X, self.y, random_state=42)
            else:
                mi_scores = mutual_info_regression(self.X, self.y, random_state=42)
            
            importance = {}
            for i, feature in enumerate(self.feature_names):
                importance[feature] = float(mi_scores[i])
            
            # Normalize
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating mutual information importance: {e}")
            return {}
    
    def _get_correlation_importance(self) -> Dict[str, float]:
        """Get correlation-based importance"""
        
        try:
            if self.y is None:
                return {}
            
            correlations = self.X.corrwith(self.y).abs()
            
            importance = {}
            for feature in self.feature_names:
                if feature in correlations.index:
                    importance[feature] = float(correlations[feature])
                else:
                    importance[feature] = 0.0
            
            # Normalize
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v / total_importance for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating correlation importance: {e}")
            return {}
    
    def _create_consensus_ranking(self, importance_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Create consensus ranking from multiple importance methods"""
        
        try:
            # Calculate average rank for each feature
            feature_ranks = {}
            
            for feature in self.feature_names:
                ranks = []
                
                for method, importances in importance_results.items():
                    if method == 'consensus':
                        continue
                    
                    if importances and feature in importances:
                        # Convert importance to rank (higher importance = lower rank number)
                        sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                        rank = next(i for i, (f, _) in enumerate(sorted_features) if f == feature) + 1
                        ranks.append(rank)
                
                if ranks:
                    feature_ranks[feature] = np.mean(ranks)
                else:
                    feature_ranks[feature] = len(self.feature_names)
            
            # Convert ranks back to importance scores (lower rank = higher importance)
            max_rank = max(feature_ranks.values())
            consensus_importance = {}
            
            for feature, rank in feature_ranks.items():
                consensus_importance[feature] = (max_rank - rank + 1) / max_rank
            
            return consensus_importance
            
        except Exception as e:
            logger.error(f"Error creating consensus ranking: {e}")
            return {}
    
    def save_explainer(self, filepath: str):
        """Save explainer to file"""
        
        explainer_data = {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'is_classifier': self.is_classifier,
            'config': self.config,
            'feature_importance_cache': self.feature_importance_cache
        }
        
        joblib.dump(explainer_data, filepath)
        logger.info(f"Global explainer saved to {filepath}")
    
    def load_explainer(self, filepath: str):
        """Load explainer from file"""
        
        explainer_data = joblib.load(filepath)
        
        self.feature_names = explainer_data['feature_names']
        self.target_name = explainer_data['target_name']
        self.is_classifier = explainer_data['is_classifier']
        self.config = explainer_data.get('config', {})
        self.feature_importance_cache = explainer_data.get('feature_importance_cache', {})
        
        logger.info(f"Global explainer loaded from {filepath}")
