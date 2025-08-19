"""
Feature attribution methods for Stage 4.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

class FeatureAttributionAnalyzer:
    """Feature attribution analysis using multiple methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_names = None
        self.attribution_cache = {}
        
    def set_model(self, model: Any, feature_names: List[str]):
        """Set model and feature names"""
        self.model = model
        self.feature_names = feature_names
        
    def permutation_importance_analysis(self, X: pd.DataFrame, y: pd.Series,
                                      scoring: str = 'roc_auc',
                                      n_repeats: int = 10) -> Dict[str, Any]:
        """Calculate permutation importance"""
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model first.")
        
        try:
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X, y,
                scoring=scoring,
                n_repeats=n_repeats,
                random_state=self.config.get('random_state', 42)
            )
            
            # Format results
            importance_data = {}
            for i, feature_name in enumerate(self.feature_names):
                importance_data[feature_name] = {
                    'importance_mean': float(perm_importance.importances_mean[i]),
                    'importance_std': float(perm_importance.importances_std[i]),
                    'importance_values': perm_importance.importances[i].tolist()
                }
            
            # Sort by importance
            sorted_importance = sorted(
                importance_data.items(),
                key=lambda x: x[1]['importance_mean'],
                reverse=True
            )
            
            return {
                'method': 'permutation_importance',
                'scoring': scoring,
                'n_repeats': n_repeats,
                'feature_importance': importance_data,
                'ranked_features': [item[0] for item in sorted_importance],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return {'error': str(e)}
    
    def drop_column_importance(self, X: pd.DataFrame, y: pd.Series,
                             scoring_func: callable = None) -> Dict[str, Any]:
        """Calculate importance by dropping each column"""
        
        if self.model is None:
            raise ValueError("Model not set. Call set_model first.")
        
        if scoring_func is None:
            scoring_func = roc_auc_score if len(np.unique(y)) == 2 else accuracy_score
        
        try:
            # Get baseline score
            if hasattr(self.model, 'predict_proba'):
                baseline_pred = self.model.predict_proba(X)[:, 1]
            else:
                baseline_pred = self.model.predict(X)
            
            baseline_score = scoring_func(y, baseline_pred)
            
            # Calculate importance for each feature
            drop_importance = {}
            
            for feature_name in self.feature_names:
                # Create dataset without this feature
                X_dropped = X.drop(columns=[feature_name])
                
                # Retrain model or use existing model with dropped feature
                # For simplicity, we'll use feature importance from existing model
                try:
                    if hasattr(self.model, 'predict_proba'):
                        dropped_pred = self.model.predict_proba(X_dropped)[:, 1]
                    else:
                        dropped_pred = self.model.predict(X_dropped)
                    
                    dropped_score = scoring_func(y, dropped_pred)
                    importance = baseline_score - dropped_score
                    
                except Exception:
                    # If model can't handle dropped feature, set importance to 0
                    importance = 0.0
                
                drop_importance[feature_name] = {
                    'baseline_score': float(baseline_score),
                    'dropped_score': float(dropped_score) if 'dropped_score' in locals() else 0.0,
                    'importance': float(importance)
                }
            
            # Sort by importance
            sorted_importance = sorted(
                drop_importance.items(),
                key=lambda x: x[1]['importance'],
                reverse=True
            )
            
            return {
                'method': 'drop_column_importance',
                'baseline_score': float(baseline_score),
                'feature_importance': drop_importance,
                'ranked_features': [item[0] for item in sorted_importance],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating drop column importance: {e}")
            return {'error': str(e)}
    
    def gradient_based_attribution(self, X: pd.DataFrame,
                                 target_class: int = 1) -> Dict[str, Any]:
        """Calculate gradient-based feature attribution"""
        
        try:
            # This is a simplified gradient-based attribution
            # In practice, you'd use libraries like Captum for deep learning models
            
            attributions = {}
            
            # For each instance, calculate approximate gradients
            for idx in range(min(len(X), 100)):  # Limit to 100 instances for efficiency
                instance = X.iloc[[idx]]
                
                # Calculate numerical gradients
                gradients = self._calculate_numerical_gradients(instance, target_class)
                
                for i, feature_name in enumerate(self.feature_names):
                    if feature_name not in attributions:
                        attributions[feature_name] = []
                    attributions[feature_name].append(gradients[i])
            
            # Aggregate attributions
            feature_attributions = {}
            for feature_name in self.feature_names:
                if feature_name in attributions:
                    values = attributions[feature_name]
                    feature_attributions[feature_name] = {
                        'mean_attribution': float(np.mean(values)),
                        'std_attribution': float(np.std(values)),
                        'abs_mean_attribution': float(np.mean(np.abs(values)))
                    }
            
            # Sort by absolute mean attribution
            sorted_attributions = sorted(
                feature_attributions.items(),
                key=lambda x: x[1]['abs_mean_attribution'],
                reverse=True
            )
            
            return {
                'method': 'gradient_based_attribution',
                'target_class': target_class,
                'feature_attributions': feature_attributions,
                'ranked_features': [item[0] for item in sorted_attributions],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating gradient-based attribution: {e}")
            return {'error': str(e)}
    
    def _calculate_numerical_gradients(self, instance: pd.DataFrame,
                                     target_class: int = 1,
                                     epsilon: float = 1e-4) -> np.ndarray:
        """Calculate numerical gradients for an instance"""
        
        gradients = np.zeros(len(self.feature_names))
        
        # Get baseline prediction
        if hasattr(self.model, 'predict_proba'):
            baseline_pred = self.model.predict_proba(instance)[0, target_class]
        else:
            baseline_pred = self.model.predict(instance)[0]
        
        # Calculate gradient for each feature
        for i, feature_name in enumerate(self.feature_names):
            # Create perturbed instance
            instance_plus = instance.copy()
            instance_minus = instance.copy()
            
            original_value = instance.iloc[0, i]
            
            # Add/subtract epsilon
            instance_plus.iloc[0, i] = original_value + epsilon
            instance_minus.iloc[0, i] = original_value - epsilon
            
            # Get predictions
            try:
                if hasattr(self.model, 'predict_proba'):
                    pred_plus = self.model.predict_proba(instance_plus)[0, target_class]
                    pred_minus = self.model.predict_proba(instance_minus)[0, target_class]
                else:
                    pred_plus = self.model.predict(instance_plus)[0]
                    pred_minus = self.model.predict(instance_minus)[0]
                
                # Calculate gradient
                gradients[i] = (pred_plus - pred_minus) / (2 * epsilon)
                
            except Exception:
                gradients[i] = 0.0
        
        return gradients
    
    def integrated_gradients_approximation(self, instance: pd.DataFrame,
                                         baseline: pd.DataFrame = None,
                                         steps: int = 50) -> Dict[str, Any]:
        """Approximate integrated gradients"""
        
        if baseline is None:
            # Use zeros as baseline
            baseline = pd.DataFrame(
                np.zeros((1, len(self.feature_names))),
                columns=self.feature_names
            )
        
        try:
            # Create interpolated path from baseline to instance
            attributions = np.zeros(len(self.feature_names))
            
            for step in range(steps):
                # Interpolate between baseline and instance
                alpha = step / steps
                interpolated = baseline + alpha * (instance - baseline)
                
                # Calculate gradients at this point
                gradients = self._calculate_numerical_gradients(interpolated)
                
                # Accumulate attributions
                attributions += gradients
            
            # Scale by step size and multiply by (instance - baseline)
            attributions = attributions / steps
            feature_diffs = (instance.iloc[0] - baseline.iloc[0]).values
            attributions = attributions * feature_diffs
            
            # Format results
            feature_attributions = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_attributions[feature_name] = {
                    'attribution': float(attributions[i]),
                    'abs_attribution': float(abs(attributions[i])),
                    'feature_value': float(instance.iloc[0, i]),
                    'baseline_value': float(baseline.iloc[0, i])
                }
            
            # Sort by absolute attribution
            sorted_attributions = sorted(
                feature_attributions.items(),
                key=lambda x: x[1]['abs_attribution'],
                reverse=True
            )
            
            return {
                'method': 'integrated_gradients_approximation',
                'steps': steps,
                'feature_attributions': feature_attributions,
                'ranked_features': [item[0] for item in sorted_attributions],
                'total_attribution': float(np.sum(attributions)),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating integrated gradients: {e}")
            return {'error': str(e)}
    
    def occlusion_analysis(self, instance: pd.DataFrame,
                          occlusion_value: str = 'mean') -> Dict[str, Any]:
        """Perform occlusion analysis"""
        
        try:
            # Get baseline prediction
            if hasattr(self.model, 'predict_proba'):
                baseline_pred = self.model.predict_proba(instance)[0, 1]
            else:
                baseline_pred = self.model.predict(instance)[0]
            
            occlusion_results = {}
            
            for feature_name in self.feature_names:
                # Create occluded instance
                occluded_instance = instance.copy()
                
                if occlusion_value == 'mean':
                    # Use mean value from training data (if available)
                    occlusion_val = 0.0  # Simplified
                elif occlusion_value == 'zero':
                    occlusion_val = 0.0
                else:
                    occlusion_val = float(occlusion_value)
                
                occluded_instance[feature_name] = occlusion_val
                
                # Get prediction with occluded feature
                try:
                    if hasattr(self.model, 'predict_proba'):
                        occluded_pred = self.model.predict_proba(occluded_instance)[0, 1]
                    else:
                        occluded_pred = self.model.predict(occluded_instance)[0]
                    
                    importance = baseline_pred - occluded_pred
                    
                except Exception:
                    importance = 0.0
                    occluded_pred = baseline_pred
                
                occlusion_results[feature_name] = {
                    'baseline_prediction': float(baseline_pred),
                    'occluded_prediction': float(occluded_pred),
                    'importance': float(importance),
                    'abs_importance': float(abs(importance)),
                    'original_value': float(instance[feature_name].iloc[0]),
                    'occlusion_value': float(occlusion_val)
                }
            
            # Sort by absolute importance
            sorted_results = sorted(
                occlusion_results.items(),
                key=lambda x: x[1]['abs_importance'],
                reverse=True
            )
            
            return {
                'method': 'occlusion_analysis',
                'occlusion_strategy': occlusion_value,
                'baseline_prediction': float(baseline_pred),
                'feature_importance': occlusion_results,
                'ranked_features': [item[0] for item in sorted_results],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error performing occlusion analysis: {e}")
            return {'error': str(e)}
    
    def compare_attribution_methods(self, X: pd.DataFrame, y: pd.Series,
                                  instance: pd.DataFrame = None) -> Dict[str, Any]:
        """Compare multiple attribution methods"""
        
        if instance is None:
            instance = X.iloc[[0]]
        
        results = {}
        
        # Permutation importance (global)
        perm_importance = self.permutation_importance_analysis(X, y)
        if 'error' not in perm_importance:
            results['permutation_importance'] = perm_importance
        
        # Gradient-based attribution (local)
        gradient_attribution = self.gradient_based_attribution(instance)
        if 'error' not in gradient_attribution:
            results['gradient_attribution'] = gradient_attribution
        
        # Occlusion analysis (local)
        occlusion_results = self.occlusion_analysis(instance)
        if 'error' not in occlusion_results:
            results['occlusion_analysis'] = occlusion_results
        
        # Integrated gradients approximation (local)
        integrated_gradients = self.integrated_gradients_approximation(instance)
        if 'error' not in integrated_gradients:
            results['integrated_gradients'] = integrated_gradients
        
        # Compare rankings
        rankings = {}
        for method_name, method_results in results.items():
            if 'ranked_features' in method_results:
                rankings[method_name] = method_results['ranked_features']
        
        # Calculate rank correlation
        rank_correlations = {}
        if len(rankings) > 1:
            from scipy.stats import spearmanr
            
            method_names = list(rankings.keys())
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    method1, method2 = method_names[i], method_names[j]
                    
                    # Get common features
                    common_features = list(set(rankings[method1]) & set(rankings[method2]))
                    
                    if len(common_features) > 2:
                        rank1 = [rankings[method1].index(f) for f in common_features]
                        rank2 = [rankings[method2].index(f) for f in common_features]
                        
                        try:
                            correlation, p_value = spearmanr(rank1, rank2)
                            rank_correlations[f"{method1}_vs_{method2}"] = {
                                'correlation': float(correlation),
                                'p_value': float(p_value),
                                'num_features': len(common_features)
                            }
                        except Exception:
                            rank_correlations[f"{method1}_vs_{method2}"] = {
                                'correlation': 0.0,
                                'p_value': 1.0,
                                'num_features': len(common_features)
                            }
        
        return {
            'comparison_timestamp': datetime.now().isoformat(),
            'methods_compared': list(results.keys()),
            'method_results': results,
            'feature_rankings': rankings,
            'rank_correlations': rank_correlations,
            'consensus_ranking': self._calculate_consensus_ranking(rankings)
        }
    
    def _calculate_consensus_ranking(self, rankings: Dict[str, List[str]]) -> List[str]:
        """Calculate consensus ranking using Borda count"""
        
        if not rankings:
            return []
        
        # Get all unique features
        all_features = set()
        for ranking in rankings.values():
            all_features.update(ranking)
        
        # Calculate Borda scores
        borda_scores = {feature: 0 for feature in all_features}
        
        for ranking in rankings.values():
            n_features = len(ranking)
            for i, feature in enumerate(ranking):
                # Higher rank gets higher score
                borda_scores[feature] += (n_features - i)
        
        # Sort by Borda score
        consensus_ranking = sorted(
            borda_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [feature for feature, score in consensus_ranking]
    
    def save_attributions(self, filepath: str):
        """Save attribution cache to file"""
        joblib.dump(self.attribution_cache, filepath)
        logger.info(f"Attribution cache saved to {filepath}")
    
    def load_attributions(self, filepath: str):
        """Load attribution cache from file"""
        self.attribution_cache = joblib.load(filepath)
        logger.info(f"Attribution cache loaded from {filepath}")
    
    def generate_attribution_report(self, X: pd.DataFrame, y: pd.Series,
                                  instance: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate comprehensive attribution report"""
        
        if instance is None:
            instance = X.iloc[[0]]
        
        # Compare all methods
        comparison_results = self.compare_attribution_methods(X, y, instance)
        
        # Generate summary
        summary = {
            'instance_analyzed': instance.iloc[0].to_dict(),
            'methods_used': comparison_results['methods_compared'],
            'consensus_top_features': comparison_results['consensus_ranking'][:10],
            'method_agreement': 'high' if any(
                corr['correlation'] > 0.7 
                for corr in comparison_results['rank_correlations'].values()
            ) else 'low'
        }
        
        return {
            'attribution_report': comparison_results,
            'summary': summary,
            'report_timestamp': datetime.now().isoformat()
        }
