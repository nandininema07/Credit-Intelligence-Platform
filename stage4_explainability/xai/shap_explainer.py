"""
SHAP-based explanations for Stage 4.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """SHAP-based model explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainer = None
        self.background_data = None
        self.feature_names = None
        self.model = None
        
    def initialize_explainer(self, model: Any, background_data: pd.DataFrame, 
                           explainer_type: str = 'auto'):
        """Initialize SHAP explainer"""
        
        self.model = model
        self.background_data = background_data
        self.feature_names = list(background_data.columns)
        
        try:
            if explainer_type == 'auto':
                explainer_type = self._detect_explainer_type(model)
            
            if explainer_type == 'tree':
                self.explainer = shap.TreeExplainer(model)
            elif explainer_type == 'linear':
                self.explainer = shap.LinearExplainer(model, background_data)
            elif explainer_type == 'kernel':
                self.explainer = shap.KernelExplainer(model.predict, background_data)
            elif explainer_type == 'deep':
                self.explainer = shap.DeepExplainer(model, background_data.values)
            elif explainer_type == 'gradient':
                self.explainer = shap.GradientExplainer(model, background_data.values)
            else:
                # Default to Kernel explainer
                self.explainer = shap.KernelExplainer(model.predict, background_data)
            
            logger.info(f"Initialized {explainer_type} SHAP explainer")
            
        except Exception as e:
            logger.warning(f"Failed to initialize {explainer_type} explainer, using Kernel: {e}")
            self.explainer = shap.KernelExplainer(model.predict, background_data)
    
    def _detect_explainer_type(self, model: Any) -> str:
        """Detect appropriate explainer type based on model"""
        
        model_name = type(model).__name__.lower()
        
        if any(tree_type in model_name for tree_type in ['tree', 'forest', 'xgb', 'lgb', 'catboost']):
            return 'tree'
        elif any(linear_type in model_name for linear_type in ['linear', 'logistic', 'ridge', 'lasso']):
            return 'linear'
        elif any(deep_type in model_name for deep_type in ['neural', 'keras', 'tensorflow', 'torch']):
            return 'deep'
        else:
            return 'kernel'
    
    def explain_instance(self, instance: pd.DataFrame, 
                        return_dict: bool = True) -> Dict[str, Any]:
        """Explain a single instance"""
        
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(instance)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
            
            # Get base value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                expected_value = 0.0
            
            if return_dict:
                return self._format_explanation_dict(
                    shap_values, expected_value, instance
                )
            else:
                return {
                    'shap_values': shap_values,
                    'expected_value': expected_value,
                    'instance': instance
                }
                
        except Exception as e:
            logger.error(f"Error explaining instance: {e}")
            return {'error': str(e)}
    
    def _format_explanation_dict(self, shap_values: np.ndarray, 
                               expected_value: float, 
                               instance: pd.DataFrame) -> Dict[str, Any]:
        """Format explanation as dictionary"""
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Take first instance if batch
        
        feature_contributions = {}
        for i, feature_name in enumerate(self.feature_names):
            feature_contributions[feature_name] = {
                'value': float(instance.iloc[0, i]),
                'shap_value': float(shap_values[i]),
                'contribution': float(shap_values[i]),
                'abs_contribution': float(abs(shap_values[i]))
            }
        
        # Sort by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]['abs_contribution'],
            reverse=True
        )
        
        return {
            'expected_value': float(expected_value),
            'prediction': float(expected_value + np.sum(shap_values)),
            'feature_contributions': feature_contributions,
            'top_features': dict(sorted_features[:10]),
            'total_contribution': float(np.sum(shap_values)),
            'explanation_timestamp': datetime.now().isoformat()
        }
    
    def explain_batch(self, instances: pd.DataFrame) -> List[Dict[str, Any]]:
        """Explain multiple instances"""
        
        explanations = []
        
        for idx in range(len(instances)):
            instance = instances.iloc[[idx]]
            explanation = self.explain_instance(instance)
            explanation['instance_id'] = idx
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_importance(self, instances: pd.DataFrame = None) -> Dict[str, float]:
        """Get global feature importance"""
        
        if instances is None:
            instances = self.background_data
        
        try:
            shap_values = self.explainer.shap_values(instances)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            feature_importance = {}
            for i, feature_name in enumerate(self.feature_names):
                feature_importance[feature_name] = float(mean_abs_shap[i])
            
            # Normalize to sum to 1
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {
                    k: v / total_importance 
                    for k, v in feature_importance.items()
                }
            
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def generate_force_plot_data(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate data for SHAP force plot visualization"""
        
        explanation = self.explain_instance(instance, return_dict=False)
        
        if 'error' in explanation:
            return explanation
        
        shap_values = explanation['shap_values']
        expected_value = explanation['expected_value']
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Prepare data for force plot
        force_plot_data = {
            'expected_value': float(expected_value),
            'shap_values': shap_values.tolist(),
            'feature_names': self.feature_names,
            'feature_values': instance.iloc[0].tolist(),
            'prediction': float(expected_value + np.sum(shap_values))
        }
        
        return force_plot_data
    
    def generate_waterfall_data(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate data for waterfall chart"""
        
        explanation = self.explain_instance(instance)
        
        if 'error' in explanation:
            return explanation
        
        feature_contributions = explanation['feature_contributions']
        expected_value = explanation['expected_value']
        
        # Sort features by contribution magnitude
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]['shap_value']),
            reverse=True
        )
        
        waterfall_data = {
            'expected_value': expected_value,
            'features': [],
            'prediction': explanation['prediction']
        }
        
        cumulative_value = expected_value
        
        for feature_name, contrib in sorted_features[:15]:  # Top 15 features
            waterfall_data['features'].append({
                'name': feature_name,
                'value': contrib['value'],
                'contribution': contrib['shap_value'],
                'cumulative': cumulative_value + contrib['shap_value']
            })
            cumulative_value += contrib['shap_value']
        
        return waterfall_data
    
    def generate_summary_plot_data(self, instances: pd.DataFrame = None, 
                                 max_display: int = 20) -> Dict[str, Any]:
        """Generate data for SHAP summary plot"""
        
        if instances is None:
            instances = self.background_data.sample(min(100, len(self.background_data)))
        
        try:
            shap_values = self.explainer.shap_values(instances)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Calculate feature importance
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Get top features
            top_indices = np.argsort(feature_importance)[-max_display:][::-1]
            
            summary_data = {
                'feature_names': [self.feature_names[i] for i in top_indices],
                'shap_values': shap_values[:, top_indices].tolist(),
                'feature_values': instances.iloc[:, top_indices].values.tolist(),
                'feature_importance': feature_importance[top_indices].tolist()
            }
            
            return summary_data
            
        except Exception as e:
            logger.error(f"Error generating summary plot data: {e}")
            return {'error': str(e)}
    
    def get_interaction_values(self, instance: pd.DataFrame, 
                             feature_pairs: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """Get SHAP interaction values"""
        
        try:
            if hasattr(self.explainer, 'shap_interaction_values'):
                interaction_values = self.explainer.shap_interaction_values(instance)
                
                if feature_pairs is None:
                    # Get top interacting feature pairs
                    feature_pairs = self._get_top_interactions(interaction_values)
                
                interactions = {}
                for feat1, feat2 in feature_pairs:
                    idx1 = self.feature_names.index(feat1)
                    idx2 = self.feature_names.index(feat2)
                    
                    interactions[f"{feat1}_x_{feat2}"] = {
                        'interaction_value': float(interaction_values[0, idx1, idx2]),
                        'feature1_main_effect': float(interaction_values[0, idx1, idx1]),
                        'feature2_main_effect': float(interaction_values[0, idx2, idx2])
                    }
                
                return interactions
            else:
                return {'error': 'Interaction values not supported for this explainer'}
                
        except Exception as e:
            logger.error(f"Error calculating interaction values: {e}")
            return {'error': str(e)}
    
    def _get_top_interactions(self, interaction_values: np.ndarray, 
                            top_k: int = 10) -> List[Tuple[str, str]]:
        """Get top feature interactions"""
        
        # Get upper triangle of interaction matrix (excluding diagonal)
        n_features = len(self.feature_names)
        interaction_strengths = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                strength = abs(interaction_values[0, i, j])
                interaction_strengths.append((strength, i, j))
        
        # Sort by interaction strength
        interaction_strengths.sort(reverse=True)
        
        # Return top k feature pairs
        top_pairs = []
        for _, i, j in interaction_strengths[:top_k]:
            top_pairs.append((self.feature_names[i], self.feature_names[j]))
        
        return top_pairs
    
    def explain_prediction_change(self, instance_before: pd.DataFrame,
                                instance_after: pd.DataFrame) -> Dict[str, Any]:
        """Explain how prediction changes between two instances"""
        
        explanation_before = self.explain_instance(instance_before)
        explanation_after = self.explain_instance(instance_after)
        
        if 'error' in explanation_before or 'error' in explanation_after:
            return {'error': 'Failed to explain one or both instances'}
        
        # Calculate changes
        prediction_change = explanation_after['prediction'] - explanation_before['prediction']
        
        feature_changes = {}
        for feature in self.feature_names:
            before_contrib = explanation_before['feature_contributions'][feature]['shap_value']
            after_contrib = explanation_after['feature_contributions'][feature]['shap_value']
            
            feature_changes[feature] = {
                'value_before': explanation_before['feature_contributions'][feature]['value'],
                'value_after': explanation_after['feature_contributions'][feature]['value'],
                'value_change': explanation_after['feature_contributions'][feature]['value'] - 
                               explanation_before['feature_contributions'][feature]['value'],
                'contribution_before': before_contrib,
                'contribution_after': after_contrib,
                'contribution_change': after_contrib - before_contrib
            }
        
        # Sort by absolute contribution change
        sorted_changes = sorted(
            feature_changes.items(),
            key=lambda x: abs(x[1]['contribution_change']),
            reverse=True
        )
        
        return {
            'prediction_before': explanation_before['prediction'],
            'prediction_after': explanation_after['prediction'],
            'prediction_change': prediction_change,
            'feature_changes': feature_changes,
            'top_drivers': dict(sorted_changes[:10]),
            'explanation_timestamp': datetime.now().isoformat()
        }
    
    def save_explainer(self, filepath: str):
        """Save explainer to file"""
        
        explainer_data = {
            'explainer': self.explainer,
            'background_data': self.background_data,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(explainer_data, filepath)
        logger.info(f"SHAP explainer saved to {filepath}")
    
    def load_explainer(self, filepath: str):
        """Load explainer from file"""
        
        explainer_data = joblib.load(filepath)
        
        self.explainer = explainer_data['explainer']
        self.background_data = explainer_data['background_data']
        self.feature_names = explainer_data['feature_names']
        self.config = explainer_data.get('config', {})
        
        logger.info(f"SHAP explainer loaded from {filepath}")
    
    def generate_explanation_report(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive explanation report"""
        
        explanation = self.explain_instance(instance)
        force_plot_data = self.generate_force_plot_data(instance)
        waterfall_data = self.generate_waterfall_data(instance)
        
        report = {
            'instance_explanation': explanation,
            'visualizations': {
                'force_plot': force_plot_data,
                'waterfall': waterfall_data
            },
            'summary': {
                'prediction': explanation.get('prediction', 0),
                'confidence': 'high' if abs(explanation.get('total_contribution', 0)) > 0.1 else 'low',
                'top_positive_features': [],
                'top_negative_features': []
            },
            'report_timestamp': datetime.now().isoformat()
        }
        
        # Extract top positive and negative features
        if 'feature_contributions' in explanation:
            contributions = explanation['feature_contributions']
            
            positive_features = [(k, v) for k, v in contributions.items() if v['shap_value'] > 0]
            negative_features = [(k, v) for k, v in contributions.items() if v['shap_value'] < 0]
            
            positive_features.sort(key=lambda x: x[1]['shap_value'], reverse=True)
            negative_features.sort(key=lambda x: x[1]['shap_value'])
            
            report['summary']['top_positive_features'] = positive_features[:5]
            report['summary']['top_negative_features'] = negative_features[:5]
        
        return report
