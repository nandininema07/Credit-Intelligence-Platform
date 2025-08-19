"""
LIME-based explanations for Stage 4.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class LIMEExplainer:
    """LIME-based model explanations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainer = None
        self.training_data = None
        self.feature_names = None
        self.categorical_features = None
        self.model = None
        
    def initialize_explainer(self, model: Any, training_data: pd.DataFrame,
                           categorical_features: List[str] = None,
                           mode: str = 'classification'):
        """Initialize LIME explainer"""
        
        self.model = model
        self.training_data = training_data
        self.feature_names = list(training_data.columns)
        self.categorical_features = categorical_features or []
        
        # Get categorical feature indices
        categorical_indices = []
        if categorical_features:
            categorical_indices = [
                self.feature_names.index(feat) for feat in categorical_features
                if feat in self.feature_names
            ]
        
        try:
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data.values,
                feature_names=self.feature_names,
                categorical_features=categorical_indices,
                mode=mode,
                discretize_continuous=self.config.get('discretize_continuous', True),
                random_state=self.config.get('random_state', 42)
            )
            
            logger.info("Initialized LIME tabular explainer")
            
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {e}")
            raise
    
    def explain_instance(self, instance: pd.DataFrame,
                        num_features: int = 10,
                        num_samples: int = 5000) -> Dict[str, Any]:
        """Explain a single instance using LIME"""
        
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer first.")
        
        try:
            # Convert instance to numpy array
            instance_array = instance.iloc[0].values
            
            # Get prediction function
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = lambda x: np.column_stack([1 - self.model.predict(x), self.model.predict(x)])
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                instance_array,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            return self._format_lime_explanation(explanation, instance)
            
        except Exception as e:
            logger.error(f"Error explaining instance with LIME: {e}")
            return {'error': str(e)}
    
    def _format_lime_explanation(self, explanation, instance: pd.DataFrame) -> Dict[str, Any]:
        """Format LIME explanation as dictionary"""
        
        # Get feature contributions
        feature_contributions = {}
        explanation_list = explanation.as_list()
        
        for feature_desc, contribution in explanation_list:
            # Parse feature description to get feature name
            feature_name = self._parse_feature_description(feature_desc)
            
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                feature_value = instance.iloc[0, feature_idx]
                
                feature_contributions[feature_name] = {
                    'value': float(feature_value),
                    'contribution': float(contribution),
                    'description': feature_desc,
                    'abs_contribution': float(abs(contribution))
                }
        
        # Get prediction probabilities
        if hasattr(explanation, 'predict_proba'):
            prediction_proba = explanation.predict_proba
        else:
            prediction_proba = [0.5, 0.5]  # Default for binary classification
        
        # Sort features by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]['abs_contribution'],
            reverse=True
        )
        
        return {
            'prediction_probability': float(prediction_proba[1]) if len(prediction_proba) > 1 else float(prediction_proba[0]),
            'feature_contributions': feature_contributions,
            'top_features': dict(sorted_features),
            'explanation_score': float(explanation.score),
            'local_prediction': float(explanation.local_pred[1]) if len(explanation.local_pred) > 1 else float(explanation.local_pred[0]),
            'explanation_timestamp': datetime.now().isoformat()
        }
    
    def _parse_feature_description(self, feature_desc: str) -> str:
        """Parse LIME feature description to extract feature name"""
        
        # LIME descriptions can be like "feature_name <= 5.0" or "feature_name > 10"
        # Extract the feature name part
        for feature_name in self.feature_names:
            if feature_name in feature_desc:
                return feature_name
        
        # If no exact match, try to extract from the beginning
        parts = feature_desc.split()
        if parts:
            return parts[0]
        
        return feature_desc
    
    def explain_batch(self, instances: pd.DataFrame,
                     num_features: int = 10,
                     num_samples: int = 5000) -> List[Dict[str, Any]]:
        """Explain multiple instances"""
        
        explanations = []
        
        for idx in range(len(instances)):
            instance = instances.iloc[[idx]]
            explanation = self.explain_instance(
                instance, num_features=num_features, num_samples=num_samples
            )
            explanation['instance_id'] = idx
            explanations.append(explanation)
        
        return explanations
    
    def get_feature_importance(self, test_instances: pd.DataFrame = None,
                             num_instances: int = 100,
                             num_features: int = 10) -> Dict[str, float]:
        """Get global feature importance using LIME"""
        
        if test_instances is None:
            # Sample from training data
            test_instances = self.training_data.sample(
                min(num_instances, len(self.training_data))
            )
        else:
            test_instances = test_instances.head(num_instances)
        
        # Collect feature contributions across instances
        feature_importance_sum = {name: 0.0 for name in self.feature_names}
        feature_count = {name: 0 for name in self.feature_names}
        
        for idx in range(len(test_instances)):
            try:
                instance = test_instances.iloc[[idx]]
                explanation = self.explain_instance(
                    instance, num_features=num_features, num_samples=1000
                )
                
                if 'error' not in explanation:
                    for feature_name, contrib_info in explanation['feature_contributions'].items():
                        feature_importance_sum[feature_name] += abs(contrib_info['contribution'])
                        feature_count[feature_name] += 1
                        
            except Exception as e:
                logger.warning(f"Failed to explain instance {idx}: {e}")
                continue
        
        # Calculate average importance
        feature_importance = {}
        for feature_name in self.feature_names:
            if feature_count[feature_name] > 0:
                feature_importance[feature_name] = feature_importance_sum[feature_name] / feature_count[feature_name]
            else:
                feature_importance[feature_name] = 0.0
        
        # Normalize
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {
                k: v / total_importance for k, v in feature_importance.items()
            }
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def generate_local_surrogate_model(self, instance: pd.DataFrame,
                                     num_samples: int = 5000) -> Dict[str, Any]:
        """Generate local surrogate model around instance"""
        
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        try:
            instance_array = instance.iloc[0].values
            
            # Generate neighborhood samples
            neighborhood_data, neighborhood_labels, distances = self.explainer.data_labels_distances(
                instance_array,
                lambda x: self.model.predict_proba(x)[:, 1] if hasattr(self.model, 'predict_proba') else self.model.predict(x),
                num_samples=num_samples,
                distance_metric='euclidean'
            )
            
            # Fit local linear model
            from sklearn.linear_model import Ridge
            local_model = Ridge(alpha=1.0)
            
            # Weight samples by distance (closer samples get higher weight)
            weights = np.exp(-distances)
            
            local_model.fit(neighborhood_data, neighborhood_labels, sample_weight=weights)
            
            # Get feature coefficients
            coefficients = {}
            for i, feature_name in enumerate(self.feature_names):
                coefficients[feature_name] = float(local_model.coef_[i])
            
            # Calculate R² score for local model
            from sklearn.metrics import r2_score
            local_predictions = local_model.predict(neighborhood_data)
            r2_score_local = r2_score(neighborhood_labels, local_predictions, sample_weight=weights)
            
            return {
                'local_model_coefficients': coefficients,
                'local_model_intercept': float(local_model.intercept_),
                'local_model_r2': float(r2_score_local),
                'neighborhood_size': len(neighborhood_data),
                'instance_prediction': float(self.model.predict_proba(instance.values)[0, 1]) if hasattr(self.model, 'predict_proba') else float(self.model.predict(instance.values)[0]),
                'local_prediction': float(local_model.predict(instance_array.reshape(1, -1))[0])
            }
            
        except Exception as e:
            logger.error(f"Error generating local surrogate model: {e}")
            return {'error': str(e)}
    
    def analyze_feature_stability(self, instance: pd.DataFrame,
                                num_explanations: int = 10,
                                num_samples: int = 1000) -> Dict[str, Any]:
        """Analyze stability of LIME explanations"""
        
        explanations = []
        
        # Generate multiple explanations with different random seeds
        for i in range(num_explanations):
            try:
                # Temporarily modify random state
                original_random_state = self.explainer.random_state
                self.explainer.random_state = i
                
                explanation = self.explain_instance(
                    instance, num_samples=num_samples
                )
                
                if 'error' not in explanation:
                    explanations.append(explanation)
                
                # Restore original random state
                self.explainer.random_state = original_random_state
                
            except Exception as e:
                logger.warning(f"Failed to generate explanation {i}: {e}")
                continue
        
        if not explanations:
            return {'error': 'No valid explanations generated'}
        
        # Analyze stability
        feature_stability = {}
        
        for feature_name in self.feature_names:
            contributions = []
            for exp in explanations:
                if feature_name in exp['feature_contributions']:
                    contributions.append(exp['feature_contributions'][feature_name]['contribution'])
            
            if contributions:
                feature_stability[feature_name] = {
                    'mean_contribution': float(np.mean(contributions)),
                    'std_contribution': float(np.std(contributions)),
                    'coefficient_of_variation': float(np.std(contributions) / np.mean(contributions)) if np.mean(contributions) != 0 else float('inf'),
                    'min_contribution': float(np.min(contributions)),
                    'max_contribution': float(np.max(contributions)),
                    'num_explanations': len(contributions)
                }
        
        # Overall stability score
        cv_values = [info['coefficient_of_variation'] for info in feature_stability.values() 
                    if not np.isinf(info['coefficient_of_variation'])]
        
        overall_stability = {
            'mean_cv': float(np.mean(cv_values)) if cv_values else float('inf'),
            'stable_features': [name for name, info in feature_stability.items() 
                              if info['coefficient_of_variation'] < 0.2],
            'unstable_features': [name for name, info in feature_stability.items() 
                                if info['coefficient_of_variation'] > 0.5]
        }
        
        return {
            'feature_stability': feature_stability,
            'overall_stability': overall_stability,
            'num_explanations_generated': len(explanations)
        }
    
    def compare_with_global_importance(self, instance: pd.DataFrame,
                                     global_importance: Dict[str, float]) -> Dict[str, Any]:
        """Compare local LIME explanation with global feature importance"""
        
        local_explanation = self.explain_instance(instance)
        
        if 'error' in local_explanation:
            return local_explanation
        
        comparison = {}
        
        for feature_name in self.feature_names:
            local_contrib = 0.0
            if feature_name in local_explanation['feature_contributions']:
                local_contrib = abs(local_explanation['feature_contributions'][feature_name]['contribution'])
            
            global_importance_val = global_importance.get(feature_name, 0.0)
            
            comparison[feature_name] = {
                'local_importance': local_contrib,
                'global_importance': global_importance_val,
                'importance_ratio': local_contrib / global_importance_val if global_importance_val > 0 else float('inf'),
                'agreement': 'high' if abs(local_contrib - global_importance_val) < 0.1 else 'low'
            }
        
        # Calculate overall agreement
        agreements = [info['agreement'] for info in comparison.values()]
        agreement_score = agreements.count('high') / len(agreements) if agreements else 0
        
        return {
            'feature_comparison': comparison,
            'overall_agreement_score': agreement_score,
            'highly_agreeing_features': [name for name, info in comparison.items() if info['agreement'] == 'high'],
            'disagreeing_features': [name for name, info in comparison.items() if info['agreement'] == 'low']
        }
    
    def generate_counterfactual_explanation(self, instance: pd.DataFrame,
                                          desired_prediction: float = None) -> Dict[str, Any]:
        """Generate counterfactual explanation using LIME insights"""
        
        local_explanation = self.explain_instance(instance)
        
        if 'error' in local_explanation:
            return local_explanation
        
        # Get current prediction
        current_prediction = local_explanation['prediction_probability']
        
        if desired_prediction is None:
            # Flip the prediction (0.5 threshold for binary classification)
            desired_prediction = 0.2 if current_prediction > 0.5 else 0.8
        
        # Identify features to modify based on LIME contributions
        feature_contributions = local_explanation['feature_contributions']
        
        # Sort features by contribution magnitude
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]['contribution']),
            reverse=True
        )
        
        counterfactual_changes = {}
        
        # Suggest changes for top contributing features
        for feature_name, contrib_info in sorted_features[:5]:
            current_value = contrib_info['value']
            contribution = contrib_info['contribution']
            
            # Suggest opposite change
            if contribution > 0 and current_prediction > desired_prediction:
                # Decrease this feature
                suggested_change = current_value * 0.8  # 20% decrease
                change_direction = 'decrease'
            elif contribution < 0 and current_prediction < desired_prediction:
                # Increase this feature
                suggested_change = current_value * 1.2  # 20% increase
                change_direction = 'increase'
            else:
                continue
            
            counterfactual_changes[feature_name] = {
                'current_value': current_value,
                'suggested_value': suggested_change,
                'change_direction': change_direction,
                'contribution_magnitude': abs(contribution),
                'expected_impact': 'positive' if (contribution > 0) == (desired_prediction > current_prediction) else 'negative'
            }
        
        return {
            'current_prediction': current_prediction,
            'desired_prediction': desired_prediction,
            'counterfactual_changes': counterfactual_changes,
            'explanation_timestamp': datetime.now().isoformat()
        }
    
    def save_explainer(self, filepath: str):
        """Save LIME explainer to file"""
        
        explainer_data = {
            'explainer': self.explainer,
            'training_data': self.training_data,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'config': self.config
        }
        
        joblib.dump(explainer_data, filepath)
        logger.info(f"LIME explainer saved to {filepath}")
    
    def load_explainer(self, filepath: str):
        """Load LIME explainer from file"""
        
        explainer_data = joblib.load(filepath)
        
        self.explainer = explainer_data['explainer']
        self.training_data = explainer_data['training_data']
        self.feature_names = explainer_data['feature_names']
        self.categorical_features = explainer_data['categorical_features']
        self.config = explainer_data.get('config', {})
        
        logger.info(f"LIME explainer loaded from {filepath}")
    
    def generate_explanation_report(self, instance: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive LIME explanation report"""
        
        # Get basic explanation
        explanation = self.explain_instance(instance)
        
        if 'error' in explanation:
            return explanation
        
        # Get local surrogate model
        surrogate_model = self.generate_local_surrogate_model(instance)
        
        # Analyze stability
        stability_analysis = self.analyze_feature_stability(instance, num_explanations=5)
        
        report = {
            'instance_explanation': explanation,
            'local_surrogate_model': surrogate_model,
            'stability_analysis': stability_analysis,
            'summary': {
                'prediction': explanation['prediction_probability'],
                'confidence': explanation['explanation_score'],
                'top_positive_features': [],
                'top_negative_features': [],
                'stability_score': stability_analysis.get('overall_stability', {}).get('mean_cv', float('inf'))
            },
            'report_timestamp': datetime.now().isoformat()
        }
        
        # Extract top positive and negative features
        contributions = explanation['feature_contributions']
        
        positive_features = [(k, v) for k, v in contributions.items() if v['contribution'] > 0]
        negative_features = [(k, v) for k, v in contributions.items() if v['contribution'] < 0]
        
        positive_features.sort(key=lambda x: x[1]['contribution'], reverse=True)
        negative_features.sort(key=lambda x: x[1]['contribution'])
        
        report['summary']['top_positive_features'] = positive_features[:3]
        report['summary']['top_negative_features'] = negative_features[:3]
        
        return report
