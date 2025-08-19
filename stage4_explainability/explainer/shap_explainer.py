"""
SHAP-based model explainer for credit risk models.
Provides detailed explanations for model predictions using SHAP values.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)

@dataclass
class ExplanationResult:
    """SHAP explanation result"""
    company_id: str
    prediction: float
    base_value: float
    shap_values: Dict[str, float]
    feature_values: Dict[str, float]
    top_positive_features: List[Tuple[str, float]]
    top_negative_features: List[Tuple[str, float]]
    explanation_text: str
    timestamp: datetime

class SHAPExplainer:
    """SHAP-based model explainer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.explainers = {}
        self.background_data = None
        self.feature_names = []
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for model explanations")
    
    async def initialize_explainer(self, model: Any, background_data: pd.DataFrame, 
                                 model_name: str = "default"):
        """Initialize SHAP explainer for a model"""
        logger.info(f"Initializing SHAP explainer for {model_name}")
        
        try:
            # Store background data
            self.background_data = background_data.sample(min(100, len(background_data)))
            self.feature_names = background_data.columns.tolist()
            
            # Choose appropriate explainer based on model type
            model_type = type(model).__name__.lower()
            
            if 'tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgb' in model_type:
                # Tree-based models
                explainer = shap.TreeExplainer(model)
            elif 'linear' in model_type or 'logistic' in model_type:
                # Linear models
                explainer = shap.LinearExplainer(model, self.background_data)
            else:
                # General explainer (slower but works for any model)
                explainer = shap.Explainer(model, self.background_data)
            
            self.explainers[model_name] = explainer
            logger.info(f"SHAP explainer initialized for {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing SHAP explainer: {str(e)}")
            raise
    
    def format_explanation_text(self, shap_values: Dict[str, float], 
                              feature_values: Dict[str, float],
                              prediction: float) -> str:
        """Generate human-readable explanation text"""
        
        # Sort features by absolute SHAP value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        explanation = f"Credit Score Prediction: {prediction:.0f}\n\n"
        explanation += "Key factors influencing this score:\n\n"
        
        for i, (feature, shap_val) in enumerate(sorted_features[:5]):
            feature_val = feature_values.get(feature, 0)
            impact = "increases" if shap_val > 0 else "decreases"
            
            explanation += f"{i+1}. {feature.replace('_', ' ').title()}: {feature_val:.2f}\n"
            explanation += f"   This {impact} the credit score by {abs(shap_val):.1f} points\n\n"
        
        return explanation
    
    async def explain_prediction(self, model: Any, features: pd.DataFrame, 
                               company_id: str, model_name: str = "default") -> ExplanationResult:
        """Generate SHAP explanation for a single prediction"""
        
        if model_name not in self.explainers:
            raise ValueError(f"Explainer for {model_name} not initialized")
        
        explainer = self.explainers[model_name]
        
        try:
            # Get SHAP values
            shap_values = explainer.shap_values(features)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-class or binary classification
                shap_vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                shap_vals = shap_values
            
            # Get base value
            if hasattr(explainer, 'expected_value'):
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[1] if len(base_value) == 2 else base_value[0]
            else:
                base_value = 0.0
            
            # Convert to dictionaries
            shap_dict = {}
            feature_dict = {}
            
            for i, feature in enumerate(self.feature_names):
                shap_dict[feature] = float(shap_vals[0][i])
                feature_dict[feature] = float(features.iloc[0, i])
            
            # Get prediction
            prediction = model.predict(features)[0]
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(features)[0][1]  # Probability of positive class
            
            # Convert to credit score scale
            credit_score = 850 - (prediction * 550)
            credit_score = max(300, min(850, credit_score))
            
            # Get top positive and negative features
            sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
            top_positive = [(k, v) for k, v in sorted_shap if v > 0][:5]
            top_negative = [(k, v) for k, v in sorted_shap if v < 0][-5:]
            
            # Generate explanation text
            explanation_text = self.format_explanation_text(shap_dict, feature_dict, credit_score)
            
            return ExplanationResult(
                company_id=company_id,
                prediction=credit_score,
                base_value=float(base_value),
                shap_values=shap_dict,
                feature_values=feature_dict,
                top_positive_features=top_positive,
                top_negative_features=top_negative,
                explanation_text=explanation_text,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            raise
    
    async def explain_batch(self, model: Any, features_df: pd.DataFrame,
                          company_ids: List[str], model_name: str = "default") -> List[ExplanationResult]:
        """Generate SHAP explanations for multiple predictions"""
        
        results = []
        
        for i, company_id in enumerate(company_ids):
            try:
                single_features = features_df.iloc[[i]]
                result = await self.explain_prediction(model, single_features, company_id, model_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error explaining prediction for {company_id}: {str(e)}")
        
        logger.info(f"Generated explanations for {len(results)} predictions")
        return results
    
    def get_global_feature_importance(self, model_name: str = "default") -> Dict[str, float]:
        """Get global feature importance from SHAP values"""
        
        if model_name not in self.explainers:
            return {}
        
        try:
            explainer = self.explainers[model_name]
            
            # Calculate SHAP values for background data
            shap_values = explainer.shap_values(self.background_data)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            else:
                shap_vals = shap_values
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_vals), axis=0)
            
            importance_dict = {}
            for i, feature in enumerate(self.feature_names):
                importance_dict[feature] = float(mean_shap[i])
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error calculating global feature importance: {str(e)}")
            return {}
    
    def save_explanation(self, result: ExplanationResult, save_path: str):
        """Save explanation result to file"""
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        filename = f"{result.company_id}_{result.timestamp.isoformat()}.json"
        filepath = os.path.join(save_path, filename)
        
        # Convert to serializable format
        data = {
            'company_id': result.company_id,
            'prediction': result.prediction,
            'base_value': result.base_value,
            'shap_values': result.shap_values,
            'feature_values': result.feature_values,
            'top_positive_features': result.top_positive_features,
            'top_negative_features': result.top_negative_features,
            'explanation_text': result.explanation_text,
            'timestamp': result.timestamp.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved explanation to {filepath}")
