"""
Real-time credit risk scoring engine.
Provides fast, low-latency scoring for credit risk assessment.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ScoringRequest:
    """Credit scoring request"""
    company_id: str
    features: Dict[str, float]
    timestamp: datetime = None
    request_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = f"{self.company_id}_{self.timestamp.isoformat()}"

@dataclass
class ScoringResult:
    """Credit scoring result"""
    company_id: str
    credit_score: float
    risk_category: str
    probability_default: float
    confidence_score: float
    feature_contributions: Dict[str, float]
    model_used: str
    timestamp: datetime
    request_id: str

class RealTimeScorer:
    """Real-time credit risk scorer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.is_loaded = False
        
    async def load_models(self, model_path: str = './models/'):
        """Load trained models for scoring"""
        import os
        
        model_files = [f for f in os.listdir(model_path) if f.endswith('.pkl') and not f.endswith('_scaler.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            model_filepath = os.path.join(model_path, model_file)
            
            try:
                model = joblib.load(model_filepath)
                self.models[model_name] = model
                
                # Load scaler if exists
                scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
                if os.path.exists(scaler_file):
                    scaler = joblib.load(scaler_file)
                    self.scalers[model_name] = scaler
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
        
        if self.models:
            self.is_loaded = True
            logger.info(f"Loaded {len(self.models)} models for scoring")
        else:
            logger.warning("No models loaded")
    
    def get_risk_category(self, credit_score: float) -> str:
        """Convert credit score to risk category"""
        if credit_score >= 750:
            return "Low Risk"
        elif credit_score >= 650:
            return "Medium Risk"
        elif credit_score >= 550:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def calculate_confidence_score(self, probabilities: np.ndarray) -> float:
        """Calculate confidence score based on prediction probabilities"""
        if len(probabilities) == 2:
            # Binary classification - confidence is max probability
            return float(np.max(probabilities))
        else:
            # Multi-class - use entropy-based confidence
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(len(probabilities))
            confidence = 1 - (entropy / max_entropy)
            return float(confidence)
    
    def calculate_feature_contributions(self, model: Any, features: np.ndarray, 
                                     feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature contributions to the prediction"""
        contributions = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                for name, importance, value in zip(feature_names, importances, features[0]):
                    contributions[name] = float(importance * value)
                    
            elif hasattr(model, 'coef_'):
                # Linear models
                coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                for name, coef, value in zip(feature_names, coefficients, features[0]):
                    contributions[name] = float(coef * value)
                    
        except Exception as e:
            logger.warning(f"Could not calculate feature contributions: {str(e)}")
        
        return contributions
    
    async def score_single(self, request: ScoringRequest, model_name: str = None) -> ScoringResult:
        """Score a single company using specified or best model"""
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        # Use best model if not specified
        if model_name is None:
            model_name = list(self.models.keys())[0]  # Use first available model
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Prepare features
        feature_df = pd.DataFrame([request.features])
        
        # Handle missing features
        expected_features = getattr(model, 'feature_names_in_', feature_df.columns)
        for feature in expected_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0.0  # Default value for missing features
        
        # Reorder columns to match training
        feature_df = feature_df.reindex(columns=expected_features, fill_value=0.0)
        
        # Scale features if scaler exists
        if model_name in self.scalers:
            features_scaled = self.scalers[model_name].transform(feature_df)
        else:
            features_scaled = feature_df.values
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            probability_default = float(probabilities[1]) if len(probabilities) == 2 else float(probabilities[prediction])
        else:
            probability_default = float(prediction)
        
        # Convert to credit score (300-850 scale)
        credit_score = 850 - (probability_default * 550)  # Inverse relationship
        credit_score = max(300, min(850, credit_score))
        
        # Calculate confidence
        confidence_score = self.calculate_confidence_score(probabilities) if hasattr(model, 'predict_proba') else 0.8
        
        # Get feature contributions
        feature_contributions = self.calculate_feature_contributions(
            model, features_scaled, feature_df.columns.tolist()
        )
        
        # Determine risk category
        risk_category = self.get_risk_category(credit_score)
        
        return ScoringResult(
            company_id=request.company_id,
            credit_score=credit_score,
            risk_category=risk_category,
            probability_default=probability_default,
            confidence_score=confidence_score,
            feature_contributions=feature_contributions,
            model_used=model_name,
            timestamp=datetime.now(),
            request_id=request.request_id
        )
    
    async def score_batch(self, requests: List[ScoringRequest], 
                         model_name: str = None) -> List[ScoringResult]:
        """Score multiple companies in batch"""
        results = []
        
        for request in requests:
            try:
                result = await self.score_single(request, model_name)
                results.append(result)
            except Exception as e:
                logger.error(f"Error scoring {request.company_id}: {str(e)}")
                # Create error result
                error_result = ScoringResult(
                    company_id=request.company_id,
                    credit_score=500.0,  # Neutral score
                    risk_category="Unknown",
                    probability_default=0.5,
                    confidence_score=0.0,
                    feature_contributions={},
                    model_used="error",
                    timestamp=datetime.now(),
                    request_id=request.request_id
                )
                results.append(error_result)
        
        logger.info(f"Scored {len(results)} companies")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self.models.keys()),
            'model_count': len(self.models),
            'has_scalers': list(self.scalers.keys()),
            'is_ready': self.is_loaded
        }
        return info
