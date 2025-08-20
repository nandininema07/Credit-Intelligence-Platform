"""
Real-time credit risk scoring engine with event-driven updates.
Provides fast, low-latency scoring for credit risk assessment and real-time event impact calculation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
        self.feature_processors = {}
        self.score_cache = {}
        self.cache_ttl = config.get('cache_ttl_seconds', 300)  # 5 minutes default
        
        # Event impact weights for different event types
        self.event_impact_weights = {
            'debt_restructuring': -0.8,
            'earnings_warning': -0.6,
            'credit_downgrade': -0.9,
            'regulatory_action': -0.7,
            'bankruptcy_filing': -1.0,
            'liquidity_crisis': -0.8,
            'positive_earnings': 0.4,
            'new_contract': 0.3,
            'expansion_news': 0.2
        }
        
        # Load models and processors
        asyncio.create_task(self._load_models())
    
    async def _load_models(self):
        """Load pre-trained models and feature processors"""
        try:
            model_path = self.config.get('model_path', 'models/')
            
            # Load different model types
            self.models['xgboost'] = joblib.load(f"{model_path}/xgboost_model.joblib")
            self.models['ensemble'] = joblib.load(f"{model_path}/ensemble_model.joblib")
            
            # Load feature processors
            self.feature_processors['scaler'] = joblib.load(f"{model_path}/feature_scaler.joblib")
            self.feature_processors['selector'] = joblib.load(f"{model_path}/feature_selector.joblib")
            
            logger.info("Models and processors loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Use dummy models for development
            self._load_dummy_models()
    
    def _load_dummy_models(self):
        # Load dummy models for development
        self.models['xgboost'] = None
        self.models['ensemble'] = None
        self.feature_processors['scaler'] = None
        self.feature_processors['selector'] = None
        logger.warning("Using dummy models for development")
        
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
    
    async def calculate_event_impact(self, company_ticker: str, event_data: Any) -> float:
        """Calculate the impact of a detected event on credit score"""
        try:
            event_type = getattr(event_data, 'event_type', None)
            if not event_type:
                return 0.0
            
            # Get base impact weight
            base_impact = self.event_impact_weights.get(event_type.value, 0.0)
            
            # Adjust by confidence and severity
            confidence = getattr(event_data, 'confidence_score', 0.5)
            severity_multiplier = {
                'critical': 1.0,
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            }.get(getattr(event_data, 'severity', 'medium').value, 0.6)
            
            # Calculate final impact
            impact = base_impact * confidence * severity_multiplier * 10.0  # Scale to score points
            
            # Apply time decay (recent events have more impact)
            event_age = datetime.now() - getattr(event_data, 'published_at', datetime.now())
            decay_factor = max(0.1, 1.0 - (event_age.days / 7.0))  # 7-day decay
            
            final_impact = impact * decay_factor
            
            logger.info(f"Event impact for {company_ticker}: {final_impact:.2f} points")
            return final_impact
            
        except Exception as e:
            logger.error(f"Error calculating event impact: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'loaded_models': list(self.models.keys()),
            'model_count': len(self.models),
            'has_scalers': list(self.scalers.keys()),
            'is_ready': self.is_loaded
        }
        return info
