"""
Model serving API for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime
import joblib
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger(__name__)

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, Any]
    model_id: Optional[str] = None
    return_probabilities: bool = True
    return_explanations: bool = False

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    probability: Optional[float] = None
    model_id: str
    timestamp: str
    explanations: Optional[Dict[str, Any]] = None
    uncertainty: Optional[Dict[str, Any]] = None

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features_list: List[Dict[str, Any]]
    model_id: Optional[str] = None
    return_probabilities: bool = True
    return_explanations: bool = False

class ModelServingAPI:
    """FastAPI-based model serving API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.app = FastAPI(title="Credit Intelligence Model API", version="1.0.0")
        self.models = {}
        self.prediction_cache = None
        self.load_balancer = None
        self._setup_middleware()
        self._setup_routes()
        
    def _setup_middleware(self):
        """Setup API middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/ready")
        async def readiness_check():
            return {
                "status": "ready",
                "loaded_models": list(self.models.keys()),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            return await self._handle_prediction(request)
        
        @self.app.post("/predict/batch")
        async def batch_predict(request: BatchPredictionRequest):
            return await self._handle_batch_prediction(request)
        
        @self.app.get("/models")
        async def list_models():
            return {
                "models": [
                    {
                        "model_id": model_id,
                        "loaded_at": info["loaded_at"],
                        "model_type": info["model_type"]
                    }
                    for model_id, info in self.models.items()
                ]
            }
        
        @self.app.post("/models/{model_id}/load")
        async def load_model(model_id: str):
            return await self._load_model(model_id)
        
        @self.app.delete("/models/{model_id}")
        async def unload_model(model_id: str):
            return await self._unload_model(model_id)
        
        @self.app.get("/models/{model_id}/info")
        async def get_model_info(model_id: str):
            return await self._get_model_info(model_id)
    
    async def _handle_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Handle single prediction request"""
        
        # Get model
        model_id = request.model_id or self._get_default_model()
        
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")
        
        model_info = self.models[model_id]
        model = model_info["model"]
        
        try:
            # Prepare features
            feature_df = pd.DataFrame([request.features])
            
            # Check cache if available
            if self.prediction_cache:
                cached_result = await self.prediction_cache.get_prediction(
                    model_id, request.features
                )
                if cached_result:
                    return PredictionResponse(**cached_result)
            
            # Make prediction
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_df)[0]
                prediction = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                probability = prediction
            else:
                prediction = model.predict(feature_df)[0]
                probability = None
            
            # Get explanations if requested
            explanations = None
            if request.return_explanations:
                explanations = await self._get_explanations(model, feature_df, model_id)
            
            # Get uncertainty if available
            uncertainty = None
            if hasattr(model, 'predict_uncertainty'):
                uncertainty = model.predict_uncertainty(feature_df)[0]
            
            response = PredictionResponse(
                prediction=float(prediction),
                probability=float(probability) if probability is not None else None,
                model_id=model_id,
                timestamp=datetime.now().isoformat(),
                explanations=explanations,
                uncertainty=uncertainty
            )
            
            # Cache result if cache available
            if self.prediction_cache:
                await self.prediction_cache.cache_prediction(
                    model_id, request.features, response.dict()
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    async def _handle_batch_prediction(self, request: BatchPredictionRequest) -> Dict[str, Any]:
        """Handle batch prediction request"""
        
        model_id = request.model_id or self._get_default_model()
        
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")
        
        model_info = self.models[model_id]
        model = model_info["model"]
        
        try:
            # Prepare features
            feature_df = pd.DataFrame(request.features_list)
            
            # Make batch predictions
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(feature_df)
                predictions = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
            else:
                predictions = model.predict(feature_df)
                probabilities = None
            
            # Prepare response
            results = []
            for i, (pred, features) in enumerate(zip(predictions, request.features_list)):
                result = {
                    "index": i,
                    "prediction": float(pred),
                    "probability": float(probabilities[i, 1]) if probabilities is not None and probabilities.shape[1] > 1 else None,
                    "features": features
                }
                results.append(result)
            
            return {
                "predictions": results,
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "batch_size": len(request.features_list)
            }
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    async def _load_model(self, model_id: str) -> Dict[str, Any]:
        """Load model into memory"""
        
        try:
            # This would typically load from model registry
            # For now, simulate loading
            model = joblib.load(f"./models/{model_id}.pkl")
            
            self.models[model_id] = {
                "model": model,
                "loaded_at": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "prediction_count": 0
            }
            
            logger.info(f"Loaded model {model_id}")
            
            return {
                "success": True,
                "model_id": model_id,
                "loaded_at": self.models[model_id]["loaded_at"]
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    async def _unload_model(self, model_id: str) -> Dict[str, Any]:
        """Unload model from memory"""
        
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")
        
        del self.models[model_id]
        
        logger.info(f"Unloaded model {model_id}")
        
        return {
            "success": True,
            "model_id": model_id,
            "unloaded_at": datetime.now().isoformat()
        }
    
    async def _get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information"""
        
        if model_id not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not loaded")
        
        model_info = self.models[model_id]
        
        return {
            "model_id": model_id,
            "model_type": model_info["model_type"],
            "loaded_at": model_info["loaded_at"],
            "prediction_count": model_info["prediction_count"],
            "memory_usage": self._get_model_memory_usage(model_id)
        }
    
    def _get_default_model(self) -> str:
        """Get default model ID"""
        
        if not self.models:
            raise HTTPException(status_code=404, detail="No models loaded")
        
        # Return first loaded model as default
        return list(self.models.keys())[0]
    
    async def _get_explanations(self, model: Any, feature_df: pd.DataFrame, 
                              model_id: str) -> Dict[str, Any]:
        """Get model explanations"""
        
        try:
            # Try SHAP explanations
            if hasattr(model, 'get_explanations'):
                return model.get_explanations(feature_df)
            
            # Try feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    feature_df.columns,
                    model.feature_importances_
                ))
                return {"feature_importance": feature_importance}
            
            # Try coefficients for linear models
            if hasattr(model, 'coef_'):
                coefficients = dict(zip(
                    feature_df.columns,
                    model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                ))
                return {"coefficients": coefficients}
            
            return {"message": "No explanations available for this model"}
            
        except Exception as e:
            logger.warning(f"Failed to get explanations: {str(e)}")
            return {"error": "Explanation generation failed"}
    
    def _get_model_memory_usage(self, model_id: str) -> Dict[str, Any]:
        """Get model memory usage (simplified)"""
        
        # This would typically use memory profiling tools
        return {
            "estimated_mb": 50,  # Placeholder
            "last_checked": datetime.now().isoformat()
        }
    
    def set_prediction_cache(self, cache):
        """Set prediction cache instance"""
        self.prediction_cache = cache
    
    def set_load_balancer(self, load_balancer):
        """Set load balancer instance"""
        self.load_balancer = load_balancer
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the serving API server"""
        
        logger.info(f"Starting model serving API on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)
    
    async def warm_up_models(self, model_ids: List[str]):
        """Warm up models by loading them"""
        
        for model_id in model_ids:
            try:
                await self._load_model(model_id)
            except Exception as e:
                logger.error(f"Failed to warm up model {model_id}: {str(e)}")
    
    def get_api_stats(self) -> Dict[str, Any]:
        """Get API statistics"""
        
        total_predictions = sum(info["prediction_count"] for info in self.models.values())
        
        return {
            "loaded_models": len(self.models),
            "total_predictions": total_predictions,
            "uptime": "calculated_uptime",  # Would calculate actual uptime
            "memory_usage": "calculated_memory",  # Would calculate actual memory
            "timestamp": datetime.now().isoformat()
        }
