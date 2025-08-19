"""
Exception handlers and custom exceptions
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from typing import Union

logger = logging.getLogger(__name__)

class CreditIntelligenceException(Exception):
    """Base exception for Credit Intelligence Platform"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class CompanyNotFoundException(CreditIntelligenceException):
    """Exception raised when company is not found"""
    def __init__(self, company_id: int):
        super().__init__(f"Company with ID {company_id} not found", "COMPANY_NOT_FOUND")

class ScoreNotFoundException(CreditIntelligenceException):
    """Exception raised when score is not found"""
    def __init__(self, company_id: int):
        super().__init__(f"Score not found for company {company_id}", "SCORE_NOT_FOUND")

class AlertNotFoundException(CreditIntelligenceException):
    """Exception raised when alert is not found"""
    def __init__(self, alert_id: int):
        super().__init__(f"Alert with ID {alert_id} not found", "ALERT_NOT_FOUND")

class DataIngestionException(CreditIntelligenceException):
    """Exception raised during data ingestion"""
    def __init__(self, message: str):
        super().__init__(message, "DATA_INGESTION_ERROR")

class ModelPredictionException(CreditIntelligenceException):
    """Exception raised during model prediction"""
    def __init__(self, message: str):
        super().__init__(message, "MODEL_PREDICTION_ERROR")

class ExplanationGenerationException(CreditIntelligenceException):
    """Exception raised during explanation generation"""
    def __init__(self, message: str):
        super().__init__(message, "EXPLANATION_GENERATION_ERROR")

class NotificationException(CreditIntelligenceException):
    """Exception raised during notification sending"""
    def __init__(self, message: str):
        super().__init__(message, "NOTIFICATION_ERROR")

def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""
    
    @app.exception_handler(CreditIntelligenceException)
    async def credit_intelligence_exception_handler(request: Request, exc: CreditIntelligenceException):
        """Handle custom Credit Intelligence exceptions"""
        logger.error(f"Credit Intelligence error: {exc.message}")
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.error_code or "CREDIT_INTELLIGENCE_ERROR",
                "message": exc.message,
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "request_id": getattr(request.state, 'request_id', None)
            }
        )
