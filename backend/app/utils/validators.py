"""
Validation utilities and helpers
"""

from typing import Any, Dict, List, Optional
import re
from datetime import datetime
from pydantic import validator

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_ticker(ticker: str) -> bool:
    """Validate stock ticker format"""
    if not ticker:
        return False
    # Ticker should be 1-10 uppercase letters
    pattern = r'^[A-Z]{1,10}$'
    return re.match(pattern, ticker.upper()) is not None

def validate_score_range(score: float) -> bool:
    """Validate credit score is in valid range"""
    return 0 <= score <= 100

def validate_company_name(name: str) -> bool:
    """Validate company name"""
    if not name or len(name.strip()) < 2:
        return False
    # Allow letters, numbers, spaces, and common business symbols
    pattern = r'^[a-zA-Z0-9\s\.\,\&\-\(\)]+$'
    return re.match(pattern, name) is not None

def validate_phone_number(phone: str) -> bool:
    """Validate phone number format"""
    # Simple validation for international phone numbers
    pattern = r'^\+?[1-9]\d{1,14}$'
    return re.match(pattern, phone.replace(' ', '').replace('-', '')) is not None

def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """Validate date range"""
    return start_date <= end_date

def validate_pagination(page: int, size: int) -> Dict[str, Any]:
    """Validate pagination parameters"""
    errors = []
    
    if page < 1:
        errors.append("Page must be greater than 0")
    
    if size < 1:
        errors.append("Size must be greater than 0")
    elif size > 100:
        errors.append("Size cannot exceed 100")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def sanitize_string(value: str, max_length: int = None) -> str:
    """Sanitize string input"""
    if not value:
        return ""
    
    # Strip whitespace
    sanitized = value.strip()
    
    # Truncate if max_length specified
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    """Validate JSON structure has required fields"""
    errors = []
    
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif data[field] is None:
            errors.append(f"Field cannot be null: {field}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

class ScoreValidator:
    """Validator for credit scores"""
    
    @staticmethod
    def validate_score_data(score: float, confidence: Optional[float] = None) -> Dict[str, Any]:
        """Validate score data"""
        errors = []
        
        if not validate_score_range(score):
            errors.append("Score must be between 0 and 100")
        
        if confidence is not None and not (0 <= confidence <= 1):
            errors.append("Confidence must be between 0 and 1")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

class AlertValidator:
    """Validator for alerts"""
    
    @staticmethod
    def validate_alert_data(title: str, message: str, severity: str) -> Dict[str, Any]:
        """Validate alert data"""
        errors = []
        
        if not title or len(title.strip()) < 3:
            errors.append("Alert title must be at least 3 characters")
        
        if not message or len(message.strip()) < 10:
            errors.append("Alert message must be at least 10 characters")
        
        valid_severities = ["critical", "high", "medium", "low"]
        if severity not in valid_severities:
            errors.append(f"Severity must be one of: {', '.join(valid_severities)}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
