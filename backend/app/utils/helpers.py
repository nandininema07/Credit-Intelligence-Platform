"""
Helper utilities and common functions
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import hashlib
import json
import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def generate_hash(data: Union[str, Dict[str, Any]]) -> str:
    """Generate MD5 hash for data"""
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """Format percentage value"""
    return f"{value:.{decimal_places}f}%"

def format_large_number(number: float) -> str:
    """Format large numbers with suffixes (K, M, B, T)"""
    if number >= 1e12:
        return f"{number/1e12:.1f}T"
    elif number >= 1e9:
        return f"{number/1e9:.1f}B"
    elif number >= 1e6:
        return f"{number/1e6:.1f}M"
    elif number >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:.0f}"

def calculate_time_ago(timestamp: datetime) -> str:
    """Calculate human-readable time ago string"""
    now = datetime.utcnow()
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "Just now"

def paginate_results(items: List[Any], page: int, size: int) -> Dict[str, Any]:
    """Paginate a list of items"""
    total = len(items)
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    
    paginated_items = items[start_idx:end_idx]
    total_pages = (total + size - 1) // size
    
    return {
        "items": paginated_items,
        "total": total,
        "page": page,
        "size": size,
        "pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }

def retry_async(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            
            raise last_exception
        
        return wrapper
    return decorator

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def extract_domain_from_email(email: str) -> Optional[str]:
    """Extract domain from email address"""
    try:
        return email.split('@')[1].lower()
    except (IndexError, AttributeError):
        return None

def generate_session_id() -> str:
    """Generate unique session ID"""
    import uuid
    return str(uuid.uuid4())

def calculate_score_change_percentage(current: float, previous: float) -> float:
    """Calculate percentage change between two scores"""
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100

def get_score_range_label(score: float) -> str:
    """Get score range label based on score value"""
    if score >= 90:
        return "excellent"
    elif score >= 80:
        return "good"
    elif score >= 70:
        return "fair"
    elif score >= 60:
        return "poor"
    else:
        return "very_poor"

def get_trend_direction(current: float, previous: float, threshold: float = 2.0) -> str:
    """Determine trend direction based on score changes"""
    change = current - previous
    
    if abs(change) < threshold:
        return "stable"
    elif change > 0:
        return "up"
    else:
        return "down"

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed based on rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside the window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[key]) < max_requests:
            self.requests[key].append(now)
            return True
        
        return False

def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """Mask sensitive data showing only last few characters"""
    if len(data) <= visible_chars:
        return mask_char * len(data)
    
    masked_length = len(data) - visible_chars
    return mask_char * masked_length + data[-visible_chars:]
