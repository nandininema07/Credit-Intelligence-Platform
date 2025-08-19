"""
Authentication utilities and dependencies
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
import logging

from app.config import get_settings
from app.middleware.auth import verify_token

logger = logging.getLogger(__name__)
security = HTTPBearer()
settings = get_settings()

async def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current authenticated user from request state"""
    try:
        if hasattr(request.state, 'user'):
            return request.state.user
        
        # Fallback for development
        if settings.DEBUG:
            return {
                "id": "dev_user",
                "username": "developer",
                "email": "dev@example.com",
                "roles": ["admin"]
            }
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated"
        )
        
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

async def get_current_user_ws(token: str) -> Dict[str, Any]:
    """Get current user for WebSocket connections"""
    try:
        if settings.DEBUG:
            return {
                "id": "dev_user",
                "username": "developer",
                "email": "dev@example.com",
                "roles": ["admin"]
            }
        
        payload = verify_token(token)
        return {
            "id": payload.get("sub"),
            "username": payload.get("username"),
            "email": payload.get("email"),
            "roles": payload.get("roles", [])
        }
        
    except Exception as e:
        logger.error(f"Error authenticating WebSocket user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid WebSocket authentication"
        )

def require_roles(required_roles: list):
    """Decorator to require specific roles"""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_roles = current_user.get("roles", [])
        
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return current_user
    
    return role_checker

async def get_admin_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current user with admin role requirement"""
    if "admin" not in current_user.get("roles", []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user
