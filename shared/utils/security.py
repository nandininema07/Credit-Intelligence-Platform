"""
Security utilities for the Credit Intelligence Platform.
"""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging
import os
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)

class SecurityManager:
    """Security management utilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.secret_key = config.get('secret_key', self._generate_secret_key())
        self.encryption_key = config.get('encryption_key', self._generate_encryption_key())
        self.cipher = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
    
    def _generate_secret_key(self) -> str:
        """Generate secure secret key"""
        return secrets.token_urlsafe(32)
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt, password_hash = hashed.split(':')
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return computed_hash.hex() == password_hash
        except:
            return False
    
    def generate_token(self, payload: Dict[str, Any], expires_hours: int = 24) -> str:
        """Generate JWT token"""
        payload['exp'] = datetime.utcnow() + timedelta(hours=expires_hours)
        payload['iat'] = datetime.utcnow()
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize user input"""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`']
        sanitized = input_data
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized.strip()
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or len(api_key) < 16:
            return False
        
        # Check for basic format requirements
        if not any(c.isalnum() for c in api_key):
            return False
        
        return True
