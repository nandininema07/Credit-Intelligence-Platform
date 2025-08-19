"""
Configuration management for Credit Intelligence Platform
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    APP_NAME: str = "Credit Intelligence Platform"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/credit_intelligence"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_POOL_SIZE: int = 10
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    
    # External APIs
    OPENAI_API_KEY: Optional[str] = None
    SLACK_BOT_TOKEN: Optional[str] = None
    SLACK_WEBHOOK_URL: Optional[str] = None
    TEAMS_WEBHOOK_URL: Optional[str] = None
    JIRA_URL: Optional[str] = None
    JIRA_USERNAME: Optional[str] = None
    JIRA_API_TOKEN: Optional[str] = None
    
    # Email settings
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_USE_TLS: bool = True
    
    # SMS settings
    TWILIO_ACCOUNT_SID: Optional[str] = None
    TWILIO_AUTH_TOKEN: Optional[str] = None
    TWILIO_PHONE_NUMBER: Optional[str] = None
    
    # Data processing settings
    MAX_COMPANIES_PER_BATCH: int = 100
    SCORE_UPDATE_INTERVAL: int = 300  # seconds
    ALERT_CHECK_INTERVAL: int = 60    # seconds
    
    # Model settings
    MODEL_REGISTRY_PATH: str = "./models"
    FEATURE_STORE_PATH: str = "./features"
    
    # Monitoring settings
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # File storage
    UPLOAD_DIR: str = "./uploads"
    EXPORT_DIR: str = "./exports"
    PDF_TEMPLATE_DIR: str = "./templates"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Stage-specific configurations
STAGE_CONFIGS = {
    "stage1_data_ingestion": {
        "scrapers": {
            "financial_data": {
                "enabled": True,
                "sources": ["yahoo_finance", "alpha_vantage", "quandl"],
                "update_frequency": 3600  # 1 hour
            },
            "news_data": {
                "enabled": True,
                "sources": ["reuters", "bloomberg", "financial_times"],
                "update_frequency": 1800  # 30 minutes
            },
            "regulatory_data": {
                "enabled": True,
                "sources": ["sec_filings", "regulatory_announcements"],
                "update_frequency": 7200  # 2 hours
            }
        },
        "data_quality": {
            "min_completeness": 0.8,
            "max_staleness_hours": 24,
            "validation_rules": ["range_check", "format_check", "consistency_check"]
        }
    },
    
    "stage2_feature_engineering": {
        "features": {
            "financial_ratios": True,
            "market_indicators": True,
            "sentiment_scores": True,
            "volatility_metrics": True,
            "peer_comparisons": True
        },
        "aggregation": {
            "rolling_windows": [7, 30, 90, 365],
            "cross_sectional_stats": ["percentile", "zscore", "rank"]
        }
    },
    
    "stage3_model_training": {
        "models": {
            "xgboost": {
                "enabled": True,
                "params": {
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "n_estimators": 100
                }
            },
            "neural_network": {
                "enabled": True,
                "architecture": "deep_feedforward",
                "hidden_layers": [256, 128, 64]
            },
            "ensemble": {
                "enabled": True,
                "methods": ["voting", "stacking"]
            }
        },
        "training": {
            "validation_split": 0.2,
            "test_split": 0.1,
            "cross_validation_folds": 5,
            "early_stopping_patience": 10
        }
    },
    
    "stage4_explainability": {
        "shap": {
            "enabled": True,
            "explainer_type": "tree",
            "max_display_features": 20
        },
        "lime": {
            "enabled": True,
            "num_features": 10,
            "num_samples": 5000
        },
        "chatbot": {
            "model": "gpt-3.5-turbo",
            "max_tokens": 500,
            "temperature": 0.7,
            "context_window": 4000
        }
    },
    
    "stage5_alerting_workflows": {
        "monitoring": {
            "score_thresholds": {
                "critical": 0.3,
                "high": 0.5,
                "medium": 0.7,
                "low": 0.9
            },
            "anomaly_detection": {
                "method": "isolation_forest",
                "contamination": 0.1,
                "sensitivity": 0.8
            }
        },
        "alerting": {
            "deduplication_window": 3600,  # 1 hour
            "max_alerts_per_company": 10,
            "cooldown_period": 1800  # 30 minutes
        },
        "notifications": {
            "channels": ["email", "slack", "webhook"],
            "retry_attempts": 3,
            "retry_delay": 300  # 5 minutes
        }
    }
}
