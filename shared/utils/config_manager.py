"""
Configuration management for the Credit Intelligence Platform.
"""

import json
import os
import yaml
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for application settings"""
    
    def __init__(self, config_path: str = None):
        if config_path and os.path.isfile(config_path):
            # Single config file path
            self.config_file = config_path
            self.config_path = os.path.dirname(config_path)
        else:
            # Config directory path
            self.config_path = config_path or os.path.join(os.getcwd(), 'config')
            self.config_file = os.path.join(self.config_path, 'config.json')
        
        self.config_data = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load configuration files"""
        # Load main config file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config_data = json.load(f)
                logger.info(f"Loaded main config: {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading main config: {e}")
        
        # Load additional config files if they exist
        additional_configs = ['api_keys.json']
        for config_file in additional_configs:
            file_path = os.path.join(self.config_path, config_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    # Merge with main config
                    config_name = config_file.split('.')[0]
                    self.config_data[config_name] = data
                    logger.info(f"Loaded additional config: {config_file}")
                except Exception as e:
                    logger.error(f"Error loading config {config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        current = self.config_data
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config_data
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        # Load from environment variables first, then config file, then defaults
        return self.config_data.get('database', {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'credit_intelligence'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        })
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for external service"""
        # Check stage1 api_keys first, then root level, then environment
        stage1_keys = self.config_data.get('stage1', {}).get('api_keys', {})
        api_keys = self.config_data.get('api_keys', {})
        
        # Try multiple environment variable patterns
        env_vars = [
            f"{service.upper()}_API_KEY",
            f"{service.upper()}_KEY", 
            f"{service.upper()}_TOKEN",
            f"{service.upper()}_BEARER_TOKEN",
            f"{service.upper()}_ACCESS_TOKEN"
        ]
        
        for env_var in env_vars:
            value = os.getenv(env_var)
            if value and value not in ['your-newsapi-key', 'your-twitter-bearer-token', 'your-alpha-vantage-key']:
                return value
        
        return (stage1_keys.get(service) or 
                api_keys.get(service))
    
    def get_all_api_keys(self) -> Dict[str, Any]:
        """Get all API keys from environment variables and config files"""
        api_keys = {}
        
        # Common API key environment variables
        env_key_mappings = {
            'NEWSAPI_KEY': 'newsapi',
            'ALPHA_VANTAGE_KEY': 'alpha_vantage',
            'FRED_KEY': 'fred',
            'FINNHUB_KEY': 'finnhub',
            'POLYGON_KEY': 'polygon',
            'TWITTER_BEARER_TOKEN': 'twitter_bearer_token',
            'REDDIT_CLIENT_ID': 'reddit_client_id',
            'REDDIT_CLIENT_SECRET': 'reddit_client_secret',
            'OPENAI_API_KEY': 'openai',
            'HUGGINGFACE_TOKEN': 'huggingface',
            'SLACK_BOT_TOKEN': 'slack_bot_token',
            'SLACK_WEBHOOK_URL': 'slack_webhook_url',
            'JIRA_API_TOKEN': 'jira_api_token',
            'SMTP_USERNAME': 'smtp_username',
            'SMTP_PASSWORD': 'smtp_password'
        }
        
        # Load from environment variables
        for env_var, key_name in env_key_mappings.items():
            value = os.getenv(env_var)
            if value and value not in ['your-newsapi-key', 'your-twitter-bearer-token', 'your-alpha-vantage-key', 'your-openai-api-key']:
                api_keys[key_name] = value
        
        # Load from config files as fallback
        stage1_keys = self.config_data.get('stage1', {}).get('api_keys', {})
        config_keys = self.config_data.get('api_keys', {})
        
        # Merge config keys (environment variables take precedence)
        for key, value in stage1_keys.items():
            if key not in api_keys:
                api_keys[key] = value
                
        for key, value in config_keys.items():
            if key not in api_keys:
                api_keys[key] = value
        
        # Convert flat keys to nested structure for backward compatibility
        nested_api_keys = {}
        
        # Simple keys
        for key in ['newsapi', 'alpha_vantage', 'fred', 'finnhub', 'polygon', 'openai', 'huggingface']:
            if key in api_keys:
                nested_api_keys[key] = api_keys[key]
        
        # Twitter nested structure
        if 'twitter_bearer_token' in api_keys:
            nested_api_keys['twitter'] = {
                'bearer_token': api_keys['twitter_bearer_token'],
                'api_key': api_keys.get('twitter_api_key', ''),
                'api_secret': api_keys.get('twitter_api_secret', ''),
                'access_token': api_keys.get('twitter_access_token', ''),
                'access_token_secret': api_keys.get('twitter_access_token_secret', '')
            }
        
        # Reddit nested structure
        if 'reddit_client_id' in api_keys or 'reddit_client_secret' in api_keys:
            nested_api_keys['reddit'] = {
                'client_id': api_keys.get('reddit_client_id', ''),
                'client_secret': api_keys.get('reddit_client_secret', ''),
                'user_agent': 'CreditIntelligence/1.0'
            }
        
        # Slack nested structure
        if 'slack_bot_token' in api_keys or 'slack_webhook_url' in api_keys:
            nested_api_keys['slack'] = {
                'bot_token': api_keys.get('slack_bot_token', ''),
                'webhook_url': api_keys.get('slack_webhook_url', '')
            }
        
        # Email nested structure
        if 'smtp_username' in api_keys or 'smtp_password' in api_keys:
            nested_api_keys['email'] = {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': api_keys.get('smtp_username', ''),
                'password': api_keys.get('smtp_password', '')
            }
        
        return nested_api_keys
    
    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """Get configuration for specific stage"""
        return self.config_data.get(stage, {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config_data.get('stage3', {
            'default_model': 'xgboost',
            'model_path': './models/',
            'feature_store_path': './feature_store/',
            'batch_size': 32,
            'timeout_seconds': 300
        })
    
    def get_alert_config(self) -> Dict[str, Any]:
        """Get alerting configuration"""
        return self.config_data.get('stage5', {
            'enabled': True,
            'notification_channels': ['email', 'slack'],
            'cooldown_minutes': 60,
            'severity_thresholds': {
                'critical': 400,
                'high': 500,
                'medium': 600
            }
        })
    
    def update_config(self, section: str, key: str, value: Any):
        """Update configuration value"""
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self.config_data[section][key] = value
        logger.info(f"Updated config: {section}.{key} = {value}")
    
    def save_config(self, section: str):
        """Save configuration section to file"""
        if section not in self.config_data:
            return
        
        file_path = os.path.join(self.config_path, f"{section}.json")
        
        try:
            os.makedirs(self.config_path, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(self.config_data[section], f, indent=2)
            logger.info(f"Saved config section: {section}")
        except Exception as e:
            logger.error(f"Error saving config {section}: {str(e)}")
