"""
Configuration management for the Credit Intelligence Platform.
"""

import json
import os
import yaml
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for application settings"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(os.getcwd(), 'config')
        self.config_data = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load configuration files"""
        config_files = [
            'config.json',
            'database.json', 
            'api_keys.json',
            'model_config.yaml',
            'alert_rules.json'
        ]
        
        for config_file in config_files:
            file_path = os.path.join(self.config_path, config_file)
            if os.path.exists(file_path):
                try:
                    if config_file.endswith('.json'):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                    elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        with open(file_path, 'r') as f:
                            data = yaml.safe_load(f)
                    
                    config_name = config_file.split('.')[0]
                    self.config_data[config_name] = data
                    logger.info(f"Loaded config: {config_file}")
                    
                except Exception as e:
                    logger.error(f"Error loading config {config_file}: {str(e)}")
    
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
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config_data.get('database', {
            'host': 'localhost',
            'port': 5432,
            'database': 'credit_intelligence',
            'user': 'postgres',
            'password': 'password'
        })
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for external service"""
        api_keys = self.config_data.get('api_keys', {})
        return api_keys.get(service) or os.getenv(f"{service.upper()}_API_KEY")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config_data.get('model_config', {
            'default_model': 'xgboost',
            'model_path': './models/',
            'feature_store_path': './feature_store/',
            'batch_size': 32,
            'timeout_seconds': 300
        })
    
    def get_alert_config(self) -> Dict[str, Any]:
        """Get alerting configuration"""
        return self.config_data.get('alert_rules', {
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
