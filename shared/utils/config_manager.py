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
        return self.config_data.get('database', {
            'host': 'localhost',
            'port': 5432,
            'database': 'credit_intelligence',
            'user': 'postgres',
            'password': 'password'
        })
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for external service"""
        # Check stage1 api_keys first, then root level, then environment
        stage1_keys = self.config_data.get('stage1', {}).get('api_keys', {})
        api_keys = self.config_data.get('api_keys', {})
        
        return (stage1_keys.get(service) or 
                api_keys.get(service) or 
                os.getenv(f"{service.upper()}_API_KEY"))
    
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
