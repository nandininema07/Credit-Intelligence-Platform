"""
Data source configurations for the ingestion pipeline.
Manages API endpoints, rate limits, and source-specific settings.
"""

from typing import Dict, Any, List
import json
import logging

logger = logging.getLogger(__name__)

class SourcesConfig:
    """Configuration manager for data sources"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.sources = self._load_default_sources()
        
        if config_file:
            self._load_from_file()
    
    def _load_default_sources(self) -> Dict[str, Any]:
        """Load default source configurations"""
        return {
            'news_sources': {
                'newsapi': {
                    'url': 'https://newsapi.org/v2/everything',
                    'rate_limit': 1000,  # requests per day
                    'languages': ['en', 'es', 'fr', 'de', 'it'],
                    'categories': ['business', 'technology'],
                    'enabled': True
                },
                'reuters': {
                    'url': 'https://www.reuters.com',
                    'rate_limit': 100,
                    'languages': ['en'],
                    'sections': ['business', 'markets', 'technology'],
                    'enabled': True
                },
                'bloomberg': {
                    'url': 'https://www.bloomberg.com',
                    'rate_limit': 50,
                    'languages': ['en'],
                    'sections': ['markets', 'companies', 'economics'],
                    'enabled': True
                },
                'financial_times': {
                    'url': 'https://www.ft.com',
                    'rate_limit': 50,
                    'languages': ['en'],
                    'sections': ['companies', 'markets'],
                    'enabled': False  # Requires subscription
                }
            },
            'social_sources': {
                'twitter': {
                    'api_version': 'v2',
                    'rate_limit': 300,  # requests per 15 minutes
                    'max_results': 100,
                    'languages': ['en', 'es', 'fr'],
                    'enabled': True
                },
                'reddit': {
                    'subreddits': [
                        'investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting',
                        'financialindependence', 'StockMarket', 'economy'
                    ],
                    'rate_limit': 60,  # requests per minute
                    'enabled': True
                },
                'youtube': {
                    'api_version': 'v3',
                    'rate_limit': 10000,  # requests per day
                    'max_results': 50,
                    'enabled': False  # Optional
                }
            },
            'financial_sources': {
                'yahoo_finance': {
                    'url': 'https://query1.finance.yahoo.com',
                    'rate_limit': 2000,  # requests per hour
                    'data_types': ['quote', 'chart', 'fundamentals'],
                    'enabled': True
                },
                'alpha_vantage': {
                    'url': 'https://www.alphavantage.co/query',
                    'rate_limit': 5,  # requests per minute (free tier)
                    'functions': ['TIME_SERIES_DAILY', 'OVERVIEW', 'INCOME_STATEMENT'],
                    'enabled': True
                },
                'quandl': {
                    'url': 'https://www.quandl.com/api/v3',
                    'rate_limit': 50000,  # requests per day
                    'enabled': False
                },
                'iex_cloud': {
                    'url': 'https://cloud.iexapis.com/stable',
                    'rate_limit': 100,  # requests per second
                    'enabled': False
                }
            },
            'regulatory_sources': {
                'sec_edgar': {
                    'url': 'https://www.sec.gov/Archives/edgar',
                    'rate_limit': 10,  # requests per second
                    'filing_types': ['10-K', '10-Q', '8-K', 'DEF 14A'],
                    'enabled': True
                },
                'finra': {
                    'url': 'https://www.finra.org',
                    'rate_limit': 60,
                    'enabled': True
                },
                'cftc': {
                    'url': 'https://www.cftc.gov',
                    'rate_limit': 60,
                    'enabled': True
                }
            },
            'international_sources': {
                'european_sources': {
                    'ecb': {
                        'url': 'https://www.ecb.europa.eu',
                        'languages': ['en', 'de', 'fr'],
                        'enabled': True
                    },
                    'financial_times_europe': {
                        'url': 'https://www.ft.com/europe',
                        'languages': ['en'],
                        'enabled': False
                    }
                },
                'asian_sources': {
                    'nikkei': {
                        'url': 'https://asia.nikkei.com',
                        'languages': ['en', 'ja'],
                        'enabled': True
                    },
                    'shanghai_daily': {
                        'url': 'https://www.shine.cn',
                        'languages': ['en', 'zh'],
                        'enabled': True
                    }
                }
            },
            'alternative_sources': {
                'satellite_data': {
                    'providers': ['planet', 'maxar'],
                    'enabled': False  # Requires special access
                },
                'patent_data': {
                    'uspto': {
                        'url': 'https://developer.uspto.gov',
                        'enabled': True
                    }
                },
                'job_postings': {
                    'linkedin_jobs': {
                        'enabled': False  # Requires API access
                    },
                    'indeed': {
                        'enabled': True
                    }
                }
            }
        }
    
    def _load_from_file(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                file_config = json.load(f)
                self._merge_configs(self.sources, file_config)
            logger.info(f"Loaded source configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading source config: {e}")
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]):
        """Merge configuration dictionaries"""
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def get_enabled_sources(self, source_type: str) -> Dict[str, Any]:
        """Get enabled sources of a specific type"""
        if source_type not in self.sources:
            return {}
        
        enabled_sources = {}
        for source_name, config in self.sources[source_type].items():
            if isinstance(config, dict) and config.get('enabled', False):
                enabled_sources[source_name] = config
            elif isinstance(config, dict):
                # Check nested sources
                nested_enabled = {}
                for nested_name, nested_config in config.items():
                    if isinstance(nested_config, dict) and nested_config.get('enabled', False):
                        nested_enabled[nested_name] = nested_config
                if nested_enabled:
                    enabled_sources[source_name] = nested_enabled
        
        return enabled_sources
    
    def get_rate_limit(self, source_type: str, source_name: str) -> int:
        """Get rate limit for a specific source"""
        try:
            return self.sources[source_type][source_name]['rate_limit']
        except KeyError:
            return 60  # Default rate limit
    
    def get_source_config(self, source_type: str, source_name: str) -> Dict[str, Any]:
        """Get configuration for a specific source"""
        try:
            return self.sources[source_type][source_name]
        except KeyError:
            return {}
    
    def update_source_status(self, source_type: str, source_name: str, enabled: bool):
        """Update source enabled status"""
        try:
            self.sources[source_type][source_name]['enabled'] = enabled
            logger.info(f"Updated {source_type}.{source_name} enabled status to {enabled}")
        except KeyError:
            logger.error(f"Source {source_type}.{source_name} not found")
    
    def get_all_languages(self) -> List[str]:
        """Get all supported languages across sources"""
        languages = set()
        
        def extract_languages(config):
            if isinstance(config, dict):
                if 'languages' in config:
                    languages.update(config['languages'])
                for value in config.values():
                    extract_languages(value)
        
        extract_languages(self.sources)
        return sorted(list(languages))
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate source configuration"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if at least one source is enabled in each category
        for source_type, sources in self.sources.items():
            enabled_count = 0
            
            def count_enabled(config):
                nonlocal enabled_count
                if isinstance(config, dict):
                    if config.get('enabled', False):
                        enabled_count += 1
                    for value in config.values():
                        if isinstance(value, dict):
                            count_enabled(value)
            
            count_enabled(sources)
            
            if enabled_count == 0:
                validation_result['warnings'].append(f"No enabled sources in {source_type}")
        
        return validation_result
    
    def export_config(self, file_path: str):
        """Export current configuration to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.sources, f, indent=2, default=str)
            logger.info(f"Exported configuration to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
    
    def get_source_priorities(self) -> Dict[str, int]:
        """Get source priorities for data collection"""
        return {
            'financial_sources': 1,  # Highest priority
            'regulatory_sources': 2,
            'news_sources': 3,
            'social_sources': 4,
            'international_sources': 5,
            'alternative_sources': 6  # Lowest priority
        }
