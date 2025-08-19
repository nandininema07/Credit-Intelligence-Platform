"""
Time series aggregation for financial features.
Handles temporal aggregation of features across different time windows.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TimeWindow:
    """Time window configuration"""
    name: str
    duration: timedelta
    aggregation_functions: List[str]

class TimeSeriesAggregator:
    """Time series feature aggregation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_windows = self._setup_time_windows()
        self.aggregation_functions = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'sum': np.sum,
            'count': len,
            'skew': lambda x: pd.Series(x).skew(),
            'kurtosis': lambda x: pd.Series(x).kurtosis(),
            'percentile_25': lambda x: np.percentile(x, 25),
            'percentile_75': lambda x: np.percentile(x, 75),
            'trend_slope': self._calculate_trend_slope,
            'volatility': lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0
        }
    
    def _setup_time_windows(self) -> List[TimeWindow]:
        """Setup default time windows"""
        default_windows = [
            TimeWindow('1h', timedelta(hours=1), ['mean', 'count', 'std']),
            TimeWindow('6h', timedelta(hours=6), ['mean', 'count', 'std', 'trend_slope']),
            TimeWindow('1d', timedelta(days=1), ['mean', 'median', 'std', 'min', 'max', 'volatility']),
            TimeWindow('3d', timedelta(days=3), ['mean', 'std', 'trend_slope', 'volatility']),
            TimeWindow('7d', timedelta(days=7), ['mean', 'std', 'trend_slope', 'skew', 'kurtosis']),
            TimeWindow('30d', timedelta(days=30), ['mean', 'median', 'std', 'trend_slope', 'volatility'])
        ]
        
        # Override with config if provided
        if 'time_windows' in self.config:
            return [
                TimeWindow(
                    name=w['name'],
                    duration=timedelta(**w['duration']),
                    aggregation_functions=w['functions']
                )
                for w in self.config['time_windows']
            ]
        
        return default_windows
    
    async def aggregate_features(self, data: pd.DataFrame, 
                               feature_columns: List[str],
                               timestamp_column: str = 'timestamp',
                               group_by_column: str = 'company') -> pd.DataFrame:
        """Aggregate features across time windows"""
        if data.empty:
            return pd.DataFrame()
        
        # Ensure timestamp column is datetime
        data[timestamp_column] = pd.to_datetime(data[timestamp_column])
        
        # Sort by timestamp
        data = data.sort_values(timestamp_column)
        
        aggregated_features = []
        
        for company in data[group_by_column].unique():
            company_data = data[data[group_by_column] == company]
            company_features = await self._aggregate_company_features(
                company_data, feature_columns, timestamp_column, company
            )
            aggregated_features.extend(company_features)
        
        if not aggregated_features:
            return pd.DataFrame()
        
        return pd.DataFrame(aggregated_features)
    
    async def _aggregate_company_features(self, company_data: pd.DataFrame,
                                        feature_columns: List[str],
                                        timestamp_column: str,
                                        company: str) -> List[Dict[str, Any]]:
        """Aggregate features for a single company"""
        features = []
        
        # Get the latest timestamp as reference point
        latest_timestamp = company_data[timestamp_column].max()
        
        for window in self.time_windows:
            window_start = latest_timestamp - window.duration
            window_data = company_data[
                company_data[timestamp_column] >= window_start
            ]
            
            if window_data.empty:
                continue
            
            window_features = {
                'company': company,
                'timestamp': latest_timestamp,
                'window': window.name,
                'window_start': window_start,
                'window_end': latest_timestamp,
                'data_points': len(window_data)
            }
            
            # Aggregate each feature column
            for feature_col in feature_columns:
                if feature_col not in window_data.columns:
                    continue
                
                feature_values = window_data[feature_col].dropna()
                if feature_values.empty:
                    continue
                
                # Apply aggregation functions
                for agg_func_name in window.aggregation_functions:
                    if agg_func_name in self.aggregation_functions:
                        try:
                            agg_value = self.aggregation_functions[agg_func_name](feature_values)
                            feature_name = f"{feature_col}_{window.name}_{agg_func_name}"
                            window_features[feature_name] = float(agg_value) if not np.isnan(agg_value) else None
                        except Exception as e:
                            logger.warning(f"Error calculating {agg_func_name} for {feature_col}: {e}")
            
            features.append(window_features)
        
        return features
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except Exception:
            return 0.0
    
    async def aggregate_sentiment_features(self, sentiment_data: pd.DataFrame,
                                         timestamp_column: str = 'timestamp',
                                         company_column: str = 'company') -> pd.DataFrame:
        """Aggregate sentiment-specific features"""
        if sentiment_data.empty:
            return pd.DataFrame()
        
        sentiment_features = []
        
        for company in sentiment_data[company_column].unique():
            company_data = sentiment_data[sentiment_data[company_column] == company]
            
            for window in self.time_windows:
                latest_timestamp = company_data[timestamp_column].max()
                window_start = latest_timestamp - window.duration
                window_data = company_data[
                    company_data[timestamp_column] >= window_start
                ]
                
                if window_data.empty:
                    continue
                
                # Calculate sentiment-specific aggregations
                sentiment_scores = window_data['sentiment_score'].dropna()
                if not sentiment_scores.empty:
                    features = {
                        'company': company,
                        'timestamp': latest_timestamp,
                        'window': window.name,
                        f'sentiment_mean_{window.name}': sentiment_scores.mean(),
                        f'sentiment_std_{window.name}': sentiment_scores.std(),
                        f'sentiment_positive_ratio_{window.name}': (sentiment_scores > 0.1).mean(),
                        f'sentiment_negative_ratio_{window.name}': (sentiment_scores < -0.1).mean(),
                        f'sentiment_neutral_ratio_{window.name}': ((sentiment_scores >= -0.1) & (sentiment_scores <= 0.1)).mean(),
                        f'sentiment_momentum_{window.name}': self._calculate_sentiment_momentum(sentiment_scores),
                        f'sentiment_volatility_{window.name}': sentiment_scores.std() if len(sentiment_scores) > 1 else 0
                    }
                    sentiment_features.append(features)
        
        return pd.DataFrame(sentiment_features) if sentiment_features else pd.DataFrame()
    
    def _calculate_sentiment_momentum(self, sentiment_scores: pd.Series) -> float:
        """Calculate sentiment momentum (recent vs older sentiment)"""
        if len(sentiment_scores) < 4:
            return 0.0
        
        # Split into recent and older halves
        mid_point = len(sentiment_scores) // 2
        recent_sentiment = sentiment_scores.iloc[mid_point:].mean()
        older_sentiment = sentiment_scores.iloc[:mid_point].mean()
        
        return recent_sentiment - older_sentiment
    
    async def aggregate_financial_features(self, financial_data: pd.DataFrame,
                                         timestamp_column: str = 'timestamp',
                                         company_column: str = 'company') -> pd.DataFrame:
        """Aggregate financial-specific features"""
        financial_columns = [
            'revenue', 'profit', 'debt_ratio', 'pe_ratio', 'market_cap',
            'stock_price', 'volume', 'volatility'
        ]
        
        # Filter to existing columns
        existing_columns = [col for col in financial_columns if col in financial_data.columns]
        
        if not existing_columns:
            return pd.DataFrame()
        
        return await self.aggregate_features(
            financial_data, existing_columns, timestamp_column, company_column
        )
    
    async def create_lagged_features(self, data: pd.DataFrame,
                                   feature_columns: List[str],
                                   lags: List[int] = [1, 2, 3, 7, 14],
                                   timestamp_column: str = 'timestamp',
                                   company_column: str = 'company') -> pd.DataFrame:
        """Create lagged features"""
        if data.empty:
            return pd.DataFrame()
        
        lagged_data = []
        
        for company in data[company_column].unique():
            company_data = data[data[company_column] == company].sort_values(timestamp_column)
            
            for lag in lags:
                for feature_col in feature_columns:
                    if feature_col in company_data.columns:
                        lagged_col = f"{feature_col}_lag_{lag}"
                        company_data[lagged_col] = company_data[feature_col].shift(lag)
            
            lagged_data.append(company_data)
        
        return pd.concat(lagged_data, ignore_index=True) if lagged_data else pd.DataFrame()
    
    async def calculate_feature_differences(self, data: pd.DataFrame,
                                          feature_columns: List[str],
                                          periods: List[int] = [1, 7, 30],
                                          timestamp_column: str = 'timestamp',
                                          company_column: str = 'company') -> pd.DataFrame:
        """Calculate feature differences over periods"""
        if data.empty:
            return pd.DataFrame()
        
        diff_data = []
        
        for company in data[company_column].unique():
            company_data = data[data[company_column] == company].sort_values(timestamp_column)
            
            for period in periods:
                for feature_col in feature_columns:
                    if feature_col in company_data.columns:
                        diff_col = f"{feature_col}_diff_{period}"
                        pct_change_col = f"{feature_col}_pct_change_{period}"
                        
                        # Absolute difference
                        company_data[diff_col] = company_data[feature_col].diff(period)
                        
                        # Percentage change
                        company_data[pct_change_col] = company_data[feature_col].pct_change(period)
            
            diff_data.append(company_data)
        
        return pd.concat(diff_data, ignore_index=True) if diff_data else pd.DataFrame()
    
    def get_aggregation_summary(self, aggregated_data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of aggregation results"""
        if aggregated_data.empty:
            return {'total_features': 0, 'companies': 0, 'time_windows': 0}
        
        feature_columns = [col for col in aggregated_data.columns 
                          if col not in ['company', 'timestamp', 'window', 'window_start', 'window_end', 'data_points']]
        
        return {
            'total_features': len(feature_columns),
            'companies': aggregated_data['company'].nunique() if 'company' in aggregated_data.columns else 0,
            'time_windows': aggregated_data['window'].nunique() if 'window' in aggregated_data.columns else 0,
            'total_records': len(aggregated_data),
            'feature_types': {
                'sentiment': len([col for col in feature_columns if 'sentiment' in col]),
                'financial': len([col for col in feature_columns if any(fin in col for fin in ['revenue', 'profit', 'price'])]),
                'technical': len([col for col in feature_columns if any(tech in col for tech in ['volatility', 'trend', 'momentum'])])
            }
        }
