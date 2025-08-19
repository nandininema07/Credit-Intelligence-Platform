"""
Rolling metrics calculator for time-based feature engineering.
Handles rolling window calculations for financial and sentiment features.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RollingMetricsCalculator:
    """Rolling window metrics calculator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_windows = config.get('rolling_windows', [5, 10, 20, 50, 100])
        
    async def calculate_rolling_metrics(self, data: pd.DataFrame,
                                      feature_columns: List[str],
                                      windows: List[int] = None,
                                      timestamp_column: str = 'timestamp',
                                      group_column: str = 'company') -> pd.DataFrame:
        """Calculate rolling metrics for features"""
        if data.empty:
            return pd.DataFrame()
        
        if windows is None:
            windows = self.default_windows
        
        # Sort by timestamp
        data = data.sort_values([group_column, timestamp_column])
        
        result_data = data.copy()
        
        for window in windows:
            for feature_col in feature_columns:
                if feature_col not in data.columns:
                    continue
                
                # Calculate rolling metrics by group
                grouped = data.groupby(group_column)[feature_col]
                
                # Rolling mean
                result_data[f'{feature_col}_rolling_mean_{window}'] = grouped.rolling(
                    window=window, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                # Rolling std
                result_data[f'{feature_col}_rolling_std_{window}'] = grouped.rolling(
                    window=window, min_periods=2
                ).std().reset_index(0, drop=True)
                
                # Rolling min/max
                result_data[f'{feature_col}_rolling_min_{window}'] = grouped.rolling(
                    window=window, min_periods=1
                ).min().reset_index(0, drop=True)
                
                result_data[f'{feature_col}_rolling_max_{window}'] = grouped.rolling(
                    window=window, min_periods=1
                ).max().reset_index(0, drop=True)
                
                # Rolling sum (for count-based features)
                if 'count' in feature_col.lower():
                    result_data[f'{feature_col}_rolling_sum_{window}'] = grouped.rolling(
                        window=window, min_periods=1
                    ).sum().reset_index(0, drop=True)
        
        return result_data
    
    async def calculate_momentum_features(self, data: pd.DataFrame,
                                        price_column: str = 'stock_price',
                                        windows: List[int] = None,
                                        group_column: str = 'company') -> pd.DataFrame:
        """Calculate momentum-based features"""
        if data.empty or price_column not in data.columns:
            return data.copy()
        
        if windows is None:
            windows = [5, 10, 20]
        
        result_data = data.copy()
        
        for window in windows:
            grouped = data.groupby(group_column)[price_column]
            
            # Price momentum (rate of change)
            result_data[f'momentum_{window}'] = grouped.pct_change(window)
            
            # Relative strength index (RSI)
            result_data[f'rsi_{window}'] = grouped.apply(
                lambda x: self._calculate_rsi(x, window)
            ).reset_index(0, drop=True)
            
            # Moving average convergence divergence (MACD) components
            if window >= 12:
                ema_fast = grouped.ewm(span=12).mean()
                ema_slow = grouped.ewm(span=26).mean()
                result_data[f'macd_{window}'] = (ema_fast - ema_slow).reset_index(0, drop=True)
        
        return result_data
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        if len(prices) < window + 1:
            return pd.Series(np.nan, index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def calculate_volatility_features(self, data: pd.DataFrame,
                                          price_column: str = 'stock_price',
                                          windows: List[int] = None,
                                          group_column: str = 'company') -> pd.DataFrame:
        """Calculate volatility-based features"""
        if data.empty or price_column not in data.columns:
            return data.copy()
        
        if windows is None:
            windows = [10, 20, 50]
        
        result_data = data.copy()
        
        for window in windows:
            grouped = data.groupby(group_column)[price_column]
            
            # Historical volatility (rolling std of returns)
            returns = grouped.pct_change()
            result_data[f'volatility_{window}'] = returns.groupby(group_column).rolling(
                window=window, min_periods=2
            ).std().reset_index(0, drop=True)
            
            # Bollinger Bands
            rolling_mean = grouped.rolling(window=window).mean()
            rolling_std = grouped.rolling(window=window).std()
            
            result_data[f'bollinger_upper_{window}'] = (rolling_mean + 2 * rolling_std).reset_index(0, drop=True)
            result_data[f'bollinger_lower_{window}'] = (rolling_mean - 2 * rolling_std).reset_index(0, drop=True)
            result_data[f'bollinger_position_{window}'] = (
                (grouped - rolling_mean) / (2 * rolling_std)
            ).reset_index(0, drop=True)
        
        return result_data
