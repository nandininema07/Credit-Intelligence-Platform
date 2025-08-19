"""
Trend analysis for financial time series data.
Analyzes price movements, volume trends, and financial metric patterns.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Financial trend analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_periods = config.get('lookback_periods', [5, 10, 20, 50])
        
    async def analyze_trends(self, financial_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze trends in financial data"""
        if not financial_data:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(financial_data)
        
        # Ensure we have timestamp column
        if 'timestamp' not in df.columns and 'date' not in df.columns:
            df['timestamp'] = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df = df.sort_values('timestamp')
        
        trends = {}
        
        # Price trend analysis
        if 'price' in df.columns:
            price_trends = self._analyze_price_trends(df['price'].values)
            trends.update({f'price_{k}': v for k, v in price_trends.items()})
        
        # Volume trend analysis
        if 'volume' in df.columns:
            volume_trends = self._analyze_volume_trends(df['volume'].values)
            trends.update({f'volume_{k}': v for k, v in volume_trends.items()})
        
        # Revenue trend analysis
        if 'revenue' in df.columns:
            revenue_trends = self._analyze_metric_trends(df['revenue'].values, 'revenue')
            trends.update(revenue_trends)
        
        # Earnings trend analysis
        if 'earnings' in df.columns or 'net_income' in df.columns:
            earnings_col = 'earnings' if 'earnings' in df.columns else 'net_income'
            earnings_trends = self._analyze_metric_trends(df[earnings_col].values, 'earnings')
            trends.update(earnings_trends)
        
        # Market cap trend analysis
        if 'market_cap' in df.columns:
            market_cap_trends = self._analyze_metric_trends(df['market_cap'].values, 'market_cap')
            trends.update(market_cap_trends)
        
        logger.info(f"Analyzed trends for {len(trends)} metrics")
        return trends
    
    def _analyze_price_trends(self, prices: np.ndarray) -> Dict[str, float]:
        """Analyze price trend patterns"""
        if len(prices) < 2:
            return {}
        
        trends = {}
        
        # Simple returns
        returns = np.diff(prices) / prices[:-1]
        trends['return_mean'] = float(np.mean(returns))
        trends['return_std'] = float(np.std(returns))
        trends['return_skew'] = float(stats.skew(returns)) if len(returns) > 2 else 0.0
        trends['return_kurtosis'] = float(stats.kurtosis(returns)) if len(returns) > 3 else 0.0
        
        # Moving averages and crossovers
        for period in self.lookback_periods:
            if len(prices) >= period:
                ma = np.mean(prices[-period:])
                trends[f'ma_{period}'] = float(ma)
                trends[f'price_vs_ma_{period}'] = float((prices[-1] - ma) / ma) if ma != 0 else 0.0
        
        # Trend strength using linear regression
        if len(prices) >= 5:
            x = np.arange(len(prices)).reshape(-1, 1)
            y = prices
            
            model = LinearRegression()
            model.fit(x, y)
            
            trends['trend_slope'] = float(model.coef_[0])
            trends['trend_r_squared'] = float(model.score(x, y))
            trends['trend_direction'] = 1.0 if model.coef_[0] > 0 else -1.0
        
        # Support and resistance levels
        trends['price_range'] = float(np.max(prices) - np.min(prices))
        trends['price_position'] = float((prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices))) if np.max(prices) != np.min(prices) else 0.5
        
        # Momentum indicators
        if len(prices) >= 10:
            momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if prices[-6] != 0 else 0
            momentum_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 and prices[-11] != 0 else 0
            
            trends['momentum_5d'] = float(momentum_5)
            trends['momentum_10d'] = float(momentum_10)
        
        return trends
    
    def _analyze_volume_trends(self, volumes: np.ndarray) -> Dict[str, float]:
        """Analyze volume trend patterns"""
        if len(volumes) < 2:
            return {}
        
        trends = {}
        
        # Volume statistics
        trends['volume_mean'] = float(np.mean(volumes))
        trends['volume_std'] = float(np.std(volumes))
        trends['volume_cv'] = float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) != 0 else 0.0
        
        # Volume moving averages
        for period in [5, 10, 20]:
            if len(volumes) >= period:
                vol_ma = np.mean(volumes[-period:])
                trends[f'volume_ma_{period}'] = float(vol_ma)
                trends[f'volume_vs_ma_{period}'] = float((volumes[-1] - vol_ma) / vol_ma) if vol_ma != 0 else 0.0
        
        # Volume trend
        if len(volumes) >= 5:
            x = np.arange(len(volumes)).reshape(-1, 1)
            y = volumes
            
            model = LinearRegression()
            model.fit(x, y)
            
            trends['volume_trend_slope'] = float(model.coef_[0])
            trends['volume_trend_r_squared'] = float(model.score(x, y))
        
        return trends
    
    def _analyze_metric_trends(self, values: np.ndarray, metric_name: str) -> Dict[str, float]:
        """Analyze trends for any financial metric"""
        if len(values) < 2:
            return {}
        
        trends = {}
        
        # Growth rates
        if len(values) >= 2:
            recent_growth = (values[-1] - values[-2]) / abs(values[-2]) if values[-2] != 0 else 0
            trends[f'{metric_name}_recent_growth'] = float(recent_growth)
        
        if len(values) >= 4:
            quarterly_growth = (values[-1] - values[-4]) / abs(values[-4]) if len(values) >= 4 and values[-4] != 0 else 0
            trends[f'{metric_name}_quarterly_growth'] = float(quarterly_growth)
        
        # Trend consistency
        if len(values) >= 5:
            x = np.arange(len(values)).reshape(-1, 1)
            y = values
            
            model = LinearRegression()
            model.fit(x, y)
            
            trends[f'{metric_name}_trend_slope'] = float(model.coef_[0])
            trends[f'{metric_name}_trend_consistency'] = float(model.score(x, y))
        
        # Volatility
        if len(values) >= 3:
            pct_changes = np.diff(values) / np.abs(values[:-1])
            pct_changes = pct_changes[np.isfinite(pct_changes)]  # Remove inf/nan
            
            if len(pct_changes) > 0:
                trends[f'{metric_name}_volatility'] = float(np.std(pct_changes))
                trends[f'{metric_name}_mean_change'] = float(np.mean(pct_changes))
        
        return trends
    
    def detect_trend_changes(self, data: pd.DataFrame, column: str, window: int = 10) -> List[Dict[str, Any]]:
        """Detect significant trend changes in a time series"""
        if len(data) < window * 2:
            return []
        
        changes = []
        values = data[column].values
        
        for i in range(window, len(values) - window):
            # Calculate trends before and after the point
            before_trend = self._calculate_trend_slope(values[i-window:i])
            after_trend = self._calculate_trend_slope(values[i:i+window])
            
            # Check for significant change
            trend_change = abs(after_trend - before_trend)
            
            if trend_change > self.config.get('trend_change_threshold', 0.1):
                change_point = {
                    'index': i,
                    'timestamp': data.iloc[i]['timestamp'] if 'timestamp' in data.columns else i,
                    'value': float(values[i]),
                    'before_trend': float(before_trend),
                    'after_trend': float(after_trend),
                    'change_magnitude': float(trend_change),
                    'change_type': 'reversal' if before_trend * after_trend < 0 else 'acceleration'
                }
                changes.append(change_point)
        
        return changes
    
    def _calculate_trend_slope(self, values: np.ndarray) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values)).reshape(-1, 1)
        y = values
        
        model = LinearRegression()
        model.fit(x, y)
        
        return float(model.coef_[0])
    
    def calculate_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict[str, float]:
        """Calculate technical analysis indicators"""
        indicators = {}
        
        if len(prices) < 2:
            return indicators
        
        # RSI (Relative Strength Index)
        if len(prices) >= 14:
            rsi = self._calculate_rsi(prices, period=14)
            indicators['rsi_14'] = float(rsi)
        
        # MACD (Moving Average Convergence Divergence)
        if len(prices) >= 26:
            macd_line, signal_line = self._calculate_macd(prices)
            indicators['macd_line'] = float(macd_line)
            indicators['macd_signal'] = float(signal_line)
            indicators['macd_histogram'] = float(macd_line - signal_line)
        
        # Bollinger Bands
        if len(prices) >= 20:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(prices, period=20)
            indicators['bb_upper'] = float(bb_upper)
            indicators['bb_middle'] = float(bb_middle)
            indicators['bb_lower'] = float(bb_lower)
            indicators['bb_position'] = float((prices[-1] - bb_lower) / (bb_upper - bb_lower)) if bb_upper != bb_lower else 0.5
        
        # Average True Range (ATR)
        if len(prices) >= 14:
            atr = self._calculate_atr(prices, period=14)
            indicators['atr_14'] = float(atr)
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD line and signal line"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD line)
        # For simplicity, using simple moving average
        if len(prices) >= slow + signal:
            macd_values = []
            for i in range(slow, len(prices)):
                ema_fast_i = self._calculate_ema(prices[:i+1], fast)
                ema_slow_i = self._calculate_ema(prices[:i+1], slow)
                macd_values.append(ema_fast_i - ema_slow_i)
            
            signal_line = np.mean(macd_values[-signal:])
        else:
            signal_line = macd_line
        
        return macd_line, signal_line
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.mean(prices[:period])  # Start with SMA
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(prices) < 2:
            return 0.0
        
        # Simplified ATR calculation using price ranges
        ranges = []
        for i in range(1, len(prices)):
            high_low = abs(prices[i] - prices[i-1])
            ranges.append(high_low)
        
        if len(ranges) >= period:
            atr = np.mean(ranges[-period:])
        else:
            atr = np.mean(ranges)
        
        return atr
