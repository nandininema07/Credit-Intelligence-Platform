"""
Volatility metrics calculation for risk assessment.
Calculates various volatility measures and risk indicators.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VolatilityMetrics:
    """Volatility and risk metrics calculator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_levels = config.get('confidence_levels', [0.95, 0.99])
        
    async def calculate_volatility(self, financial_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive volatility metrics"""
        if not financial_data:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(financial_data)
        
        volatility_metrics = {}
        
        # Price volatility
        if 'price' in df.columns:
            price_vol = self._calculate_price_volatility(df['price'].values)
            volatility_metrics.update({f'price_{k}': v for k, v in price_vol.items()})
        
        # Return volatility
        if 'price' in df.columns and len(df) > 1:
            returns = df['price'].pct_change().dropna().values
            return_vol = self._calculate_return_volatility(returns)
            volatility_metrics.update({f'return_{k}': v for k, v in return_vol.items()})
        
        # Volume volatility
        if 'volume' in df.columns:
            volume_vol = self._calculate_volume_volatility(df['volume'].values)
            volatility_metrics.update({f'volume_{k}': v for k, v in volume_vol.items()})
        
        # Earnings volatility
        if 'earnings' in df.columns or 'net_income' in df.columns:
            earnings_col = 'earnings' if 'earnings' in df.columns else 'net_income'
            earnings_vol = self._calculate_earnings_volatility(df[earnings_col].values)
            volatility_metrics.update({f'earnings_{k}': v for k, v in earnings_vol.items()})
        
        logger.info(f"Calculated {len(volatility_metrics)} volatility metrics")
        return volatility_metrics
    
    def _calculate_price_volatility(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate price-based volatility metrics"""
        if len(prices) < 2:
            return {}
        
        metrics = {}
        
        # Standard deviation of prices
        metrics['std_dev'] = float(np.std(prices))
        metrics['coefficient_of_variation'] = float(np.std(prices) / np.mean(prices)) if np.mean(prices) != 0 else 0.0
        
        # Price range metrics
        metrics['range'] = float(np.max(prices) - np.min(prices))
        metrics['range_ratio'] = float((np.max(prices) - np.min(prices)) / np.mean(prices)) if np.mean(prices) != 0 else 0.0
        
        # Rolling volatility (if enough data)
        if len(prices) >= 20:
            rolling_std = []
            window = min(20, len(prices) // 2)
            
            for i in range(window, len(prices)):
                rolling_std.append(np.std(prices[i-window:i]))
            
            if rolling_std:
                metrics['rolling_volatility_mean'] = float(np.mean(rolling_std))
                metrics['rolling_volatility_std'] = float(np.std(rolling_std))
        
        return metrics
    
    def _calculate_return_volatility(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate return-based volatility metrics"""
        if len(returns) < 2:
            return {}
        
        metrics = {}
        
        # Basic return statistics
        metrics['volatility'] = float(np.std(returns))
        metrics['annualized_volatility'] = float(np.std(returns) * np.sqrt(252))  # Assuming daily returns
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['downside_deviation'] = float(np.std(negative_returns))
        else:
            metrics['downside_deviation'] = 0.0
        
        # Upside deviation
        positive_returns = returns[returns > 0]
        if len(positive_returns) > 0:
            metrics['upside_deviation'] = float(np.std(positive_returns))
        else:
            metrics['upside_deviation'] = 0.0
        
        # Semi-variance
        mean_return = np.mean(returns)
        below_mean_returns = returns[returns < mean_return]
        if len(below_mean_returns) > 0:
            metrics['semi_variance'] = float(np.var(below_mean_returns))
        else:
            metrics['semi_variance'] = 0.0
        
        # Value at Risk (VaR)
        for confidence_level in self.confidence_levels:
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(returns, var_percentile)
            metrics[f'var_{int(confidence_level*100)}'] = float(var_value)
        
        # Conditional Value at Risk (CVaR)
        for confidence_level in self.confidence_levels:
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(returns, var_percentile)
            tail_returns = returns[returns <= var_value]
            if len(tail_returns) > 0:
                cvar_value = np.mean(tail_returns)
                metrics[f'cvar_{int(confidence_level*100)}'] = float(cvar_value)
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = float(np.min(drawdowns))
        
        # Skewness and Kurtosis
        if len(returns) > 2:
            metrics['skewness'] = float(stats.skew(returns))
        if len(returns) > 3:
            metrics['kurtosis'] = float(stats.kurtosis(returns))
            metrics['excess_kurtosis'] = float(stats.kurtosis(returns, fisher=True))
        
        return metrics
    
    def _calculate_volume_volatility(self, volumes: np.ndarray) -> Dict[str, float]:
        """Calculate volume-based volatility metrics"""
        if len(volumes) < 2:
            return {}
        
        metrics = {}
        
        # Volume standard deviation
        metrics['volume_std'] = float(np.std(volumes))
        metrics['volume_cv'] = float(np.std(volumes) / np.mean(volumes)) if np.mean(volumes) != 0 else 0.0
        
        # Volume changes
        if len(volumes) > 1:
            volume_changes = np.diff(volumes) / volumes[:-1]
            volume_changes = volume_changes[np.isfinite(volume_changes)]
            
            if len(volume_changes) > 0:
                metrics['volume_change_volatility'] = float(np.std(volume_changes))
                metrics['volume_change_mean'] = float(np.mean(volume_changes))
        
        return metrics
    
    def _calculate_earnings_volatility(self, earnings: np.ndarray) -> Dict[str, float]:
        """Calculate earnings-based volatility metrics"""
        if len(earnings) < 2:
            return {}
        
        metrics = {}
        
        # Earnings standard deviation
        metrics['earnings_std'] = float(np.std(earnings))
        
        # Earnings growth volatility
        if len(earnings) > 1:
            earnings_growth = np.diff(earnings) / np.abs(earnings[:-1])
            earnings_growth = earnings_growth[np.isfinite(earnings_growth)]
            
            if len(earnings_growth) > 0:
                metrics['earnings_growth_volatility'] = float(np.std(earnings_growth))
                metrics['earnings_growth_mean'] = float(np.mean(earnings_growth))
        
        return metrics
    
    def calculate_garch_volatility(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate GARCH volatility (simplified implementation)"""
        if len(returns) < 10:
            return {}
        
        try:
            # Simplified GARCH(1,1) estimation
            # In practice, would use specialized libraries like arch
            
            # Calculate squared returns
            squared_returns = returns ** 2
            
            # Simple moving average as proxy for conditional variance
            window = min(10, len(returns) // 2)
            conditional_variance = []
            
            for i in range(window, len(squared_returns)):
                var_estimate = np.mean(squared_returns[i-window:i])
                conditional_variance.append(var_estimate)
            
            if conditional_variance:
                garch_vol = np.sqrt(np.mean(conditional_variance))
                return {
                    'garch_volatility': float(garch_vol),
                    'garch_persistence': float(np.corrcoef(conditional_variance[:-1], conditional_variance[1:])[0, 1]) if len(conditional_variance) > 1 else 0.0
                }
        
        except Exception as e:
            logger.error(f"Error calculating GARCH volatility: {e}")
        
        return {}
    
    def calculate_realized_volatility(self, high_freq_returns: np.ndarray) -> float:
        """Calculate realized volatility from high-frequency returns"""
        if len(high_freq_returns) < 2:
            return 0.0
        
        # Sum of squared returns
        realized_var = np.sum(high_freq_returns ** 2)
        realized_vol = np.sqrt(realized_var)
        
        return float(realized_vol)
    
    def calculate_volatility_smile(self, option_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate volatility smile metrics (if option data available)"""
        # Placeholder for volatility smile analysis
        # Would require actual option pricing data
        return {
            'volatility_smile_slope': 0.0,
            'volatility_smile_convexity': 0.0,
            'at_the_money_volatility': 0.0
        }
    
    def calculate_correlation_volatility(self, returns1: np.ndarray, returns2: np.ndarray) -> Dict[str, float]:
        """Calculate correlation-based volatility metrics"""
        if len(returns1) != len(returns2) or len(returns1) < 2:
            return {}
        
        metrics = {}
        
        # Rolling correlation
        if len(returns1) >= 20:
            window = min(20, len(returns1) // 2)
            rolling_corr = []
            
            for i in range(window, len(returns1)):
                corr = np.corrcoef(returns1[i-window:i], returns2[i-window:i])[0, 1]
                if not np.isnan(corr):
                    rolling_corr.append(corr)
            
            if rolling_corr:
                metrics['correlation_volatility'] = float(np.std(rolling_corr))
                metrics['mean_correlation'] = float(np.mean(rolling_corr))
        
        return metrics
    
    def calculate_regime_volatility(self, returns: np.ndarray, n_regimes: int = 2) -> Dict[str, Any]:
        """Calculate volatility in different market regimes"""
        if len(returns) < 10:
            return {}
        
        try:
            # Simple regime detection based on volatility levels
            rolling_vol = []
            window = min(10, len(returns) // 3)
            
            for i in range(window, len(returns)):
                vol = np.std(returns[i-window:i])
                rolling_vol.append(vol)
            
            if not rolling_vol:
                return {}
            
            # Classify regimes based on volatility percentiles
            vol_threshold = np.percentile(rolling_vol, 50)  # Median split
            
            low_vol_returns = []
            high_vol_returns = []
            
            for i, vol in enumerate(rolling_vol):
                if vol <= vol_threshold:
                    low_vol_returns.extend(returns[i:i+window])
                else:
                    high_vol_returns.extend(returns[i:i+window])
            
            regimes = {}
            
            if low_vol_returns:
                regimes['low_volatility_regime'] = {
                    'volatility': float(np.std(low_vol_returns)),
                    'mean_return': float(np.mean(low_vol_returns)),
                    'observations': len(low_vol_returns)
                }
            
            if high_vol_returns:
                regimes['high_volatility_regime'] = {
                    'volatility': float(np.std(high_vol_returns)),
                    'mean_return': float(np.mean(high_vol_returns)),
                    'observations': len(high_vol_returns)
                }
            
            return regimes
            
        except Exception as e:
            logger.error(f"Error calculating regime volatility: {e}")
            return {}
    
    def calculate_volatility_forecast(self, returns: np.ndarray, horizon: int = 5) -> Dict[str, float]:
        """Forecast future volatility"""
        if len(returns) < 10:
            return {}
        
        try:
            # Simple exponential smoothing for volatility forecast
            squared_returns = returns ** 2
            
            # Calculate exponentially weighted moving average
            alpha = 0.1  # Smoothing parameter
            ewma_var = squared_returns[0]
            
            for sq_return in squared_returns[1:]:
                ewma_var = alpha * sq_return + (1 - alpha) * ewma_var
            
            # Forecast assuming persistence
            forecasted_volatility = np.sqrt(ewma_var)
            
            return {
                'volatility_forecast': float(forecasted_volatility),
                'forecast_horizon': horizon,
                'forecast_confidence': 0.7  # Placeholder confidence level
            }
            
        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return {}
