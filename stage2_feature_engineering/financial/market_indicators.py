"""
Market indicators calculation for macroeconomic and market analysis.
Calculates market-wide indicators and economic metrics.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketIndicators:
    """Market and economic indicators calculator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def calculate_indicators(self, financial_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive market indicators"""
        if not financial_data:
            return {}
        
        df = pd.DataFrame(financial_data)
        indicators = {}
        
        # Market performance indicators
        market_perf = self._calculate_market_performance(df)
        indicators.update({f'market_{k}': v for k, v in market_perf.items()})
        
        # Liquidity indicators
        liquidity_ind = self._calculate_liquidity_indicators(df)
        indicators.update({f'liquidity_{k}': v for k, v in liquidity_ind.items()})
        
        # Momentum indicators
        momentum_ind = self._calculate_momentum_indicators(df)
        indicators.update({f'momentum_{k}': v for k, v in momentum_ind.items()})
        
        # Breadth indicators
        breadth_ind = self._calculate_breadth_indicators(df)
        indicators.update({f'breadth_{k}': v for k, v in breadth_ind.items()})
        
        logger.info(f"Calculated {len(indicators)} market indicators")
        return indicators
    
    def _calculate_market_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market performance indicators"""
        indicators = {}
        
        if 'market_cap' in df.columns:
            market_caps = df['market_cap'].dropna()
            if len(market_caps) > 0:
                indicators['total_market_cap'] = float(market_caps.sum())
                indicators['avg_market_cap'] = float(market_caps.mean())
                indicators['market_cap_concentration'] = float(market_caps.std() / market_caps.mean()) if market_caps.mean() != 0 else 0.0
        
        if 'price' in df.columns:
            prices = df['price'].dropna()
            if len(prices) > 1:
                price_changes = prices.pct_change().dropna()
                indicators['market_return'] = float(price_changes.mean())
                indicators['market_volatility'] = float(price_changes.std())
        
        if 'pe_ratio' in df.columns:
            pe_ratios = df['pe_ratio'].dropna()
            if len(pe_ratios) > 0:
                indicators['avg_pe_ratio'] = float(pe_ratios.mean())
                indicators['median_pe_ratio'] = float(pe_ratios.median())
        
        return indicators
    
    def _calculate_liquidity_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market liquidity indicators"""
        indicators = {}
        
        if 'volume' in df.columns:
            volumes = df['volume'].dropna()
            if len(volumes) > 0:
                indicators['total_volume'] = float(volumes.sum())
                indicators['avg_volume'] = float(volumes.mean())
                indicators['volume_concentration'] = float(volumes.std() / volumes.mean()) if volumes.mean() != 0 else 0.0
        
        if 'bid_ask_spread' in df.columns:
            spreads = df['bid_ask_spread'].dropna()
            if len(spreads) > 0:
                indicators['avg_bid_ask_spread'] = float(spreads.mean())
                indicators['median_bid_ask_spread'] = float(spreads.median())
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum indicators"""
        indicators = {}
        
        if 'price' in df.columns and len(df) > 1:
            prices = df['price'].dropna()
            if len(prices) > 1:
                returns = prices.pct_change().dropna()
                
                # Momentum score
                positive_returns = (returns > 0).sum()
                total_returns = len(returns)
                indicators['momentum_score'] = float(positive_returns / total_returns) if total_returns > 0 else 0.0
                
                # Average return
                indicators['avg_return'] = float(returns.mean())
                
                # Return skewness
                if len(returns) > 2:
                    indicators['return_skewness'] = float(returns.skew())
        
        return indicators
    
    def _calculate_breadth_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        indicators = {}
        
        if 'price' in df.columns and len(df) > 1:
            # Advance/Decline ratio
            price_changes = df['price'].pct_change().dropna()
            advances = (price_changes > 0).sum()
            declines = (price_changes < 0).sum()
            
            indicators['advance_decline_ratio'] = float(advances / declines) if declines > 0 else float('inf')
            indicators['advance_percentage'] = float(advances / len(price_changes)) if len(price_changes) > 0 else 0.0
        
        return indicators
