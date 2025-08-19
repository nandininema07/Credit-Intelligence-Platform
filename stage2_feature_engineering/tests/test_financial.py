"""
Tests for financial components in Stage 2 feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from ..financial.ratio_calculator import RatioCalculator
from ..financial.trend_analyzer import TrendAnalyzer
from ..financial.volatility_metrics import VolatilityMetrics
from ..financial.market_indicators import MarketIndicators

@pytest.fixture
def sample_financial_data():
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'date': dates,
        'company': ['AAPL'] * 100,
        'revenue': np.random.uniform(80, 120, 100),
        'profit': np.random.uniform(15, 25, 100),
        'debt': np.random.uniform(50, 70, 100),
        'equity': np.random.uniform(100, 150, 100),
        'stock_price': 150 + np.cumsum(np.random.normal(0, 2, 100)),
        'volume': np.random.uniform(1000000, 5000000, 100),
        'market_cap': np.random.uniform(2000000, 3000000, 100)
    })

@pytest.fixture
def sample_config():
    return {
        'ratios': {
            'debt_to_equity': True,
            'profit_margin': True,
            'roe': True
        },
        'volatility_windows': [10, 20, 50],
        'trend_periods': [5, 10, 20]
    }

class TestRatioCalculator:
    """Test financial ratio calculations"""
    
    def test_ratio_calculator_init(self, sample_config):
        calculator = RatioCalculator(sample_config)
        assert calculator.config == sample_config
    
    def test_debt_to_equity_ratio(self, sample_config, sample_financial_data):
        calculator = RatioCalculator(sample_config)
        
        result = calculator.calculate_debt_to_equity(
            sample_financial_data['debt'],
            sample_financial_data['equity']
        )
        
        assert len(result) == len(sample_financial_data)
        assert all(isinstance(x, (int, float)) or pd.isna(x) for x in result)
        assert all(x >= 0 for x in result if not pd.isna(x))
    
    def test_profit_margin(self, sample_config, sample_financial_data):
        calculator = RatioCalculator(sample_config)
        
        result = calculator.calculate_profit_margin(
            sample_financial_data['profit'],
            sample_financial_data['revenue']
        )
        
        assert len(result) == len(sample_financial_data)
        assert all(0 <= x <= 1 for x in result if not pd.isna(x))
    
    def test_return_on_equity(self, sample_config, sample_financial_data):
        calculator = RatioCalculator(sample_config)
        
        result = calculator.calculate_roe(
            sample_financial_data['profit'],
            sample_financial_data['equity']
        )
        
        assert len(result) == len(sample_financial_data)
        assert all(isinstance(x, (int, float)) or pd.isna(x) for x in result)
    
    def test_calculate_all_ratios(self, sample_config, sample_financial_data):
        calculator = RatioCalculator(sample_config)
        
        result = calculator.calculate_all_ratios(sample_financial_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'debt_to_equity' in result.columns
        assert 'profit_margin' in result.columns
        assert 'roe' in result.columns
        assert len(result) == len(sample_financial_data)
    
    def test_ratio_with_zero_values(self, sample_config):
        calculator = RatioCalculator(sample_config)
        
        # Test division by zero handling
        debt = pd.Series([10, 20, 30])
        equity = pd.Series([5, 0, 10])  # Zero equity
        
        result = calculator.calculate_debt_to_equity(debt, equity)
        
        assert len(result) == 3
        assert not pd.isna(result.iloc[0])  # Normal case
        assert pd.isna(result.iloc[1]) or np.isinf(result.iloc[1])  # Division by zero
        assert not pd.isna(result.iloc[2])  # Normal case

class TestTrendAnalyzer:
    """Test trend analysis functionality"""
    
    def test_trend_analyzer_init(self, sample_config):
        analyzer = TrendAnalyzer(sample_config)
        assert analyzer.config == sample_config
    
    def test_calculate_trend(self, sample_config, sample_financial_data):
        analyzer = TrendAnalyzer(sample_config)
        
        trend = analyzer.calculate_trend(
            sample_financial_data['stock_price'],
            window=10
        )
        
        assert len(trend) == len(sample_financial_data)
        assert all(isinstance(x, (int, float)) or pd.isna(x) for x in trend)
    
    def test_moving_averages(self, sample_config, sample_financial_data):
        analyzer = TrendAnalyzer(sample_config)
        
        ma = analyzer.calculate_moving_average(
            sample_financial_data['stock_price'],
            window=20
        )
        
        assert len(ma) == len(sample_financial_data)
        # First 19 values should be NaN
        assert pd.isna(ma.iloc[:19]).all()
        # Rest should have values
        assert not pd.isna(ma.iloc[19:]).any()
    
    def test_trend_strength(self, sample_config):
        analyzer = TrendAnalyzer(sample_config)
        
        # Create artificial trend data
        uptrend = pd.Series(range(100))  # Strong uptrend
        downtrend = pd.Series(range(100, 0, -1))  # Strong downtrend
        sideways = pd.Series([50] * 100)  # No trend
        
        up_strength = analyzer.calculate_trend_strength(uptrend)
        down_strength = analyzer.calculate_trend_strength(downtrend)
        side_strength = analyzer.calculate_trend_strength(sideways)
        
        assert up_strength > 0.5  # Strong positive trend
        assert down_strength < -0.5  # Strong negative trend
        assert abs(side_strength) < 0.1  # Weak trend

class TestVolatilityMetrics:
    """Test volatility calculations"""
    
    def test_volatility_metrics_init(self, sample_config):
        calculator = VolatilityMetrics(sample_config)
        assert calculator.config == sample_config
    
    def test_historical_volatility(self, sample_config, sample_financial_data):
        calculator = VolatilityMetrics(sample_config)
        
        volatility = calculator.calculate_historical_volatility(
            sample_financial_data['stock_price'],
            window=20
        )
        
        assert len(volatility) == len(sample_financial_data)
        assert all(x >= 0 for x in volatility if not pd.isna(x))
    
    def test_returns_calculation(self, sample_config, sample_financial_data):
        calculator = VolatilityMetrics(sample_config)
        
        returns = calculator.calculate_returns(sample_financial_data['stock_price'])
        
        assert len(returns) == len(sample_financial_data)
        # First return should be NaN
        assert pd.isna(returns.iloc[0])
        # Rest should have values
        assert not pd.isna(returns.iloc[1:]).any()
    
    def test_volatility_with_constant_prices(self, sample_config):
        calculator = VolatilityMetrics(sample_config)
        
        # Constant prices should have zero volatility
        constant_prices = pd.Series([100] * 50)
        volatility = calculator.calculate_historical_volatility(constant_prices, window=10)
        
        # Should be zero or very close to zero (excluding NaN values)
        non_nan_volatility = volatility.dropna()
        assert all(abs(x) < 1e-10 for x in non_nan_volatility)

class TestMarketIndicators:
    """Test market indicator calculations"""
    
    def test_market_indicators_init(self, sample_config):
        calculator = MarketIndicators(sample_config)
        assert calculator.config == sample_config
    
    def test_rsi_calculation(self, sample_config, sample_financial_data):
        calculator = MarketIndicators(sample_config)
        
        rsi = calculator.calculate_rsi(
            sample_financial_data['stock_price'],
            window=14
        )
        
        assert len(rsi) == len(sample_financial_data)
        # RSI should be between 0 and 100
        non_nan_rsi = rsi.dropna()
        assert all(0 <= x <= 100 for x in non_nan_rsi)
    
    def test_bollinger_bands(self, sample_config, sample_financial_data):
        calculator = MarketIndicators(sample_config)
        
        upper, middle, lower = calculator.calculate_bollinger_bands(
            sample_financial_data['stock_price'],
            window=20
        )
        
        assert len(upper) == len(sample_financial_data)
        assert len(middle) == len(sample_financial_data)
        assert len(lower) == len(sample_financial_data)
        
        # Upper should be >= middle >= lower (where not NaN)
        for i in range(len(upper)):
            if not any(pd.isna([upper.iloc[i], middle.iloc[i], lower.iloc[i]])):
                assert upper.iloc[i] >= middle.iloc[i] >= lower.iloc[i]
    
    def test_macd_calculation(self, sample_config, sample_financial_data):
        calculator = MarketIndicators(sample_config)
        
        macd, signal = calculator.calculate_macd(
            sample_financial_data['stock_price']
        )
        
        assert len(macd) == len(sample_financial_data)
        assert len(signal) == len(sample_financial_data)

@pytest.mark.integration
class TestFinancialIntegration:
    """Test financial component integration"""
    
    def test_financial_pipeline_integration(self, sample_config, sample_financial_data):
        """Test integrated financial processing pipeline"""
        
        # Initialize components
        ratio_calc = RatioCalculator(sample_config)
        trend_analyzer = TrendAnalyzer(sample_config)
        volatility_calc = VolatilityMetrics(sample_config)
        market_indicators = MarketIndicators(sample_config)
        
        # Process through pipeline
        ratios = ratio_calc.calculate_all_ratios(sample_financial_data)
        trends = trend_analyzer.calculate_all_trends(sample_financial_data)
        volatility = volatility_calc.calculate_all_volatility_metrics(sample_financial_data)
        indicators = market_indicators.calculate_all_indicators(sample_financial_data)
        
        # Combine results
        combined = pd.concat([ratios, trends, volatility, indicators], axis=1)
        
        assert len(combined) == len(sample_financial_data)
        assert combined.shape[1] > sample_financial_data.shape[1]  # More features added
    
    def test_financial_error_handling(self, sample_config):
        """Test financial component error handling"""
        calculator = RatioCalculator(sample_config)
        
        # Test with empty data
        empty_series = pd.Series([], dtype=float)
        result = calculator.calculate_debt_to_equity(empty_series, empty_series)
        assert len(result) == 0
        
        # Test with mismatched lengths
        short_series = pd.Series([1, 2])
        long_series = pd.Series([1, 2, 3, 4])
        
        try:
            result = calculator.calculate_debt_to_equity(short_series, long_series)
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            assert isinstance(e, (ValueError, IndexError))
    
    def test_financial_data_validation(self, sample_config):
        """Test financial data validation"""
        calculator = RatioCalculator(sample_config)
        
        # Test with negative values
        negative_data = pd.Series([-10, -20, -30])
        positive_data = pd.Series([10, 20, 30])
        
        result = calculator.calculate_debt_to_equity(negative_data, positive_data)
        
        # Should handle negative values appropriately
        assert len(result) == 3
        assert all(isinstance(x, (int, float)) or pd.isna(x) for x in result)
