"""
Feature aggregation module for Stage 2.
Handles time series aggregation, cross-sectional features, and rolling metrics.
"""

from .time_series_agg import TimeSeriesAggregator
from .cross_sectional_agg import CrossSectionalAggregator
from .rolling_metrics import RollingMetricsCalculator

__all__ = [
    'TimeSeriesAggregator',
    'CrossSectionalAggregator', 
    'RollingMetricsCalculator'
]
