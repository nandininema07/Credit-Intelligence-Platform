"""
Trend monitoring for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TrendDirection(Enum):
    """Trend direction types"""
    UPWARD = "upward"
    DOWNWARD = "downward"
    STABLE = "stable"
    VOLATILE = "volatile"

class TrendStrength(Enum):
    """Trend strength levels"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    company_id: str
    factor: str
    direction: TrendDirection
    strength: TrendStrength
    slope: float
    r_squared: float
    duration_days: int
    start_value: float
    end_value: float
    change_amount: float
    change_percentage: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]

class TrendMonitor:
    """Monitor trends in credit data over time"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.historical_data = {}
        self.trend_cache = {}
        self.trend_alerts = []
        self.statistics = {
            'trends_analyzed': 0,
            'upward_trends': 0,
            'downward_trends': 0,
            'trend_reversals': 0,
            'volatile_periods': 0
        }
        self._initialize_monitor()
    
    def _initialize_monitor(self):
        """Initialize trend monitor"""
        
        self.settings = {
            'min_data_points': self.config.get('min_data_points', 10),
            'trend_window_days': self.config.get('trend_window_days', 30),
            'short_term_days': self.config.get('short_term_days', 7),
            'medium_term_days': self.config.get('medium_term_days', 30),
            'long_term_days': self.config.get('long_term_days', 90),
            'volatility_threshold': self.config.get('volatility_threshold', 0.15),
            'trend_strength_threshold': self.config.get('trend_strength_threshold', 0.7),
            'reversal_threshold': self.config.get('reversal_threshold', 0.5)
        }
    
    async def analyze_trends(self, company_id: str, data: Dict[str, Any]) -> List[TrendAnalysis]:
        """Analyze trends for all factors in the data"""
        
        try:
            # Update historical data
            await self._update_historical_data(company_id, data)
            
            trend_analyses = []
            
            # Analyze trends for each factor
            for factor, value in data.items():
                if not isinstance(value, (int, float)):
                    continue
                
                # Get different time window analyses
                analyses = await self._analyze_factor_trends(company_id, factor)
                trend_analyses.extend(analyses)
            
            # Update statistics
            for analysis in trend_analyses:
                self._update_statistics(analysis)
            
            # Check for trend alerts
            alerts = await self._check_trend_alerts(trend_analyses)
            self.trend_alerts.extend(alerts)
            
            return trend_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return []
    
    async def _update_historical_data(self, company_id: str, data: Dict[str, Any]):
        """Update historical data for trend analysis"""
        
        try:
            if company_id not in self.historical_data:
                self.historical_data[company_id] = {}
            
            timestamp = datetime.now()
            
            for factor, value in data.items():
                if not isinstance(value, (int, float)):
                    continue
                
                if factor not in self.historical_data[company_id]:
                    self.historical_data[company_id][factor] = []
                
                self.historical_data[company_id][factor].append({
                    'timestamp': timestamp,
                    'value': value
                })
                
                # Limit historical data
                max_points = self.config.get('max_historical_points', 2000)
                if len(self.historical_data[company_id][factor]) > max_points:
                    self.historical_data[company_id][factor] = \
                        self.historical_data[company_id][factor][-max_points//2:]
            
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
    
    async def _analyze_factor_trends(self, company_id: str, factor: str) -> List[TrendAnalysis]:
        """Analyze trends for a specific factor across different time windows"""
        
        try:
            analyses = []
            
            # Define time windows
            time_windows = [
                ('short_term', self.settings['short_term_days']),
                ('medium_term', self.settings['medium_term_days']),
                ('long_term', self.settings['long_term_days'])
            ]
            
            for window_name, days in time_windows:
                analysis = await self._analyze_single_trend(company_id, factor, days, window_name)
                if analysis:
                    analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error analyzing factor trends: {e}")
            return []
    
    async def _analyze_single_trend(self, company_id: str, factor: str, 
                                  days: int, window_name: str) -> Optional[TrendAnalysis]:
        """Analyze trend for a single time window"""
        
        try:
            # Get historical data for the time window
            data_points = await self._get_time_window_data(company_id, factor, days)
            
            if len(data_points) < self.settings['min_data_points']:
                return None
            
            # Extract values and timestamps
            timestamps = [point['timestamp'] for point in data_points]
            values = [point['value'] for point in data_points]
            
            # Convert timestamps to numeric values for regression
            start_time = timestamps[0]
            x_values = [(ts - start_time).total_seconds() / 86400 for ts in timestamps]  # Days
            
            # Perform linear regression
            slope, r_squared = self._calculate_linear_trend(x_values, values)
            
            # Determine trend direction and strength
            direction = self._determine_trend_direction(slope, values)
            strength = self._determine_trend_strength(r_squared, slope, values)
            
            # Calculate change metrics
            start_value = values[0]
            end_value = values[-1]
            change_amount = end_value - start_value
            change_percentage = (change_amount / start_value * 100) if start_value != 0 else 0
            
            # Calculate confidence
            confidence = self._calculate_trend_confidence(r_squared, len(values), direction)
            
            # Create trend analysis
            analysis = TrendAnalysis(
                company_id=company_id,
                factor=factor,
                direction=direction,
                strength=strength,
                slope=slope,
                r_squared=r_squared,
                duration_days=days,
                start_value=start_value,
                end_value=end_value,
                change_amount=change_amount,
                change_percentage=change_percentage,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'window_name': window_name,
                    'data_points': len(values),
                    'volatility': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    'min_value': min(values),
                    'max_value': max(values),
                    'mean_value': np.mean(values)
                }
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing single trend: {e}")
            return None
    
    async def _get_time_window_data(self, company_id: str, factor: str, days: int) -> List[Dict[str, Any]]:
        """Get data points for a specific time window"""
        
        try:
            if (company_id not in self.historical_data or 
                factor not in self.historical_data[company_id]):
                return []
            
            cutoff_time = datetime.now() - timedelta(days=days)
            
            return [
                point for point in self.historical_data[company_id][factor]
                if point['timestamp'] >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error getting time window data: {e}")
            return []
    
    def _calculate_linear_trend(self, x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
        """Calculate linear trend using least squares regression"""
        
        try:
            if len(x_values) < 2:
                return 0.0, 0.0
            
            # Calculate slope and intercept
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            # Calculate slope
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0.0, 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Calculate R-squared
            y_mean = sum_y / n
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            
            if ss_tot == 0:
                r_squared = 1.0 if slope == 0 else 0.0
            else:
                intercept = (sum_y - slope * sum_x) / n
                ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
                r_squared = 1 - (ss_res / ss_tot)
            
            return slope, max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            logger.error(f"Error calculating linear trend: {e}")
            return 0.0, 0.0
    
    def _determine_trend_direction(self, slope: float, values: List[float]) -> TrendDirection:
        """Determine trend direction based on slope and volatility"""
        
        try:
            # Calculate volatility
            volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            if volatility > self.settings['volatility_threshold']:
                return TrendDirection.VOLATILE
            
            # Determine direction based on slope
            if abs(slope) < 0.01:  # Very small slope
                return TrendDirection.STABLE
            elif slope > 0:
                return TrendDirection.UPWARD
            else:
                return TrendDirection.DOWNWARD
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return TrendDirection.STABLE
    
    def _determine_trend_strength(self, r_squared: float, slope: float, values: List[float]) -> TrendStrength:
        """Determine trend strength based on R-squared and slope magnitude"""
        
        try:
            # Normalize slope by value range
            value_range = max(values) - min(values)
            normalized_slope = abs(slope) / value_range if value_range > 0 else 0
            
            # Combine R-squared and normalized slope
            strength_score = r_squared * (1 + normalized_slope)
            
            if strength_score > 0.8:
                return TrendStrength.VERY_STRONG
            elif strength_score > 0.6:
                return TrendStrength.STRONG
            elif strength_score > 0.4:
                return TrendStrength.MODERATE
            else:
                return TrendStrength.WEAK
                
        except Exception as e:
            logger.error(f"Error determining trend strength: {e}")
            return TrendStrength.WEAK
    
    def _calculate_trend_confidence(self, r_squared: float, data_points: int, 
                                  direction: TrendDirection) -> float:
        """Calculate confidence in the trend analysis"""
        
        try:
            base_confidence = r_squared
            
            # Adjust for number of data points
            data_factor = min(1.0, data_points / 20)  # Full confidence at 20+ points
            
            # Adjust for trend type
            direction_factor = 1.0
            if direction == TrendDirection.VOLATILE:
                direction_factor = 0.7
            elif direction == TrendDirection.STABLE:
                direction_factor = 0.8
            
            confidence = base_confidence * data_factor * direction_factor
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating trend confidence: {e}")
            return 0.5
    
    def _update_statistics(self, analysis: TrendAnalysis):
        """Update trend monitoring statistics"""
        
        try:
            self.statistics['trends_analyzed'] += 1
            
            if analysis.direction == TrendDirection.UPWARD:
                self.statistics['upward_trends'] += 1
            elif analysis.direction == TrendDirection.DOWNWARD:
                self.statistics['downward_trends'] += 1
            elif analysis.direction == TrendDirection.VOLATILE:
                self.statistics['volatile_periods'] += 1
                
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def _check_trend_alerts(self, analyses: List[TrendAnalysis]) -> List[Dict[str, Any]]:
        """Check for trend-based alerts"""
        
        try:
            alerts = []
            
            for analysis in analyses:
                # Check for significant downward trends
                if (analysis.direction == TrendDirection.DOWNWARD and 
                    analysis.strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG] and
                    analysis.confidence > 0.7):
                    
                    alerts.append({
                        'type': 'downward_trend',
                        'company_id': analysis.company_id,
                        'factor': analysis.factor,
                        'severity': 'high' if analysis.strength == TrendStrength.VERY_STRONG else 'medium',
                        'analysis': analysis,
                        'timestamp': datetime.now()
                    })
                
                # Check for trend reversals
                reversal = await self._detect_trend_reversal(analysis)
                if reversal:
                    alerts.append(reversal)
                
                # Check for high volatility
                if (analysis.direction == TrendDirection.VOLATILE and 
                    analysis.metadata.get('volatility', 0) > self.settings['volatility_threshold']):
                    
                    alerts.append({
                        'type': 'high_volatility',
                        'company_id': analysis.company_id,
                        'factor': analysis.factor,
                        'severity': 'medium',
                        'analysis': analysis,
                        'timestamp': datetime.now()
                    })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking trend alerts: {e}")
            return []
    
    async def _detect_trend_reversal(self, current_analysis: TrendAnalysis) -> Optional[Dict[str, Any]]:
        """Detect if there's a trend reversal"""
        
        try:
            # Get previous trend analysis for comparison
            cache_key = f"{current_analysis.company_id}_{current_analysis.factor}_{current_analysis.metadata['window_name']}"
            
            if cache_key not in self.trend_cache:
                self.trend_cache[cache_key] = current_analysis
                return None
            
            previous_analysis = self.trend_cache[cache_key]
            
            # Check for direction reversal
            if (previous_analysis.direction == TrendDirection.UPWARD and 
                current_analysis.direction == TrendDirection.DOWNWARD) or \
               (previous_analysis.direction == TrendDirection.DOWNWARD and 
                current_analysis.direction == TrendDirection.UPWARD):
                
                # Check if reversal is significant
                if (current_analysis.confidence > self.settings['reversal_threshold'] and
                    current_analysis.strength in [TrendStrength.MODERATE, TrendStrength.STRONG, TrendStrength.VERY_STRONG]):
                    
                    self.statistics['trend_reversals'] += 1
                    
                    # Update cache
                    self.trend_cache[cache_key] = current_analysis
                    
                    return {
                        'type': 'trend_reversal',
                        'company_id': current_analysis.company_id,
                        'factor': current_analysis.factor,
                        'severity': 'high',
                        'previous_direction': previous_analysis.direction.value,
                        'current_direction': current_analysis.direction.value,
                        'analysis': current_analysis,
                        'timestamp': datetime.now()
                    }
            
            # Update cache
            self.trend_cache[cache_key] = current_analysis
            return None
            
        except Exception as e:
            logger.error(f"Error detecting trend reversal: {e}")
            return None
    
    async def get_trend_analysis(self, company_id: str, factor: str = None, 
                               days: int = 30) -> List[TrendAnalysis]:
        """Get trend analysis for a company and optional factor"""
        
        try:
            if company_id not in self.historical_data:
                return []
            
            analyses = []
            factors = [factor] if factor else self.historical_data[company_id].keys()
            
            for f in factors:
                analysis = await self._analyze_single_trend(company_id, f, days, 'custom')
                if analysis:
                    analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error getting trend analysis: {e}")
            return []
    
    async def get_trend_alerts(self, company_id: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trend alerts with optional filtering"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_alerts = []
            
            for alert in self.trend_alerts:
                if alert['timestamp'] < cutoff_time:
                    continue
                
                if company_id and alert['company_id'] != company_id:
                    continue
                
                filtered_alerts.append(alert)
            
            # Sort by timestamp (most recent first)
            filtered_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return filtered_alerts
            
        except Exception as e:
            logger.error(f"Error getting trend alerts: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get trend monitoring statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Add current counts
            stats.update({
                'companies_monitored': len(self.historical_data),
                'factors_monitored': sum(len(factors) for factors in self.historical_data.values()),
                'cached_trends': len(self.trend_cache),
                'recent_alerts_24h': len(await self.get_trend_alerts(hours=24)),
                'settings': self.settings.copy()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    async def update_settings(self, new_settings: Dict[str, Any]):
        """Update trend monitoring settings"""
        
        try:
            self.settings.update(new_settings)
            logger.info(f"Updated trend monitoring settings: {new_settings}")
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
    
    async def clear_cache(self, company_id: str = None):
        """Clear trend cache"""
        
        try:
            if company_id:
                # Clear cache for specific company
                keys_to_remove = [key for key in self.trend_cache.keys() if key.startswith(f"{company_id}_")]
                for key in keys_to_remove:
                    del self.trend_cache[key]
            else:
                # Clear all cache
                self.trend_cache.clear()
            
            logger.info("Cleared trend cache")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    async def export_trend_data(self, company_id: str, factor: str = None) -> Dict[str, Any]:
        """Export trend data for analysis"""
        
        try:
            if company_id not in self.historical_data:
                return {'error': 'Company not found'}
            
            export_data = {
                'company_id': company_id,
                'exported_at': datetime.now().isoformat(),
                'factors': {}
            }
            
            factors = [factor] if factor else self.historical_data[company_id].keys()
            
            for f in factors:
                if f in self.historical_data[company_id]:
                    export_data['factors'][f] = {
                        'data_points': [
                            {
                                'timestamp': point['timestamp'].isoformat(),
                                'value': point['value']
                            }
                            for point in self.historical_data[company_id][f]
                        ],
                        'trend_analyses': []
                    }
                    
                    # Add trend analyses for different time windows
                    for days in [7, 30, 90]:
                        analysis = await self._analyze_single_trend(company_id, f, days, f'{days}d')
                        if analysis:
                            export_data['factors'][f]['trend_analyses'].append({
                                'window_days': days,
                                'direction': analysis.direction.value,
                                'strength': analysis.strength.value,
                                'slope': analysis.slope,
                                'r_squared': analysis.r_squared,
                                'change_percentage': analysis.change_percentage,
                                'confidence': analysis.confidence
                            })
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting trend data: {e}")
            return {'error': str(e)}
