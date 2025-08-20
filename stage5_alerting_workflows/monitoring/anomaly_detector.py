"""
Anomaly detection for Stage 5 alerting workflows.
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

class AnomalyType(Enum):
    """Types of anomalies"""
    POINT_ANOMALY = "point_anomaly"
    CONTEXTUAL_ANOMALY = "contextual_anomaly"
    COLLECTIVE_ANOMALY = "collective_anomaly"

class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Anomaly:
    """Detected anomaly"""
    id: str
    company_id: str
    factor: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: datetime
    value: float
    expected_value: float
    deviation_score: float
    confidence: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class AnomalyDetector:
    """Detect anomalies in credit data using multiple methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.historical_data = {}
        self.anomaly_models = {}
        self.detected_anomalies = []
        self.statistics = {
            'total_anomalies': 0,
            'point_anomalies': 0,
            'contextual_anomalies': 0,
            'collective_anomalies': 0,
            'false_positives': 0
        }
        self._initialize_detector()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("AnomalyDetector initialized successfully")
        return True
    
    def _initialize_detector(self):
        """Initialize anomaly detector"""
        
        self.detection_methods = {
            'statistical': self._statistical_detection,
            'isolation_forest': self._isolation_forest_detection,
            'z_score': self._z_score_detection,
            'iqr': self._iqr_detection,
            'moving_average': self._moving_average_detection
        }
        
        self.thresholds = {
            'z_score_threshold': self.config.get('z_score_threshold', 2.5),
            'iqr_multiplier': self.config.get('iqr_multiplier', 1.5),
            'isolation_contamination': self.config.get('isolation_contamination', 0.1),
            'moving_window_size': self.config.get('moving_window_size', 20),
            'min_data_points': self.config.get('min_data_points', 10)
        }
    
    async def detect_anomalies(self, company_id: str, data: Dict[str, Any]) -> List[Anomaly]:
        """Detect anomalies in the provided data"""
        
        try:
            anomalies = []
            
            # Update historical data
            await self._update_historical_data(company_id, data)
            
            # Check each factor for anomalies
            for factor, value in data.items():
                if not isinstance(value, (int, float)):
                    continue
                
                # Run multiple detection methods
                factor_anomalies = await self._detect_factor_anomalies(
                    company_id, factor, value
                )
                
                anomalies.extend(factor_anomalies)
            
            # Update statistics
            for anomaly in anomalies:
                self._update_statistics(anomaly)
            
            # Store detected anomalies
            self.detected_anomalies.extend(anomalies)
            
            # Limit stored anomalies
            if len(self.detected_anomalies) > 10000:
                self.detected_anomalies = self.detected_anomalies[-5000:]
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def _update_historical_data(self, company_id: str, data: Dict[str, Any]):
        """Update historical data for the company"""
        
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
                
                # Limit historical data size
                max_points = self.config.get('max_historical_points', 1000)
                if len(self.historical_data[company_id][factor]) > max_points:
                    self.historical_data[company_id][factor] = \
                        self.historical_data[company_id][factor][-max_points//2:]
            
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
    
    async def _detect_factor_anomalies(self, company_id: str, factor: str, 
                                     current_value: float) -> List[Anomaly]:
        """Detect anomalies for a specific factor"""
        
        try:
            anomalies = []
            
            # Get historical data for this factor
            historical_values = await self._get_historical_values(company_id, factor)
            
            if len(historical_values) < self.thresholds['min_data_points']:
                return anomalies
            
            # Run detection methods
            detection_results = {}
            
            for method_name, method_func in self.detection_methods.items():
                try:
                    result = await method_func(historical_values, current_value)
                    detection_results[method_name] = result
                except Exception as e:
                    logger.error(f"Error in {method_name} detection: {e}")
                    detection_results[method_name] = None
            
            # Combine results and determine if anomaly exists
            anomaly = await self._combine_detection_results(
                company_id, factor, current_value, historical_values, detection_results
            )
            
            if anomaly:
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting factor anomalies: {e}")
            return []
    
    async def _get_historical_values(self, company_id: str, factor: str) -> List[float]:
        """Get historical values for a factor"""
        
        try:
            if (company_id not in self.historical_data or 
                factor not in self.historical_data[company_id]):
                return []
            
            return [entry['value'] for entry in self.historical_data[company_id][factor]]
            
        except Exception as e:
            logger.error(f"Error getting historical values: {e}")
            return []
    
    async def _statistical_detection(self, historical_values: List[float], 
                                   current_value: float) -> Dict[str, Any]:
        """Statistical anomaly detection"""
        
        try:
            if len(historical_values) < 3:
                return None
            
            mean_val = np.mean(historical_values)
            std_val = np.std(historical_values)
            
            if std_val == 0:
                return None
            
            z_score = abs(current_value - mean_val) / std_val
            
            return {
                'is_anomaly': z_score > self.thresholds['z_score_threshold'],
                'score': z_score,
                'expected_value': mean_val,
                'method': 'statistical'
            }
            
        except Exception as e:
            logger.error(f"Error in statistical detection: {e}")
            return None
    
    async def _isolation_forest_detection(self, historical_values: List[float], 
                                        current_value: float) -> Dict[str, Any]:
        """Isolation forest anomaly detection"""
        
        try:
            # Simplified isolation forest implementation
            # In production, use sklearn.ensemble.IsolationForest
            
            if len(historical_values) < 10:
                return None
            
            # Calculate isolation score based on value position
            sorted_values = sorted(historical_values)
            n = len(sorted_values)
            
            # Find position of current value
            position = 0
            for i, val in enumerate(sorted_values):
                if current_value <= val:
                    position = i
                    break
            else:
                position = n
            
            # Calculate isolation score (simplified)
            isolation_score = min(position, n - position) / (n / 2)
            
            return {
                'is_anomaly': isolation_score < self.thresholds['isolation_contamination'],
                'score': 1 - isolation_score,
                'expected_value': np.median(historical_values),
                'method': 'isolation_forest'
            }
            
        except Exception as e:
            logger.error(f"Error in isolation forest detection: {e}")
            return None
    
    async def _z_score_detection(self, historical_values: List[float], 
                               current_value: float) -> Dict[str, Any]:
        """Z-score based anomaly detection"""
        
        try:
            if len(historical_values) < 3:
                return None
            
            mean_val = np.mean(historical_values)
            std_val = np.std(historical_values)
            
            if std_val == 0:
                return None
            
            z_score = (current_value - mean_val) / std_val
            
            return {
                'is_anomaly': abs(z_score) > self.thresholds['z_score_threshold'],
                'score': abs(z_score),
                'expected_value': mean_val,
                'method': 'z_score'
            }
            
        except Exception as e:
            logger.error(f"Error in z-score detection: {e}")
            return None
    
    async def _iqr_detection(self, historical_values: List[float], 
                           current_value: float) -> Dict[str, Any]:
        """IQR-based anomaly detection"""
        
        try:
            if len(historical_values) < 5:
                return None
            
            q1 = np.percentile(historical_values, 25)
            q3 = np.percentile(historical_values, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                return None
            
            lower_bound = q1 - self.thresholds['iqr_multiplier'] * iqr
            upper_bound = q3 + self.thresholds['iqr_multiplier'] * iqr
            
            is_anomaly = current_value < lower_bound or current_value > upper_bound
            
            return {
                'is_anomaly': is_anomaly,
                'score': max(
                    (lower_bound - current_value) / iqr if current_value < lower_bound else 0,
                    (current_value - upper_bound) / iqr if current_value > upper_bound else 0
                ),
                'expected_value': np.median(historical_values),
                'method': 'iqr'
            }
            
        except Exception as e:
            logger.error(f"Error in IQR detection: {e}")
            return None
    
    async def _moving_average_detection(self, historical_values: List[float], 
                                      current_value: float) -> Dict[str, Any]:
        """Moving average based anomaly detection"""
        
        try:
            window_size = min(self.thresholds['moving_window_size'], len(historical_values))
            
            if window_size < 3:
                return None
            
            recent_values = historical_values[-window_size:]
            moving_avg = np.mean(recent_values)
            moving_std = np.std(recent_values)
            
            if moving_std == 0:
                return None
            
            deviation = abs(current_value - moving_avg) / moving_std
            
            return {
                'is_anomaly': deviation > 2.0,
                'score': deviation,
                'expected_value': moving_avg,
                'method': 'moving_average'
            }
            
        except Exception as e:
            logger.error(f"Error in moving average detection: {e}")
            return None
    
    async def _combine_detection_results(self, company_id: str, factor: str, 
                                       current_value: float, historical_values: List[float],
                                       detection_results: Dict[str, Any]) -> Optional[Anomaly]:
        """Combine results from multiple detection methods"""
        
        try:
            # Filter out None results
            valid_results = {k: v for k, v in detection_results.items() if v is not None}
            
            if not valid_results:
                return None
            
            # Count positive detections
            positive_detections = sum(1 for result in valid_results.values() 
                                    if result['is_anomaly'])
            
            # Require majority consensus
            if positive_detections < len(valid_results) / 2:
                return None
            
            # Calculate combined score
            scores = [result['score'] for result in valid_results.values()]
            combined_score = np.mean(scores)
            
            # Calculate expected value
            expected_values = [result['expected_value'] for result in valid_results.values()]
            expected_value = np.mean(expected_values)
            
            # Determine severity
            severity = self._determine_severity(combined_score, factor)
            
            # Determine anomaly type
            anomaly_type = self._determine_anomaly_type(
                current_value, historical_values, factor
            )
            
            # Calculate confidence
            confidence = min(0.95, positive_detections / len(valid_results))
            
            # Create anomaly
            anomaly = Anomaly(
                id=f"{company_id}_{factor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                company_id=company_id,
                factor=factor,
                anomaly_type=anomaly_type,
                severity=severity,
                timestamp=datetime.now(),
                value=current_value,
                expected_value=expected_value,
                deviation_score=combined_score,
                confidence=confidence,
                context={
                    'detection_methods': list(valid_results.keys()),
                    'positive_detections': positive_detections,
                    'total_methods': len(valid_results),
                    'historical_mean': np.mean(historical_values),
                    'historical_std': np.std(historical_values)
                },
                metadata={
                    'detection_results': valid_results,
                    'historical_data_points': len(historical_values)
                }
            )
            
            return anomaly
            
        except Exception as e:
            logger.error(f"Error combining detection results: {e}")
            return None
    
    def _determine_severity(self, score: float, factor: str) -> AnomalySeverity:
        """Determine anomaly severity based on score and factor"""
        
        try:
            # Adjust thresholds based on factor importance
            factor_weights = {
                'credit_score': 1.2,
                'payment_history': 1.1,
                'credit_utilization': 1.0,
                'debt_to_income': 1.0
            }
            
            weight = factor_weights.get(factor, 1.0)
            adjusted_score = score * weight
            
            if adjusted_score > 4.0:
                return AnomalySeverity.CRITICAL
            elif adjusted_score > 3.0:
                return AnomalySeverity.HIGH
            elif adjusted_score > 2.0:
                return AnomalySeverity.MEDIUM
            else:
                return AnomalySeverity.LOW
                
        except Exception as e:
            logger.error(f"Error determining severity: {e}")
            return AnomalySeverity.LOW
    
    def _determine_anomaly_type(self, current_value: float, 
                              historical_values: List[float], factor: str) -> AnomalyType:
        """Determine the type of anomaly"""
        
        try:
            # Simple heuristics for anomaly type classification
            recent_values = historical_values[-5:] if len(historical_values) >= 5 else historical_values
            
            # Check if it's a collective anomaly (pattern change)
            if len(recent_values) >= 3:
                recent_trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                overall_trend = np.polyfit(range(len(historical_values)), historical_values, 1)[0]
                
                if abs(recent_trend - overall_trend) > 0.5:
                    return AnomalyType.COLLECTIVE_ANOMALY
            
            # Check if it's contextual (depends on recent context)
            if len(recent_values) >= 2:
                recent_mean = np.mean(recent_values)
                overall_mean = np.mean(historical_values)
                
                if abs(recent_mean - overall_mean) > np.std(historical_values):
                    return AnomalyType.CONTEXTUAL_ANOMALY
            
            # Default to point anomaly
            return AnomalyType.POINT_ANOMALY
            
        except Exception as e:
            logger.error(f"Error determining anomaly type: {e}")
            return AnomalyType.POINT_ANOMALY
    
    def _update_statistics(self, anomaly: Anomaly):
        """Update detection statistics"""
        
        try:
            self.statistics['total_anomalies'] += 1
            
            if anomaly.anomaly_type == AnomalyType.POINT_ANOMALY:
                self.statistics['point_anomalies'] += 1
            elif anomaly.anomaly_type == AnomalyType.CONTEXTUAL_ANOMALY:
                self.statistics['contextual_anomalies'] += 1
            elif anomaly.anomaly_type == AnomalyType.COLLECTIVE_ANOMALY:
                self.statistics['collective_anomalies'] += 1
                
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
    
    async def get_anomalies(self, company_id: str = None, hours: int = 24,
                          severity: AnomalySeverity = None) -> List[Anomaly]:
        """Get detected anomalies with optional filtering"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            filtered_anomalies = []
            
            for anomaly in self.detected_anomalies:
                # Filter by time
                if anomaly.timestamp < cutoff_time:
                    continue
                
                # Filter by company
                if company_id and anomaly.company_id != company_id:
                    continue
                
                # Filter by severity
                if severity and anomaly.severity != severity:
                    continue
                
                filtered_anomalies.append(anomaly)
            
            # Sort by timestamp (most recent first)
            filtered_anomalies.sort(key=lambda x: x.timestamp, reverse=True)
            
            return filtered_anomalies
            
        except Exception as e:
            logger.error(f"Error getting anomalies: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get anomaly detection statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Add current counts
            stats.update({
                'companies_monitored': len(self.historical_data),
                'total_data_points': sum(
                    sum(len(factor_data) for factor_data in company_data.values())
                    for company_data in self.historical_data.values()
                ),
                'recent_anomalies_24h': len(await self.get_anomalies(hours=24)),
                'detection_methods': list(self.detection_methods.keys()),
                'thresholds': self.thresholds.copy()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    async def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update detection thresholds"""
        
        try:
            self.thresholds.update(new_thresholds)
            logger.info(f"Updated anomaly detection thresholds: {new_thresholds}")
            
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
    
    async def clear_historical_data(self, company_id: str = None, older_than_days: int = None):
        """Clear historical data"""
        
        try:
            if company_id:
                # Clear data for specific company
                if company_id in self.historical_data:
                    if older_than_days:
                        cutoff_time = datetime.now() - timedelta(days=older_than_days)
                        for factor in self.historical_data[company_id]:
                            self.historical_data[company_id][factor] = [
                                entry for entry in self.historical_data[company_id][factor]
                                if entry['timestamp'] > cutoff_time
                            ]
                    else:
                        del self.historical_data[company_id]
            else:
                # Clear all data or data older than specified days
                if older_than_days:
                    cutoff_time = datetime.now() - timedelta(days=older_than_days)
                    for company_id in self.historical_data:
                        for factor in self.historical_data[company_id]:
                            self.historical_data[company_id][factor] = [
                                entry for entry in self.historical_data[company_id][factor]
                                if entry['timestamp'] > cutoff_time
                            ]
                else:
                    self.historical_data.clear()
            
            logger.info("Cleared historical data")
            
        except Exception as e:
            logger.error(f"Error clearing historical data: {e}")
