"""
Credit score monitoring for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ScoreChangeType(Enum):
    """Types of score changes"""
    INCREASE = "increase"
    DECREASE = "decrease"
    STABLE = "stable"
    VOLATILE = "volatile"

@dataclass
class ScoreEvent:
    """Score monitoring event"""
    company_id: str
    timestamp: datetime
    previous_score: float
    current_score: float
    change_amount: float
    change_percentage: float
    change_type: ScoreChangeType
    confidence: float
    factors_changed: List[str]
    metadata: Dict[str, Any]

class ScoreMonitor:
    """Monitor credit score changes and detect significant events"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.score_history = {}
        self.monitoring_active = False
        self.event_callbacks = []
        self.statistics = {
            'total_events': 0,
            'score_increases': 0,
            'score_decreases': 0,
            'volatile_periods': 0,
            'companies_monitored': 0
        }
        self._initialize_monitor()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("ScoreMonitor initialized successfully")
        return True
    
    def _initialize_monitor(self):
        """Initialize score monitor"""
        
        # Default monitoring thresholds
        self.thresholds = {
            'significant_change': self.config.get('significant_change_threshold', 5.0),
            'major_change': self.config.get('major_change_threshold', 10.0),
            'critical_change': self.config.get('critical_change_threshold', 20.0),
            'volatility_threshold': self.config.get('volatility_threshold', 3.0),
            'monitoring_window_hours': self.config.get('monitoring_window_hours', 24),
            'min_confidence': self.config.get('min_confidence', 0.7)
        }
        
        # Monitoring settings
        self.settings = {
            'check_interval_seconds': self.config.get('check_interval_seconds', 300),
            'max_history_days': self.config.get('max_history_days', 90),
            'enable_trend_analysis': self.config.get('enable_trend_analysis', True),
            'enable_anomaly_detection': self.config.get('enable_anomaly_detection', True)
        }
    
    async def start_monitoring(self):
        """Start continuous score monitoring"""
        
        try:
            self.monitoring_active = True
            logger.info("Score monitoring started")
            
            while self.monitoring_active:
                await self._monitoring_cycle()
                await asyncio.sleep(self.settings['check_interval_seconds'])
                
        except Exception as e:
            logger.error(f"Error in score monitoring: {e}")
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """Stop score monitoring"""
        
        self.monitoring_active = False
        logger.info("Score monitoring stopped")
    
    async def _monitoring_cycle(self):
        """Single monitoring cycle"""
        
        try:
            # Get list of companies to monitor
            companies = await self._get_monitored_companies()
            
            for company_id in companies:
                try:
                    await self._check_company_score(company_id)
                except Exception as e:
                    logger.error(f"Error monitoring company {company_id}: {e}")
            
            # Cleanup old history
            await self._cleanup_old_history()
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
    
    async def _get_monitored_companies(self) -> List[str]:
        """Get list of companies to monitor"""
        
        try:
            # This would typically query a database or configuration
            # For now, return companies that have recent score history
            return list(self.score_history.keys())
            
        except Exception as e:
            logger.error(f"Error getting monitored companies: {e}")
            return []
    
    async def _check_company_score(self, company_id: str):
        """Check score changes for a specific company"""
        
        try:
            # Get current score (this would typically come from Stage 3)
            current_score = await self._get_current_score(company_id)
            
            if current_score is None:
                return
            
            # Get previous score
            previous_score = await self._get_previous_score(company_id)
            
            if previous_score is None:
                # First score for this company
                await self._record_score(company_id, current_score)
                return
            
            # Calculate change
            change_amount = current_score - previous_score
            change_percentage = (change_amount / previous_score) * 100 if previous_score != 0 else 0
            
            # Determine change type
            change_type = self._classify_change(change_amount, change_percentage)
            
            # Check if this is a significant event
            if await self._is_significant_event(company_id, current_score, previous_score, change_amount, change_percentage):
                
                # Get factors that changed
                factors_changed = await self._identify_changed_factors(company_id)
                
                # Create score event
                event = ScoreEvent(
                    company_id=company_id,
                    timestamp=datetime.now(),
                    previous_score=previous_score,
                    current_score=current_score,
                    change_amount=change_amount,
                    change_percentage=change_percentage,
                    change_type=change_type,
                    confidence=await self._calculate_confidence(company_id, change_amount),
                    factors_changed=factors_changed,
                    metadata=await self._get_event_metadata(company_id)
                )
                
                # Process the event
                await self._process_score_event(event)
            
            # Record the new score
            await self._record_score(company_id, current_score)
            
        except Exception as e:
            logger.error(f"Error checking company score {company_id}: {e}")
    
    async def _get_current_score(self, company_id: str) -> Optional[float]:
        """Get current credit score for company"""
        
        try:
            # This would typically query the scoring service from Stage 3
            # For demonstration, simulate score retrieval
            
            # In real implementation, this would be:
            # return await scoring_service.get_current_score(company_id)
            
            # Simulate score with some randomness for testing
            if company_id not in self.score_history:
                return np.random.uniform(400, 800)
            
            last_score = self.score_history[company_id][-1]['score']
            # Add small random change
            change = np.random.normal(0, 2)
            new_score = max(300, min(850, last_score + change))
            
            return new_score
            
        except Exception as e:
            logger.error(f"Error getting current score for {company_id}: {e}")
            return None
    
    async def _get_previous_score(self, company_id: str) -> Optional[float]:
        """Get previous score for company"""
        
        try:
            if company_id not in self.score_history or not self.score_history[company_id]:
                return None
            
            return self.score_history[company_id][-1]['score']
            
        except Exception as e:
            logger.error(f"Error getting previous score for {company_id}: {e}")
            return None
    
    async def _record_score(self, company_id: str, score: float):
        """Record score in history"""
        
        try:
            if company_id not in self.score_history:
                self.score_history[company_id] = []
            
            self.score_history[company_id].append({
                'score': score,
                'timestamp': datetime.now(),
                'recorded_at': datetime.now().isoformat()
            })
            
            # Limit history size
            max_entries = self.settings['max_history_days'] * 24 * 4  # Assuming 4 checks per hour
            if len(self.score_history[company_id]) > max_entries:
                self.score_history[company_id] = self.score_history[company_id][-max_entries:]
            
        except Exception as e:
            logger.error(f"Error recording score for {company_id}: {e}")
    
    def _classify_change(self, change_amount: float, change_percentage: float) -> ScoreChangeType:
        """Classify the type of score change"""
        
        try:
            abs_change = abs(change_amount)
            abs_percentage = abs(change_percentage)
            
            if abs_change < 1.0 and abs_percentage < 0.5:
                return ScoreChangeType.STABLE
            elif abs_percentage > self.thresholds['volatility_threshold']:
                return ScoreChangeType.VOLATILE
            elif change_amount > 0:
                return ScoreChangeType.INCREASE
            else:
                return ScoreChangeType.DECREASE
                
        except Exception as e:
            logger.error(f"Error classifying change: {e}")
            return ScoreChangeType.STABLE
    
    async def _is_significant_event(self, company_id: str, current_score: float, 
                                  previous_score: float, change_amount: float, 
                                  change_percentage: float) -> bool:
        """Determine if score change is significant enough to generate event"""
        
        try:
            abs_change = abs(change_amount)
            abs_percentage = abs(change_percentage)
            
            # Check against thresholds
            if abs_change >= self.thresholds['significant_change']:
                return True
            
            if abs_percentage >= (self.thresholds['significant_change'] / previous_score * 100):
                return True
            
            # Check for volatility patterns
            if await self._is_volatile_period(company_id):
                return True
            
            # Check for trend reversals
            if await self._is_trend_reversal(company_id, change_amount):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking significance: {e}")
            return False
    
    async def _is_volatile_period(self, company_id: str) -> bool:
        """Check if company is in volatile period"""
        
        try:
            if company_id not in self.score_history:
                return False
            
            history = self.score_history[company_id]
            if len(history) < 5:
                return False
            
            # Check recent volatility
            recent_scores = [h['score'] for h in history[-5:]]
            volatility = np.std(recent_scores)
            
            return volatility > self.thresholds['volatility_threshold']
            
        except Exception as e:
            logger.error(f"Error checking volatility: {e}")
            return False
    
    async def _is_trend_reversal(self, company_id: str, current_change: float) -> bool:
        """Check if current change represents a trend reversal"""
        
        try:
            if company_id not in self.score_history:
                return False
            
            history = self.score_history[company_id]
            if len(history) < 3:
                return False
            
            # Get recent changes
            recent_changes = []
            for i in range(len(history) - 1, max(0, len(history) - 4), -1):
                if i > 0:
                    change = history[i]['score'] - history[i-1]['score']
                    recent_changes.append(change)
            
            if len(recent_changes) < 2:
                return False
            
            # Check if current change is opposite to recent trend
            recent_trend = np.mean(recent_changes)
            
            if recent_trend > 0 and current_change < -self.thresholds['significant_change']:
                return True
            elif recent_trend < 0 and current_change > self.thresholds['significant_change']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trend reversal: {e}")
            return False
    
    async def _identify_changed_factors(self, company_id: str) -> List[str]:
        """Identify which factors contributed to score change"""
        
        try:
            # This would typically integrate with Stage 4 explainability
            # to get factor attribution for the score change
            
            # For demonstration, return common factors
            possible_factors = [
                'payment_history', 'credit_utilization', 'debt_to_income',
                'market_sentiment', 'financial_ratios', 'news_sentiment',
                'industry_trends', 'regulatory_changes'
            ]
            
            # Simulate factor identification
            num_factors = np.random.randint(1, 4)
            return np.random.choice(possible_factors, num_factors, replace=False).tolist()
            
        except Exception as e:
            logger.error(f"Error identifying changed factors: {e}")
            return []
    
    async def _calculate_confidence(self, company_id: str, change_amount: float) -> float:
        """Calculate confidence in the score change"""
        
        try:
            base_confidence = 0.8
            
            # Adjust based on change magnitude
            abs_change = abs(change_amount)
            if abs_change > self.thresholds['major_change']:
                base_confidence += 0.1
            elif abs_change < self.thresholds['significant_change'] / 2:
                base_confidence -= 0.2
            
            # Adjust based on historical volatility
            if await self._is_volatile_period(company_id):
                base_confidence -= 0.1
            
            # Adjust based on data recency
            # (In real implementation, check how recent the underlying data is)
            
            return max(0.1, min(0.95, base_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def _get_event_metadata(self, company_id: str) -> Dict[str, Any]:
        """Get additional metadata for the event"""
        
        try:
            metadata = {
                'monitoring_timestamp': datetime.now().isoformat(),
                'company_id': company_id,
                'monitoring_window_hours': self.thresholds['monitoring_window_hours'],
                'thresholds_used': self.thresholds.copy()
            }
            
            # Add historical context
            if company_id in self.score_history:
                history = self.score_history[company_id]
                if len(history) >= 2:
                    recent_scores = [h['score'] for h in history[-10:]]
                    metadata.update({
                        'recent_average': np.mean(recent_scores),
                        'recent_volatility': np.std(recent_scores),
                        'recent_trend': recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0
                    })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting event metadata: {e}")
            return {}
    
    async def _process_score_event(self, event: ScoreEvent):
        """Process a significant score event"""
        
        try:
            # Update statistics
            self.statistics['total_events'] += 1
            
            if event.change_type == ScoreChangeType.INCREASE:
                self.statistics['score_increases'] += 1
            elif event.change_type == ScoreChangeType.DECREASE:
                self.statistics['score_decreases'] += 1
            elif event.change_type == ScoreChangeType.VOLATILE:
                self.statistics['volatile_periods'] += 1
            
            # Log the event
            logger.info(f"Score event detected: {event.company_id} - {event.change_type.value} "
                       f"({event.change_amount:+.1f} points, {event.change_percentage:+.1f}%)")
            
            # Notify registered callbacks
            for callback in self.event_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")
            
        except Exception as e:
            logger.error(f"Error processing score event: {e}")
    
    async def _cleanup_old_history(self):
        """Clean up old score history"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self.settings['max_history_days'])
            
            for company_id in list(self.score_history.keys()):
                if company_id in self.score_history:
                    # Filter out old entries
                    self.score_history[company_id] = [
                        entry for entry in self.score_history[company_id]
                        if entry['timestamp'] > cutoff_date
                    ]
                    
                    # Remove companies with no recent history
                    if not self.score_history[company_id]:
                        del self.score_history[company_id]
            
        except Exception as e:
            logger.error(f"Error cleaning up history: {e}")
    
    def register_event_callback(self, callback):
        """Register callback for score events"""
        
        if callback not in self.event_callbacks:
            self.event_callbacks.append(callback)
    
    def unregister_event_callback(self, callback):
        """Unregister event callback"""
        
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)
    
    async def get_company_score_history(self, company_id: str, 
                                      hours: int = 24) -> List[Dict[str, Any]]:
        """Get score history for a company"""
        
        try:
            if company_id not in self.score_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            return [
                entry for entry in self.score_history[company_id]
                if entry['timestamp'] > cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error getting score history: {e}")
            return []
    
    async def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        
        try:
            stats = self.statistics.copy()
            stats.update({
                'monitoring_active': self.monitoring_active,
                'companies_monitored': len(self.score_history),
                'total_score_records': sum(len(history) for history in self.score_history.values()),
                'monitoring_uptime': datetime.now().isoformat(),
                'thresholds': self.thresholds.copy(),
                'settings': self.settings.copy()
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
    
    async def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update monitoring thresholds"""
        
        try:
            self.thresholds.update(new_thresholds)
            logger.info(f"Updated monitoring thresholds: {new_thresholds}")
            
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")
    
    async def add_company_to_monitoring(self, company_id: str, initial_score: float = None):
        """Add company to monitoring"""
        
        try:
            if company_id not in self.score_history:
                self.score_history[company_id] = []
                
                if initial_score is not None:
                    await self._record_score(company_id, initial_score)
                
                logger.info(f"Added company {company_id} to monitoring")
                
        except Exception as e:
            logger.error(f"Error adding company to monitoring: {e}")
    
    async def remove_company_from_monitoring(self, company_id: str):
        """Remove company from monitoring"""
        
        try:
            if company_id in self.score_history:
                del self.score_history[company_id]
                logger.info(f"Removed company {company_id} from monitoring")
                
        except Exception as e:
            logger.error(f"Error removing company from monitoring: {e}")
