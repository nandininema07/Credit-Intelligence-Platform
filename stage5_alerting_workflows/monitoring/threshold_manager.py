"""
Threshold management for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ThresholdType(Enum):
    """Types of thresholds"""
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    STANDARD_DEVIATION = "standard_deviation"
    PERCENTILE = "percentile"

class ThresholdSeverity(Enum):
    """Threshold severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Threshold:
    """Threshold configuration"""
    id: str
    name: str
    description: str
    threshold_type: ThresholdType
    severity: ThresholdSeverity
    value: float
    comparison: str  # 'greater_than', 'less_than', 'equal'
    enabled: bool
    company_ids: List[str]
    factors: List[str]
    time_window_minutes: int
    cooldown_minutes: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class ThresholdManager:
    """Manage dynamic thresholds for credit monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = {}
        self.breach_history = {}
        self.statistics = {
            'total_thresholds': 0,
            'active_thresholds': 0,
            'breaches_detected': 0
        }
        self._initialize_defaults()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("ThresholdManager initialized successfully")
        return True
    
    def _initialize_defaults(self):
        """Initialize default thresholds"""
        
        default_thresholds = [
            {
                'id': 'score_drop_major',
                'name': 'Major Score Drop',
                'description': 'Detect major credit score decreases',
                'threshold_type': ThresholdType.ABSOLUTE,
                'severity': ThresholdSeverity.HIGH,
                'value': 15.0,
                'comparison': 'less_than',
                'factors': ['credit_score']
            },
            {
                'id': 'volatility_high',
                'name': 'High Volatility',
                'description': 'Detect high score volatility',
                'threshold_type': ThresholdType.STANDARD_DEVIATION,
                'severity': ThresholdSeverity.MEDIUM,
                'value': 2.0,
                'comparison': 'greater_than',
                'factors': ['credit_score']
            }
        ]
        
        for config in default_thresholds:
            asyncio.create_task(self._create_default_threshold(config))
    
    async def _create_default_threshold(self, config: Dict[str, Any]):
        """Create a default threshold"""
        
        try:
            threshold = Threshold(
                id=config['id'],
                name=config['name'],
                description=config['description'],
                threshold_type=config['threshold_type'],
                severity=config['severity'],
                value=config['value'],
                comparison=config['comparison'],
                enabled=True,
                company_ids=[],
                factors=config['factors'],
                time_window_minutes=60,
                cooldown_minutes=30,
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.thresholds[config['id']] = threshold
            self.breach_history[config['id']] = []
            self.statistics['total_thresholds'] += 1
            self.statistics['active_thresholds'] += 1
            
        except Exception as e:
            logger.error(f"Error creating default threshold: {e}")
    
    async def create_threshold(self, threshold_id: str, name: str, description: str,
                             threshold_type: ThresholdType, severity: ThresholdSeverity,
                             value: float, comparison: str, factors: List[str],
                             time_window_minutes: int = 60, cooldown_minutes: int = 30,
                             company_ids: List[str] = None) -> Threshold:
        """Create a new threshold"""
        
        try:
            if threshold_id in self.thresholds:
                raise ValueError(f"Threshold {threshold_id} already exists")
            
            threshold = Threshold(
                id=threshold_id,
                name=name,
                description=description,
                threshold_type=threshold_type,
                severity=severity,
                value=value,
                comparison=comparison,
                enabled=True,
                company_ids=company_ids or [],
                factors=factors,
                time_window_minutes=time_window_minutes,
                cooldown_minutes=cooldown_minutes,
                metadata={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.thresholds[threshold_id] = threshold
            self.breach_history[threshold_id] = []
            self.statistics['total_thresholds'] += 1
            self.statistics['active_thresholds'] += 1
            
            logger.info(f"Created threshold: {threshold_id}")
            return threshold
            
        except Exception as e:
            logger.error(f"Error creating threshold: {e}")
            raise
    
    async def check_thresholds(self, company_id: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any thresholds are breached"""
        
        try:
            breaches = []
            
            for threshold_id, threshold in self.thresholds.items():
                if not threshold.enabled:
                    continue
                
                if threshold.company_ids and company_id not in threshold.company_ids:
                    continue
                
                if await self._is_in_cooldown(threshold_id, company_id):
                    continue
                
                for factor in threshold.factors:
                    if factor not in data:
                        continue
                    
                    if await self._evaluate_threshold(threshold, data[factor]):
                        breach_info = {
                            'threshold_id': threshold_id,
                            'threshold_name': threshold.name,
                            'severity': threshold.severity.value,
                            'factor': factor,
                            'current_value': data[factor],
                            'threshold_value': threshold.value,
                            'company_id': company_id,
                            'timestamp': datetime.now()
                        }
                        
                        breaches.append(breach_info)
                        await self._record_breach(threshold_id, breach_info)
            
            return breaches
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
            return []
    
    async def _evaluate_threshold(self, threshold: Threshold, current_value: float) -> bool:
        """Evaluate if threshold is breached"""
        
        try:
            if threshold.comparison == 'greater_than':
                return current_value > threshold.value
            elif threshold.comparison == 'less_than':
                return current_value < threshold.value
            elif threshold.comparison == 'equal':
                return abs(current_value - threshold.value) < 0.001
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating threshold: {e}")
            return False
    
    async def _is_in_cooldown(self, threshold_id: str, company_id: str) -> bool:
        """Check if threshold is in cooldown period"""
        
        try:
            if threshold_id not in self.breach_history:
                return False
            
            threshold = self.thresholds[threshold_id]
            cooldown_period = timedelta(minutes=threshold.cooldown_minutes)
            current_time = datetime.now()
            
            recent_breaches = [
                breach for breach in self.breach_history[threshold_id]
                if breach['company_id'] == company_id and 
                current_time - breach['timestamp'] < cooldown_period
            ]
            
            return len(recent_breaches) > 0
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return False
    
    async def _record_breach(self, threshold_id: str, breach_info: Dict[str, Any]):
        """Record a threshold breach"""
        
        try:
            self.breach_history[threshold_id].append(breach_info)
            self.statistics['breaches_detected'] += 1
            
            # Limit history size
            if len(self.breach_history[threshold_id]) > 1000:
                self.breach_history[threshold_id] = self.breach_history[threshold_id][-500:]
            
        except Exception as e:
            logger.error(f"Error recording breach: {e}")
    
    async def get_all_thresholds(self) -> List[Threshold]:
        """Get all thresholds"""
        return list(self.thresholds.values())
    
    async def get_breach_history(self, threshold_id: str = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Get breach history"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            breaches = []
            
            threshold_ids = [threshold_id] if threshold_id else self.breach_history.keys()
            
            for tid in threshold_ids:
                if tid not in self.breach_history:
                    continue
                
                for breach in self.breach_history[tid]:
                    if breach['timestamp'] >= cutoff_time:
                        breaches.append(breach)
            
            breaches.sort(key=lambda x: x['timestamp'], reverse=True)
            return breaches
            
        except Exception as e:
            logger.error(f"Error getting breach history: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get threshold manager statistics"""
        
        stats = self.statistics.copy()
        stats.update({
            'total_thresholds': len(self.thresholds),
            'active_thresholds': sum(1 for t in self.thresholds.values() if t.enabled)
        })
        
        return stats
