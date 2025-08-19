"""
Alert deduplication for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)

class DeduplicationStrategy(Enum):
    """Deduplication strategies"""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    TIME_WINDOW = "time_window"
    CONTENT_HASH = "content_hash"
    COMPANY_FACTOR = "company_factor"

@dataclass
class DeduplicationRule:
    """Deduplication rule configuration"""
    id: str
    name: str
    strategy: DeduplicationStrategy
    time_window_minutes: int
    similarity_threshold: float
    fields_to_compare: List[str]
    enabled: bool
    priority: int

class AlertDeduplicator:
    """Handle alert deduplication to reduce noise"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dedup_rules = {}
        self.alert_signatures = {}
        self.suppressed_alerts = []
        self.statistics = {
            'alerts_processed': 0,
            'alerts_deduplicated': 0,
            'alerts_suppressed': 0,
            'unique_alerts': 0
        }
        self._initialize_deduplicator()
    
    def _initialize_deduplicator(self):
        """Initialize deduplication system"""
        
        # Default deduplication rules
        default_rules = [
            {
                'id': 'exact_company_factor',
                'name': 'Exact Company-Factor Match',
                'strategy': DeduplicationStrategy.COMPANY_FACTOR,
                'time_window_minutes': 15,
                'similarity_threshold': 1.0,
                'fields_to_compare': ['company_id', 'factor', 'severity'],
                'priority': 1
            },
            {
                'id': 'similar_content',
                'name': 'Similar Content Match',
                'strategy': DeduplicationStrategy.CONTENT_HASH,
                'time_window_minutes': 30,
                'similarity_threshold': 0.8,
                'fields_to_compare': ['title', 'description', 'company_id'],
                'priority': 2
            },
            {
                'id': 'time_window_exact',
                'name': 'Time Window Exact Match',
                'strategy': DeduplicationStrategy.TIME_WINDOW,
                'time_window_minutes': 60,
                'similarity_threshold': 1.0,
                'fields_to_compare': ['company_id', 'factor', 'title'],
                'priority': 3
            }
        ]
        
        for rule_config in default_rules:
            self.dedup_rules[rule_config['id']] = DeduplicationRule(
                id=rule_config['id'],
                name=rule_config['name'],
                strategy=DeduplicationStrategy(rule_config['strategy']),
                time_window_minutes=rule_config['time_window_minutes'],
                similarity_threshold=rule_config['similarity_threshold'],
                fields_to_compare=rule_config['fields_to_compare'],
                enabled=True,
                priority=rule_config['priority']
            )
    
    async def check_duplicate(self, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if alert is a duplicate"""
        
        try:
            self.statistics['alerts_processed'] += 1
            
            # Sort rules by priority
            sorted_rules = sorted(self.dedup_rules.values(), key=lambda x: x.priority)
            
            for rule in sorted_rules:
                if not rule.enabled:
                    continue
                
                duplicate_info = await self._check_rule(rule, alert_data)
                
                if duplicate_info:
                    self.statistics['alerts_deduplicated'] += 1
                    
                    return {
                        'is_duplicate': True,
                        'rule_id': rule.id,
                        'rule_name': rule.name,
                        'strategy': rule.strategy.value,
                        'original_alert_id': duplicate_info.get('original_alert_id'),
                        'similarity_score': duplicate_info.get('similarity_score', 1.0),
                        'time_difference_minutes': duplicate_info.get('time_difference_minutes', 0),
                        'matched_fields': duplicate_info.get('matched_fields', [])
                    }
            
            # Not a duplicate - record signature
            await self._record_alert_signature(alert_data)
            self.statistics['unique_alerts'] += 1
            
            return {'is_duplicate': False}
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return {'is_duplicate': False}
    
    async def _check_rule(self, rule: DeduplicationRule, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check a specific deduplication rule"""
        
        try:
            if rule.strategy == DeduplicationStrategy.EXACT_MATCH:
                return await self._check_exact_match(rule, alert_data)
            elif rule.strategy == DeduplicationStrategy.FUZZY_MATCH:
                return await self._check_fuzzy_match(rule, alert_data)
            elif rule.strategy == DeduplicationStrategy.TIME_WINDOW:
                return await self._check_time_window(rule, alert_data)
            elif rule.strategy == DeduplicationStrategy.CONTENT_HASH:
                return await self._check_content_hash(rule, alert_data)
            elif rule.strategy == DeduplicationStrategy.COMPANY_FACTOR:
                return await self._check_company_factor(rule, alert_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking rule {rule.id}: {e}")
            return None
    
    async def _check_exact_match(self, rule: DeduplicationRule, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for exact match duplicates"""
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=rule.time_window_minutes)
            
            # Create signature for current alert
            current_signature = self._create_signature(alert_data, rule.fields_to_compare)
            
            # Check against existing signatures
            for signature, signature_data in self.alert_signatures.items():
                if signature_data['timestamp'] < cutoff_time:
                    continue
                
                if signature == current_signature:
                    return {
                        'original_alert_id': signature_data['alert_id'],
                        'similarity_score': 1.0,
                        'time_difference_minutes': (datetime.now() - signature_data['timestamp']).total_seconds() / 60,
                        'matched_fields': rule.fields_to_compare
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in exact match check: {e}")
            return None
    
    async def _check_fuzzy_match(self, rule: DeduplicationRule, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for fuzzy match duplicates"""
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=rule.time_window_minutes)
            
            for signature, signature_data in self.alert_signatures.items():
                if signature_data['timestamp'] < cutoff_time:
                    continue
                
                similarity = self._calculate_similarity(alert_data, signature_data['alert_data'], rule.fields_to_compare)
                
                if similarity >= rule.similarity_threshold:
                    return {
                        'original_alert_id': signature_data['alert_id'],
                        'similarity_score': similarity,
                        'time_difference_minutes': (datetime.now() - signature_data['timestamp']).total_seconds() / 60,
                        'matched_fields': rule.fields_to_compare
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in fuzzy match check: {e}")
            return None
    
    async def _check_time_window(self, rule: DeduplicationRule, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for duplicates within time window"""
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=rule.time_window_minutes)
            
            for signature, signature_data in self.alert_signatures.items():
                if signature_data['timestamp'] < cutoff_time:
                    continue
                
                # Check if key fields match
                matches = 0
                for field in rule.fields_to_compare:
                    if (alert_data.get(field) == signature_data['alert_data'].get(field) and 
                        alert_data.get(field) is not None):
                        matches += 1
                
                similarity = matches / len(rule.fields_to_compare) if rule.fields_to_compare else 0
                
                if similarity >= rule.similarity_threshold:
                    return {
                        'original_alert_id': signature_data['alert_id'],
                        'similarity_score': similarity,
                        'time_difference_minutes': (datetime.now() - signature_data['timestamp']).total_seconds() / 60,
                        'matched_fields': [f for f in rule.fields_to_compare 
                                         if alert_data.get(f) == signature_data['alert_data'].get(f)]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in time window check: {e}")
            return None
    
    async def _check_content_hash(self, rule: DeduplicationRule, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for duplicates using content hash"""
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=rule.time_window_minutes)
            
            # Create content hash
            content_hash = self._create_content_hash(alert_data, rule.fields_to_compare)
            
            for signature, signature_data in self.alert_signatures.items():
                if signature_data['timestamp'] < cutoff_time:
                    continue
                
                stored_hash = signature_data.get('content_hash')
                if stored_hash and stored_hash == content_hash:
                    return {
                        'original_alert_id': signature_data['alert_id'],
                        'similarity_score': 1.0,
                        'time_difference_minutes': (datetime.now() - signature_data['timestamp']).total_seconds() / 60,
                        'matched_fields': rule.fields_to_compare
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in content hash check: {e}")
            return None
    
    async def _check_company_factor(self, rule: DeduplicationRule, alert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for duplicates based on company and factor"""
        
        try:
            cutoff_time = datetime.now() - timedelta(minutes=rule.time_window_minutes)
            
            company_id = alert_data.get('company_id')
            factor = alert_data.get('factor')
            
            if not company_id or not factor:
                return None
            
            for signature, signature_data in self.alert_signatures.items():
                if signature_data['timestamp'] < cutoff_time:
                    continue
                
                stored_data = signature_data['alert_data']
                
                if (stored_data.get('company_id') == company_id and 
                    stored_data.get('factor') == factor):
                    
                    # Check additional fields if specified
                    additional_match = True
                    matched_fields = ['company_id', 'factor']
                    
                    for field in rule.fields_to_compare:
                        if field not in ['company_id', 'factor']:
                            if alert_data.get(field) != stored_data.get(field):
                                additional_match = False
                                break
                            else:
                                matched_fields.append(field)
                    
                    if additional_match:
                        return {
                            'original_alert_id': signature_data['alert_id'],
                            'similarity_score': 1.0,
                            'time_difference_minutes': (datetime.now() - signature_data['timestamp']).total_seconds() / 60,
                            'matched_fields': matched_fields
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in company-factor check: {e}")
            return None
    
    def _create_signature(self, alert_data: Dict[str, Any], fields: List[str]) -> str:
        """Create alert signature from specified fields"""
        
        try:
            signature_data = {}
            
            for field in fields:
                value = alert_data.get(field)
                if value is not None:
                    signature_data[field] = str(value)
            
            # Create hash from sorted data
            signature_string = json.dumps(signature_data, sort_keys=True)
            return hashlib.md5(signature_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error creating signature: {e}")
            return ""
    
    def _create_content_hash(self, alert_data: Dict[str, Any], fields: List[str]) -> str:
        """Create content hash from specified fields"""
        
        try:
            content_parts = []
            
            for field in fields:
                value = alert_data.get(field)
                if value is not None:
                    # Normalize text content
                    if isinstance(value, str):
                        value = value.lower().strip()
                    content_parts.append(str(value))
            
            content_string = '|'.join(content_parts)
            return hashlib.sha256(content_string.encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error creating content hash: {e}")
            return ""
    
    def _calculate_similarity(self, alert1: Dict[str, Any], alert2: Dict[str, Any], fields: List[str]) -> float:
        """Calculate similarity between two alerts"""
        
        try:
            if not fields:
                return 0.0
            
            matches = 0
            total_fields = 0
            
            for field in fields:
                value1 = alert1.get(field)
                value2 = alert2.get(field)
                
                if value1 is not None and value2 is not None:
                    total_fields += 1
                    
                    if isinstance(value1, str) and isinstance(value2, str):
                        # Text similarity (simple approach)
                        similarity = self._text_similarity(value1, value2)
                        matches += similarity
                    else:
                        # Exact match for non-text fields
                        if value1 == value2:
                            matches += 1
            
            return matches / total_fields if total_fields > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple Jaccard similarity)"""
        
        try:
            # Convert to lowercase and split into words
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 and not words2:
                return 1.0
            
            if not words1 or not words2:
                return 0.0
            
            # Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating text similarity: {e}")
            return 0.0
    
    async def _record_alert_signature(self, alert_data: Dict[str, Any]):
        """Record alert signature for future deduplication"""
        
        try:
            alert_id = alert_data.get('id', f"alert_{datetime.now().timestamp()}")
            
            # Create multiple signatures for different strategies
            signatures_to_store = []
            
            for rule in self.dedup_rules.values():
                if not rule.enabled:
                    continue
                
                signature = self._create_signature(alert_data, rule.fields_to_compare)
                content_hash = self._create_content_hash(alert_data, rule.fields_to_compare)
                
                signature_key = f"{rule.id}_{signature}"
                
                signatures_to_store.append({
                    'key': signature_key,
                    'data': {
                        'alert_id': alert_id,
                        'alert_data': alert_data.copy(),
                        'timestamp': datetime.now(),
                        'rule_id': rule.id,
                        'signature': signature,
                        'content_hash': content_hash
                    }
                })
            
            # Store signatures
            for sig_info in signatures_to_store:
                self.alert_signatures[sig_info['key']] = sig_info['data']
            
            # Cleanup old signatures
            await self._cleanup_old_signatures()
            
        except Exception as e:
            logger.error(f"Error recording alert signature: {e}")
    
    async def _cleanup_old_signatures(self):
        """Clean up old alert signatures"""
        
        try:
            # Find maximum time window across all rules
            max_window = max(rule.time_window_minutes for rule in self.dedup_rules.values() if rule.enabled)
            cutoff_time = datetime.now() - timedelta(minutes=max_window * 2)  # Keep extra buffer
            
            # Remove old signatures
            keys_to_remove = []
            
            for key, signature_data in self.alert_signatures.items():
                if signature_data['timestamp'] < cutoff_time:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.alert_signatures[key]
            
            if keys_to_remove:
                logger.debug(f"Cleaned up {len(keys_to_remove)} old signatures")
                
        except Exception as e:
            logger.error(f"Error cleaning up signatures: {e}")
    
    async def suppress_duplicate(self, alert_data: Dict[str, Any], duplicate_info: Dict[str, Any]):
        """Suppress a duplicate alert"""
        
        try:
            suppression_record = {
                'alert_data': alert_data,
                'duplicate_info': duplicate_info,
                'suppressed_at': datetime.now()
            }
            
            self.suppressed_alerts.append(suppression_record)
            self.statistics['alerts_suppressed'] += 1
            
            # Limit suppressed alerts history
            if len(self.suppressed_alerts) > 10000:
                self.suppressed_alerts = self.suppressed_alerts[-5000:]
            
            logger.info(f"Suppressed duplicate alert: {duplicate_info.get('rule_name', 'Unknown rule')}")
            
        except Exception as e:
            logger.error(f"Error suppressing duplicate: {e}")
    
    async def add_deduplication_rule(self, rule_id: str, name: str, strategy: DeduplicationStrategy,
                                   time_window_minutes: int, similarity_threshold: float,
                                   fields_to_compare: List[str], priority: int = 10) -> bool:
        """Add new deduplication rule"""
        
        try:
            rule = DeduplicationRule(
                id=rule_id,
                name=name,
                strategy=strategy,
                time_window_minutes=time_window_minutes,
                similarity_threshold=similarity_threshold,
                fields_to_compare=fields_to_compare,
                enabled=True,
                priority=priority
            )
            
            self.dedup_rules[rule_id] = rule
            logger.info(f"Added deduplication rule: {rule_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding deduplication rule: {e}")
            return False
    
    async def remove_deduplication_rule(self, rule_id: str) -> bool:
        """Remove deduplication rule"""
        
        try:
            if rule_id in self.dedup_rules:
                del self.dedup_rules[rule_id]
                logger.info(f"Removed deduplication rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing deduplication rule: {e}")
            return False
    
    async def get_deduplication_rules(self) -> List[DeduplicationRule]:
        """Get all deduplication rules"""
        
        return list(self.dedup_rules.values())
    
    async def get_suppressed_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get suppressed alerts"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            return [
                alert for alert in self.suppressed_alerts
                if alert['suppressed_at'] >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error getting suppressed alerts: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        
        try:
            stats = self.statistics.copy()
            
            stats.update({
                'deduplication_rules': len(self.dedup_rules),
                'enabled_rules': sum(1 for r in self.dedup_rules.values() if r.enabled),
                'stored_signatures': len(self.alert_signatures),
                'suppressed_alerts_stored': len(self.suppressed_alerts),
                'deduplication_rate': (self.statistics['alerts_deduplicated'] / 
                                     max(1, self.statistics['alerts_processed'])) * 100
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
