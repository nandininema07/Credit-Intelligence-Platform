"""
Financial event detection from text data.
Identifies key financial events like earnings announcements, mergers, regulatory actions.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class FinancialEvent:
    """Financial event data structure"""
    event_type: str
    description: str
    confidence: float
    entities: List[str]
    timestamp: datetime
    severity: str  # low, medium, high, critical
    impact: str    # positive, negative, neutral
    keywords: List[str]

class EventDetector:
    """Financial event detection from text"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.event_patterns = self._define_event_patterns()
        self.severity_keywords = self._define_severity_keywords()
        self.impact_keywords = self._define_impact_keywords()
        
    def _define_event_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Define patterns for different types of financial events"""
        return {
            'earnings': {
                'patterns': [
                    r'\bearnings?\s+(?:report|announcement|release|results?)\b',
                    r'\bquarterly\s+(?:earnings?|results?)\b',
                    r'\beps\s+(?:beat|miss|estimate)\b',
                    r'\brevenue\s+(?:beat|miss|estimate)\b',
                    r'\bguidence\s+(?:raised|lowered|updated)\b'
                ],
                'keywords': ['earnings', 'quarterly', 'eps', 'revenue', 'guidance', 'forecast'],
                'base_severity': 'medium',
                'category': 'financial_performance'
            },
            'merger_acquisition': {
                'patterns': [
                    r'\b(?:merger|acquisition|takeover|buyout)\b',
                    r'\bacquire[ds]?\s+(?:by|for)\b',
                    r'\bmerge[ds]?\s+with\b',
                    r'\btakeover\s+(?:bid|offer)\b',
                    r'\bbuyout\s+(?:deal|offer)\b'
                ],
                'keywords': ['merger', 'acquisition', 'takeover', 'buyout', 'deal'],
                'base_severity': 'high',
                'category': 'corporate_action'
            },
            'regulatory': {
                'patterns': [
                    r'\b(?:sec|fda|ftc|doj)\s+(?:investigation|probe|inquiry)\b',
                    r'\bregulatory\s+(?:approval|rejection|investigation)\b',
                    r'\blawsuit\s+(?:filed|settled)\b',
                    r'\bfine\s+(?:imposed|levied)\b',
                    r'\bcompliance\s+(?:violation|issue)\b'
                ],
                'keywords': ['sec', 'regulatory', 'lawsuit', 'fine', 'compliance', 'investigation'],
                'base_severity': 'high',
                'category': 'regulatory'
            },
            'management_change': {
                'patterns': [
                    r'\b(?:ceo|cfo|president|chairman)\s+(?:resigned?|appointed|named)\b',
                    r'\bmanagement\s+(?:change|restructuring)\b',
                    r'\bboard\s+(?:member|director)\s+(?:resigned?|appointed)\b',
                    r'\bexecutive\s+(?:departure|appointment)\b'
                ],
                'keywords': ['ceo', 'cfo', 'management', 'board', 'executive', 'resignation', 'appointment'],
                'base_severity': 'medium',
                'category': 'management'
            },
            'financial_distress': {
                'patterns': [
                    r'\bbankruptcy\s+(?:filing|protection|proceedings)\b',
                    r'\bdefault\s+(?:on|risk)\b',
                    r'\bdebt\s+(?:restructuring|covenant)\b',
                    r'\bliquidity\s+(?:crisis|concern)\b',
                    r'\bcredit\s+(?:downgrade|rating)\b'
                ],
                'keywords': ['bankruptcy', 'default', 'debt', 'liquidity', 'credit', 'downgrade'],
                'base_severity': 'critical',
                'category': 'financial_health'
            },
            'product_launch': {
                'patterns': [
                    r'\bproduct\s+(?:launch|release|announcement)\b',
                    r'\bnew\s+(?:product|service|offering)\b',
                    r'\binnovation\s+(?:announcement|breakthrough)\b',
                    r'\bpatent\s+(?:filed|granted|approved)\b'
                ],
                'keywords': ['product', 'launch', 'innovation', 'patent', 'new'],
                'base_severity': 'low',
                'category': 'business_development'
            },
            'market_movement': {
                'patterns': [
                    r'\bstock\s+(?:surge[ds]?|plunge[ds]?|soar[eds]?|crash[eds]?)\b',
                    r'\bshare\s+price\s+(?:up|down|rise[ds]?|fall[s]?)\b',
                    r'\bmarket\s+(?:rally|selloff|correction)\b',
                    r'\bvolatility\s+(?:spike|increase)\b'
                ],
                'keywords': ['stock', 'share', 'price', 'market', 'volatility'],
                'base_severity': 'medium',
                'category': 'market_performance'
            },
            'dividend_action': {
                'patterns': [
                    r'\bdividend\s+(?:increase[ds]?|cut|suspended?|announced?)\b',
                    r'\bshare\s+(?:buyback|repurchase)\s+(?:program|announced?)\b',
                    r'\bstock\s+split\s+(?:announced?|declared?)\b',
                    r'\bspecial\s+dividend\b'
                ],
                'keywords': ['dividend', 'buyback', 'repurchase', 'split'],
                'base_severity': 'low',
                'category': 'shareholder_action'
            }
        }
    
    def _define_severity_keywords(self) -> Dict[str, List[str]]:
        """Define keywords that modify event severity"""
        return {
            'critical': [
                'crisis', 'emergency', 'urgent', 'critical', 'severe', 'major',
                'significant', 'substantial', 'massive', 'unprecedented'
            ],
            'high': [
                'important', 'notable', 'considerable', 'material', 'substantial',
                'significant', 'major', 'large', 'big'
            ],
            'medium': [
                'moderate', 'reasonable', 'fair', 'standard', 'typical', 'normal'
            ],
            'low': [
                'minor', 'small', 'slight', 'minimal', 'limited', 'modest'
            ]
        }
    
    def _define_impact_keywords(self) -> Dict[str, List[str]]:
        """Define keywords that indicate event impact"""
        return {
            'positive': [
                'positive', 'good', 'strong', 'excellent', 'outstanding', 'impressive',
                'beat', 'exceed', 'outperform', 'growth', 'increase', 'gain',
                'success', 'achievement', 'breakthrough', 'improvement'
            ],
            'negative': [
                'negative', 'bad', 'weak', 'poor', 'disappointing', 'concerning',
                'miss', 'underperform', 'decline', 'decrease', 'loss', 'drop',
                'failure', 'problem', 'issue', 'concern', 'warning'
            ],
            'neutral': [
                'neutral', 'stable', 'unchanged', 'maintain', 'steady', 'consistent'
            ]
        }
    
    async def detect_events(self, texts: List[str]) -> Dict[str, Any]:
        """Detect financial events from text data"""
        if not texts:
            return {'events': [], 'event_counts': {}, 'total_events': 0}
        
        all_events = []
        
        for text in texts:
            text_events = await self._detect_events_in_text(text)
            all_events.extend(text_events)
        
        # Aggregate and analyze events
        event_analysis = self._analyze_events(all_events)
        
        return event_analysis
    
    async def _detect_events_in_text(self, text: str) -> List[FinancialEvent]:
        """Detect events in a single text"""
        events = []
        text_lower = text.lower()
        
        for event_type, config in self.event_patterns.items():
            for pattern in config['patterns']:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    # Extract context around the match
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]
                    
                    # Calculate confidence based on keyword density
                    confidence = self._calculate_confidence(context, config['keywords'])
                    
                    # Determine severity
                    severity = self._determine_severity(context, config['base_severity'])
                    
                    # Determine impact
                    impact = self._determine_impact(context)
                    
                    # Extract entities (simple approach)
                    entities = self._extract_event_entities(context)
                    
                    event = FinancialEvent(
                        event_type=event_type,
                        description=match.group(),
                        confidence=confidence,
                        entities=entities,
                        timestamp=datetime.now(),
                        severity=severity,
                        impact=impact,
                        keywords=config['keywords']
                    )
                    
                    events.append(event)
        
        return events
    
    def _calculate_confidence(self, context: str, keywords: List[str]) -> float:
        """Calculate confidence score for event detection"""
        context_lower = context.lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in keywords if keyword in context_lower)
        
        # Base confidence from pattern match
        base_confidence = 0.7
        
        # Boost confidence based on keyword density
        keyword_boost = min(keyword_matches * 0.1, 0.3)
        
        # Check for financial context indicators
        financial_indicators = [
            'company', 'stock', 'share', 'market', 'financial', 'business',
            'revenue', 'profit', 'earnings', 'quarter', 'annual'
        ]
        
        financial_matches = sum(1 for indicator in financial_indicators 
                              if indicator in context_lower)
        
        financial_boost = min(financial_matches * 0.05, 0.2)
        
        total_confidence = min(base_confidence + keyword_boost + financial_boost, 1.0)
        
        return total_confidence
    
    def _determine_severity(self, context: str, base_severity: str) -> str:
        """Determine event severity based on context"""
        context_lower = context.lower()
        
        # Check for severity modifying keywords
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                return severity
        
        return base_severity
    
    def _determine_impact(self, context: str) -> str:
        """Determine event impact (positive/negative/neutral)"""
        context_lower = context.lower()
        
        impact_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for impact_type, keywords in self.impact_keywords.items():
            for keyword in keywords:
                if keyword in context_lower:
                    impact_scores[impact_type] += 1
        
        # Return the impact type with highest score
        max_impact = max(impact_scores.items(), key=lambda x: x[1])
        
        if max_impact[1] == 0:
            return 'neutral'
        
        return max_impact[0]
    
    def _extract_event_entities(self, context: str) -> List[str]:
        """Extract entities related to the event"""
        entities = []
        
        # Simple entity extraction - look for capitalized words
        words = context.split()
        
        for i, word in enumerate(words):
            # Skip common words
            if word.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
                continue
                
            # Look for capitalized words (potential entities)
            if word[0].isupper() and len(word) > 2:
                # Check if it's part of a multi-word entity
                entity = word
                j = i + 1
                while j < len(words) and j < i + 3:  # Max 3 words
                    if words[j][0].isupper():
                        entity += f" {words[j]}"
                        j += 1
                    else:
                        break
                
                if len(entity.split()) >= 1:  # At least one word
                    entities.append(entity)
        
        # Remove duplicates and limit to top 5
        unique_entities = list(set(entities))[:5]
        
        return unique_entities
    
    def _analyze_events(self, events: List[FinancialEvent]) -> Dict[str, Any]:
        """Analyze detected events and provide summary statistics"""
        if not events:
            return {
                'events': [],
                'total_events': 0,
                'event_counts': {},
                'severity_distribution': {},
                'impact_distribution': {},
                'confidence_stats': {}
            }
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in events:
            event_counts[event.event_type] += 1
        
        # Count by severity
        severity_counts = defaultdict(int)
        for event in events:
            severity_counts[event.severity] += 1
        
        # Count by impact
        impact_counts = defaultdict(int)
        for event in events:
            impact_counts[event.impact] += 1
        
        # Calculate confidence statistics
        confidences = [event.confidence for event in events]
        confidence_stats = {
            'mean': sum(confidences) / len(confidences),
            'min': min(confidences),
            'max': max(confidences),
            'high_confidence_count': sum(1 for c in confidences if c > 0.8)
        }
        
        # Convert events to dictionaries for JSON serialization
        events_dict = []
        for event in events:
            events_dict.append({
                'event_type': event.event_type,
                'description': event.description,
                'confidence': event.confidence,
                'entities': event.entities,
                'timestamp': event.timestamp.isoformat(),
                'severity': event.severity,
                'impact': event.impact,
                'keywords': event.keywords
            })
        
        return {
            'events': events_dict,
            'total_events': len(events),
            'financial_event_count': sum(1 for e in events if e.event_type in ['earnings', 'merger_acquisition', 'financial_distress']),
            'risk_event_count': sum(1 for e in events if e.severity in ['high', 'critical'] and e.impact == 'negative'),
            'opportunity_event_count': sum(1 for e in events if e.impact == 'positive'),
            'event_counts': dict(event_counts),
            'severity_distribution': dict(severity_counts),
            'impact_distribution': dict(impact_counts),
            'confidence_stats': confidence_stats
        }
    
    def get_event_timeline(self, events: List[Dict[str, Any]], 
                          days_back: int = 30) -> Dict[str, Any]:
        """Create timeline analysis of events"""
        if not events:
            return {'timeline': {}, 'trends': {}}
        
        # Group events by date
        timeline = defaultdict(list)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for event in events:
            event_date = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
            if event_date >= cutoff_date:
                date_key = event_date.strftime('%Y-%m-%d')
                timeline[date_key].append(event)
        
        # Calculate daily event counts by type
        daily_counts = {}
        for date, day_events in timeline.items():
            daily_counts[date] = defaultdict(int)
            for event in day_events:
                daily_counts[date][event['event_type']] += 1
        
        # Identify trends
        trends = {}
        for event_type in self.event_patterns.keys():
            recent_counts = []
            dates = sorted(daily_counts.keys())[-7:]  # Last 7 days
            
            for date in dates:
                count = daily_counts.get(date, {}).get(event_type, 0)
                recent_counts.append(count)
            
            if len(recent_counts) >= 2:
                trend = 'increasing' if recent_counts[-1] > recent_counts[0] else 'decreasing'
                if recent_counts[-1] == recent_counts[0]:
                    trend = 'stable'
                trends[event_type] = trend
        
        return {
            'timeline': dict(timeline),
            'daily_counts': dict(daily_counts),
            'trends': trends,
            'total_days': len(timeline)
        }
