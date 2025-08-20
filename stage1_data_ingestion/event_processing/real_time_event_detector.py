"""
Real-time event detection and classification for credit-impacting events.
Processes unstructured data to identify events that should trigger score updates.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class EventSeverity(Enum):
    CRITICAL = "critical"      # Immediate score impact
    HIGH = "high"             # Significant score impact
    MEDIUM = "medium"         # Moderate score impact
    LOW = "low"              # Minor score impact

class EventType(Enum):
    DEBT_RESTRUCTURING = "debt_restructuring"
    EARNINGS_WARNING = "earnings_warning"
    CREDIT_DOWNGRADE = "credit_downgrade"
    REGULATORY_ACTION = "regulatory_action"
    MANAGEMENT_CHANGE = "management_change"
    MERGER_ACQUISITION = "merger_acquisition"
    BANKRUPTCY_FILING = "bankruptcy_filing"
    DIVIDEND_CUT = "dividend_cut"
    DEBT_DEFAULT = "debt_default"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    POSITIVE_EARNINGS = "positive_earnings"
    NEW_CONTRACT = "new_contract"
    EXPANSION_NEWS = "expansion_news"

@dataclass
class CreditEvent:
    """Credit-impacting event detected from unstructured data"""
    event_id: str
    company_ticker: str
    event_type: EventType
    severity: EventSeverity
    title: str
    description: str
    source: str
    url: str
    detected_at: datetime
    published_at: datetime
    confidence_score: float
    impact_direction: str  # "negative", "positive", "neutral"
    estimated_score_impact: float  # Expected change in credit score
    keywords_matched: List[str]
    metadata: Dict[str, Any]

class RealTimeEventDetector:
    """Detects credit-impacting events from unstructured data streams"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Event detection patterns
        self.event_patterns = {
            EventType.DEBT_RESTRUCTURING: {
                'keywords': [
                    'debt restructuring', 'restructure debt', 'debt reorganization',
                    'chapter 11', 'bankruptcy protection', 'creditor agreement',
                    'debt modification', 'payment terms', 'covenant breach'
                ],
                'severity': EventSeverity.CRITICAL,
                'impact_direction': 'negative',
                'score_impact': -15.0
            },
            EventType.EARNINGS_WARNING: {
                'keywords': [
                    'earnings warning', 'profit warning', 'guidance cut',
                    'lower than expected', 'below estimates', 'disappointing results',
                    'revenue shortfall', 'margin pressure', 'outlook reduced'
                ],
                'severity': EventSeverity.HIGH,
                'impact_direction': 'negative',
                'score_impact': -8.0
            },
            EventType.CREDIT_DOWNGRADE: {
                'keywords': [
                    'credit downgrade', 'rating cut', 'moody\'s downgrade',
                    's&p downgrade', 'fitch downgrade', 'outlook negative',
                    'credit watch negative', 'rating lowered'
                ],
                'severity': EventSeverity.CRITICAL,
                'impact_direction': 'negative',
                'score_impact': -12.0
            },
            EventType.REGULATORY_ACTION: {
                'keywords': [
                    'regulatory action', 'sec investigation', 'fda warning',
                    'consent decree', 'regulatory fine', 'compliance violation',
                    'license suspension', 'regulatory scrutiny'
                ],
                'severity': EventSeverity.HIGH,
                'impact_direction': 'negative',
                'score_impact': -10.0
            },
            EventType.BANKRUPTCY_FILING: {
                'keywords': [
                    'bankruptcy filing', 'chapter 7', 'chapter 11',
                    'insolvency', 'liquidation', 'administration',
                    'receivership', 'wind down'
                ],
                'severity': EventSeverity.CRITICAL,
                'impact_direction': 'negative',
                'score_impact': -25.0
            },
            EventType.LIQUIDITY_CRISIS: {
                'keywords': [
                    'liquidity crisis', 'cash shortage', 'working capital',
                    'credit facility', 'covenant violation', 'cash burn',
                    'funding gap', 'liquidity concerns'
                ],
                'severity': EventSeverity.HIGH,
                'impact_direction': 'negative',
                'score_impact': -12.0
            },
            EventType.POSITIVE_EARNINGS: {
                'keywords': [
                    'beats estimates', 'exceeds expectations', 'strong earnings',
                    'record profits', 'revenue growth', 'margin expansion',
                    'guidance raised', 'outlook improved'
                ],
                'severity': EventSeverity.MEDIUM,
                'impact_direction': 'positive',
                'score_impact': 5.0
            },
            EventType.NEW_CONTRACT: {
                'keywords': [
                    'major contract', 'new deal', 'contract award',
                    'partnership agreement', 'strategic alliance',
                    'long-term contract', 'revenue secured'
                ],
                'severity': EventSeverity.LOW,
                'impact_direction': 'positive',
                'score_impact': 3.0
            }
        }
        
        # Company-specific keywords for better matching
        self.company_aliases = {
            'AAPL': ['apple', 'apple inc', 'cupertino'],
            'MSFT': ['microsoft', 'microsoft corp', 'redmond'],
            'GOOGL': ['google', 'alphabet', 'alphabet inc'],
            'AMZN': ['amazon', 'amazon.com', 'bezos'],
            'TSLA': ['tesla', 'tesla inc', 'musk', 'elon musk'],
            'META': ['meta', 'facebook', 'meta platforms'],
            'JPM': ['jpmorgan', 'jp morgan', 'chase'],
            'BAC': ['bank of america', 'bofa', 'merrill lynch']
        }
    
    def extract_company_mentions(self, text: str) -> List[str]:
        """Extract company ticker mentions from text"""
        mentioned_tickers = []
        text_lower = text.lower()
        
        for ticker, aliases in self.company_aliases.items():
            for alias in aliases:
                if alias in text_lower:
                    mentioned_tickers.append(ticker)
                    break
            
            # Also check for direct ticker mentions
            if ticker.lower() in text_lower:
                mentioned_tickers.append(ticker)
        
        return list(set(mentioned_tickers))
    
    def detect_events_in_text(self, text: str, title: str, source: str, 
                            url: str, published_at: datetime) -> List[CreditEvent]:
        """Detect credit events in a piece of text"""
        events = []
        text_lower = text.lower()
        title_lower = title.lower()
        combined_text = f"{title_lower} {text_lower}"
        
        # Extract company mentions
        mentioned_companies = self.extract_company_mentions(combined_text)
        
        if not mentioned_companies:
            return events
        
        # Check for event patterns
        for event_type, pattern_info in self.event_patterns.items():
            keywords_matched = []
            
            for keyword in pattern_info['keywords']:
                if keyword in combined_text:
                    keywords_matched.append(keyword)
            
            if keywords_matched:
                # Calculate confidence based on keyword matches and context
                confidence = min(len(keywords_matched) * 0.3 + 0.4, 1.0)
                
                # Create events for each mentioned company
                for ticker in mentioned_companies:
                    event = CreditEvent(
                        event_id=f"{ticker}_{event_type.value}_{int(published_at.timestamp())}",
                        company_ticker=ticker,
                        event_type=event_type,
                        severity=pattern_info['severity'],
                        title=title,
                        description=text[:500] + "..." if len(text) > 500 else text,
                        source=source,
                        url=url,
                        detected_at=datetime.now(),
                        published_at=published_at,
                        confidence_score=confidence,
                        impact_direction=pattern_info['impact_direction'],
                        estimated_score_impact=pattern_info['score_impact'],
                        keywords_matched=keywords_matched,
                        metadata={
                            'pattern_matched': event_type.value,
                            'keyword_count': len(keywords_matched),
                            'text_length': len(text)
                        }
                    )
                    events.append(event)
        
        return events
    
    async def process_data_points(self, data_points: List[Any]) -> List[CreditEvent]:
        """Process data points to detect credit events"""
        all_events = []
        
        for data_point in data_points:
            try:
                # Extract relevant fields
                title = getattr(data_point, 'title', '')
                content = getattr(data_point, 'content', '')
                source = getattr(data_point, 'source_name', 'unknown')
                url = getattr(data_point, 'url', '')
                published_at = getattr(data_point, 'published_date', datetime.now())
                
                # Detect events in this data point
                events = self.detect_events_in_text(
                    content, title, source, url, published_at
                )
                
                all_events.extend(events)
                
            except Exception as e:
                logger.error(f"Error processing data point for event detection: {e}")
                continue
        
        # Filter and deduplicate events
        filtered_events = self.filter_and_deduplicate_events(all_events)
        
        logger.info(f"Detected {len(filtered_events)} credit events from {len(data_points)} data points")
        return filtered_events
    
    def filter_and_deduplicate_events(self, events: List[CreditEvent]) -> List[CreditEvent]:
        """Filter events by confidence and remove duplicates"""
        # Filter by minimum confidence threshold
        min_confidence = self.config.get('min_event_confidence', 0.6)
        high_confidence_events = [e for e in events if e.confidence_score >= min_confidence]
        
        # Deduplicate by company + event type + time window
        deduplicated = {}
        time_window = timedelta(hours=6)  # 6-hour deduplication window
        
        for event in high_confidence_events:
            key = f"{event.company_ticker}_{event.event_type.value}"
            
            if key not in deduplicated:
                deduplicated[key] = event
            else:
                existing_event = deduplicated[key]
                time_diff = abs(event.published_at - existing_event.published_at)
                
                # Keep the more recent or higher confidence event
                if (time_diff < time_window and 
                    event.confidence_score > existing_event.confidence_score):
                    deduplicated[key] = event
                elif time_diff >= time_window:
                    # Different time window, keep both but with unique keys
                    deduplicated[f"{key}_{int(event.published_at.timestamp())}"] = event
        
        return list(deduplicated.values())
    
    def get_event_summary(self, events: List[CreditEvent]) -> Dict[str, Any]:
        """Get summary statistics of detected events"""
        summary = {
            'total_events': len(events),
            'by_severity': {},
            'by_type': {},
            'by_company': {},
            'by_impact': {'positive': 0, 'negative': 0, 'neutral': 0},
            'average_confidence': 0.0,
            'critical_events': []
        }
        
        if not events:
            return summary
        
        total_confidence = 0
        for event in events:
            # By severity
            severity = event.severity.value
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            # By type
            event_type = event.event_type.value
            summary['by_type'][event_type] = summary['by_type'].get(event_type, 0) + 1
            
            # By company
            ticker = event.company_ticker
            summary['by_company'][ticker] = summary['by_company'].get(ticker, 0) + 1
            
            # By impact
            summary['by_impact'][event.impact_direction] += 1
            
            # Confidence
            total_confidence += event.confidence_score
            
            # Critical events
            if event.severity == EventSeverity.CRITICAL:
                summary['critical_events'].append({
                    'company': event.company_ticker,
                    'type': event.event_type.value,
                    'title': event.title,
                    'impact': event.estimated_score_impact
                })
        
        summary['average_confidence'] = total_confidence / len(events)
        return summary

# Integration class for the main pipeline
class EventDrivenScoreUpdater:
    """Updates credit scores based on detected events"""
    
    def __init__(self, event_detector: RealTimeEventDetector):
        self.event_detector = event_detector
        self.score_adjustments = {}  # Track recent score adjustments
    
    async def process_events_for_scoring(self, events: List[CreditEvent]) -> Dict[str, float]:
        """Calculate score adjustments based on detected events"""
        score_adjustments = {}
        
        for event in events:
            ticker = event.company_ticker
            
            # Apply time decay to event impact (recent events have more impact)
            time_since_event = datetime.now() - event.published_at
            decay_factor = max(0.1, 1.0 - (time_since_event.days / 30.0))  # 30-day decay
            
            # Calculate adjusted impact
            adjusted_impact = event.estimated_score_impact * decay_factor * event.confidence_score
            
            # Accumulate adjustments for the same company
            if ticker not in score_adjustments:
                score_adjustments[ticker] = 0.0
            
            score_adjustments[ticker] += adjusted_impact
        
        # Cap maximum adjustment per update cycle
        max_adjustment = 20.0
        for ticker in score_adjustments:
            score_adjustments[ticker] = max(-max_adjustment, 
                                          min(max_adjustment, score_adjustments[ticker]))
        
        return score_adjustments
