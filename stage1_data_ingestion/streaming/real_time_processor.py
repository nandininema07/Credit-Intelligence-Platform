"""
Real-time streaming data processor for continuous credit intelligence updates.
Processes incoming data streams and triggers immediate score updates.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

@dataclass
class StreamingUpdate:
    """Real-time update message"""
    update_id: str
    company_ticker: str
    update_type: str  # 'score_change', 'event_detected', 'data_update'
    data: Dict[str, Any]
    timestamp: datetime
    priority: int  # 1=critical, 2=high, 3=medium, 4=low

class RealTimeProcessor:
    """Processes streaming data for immediate credit intelligence updates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.update_queue = asyncio.Queue(maxsize=1000)
        self.subscribers = []  # WebSocket connections, callbacks, etc.
        
        # Processing statistics
        self.stats = {
            'updates_processed': 0,
            'events_detected': 0,
            'scores_updated': 0,
            'start_time': None,
            'last_update': None
        }
        
        # Rate limiting for updates
        self.last_company_update = {}  # Track last update time per company
        self.min_update_interval = timedelta(seconds=30)  # Min 30s between updates per company
        
        # Initialize components
        self.event_detector = None
        self.score_calculator = None
        
    async def initialize(self):
        """Initialize the real-time processor"""
        try:
            from ..event_processing.real_time_event_detector import RealTimeEventDetector
            from ...stage3_model_training.scoring.real_time_scorer import RealTimeScorer
            
            self.event_detector = RealTimeEventDetector(self.config)
            self.score_calculator = RealTimeScorer(self.config)
            
            logger.info("Real-time processor initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Some components not available: {e}")
    
    async def start_processing(self):
        """Start the real-time processing loop"""
        if self.is_running:
            logger.warning("Real-time processor already running")
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info("Starting real-time processing...")
        
        # Start processing tasks
        tasks = [
            asyncio.create_task(self._process_update_queue()),
            asyncio.create_task(self._periodic_health_check()),
            asyncio.create_task(self._cleanup_old_data())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
        finally:
            self.is_running = False
    
    async def stop_processing(self):
        """Stop the real-time processing"""
        logger.info("Stopping real-time processing...")
        self.is_running = False
        
        # Clear the queue
        while not self.update_queue.empty():
            try:
                self.update_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def queue_update(self, update: StreamingUpdate):
        """Queue an update for processing"""
        try:
            await self.update_queue.put(update)
            logger.debug(f"Queued update: {update.update_type} for {update.company_ticker}")
        except asyncio.QueueFull:
            logger.warning("Update queue full, dropping oldest update")
            try:
                self.update_queue.get_nowait()  # Remove oldest
                await self.update_queue.put(update)  # Add new
            except asyncio.QueueEmpty:
                pass
    
    async def _process_update_queue(self):
        """Process updates from the queue"""
        while self.is_running:
            try:
                # Wait for update with timeout
                update = await asyncio.wait_for(
                    self.update_queue.get(), 
                    timeout=1.0
                )
                
                await self._process_single_update(update)
                self.stats['updates_processed'] += 1
                self.stats['last_update'] = datetime.now()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing update: {e}")
                continue
    
    async def _process_single_update(self, update: StreamingUpdate):
        """Process a single streaming update"""
        try:
            # Check rate limiting
            if not self._should_process_update(update):
                logger.debug(f"Rate limited update for {update.company_ticker}")
                return
            
            # Process based on update type
            if update.update_type == 'data_update':
                await self._process_data_update(update)
            elif update.update_type == 'event_detected':
                await self._process_event_update(update)
            elif update.update_type == 'score_change':
                await self._process_score_update(update)
            
            # Update rate limiting tracker
            self.last_company_update[update.company_ticker] = datetime.now()
            
            # Notify subscribers
            await self._notify_subscribers(update)
            
        except Exception as e:
            logger.error(f"Error processing update {update.update_id}: {e}")
    
    def _should_process_update(self, update: StreamingUpdate) -> bool:
        """Check if update should be processed based on rate limiting"""
        # Critical updates always process
        if update.priority == 1:
            return True
        
        # Check company-specific rate limiting
        last_update = self.last_company_update.get(update.company_ticker)
        if last_update:
            time_since_last = datetime.now() - last_update
            if time_since_last < self.min_update_interval:
                return False
        
        return True
    
    async def _process_data_update(self, update: StreamingUpdate):
        """Process new data and detect events"""
        try:
            data_points = update.data.get('data_points', [])
            
            if self.event_detector and data_points:
                # Detect events in new data
                events = await self.event_detector.process_data_points(data_points)
                
                if events:
                    self.stats['events_detected'] += len(events)
                    
                    # Queue event updates
                    for event in events:
                        event_update = StreamingUpdate(
                            update_id=f"event_{event.event_id}",
                            company_ticker=event.company_ticker,
                            update_type='event_detected',
                            data={
                                'event': event,
                                'impact': event.estimated_score_impact,
                                'confidence': event.confidence_score
                            },
                            timestamp=datetime.now(),
                            priority=1 if event.severity.value == 'critical' else 2
                        )
                        await self.queue_update(event_update)
            
        except Exception as e:
            logger.error(f"Error processing data update: {e}")
    
    async def _process_event_update(self, update: StreamingUpdate):
        """Process detected event and update scores"""
        try:
            event_data = update.data.get('event')
            if not event_data:
                return
            
            # Calculate score impact
            if self.score_calculator:
                score_change = await self.score_calculator.calculate_event_impact(
                    update.company_ticker,
                    event_data
                )
                
                if abs(score_change) > 0.1:  # Minimum threshold for score updates
                    # Queue score update
                    score_update = StreamingUpdate(
                        update_id=f"score_{update.company_ticker}_{int(time.time())}",
                        company_ticker=update.company_ticker,
                        update_type='score_change',
                        data={
                            'score_change': score_change,
                            'reason': f"Event: {event_data.event_type.value}",
                            'event_id': event_data.event_id,
                            'confidence': event_data.confidence_score
                        },
                        timestamp=datetime.now(),
                        priority=1 if abs(score_change) > 5.0 else 2
                    )
                    await self.queue_update(score_update)
            
        except Exception as e:
            logger.error(f"Error processing event update: {e}")
    
    async def _process_score_update(self, update: StreamingUpdate):
        """Process score change and persist"""
        try:
            score_change = update.data.get('score_change', 0)
            reason = update.data.get('reason', 'Unknown')
            
            # Log significant score changes
            if abs(score_change) > 2.0:
                logger.info(
                    f"Significant score change for {update.company_ticker}: "
                    f"{score_change:+.2f} - {reason}"
                )
            
            self.stats['scores_updated'] += 1
            
            # Here you would persist the score change to database
            # await self._persist_score_change(update)
            
        except Exception as e:
            logger.error(f"Error processing score update: {e}")
    
    async def _notify_subscribers(self, update: StreamingUpdate):
        """Notify all subscribers of the update"""
        if not self.subscribers:
            return
        
        notification = {
            'type': update.update_type,
            'company': update.company_ticker,
            'data': update.data,
            'timestamp': update.timestamp.isoformat(),
            'priority': update.priority
        }
        
        # Notify all subscribers (WebSocket connections, etc.)
        for subscriber in self.subscribers[:]:  # Copy list to avoid modification issues
            try:
                if callable(subscriber):
                    await subscriber(notification)
                elif hasattr(subscriber, 'send'):
                    await subscriber.send(json.dumps(notification))
            except Exception as e:
                logger.warning(f"Failed to notify subscriber: {e}")
                # Remove failed subscriber
                try:
                    self.subscribers.remove(subscriber)
                except ValueError:
                    pass
    
    async def _periodic_health_check(self):
        """Periodic health check and statistics logging"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Log statistics
                uptime = datetime.now() - self.stats['start_time']
                logger.info(
                    f"Real-time processor stats - "
                    f"Uptime: {uptime}, "
                    f"Updates: {self.stats['updates_processed']}, "
                    f"Events: {self.stats['events_detected']}, "
                    f"Scores: {self.stats['scores_updated']}, "
                    f"Queue size: {self.update_queue.qsize()}"
                )
                
            except Exception as e:
                logger.error(f"Error in health check: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old data periodically"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Clean up every hour
                
                # Clean up old rate limiting data
                cutoff_time = datetime.now() - timedelta(hours=1)
                old_companies = [
                    company for company, last_time in self.last_company_update.items()
                    if last_time < cutoff_time
                ]
                
                for company in old_companies:
                    del self.last_company_update[company]
                
                if old_companies:
                    logger.debug(f"Cleaned up rate limiting data for {len(old_companies)} companies")
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
    
    def subscribe(self, callback_or_connection):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback_or_connection)
        logger.debug(f"New subscriber added, total: {len(self.subscribers)}")
    
    def unsubscribe(self, callback_or_connection):
        """Unsubscribe from real-time updates"""
        try:
            self.subscribers.remove(callback_or_connection)
            logger.debug(f"Subscriber removed, total: {len(self.subscribers)}")
        except ValueError:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.stats.copy()
        if stats['start_time']:
            stats['uptime_seconds'] = (datetime.now() - stats['start_time']).total_seconds()
        stats['queue_size'] = self.update_queue.qsize()
        stats['subscriber_count'] = len(self.subscribers)
        return stats

# Integration with data collection pipeline
class StreamingDataIntegrator:
    """Integrates streaming processor with data collection pipeline"""
    
    def __init__(self, processor: RealTimeProcessor):
        self.processor = processor
    
    async def process_collected_data(self, data_points: List[Any]):
        """Process newly collected data points for streaming updates"""
        if not data_points:
            return
        
        # Group data points by company
        company_data = {}
        for dp in data_points:
            ticker = getattr(dp, 'company_ticker', None)
            if ticker:
                if ticker not in company_data:
                    company_data[ticker] = []
                company_data[ticker].append(dp)
        
        # Create streaming updates for each company
        for ticker, company_points in company_data.items():
            update = StreamingUpdate(
                update_id=f"data_{ticker}_{int(time.time())}",
                company_ticker=ticker,
                update_type='data_update',
                data={'data_points': company_points},
                timestamp=datetime.now(),
                priority=3  # Medium priority for regular data updates
            )
            
            await self.processor.queue_update(update)
    
    async def trigger_immediate_score_update(self, ticker: str, reason: str):
        """Trigger immediate score recalculation for a company"""
        update = StreamingUpdate(
            update_id=f"immediate_{ticker}_{int(time.time())}",
            company_ticker=ticker,
            update_type='score_change',
            data={'reason': reason, 'immediate': True},
            timestamp=datetime.now(),
            priority=1  # Critical priority
        )
        
        await self.processor.queue_update(update)
