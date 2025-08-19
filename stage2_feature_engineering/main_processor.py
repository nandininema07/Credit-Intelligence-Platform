"""
Main feature processing pipeline for Stage 2.
Orchestrates NLP processing, financial calculations, and feature generation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass

from .nlp.sentiment_analyzer import SentimentAnalyzer
from .nlp.topic_extractor import TopicExtractor
from .nlp.entity_linker import EntityLinker
from .nlp.event_detector import EventDetector
from .nlp.text_embeddings import TextEmbeddings

from .financial.ratio_calculator import RatioCalculator
from .financial.trend_analyzer import TrendAnalyzer
from .financial.volatility_metrics import VolatilityMetrics
from .financial.market_indicators import MarketIndicators

from .feature_store.feature_store import FeatureStore
from .aggregation.time_series_agg import TimeSeriesAggregator
from .transformers.scalers import FeatureScaler

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result of feature processing"""
    company: str
    features: Dict[str, float]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    errors: List[str]

class MainProcessor:
    """Main feature engineering pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all processing components"""
        # NLP Components
        self.sentiment_analyzer = SentimentAnalyzer(self.config.get('nlp', {}))
        self.topic_extractor = TopicExtractor(self.config.get('nlp', {}))
        self.entity_linker = EntityLinker(self.config.get('nlp', {}))
        self.event_detector = EventDetector(self.config.get('nlp', {}))
        self.text_embeddings = TextEmbeddings(self.config.get('nlp', {}))
        
        # Financial Components
        self.ratio_calculator = RatioCalculator(self.config.get('financial', {}))
        self.trend_analyzer = TrendAnalyzer(self.config.get('financial', {}))
        self.volatility_metrics = VolatilityMetrics(self.config.get('financial', {}))
        self.market_indicators = MarketIndicators(self.config.get('financial', {}))
        
        # Feature Store and Aggregation
        self.feature_store = FeatureStore(self.config.get('feature_store', {}))
        self.time_series_agg = TimeSeriesAggregator(self.config.get('aggregation', {}))
        self.feature_scaler = FeatureScaler(self.config.get('transformers', {}))
        
        logger.info("Feature processing components initialized")
    
    async def process_company_data(self, company: str, raw_data: List[Dict[str, Any]]) -> ProcessingResult:
        """Process all data for a single company"""
        start_time = datetime.now()
        errors = []
        features = {}
        
        try:
            # Separate data by type
            text_data = [item for item in raw_data if item.get('data_type') in ['news', 'social', 'regulatory']]
            financial_data = [item for item in raw_data if item.get('data_type') == 'financial']
            
            # Process text data
            if text_data:
                text_features = await self._process_text_data(company, text_data)
                features.update(text_features)
            
            # Process financial data
            if financial_data:
                financial_features = await self._process_financial_data(company, financial_data)
                features.update(financial_features)
            
            # Generate aggregated features
            aggregated_features = await self._generate_aggregated_features(company, features)
            features.update(aggregated_features)
            
            # Scale features
            scaled_features = await self.feature_scaler.scale_features(features)
            
            # Store features
            await self.feature_store.store_features(company, scaled_features, datetime.now())
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                company=company,
                features=scaled_features,
                metadata={
                    'text_records': len(text_data),
                    'financial_records': len(financial_data),
                    'feature_count': len(scaled_features)
                },
                processing_time=processing_time,
                success=True,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error processing company {company}: {e}")
            errors.append(str(e))
            
            return ProcessingResult(
                company=company,
                features={},
                metadata={},
                processing_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                errors=errors
            )
    
    async def _process_text_data(self, company: str, text_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Process text data through NLP pipeline"""
        features = {}
        
        # Combine all text content
        texts = []
        for item in text_data:
            content = item.get('content', {})
            if isinstance(content, dict):
                text = content.get('title', '') + ' ' + content.get('content', '')
            else:
                text = str(content)
            texts.append(text)
        
        if not texts:
            return features
        
        # Run NLP processing in parallel
        tasks = [
            self.sentiment_analyzer.analyze_batch(texts),
            self.topic_extractor.extract_topics(texts),
            self.entity_linker.link_entities(texts, company),
            self.event_detector.detect_events(texts),
            self.text_embeddings.generate_embeddings(texts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process sentiment results
        if isinstance(results[0], dict):
            sentiment_data = results[0]
            features.update({
                'sentiment_mean': sentiment_data.get('mean_sentiment', 0.0),
                'sentiment_std': sentiment_data.get('sentiment_std', 0.0),
                'positive_ratio': sentiment_data.get('positive_ratio', 0.0),
                'negative_ratio': sentiment_data.get('negative_ratio', 0.0),
                'neutral_ratio': sentiment_data.get('neutral_ratio', 0.0)
            })
        
        # Process topic results
        if isinstance(results[1], dict):
            topic_data = results[1]
            for i, topic_weight in enumerate(topic_data.get('topic_weights', [])[:5]):
                features[f'topic_{i}_weight'] = topic_weight
        
        # Process entity linking results
        if isinstance(results[2], dict):
            entity_data = results[2]
            features.update({
                'entity_mentions': entity_data.get('mention_count', 0),
                'entity_confidence': entity_data.get('avg_confidence', 0.0)
            })
        
        # Process event detection results
        if isinstance(results[3], dict):
            event_data = results[3]
            features.update({
                'financial_events': event_data.get('financial_event_count', 0),
                'risk_events': event_data.get('risk_event_count', 0),
                'opportunity_events': event_data.get('opportunity_event_count', 0)
            })
        
        # Process embeddings
        if isinstance(results[4], np.ndarray):
            embeddings = results[4]
            # Use first few dimensions as features
            for i in range(min(10, embeddings.shape[1])):
                features[f'embedding_{i}'] = float(embeddings[:, i].mean())
        
        return features
    
    async def _process_financial_data(self, company: str, financial_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Process financial data"""
        features = {}
        
        # Extract financial metrics
        financial_records = []
        for item in financial_data:
            content = item.get('content', {})
            if isinstance(content, dict):
                financial_records.append(content)
        
        if not financial_records:
            return features
        
        # Run financial processing
        tasks = [
            self.ratio_calculator.calculate_ratios(financial_records),
            self.trend_analyzer.analyze_trends(financial_records),
            self.volatility_metrics.calculate_volatility(financial_records),
            self.market_indicators.calculate_indicators(financial_records)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process ratio results
        if isinstance(results[0], dict):
            features.update(results[0])
        
        # Process trend results
        if isinstance(results[1], dict):
            features.update(results[1])
        
        # Process volatility results
        if isinstance(results[2], dict):
            features.update(results[2])
        
        # Process market indicator results
        if isinstance(results[3], dict):
            features.update(results[3])
        
        return features
    
    async def _generate_aggregated_features(self, company: str, features: Dict[str, float]) -> Dict[str, float]:
        """Generate time-series aggregated features"""
        try:
            # Get historical features for time-series aggregation
            historical_features = await self.feature_store.get_historical_features(
                company, 
                days_back=30
            )
            
            if historical_features:
                aggregated = await self.time_series_agg.aggregate_features(
                    historical_features, 
                    features
                )
                return aggregated
            
            return {}
            
        except Exception as e:
            logger.error(f"Error generating aggregated features: {e}")
            return {}
    
    async def process_batch(self, company_data: Dict[str, List[Dict[str, Any]]]) -> List[ProcessingResult]:
        """Process multiple companies in batch"""
        tasks = [
            self.process_company_data(company, data) 
            for company, data in company_data.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, ProcessingResult):
                processed_results.append(result)
            else:
                logger.error(f"Error in batch processing: {result}")
        
        logger.info(f"Processed {len(processed_results)} companies successfully")
        return processed_results
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return await self.feature_store.get_feature_importance()
    
    async def validate_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Validate feature quality and completeness"""
        validation_results = {
            'valid': True,
            'issues': [],
            'completeness': 0.0,
            'quality_score': 0.0
        }
        
        # Check for missing values
        missing_count = sum(1 for v in features.values() if v is None or np.isnan(v))
        validation_results['completeness'] = 1.0 - (missing_count / len(features))
        
        # Check for outliers
        values = [v for v in features.values() if v is not None and not np.isnan(v)]
        if values:
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            outlier_count = sum(1 for v in values if v < (q25 - 1.5 * iqr) or v > (q75 + 1.5 * iqr))
            outlier_ratio = outlier_count / len(values)
            
            if outlier_ratio > 0.1:
                validation_results['issues'].append(f"High outlier ratio: {outlier_ratio:.2%}")
        
        # Calculate overall quality score
        validation_results['quality_score'] = validation_results['completeness'] * (1.0 - min(outlier_ratio, 0.5))
        
        if validation_results['quality_score'] < 0.7:
            validation_results['valid'] = False
            validation_results['issues'].append("Low overall quality score")
        
        return validation_results
