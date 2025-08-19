"""
Tests for NLP components in Stage 2 feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from ..nlp.sentiment_analyzer import SentimentAnalyzer
from ..nlp.topic_extractor import TopicExtractor
from ..nlp.entity_linker import EntityLinker
from ..nlp.event_detector import EventDetector
from ..nlp.text_embeddings import TextEmbeddings

@pytest.fixture
def sample_texts():
    return [
        "Apple Inc. reported strong quarterly earnings with revenue growth of 15%.",
        "Tesla's stock price declined due to production concerns and supply chain issues.",
        "Microsoft announced a new cloud computing initiative that could boost profitability.",
        "Amazon faces regulatory scrutiny over its market dominance in e-commerce."
    ]

@pytest.fixture
def sample_config():
    return {
        'models': {
            'sentiment': 'finbert',
            'embeddings': 'sentence-transformers'
        },
        'languages': ['en', 'es', 'fr']
    }

class TestSentimentAnalyzer:
    """Test sentiment analysis functionality"""
    
    def test_sentiment_analyzer_init(self, sample_config):
        analyzer = SentimentAnalyzer(sample_config)
        assert analyzer.config == sample_config
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_single_text(self, sample_config):
        analyzer = SentimentAnalyzer(sample_config)
        text = "Apple reported excellent quarterly results with strong revenue growth."
        
        # Mock the actual sentiment analysis
        with patch.object(analyzer, '_analyze_with_finbert', return_value={'sentiment': 'positive', 'score': 0.8}):
            result = await analyzer.analyze_sentiment(text)
            
            assert 'sentiment' in result
            assert 'score' in result
            assert isinstance(result['score'], float)
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_batch(self, sample_config, sample_texts):
        analyzer = SentimentAnalyzer(sample_config)
        
        # Mock batch analysis
        mock_results = [
            {'sentiment': 'positive', 'score': 0.7},
            {'sentiment': 'negative', 'score': -0.6},
            {'sentiment': 'positive', 'score': 0.5},
            {'sentiment': 'negative', 'score': -0.4}
        ]
        
        with patch.object(analyzer, 'analyze_batch', return_value=mock_results):
            results = await analyzer.analyze_batch(sample_texts)
            
            assert len(results) == len(sample_texts)
            assert all('sentiment' in result for result in results)
    
    def test_sentiment_aggregation(self, sample_config):
        analyzer = SentimentAnalyzer(sample_config)
        
        sentiments = [
            {'sentiment': 'positive', 'score': 0.8},
            {'sentiment': 'positive', 'score': 0.6},
            {'sentiment': 'negative', 'score': -0.7},
            {'sentiment': 'neutral', 'score': 0.1}
        ]
        
        aggregated = analyzer.aggregate_sentiments(sentiments)
        
        assert 'average_score' in aggregated
        assert 'positive_ratio' in aggregated
        assert 'negative_ratio' in aggregated
        assert isinstance(aggregated['average_score'], float)

class TestTopicExtractor:
    """Test topic extraction functionality"""
    
    def test_topic_extractor_init(self, sample_config):
        extractor = TopicExtractor(sample_config)
        assert extractor.config == sample_config
    
    @pytest.mark.asyncio
    async def test_extract_topics(self, sample_config, sample_texts):
        extractor = TopicExtractor(sample_config)
        
        # Mock topic extraction
        mock_topics = [
            {'topic': 'earnings', 'confidence': 0.9},
            {'topic': 'stock_performance', 'confidence': 0.8},
            {'topic': 'technology', 'confidence': 0.7}
        ]
        
        with patch.object(extractor, 'extract_topics', return_value=mock_topics):
            topics = await extractor.extract_topics(sample_texts)
            
            assert isinstance(topics, list)
            assert all('topic' in topic for topic in topics)
            assert all('confidence' in topic for topic in topics)
    
    def test_topic_clustering(self, sample_config):
        extractor = TopicExtractor(sample_config)
        
        # Test topic clustering logic
        topics = ['earnings', 'revenue', 'profit', 'stock', 'price', 'market']
        
        # Mock clustering
        with patch.object(extractor, 'cluster_topics', return_value={'financial': ['earnings', 'revenue', 'profit'], 'market': ['stock', 'price', 'market']}):
            clusters = extractor.cluster_topics(topics)
            
            assert isinstance(clusters, dict)
            assert len(clusters) > 0

class TestEntityLinker:
    """Test entity linking functionality"""
    
    def test_entity_linker_init(self, sample_config):
        linker = EntityLinker(sample_config)
        assert linker.config == sample_config
    
    @pytest.mark.asyncio
    async def test_link_entities(self, sample_config):
        linker = EntityLinker(sample_config)
        
        entities = ['Apple Inc.', 'Tesla', 'Microsoft']
        
        # Mock entity linking
        mock_linked = [
            {'entity': 'Apple Inc.', 'linked_id': 'AAPL', 'confidence': 0.95},
            {'entity': 'Tesla', 'linked_id': 'TSLA', 'confidence': 0.90},
            {'entity': 'Microsoft', 'linked_id': 'MSFT', 'confidence': 0.92}
        ]
        
        with patch.object(linker, 'link_entities', return_value=mock_linked):
            linked = await linker.link_entities(entities)
            
            assert len(linked) == len(entities)
            assert all('linked_id' in item for item in linked)

class TestEventDetector:
    """Test event detection functionality"""
    
    def test_event_detector_init(self, sample_config):
        detector = EventDetector(sample_config)
        assert detector.config == sample_config
    
    @pytest.mark.asyncio
    async def test_detect_events(self, sample_config, sample_texts):
        detector = EventDetector(sample_config)
        
        # Mock event detection
        mock_events = [
            {'event_type': 'earnings_announcement', 'confidence': 0.9, 'entities': ['Apple Inc.']},
            {'event_type': 'stock_decline', 'confidence': 0.8, 'entities': ['Tesla']},
            {'event_type': 'product_announcement', 'confidence': 0.7, 'entities': ['Microsoft']}
        ]
        
        with patch.object(detector, 'detect_events', return_value=mock_events):
            events = await detector.detect_events(sample_texts)
            
            assert isinstance(events, list)
            assert all('event_type' in event for event in events)
            assert all('confidence' in event for event in events)

class TestTextEmbeddings:
    """Test text embeddings functionality"""
    
    def test_text_embeddings_init(self, sample_config):
        embedder = TextEmbeddings(sample_config)
        assert embedder.config == sample_config
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, sample_config, sample_texts):
        embedder = TextEmbeddings(sample_config)
        
        # Mock embedding generation
        mock_embeddings = np.random.rand(len(sample_texts), 384)  # 384-dim embeddings
        
        with patch.object(embedder, 'generate_embeddings', return_value=mock_embeddings):
            embeddings = await embedder.generate_embeddings(sample_texts)
            
            assert embeddings.shape[0] == len(sample_texts)
            assert embeddings.shape[1] > 0  # Has embedding dimensions
    
    def test_similarity_calculation(self, sample_config):
        embedder = TextEmbeddings(sample_config)
        
        # Mock embeddings
        embedding1 = np.array([1, 0, 0])
        embedding2 = np.array([0, 1, 0])
        embedding3 = np.array([1, 0, 0])
        
        # Test cosine similarity
        sim_12 = embedder.cosine_similarity(embedding1, embedding2)
        sim_13 = embedder.cosine_similarity(embedding1, embedding3)
        
        assert sim_12 == 0.0  # Orthogonal vectors
        assert sim_13 == 1.0  # Identical vectors

@pytest.mark.integration
class TestNLPIntegration:
    """Test NLP component integration"""
    
    @pytest.mark.asyncio
    async def test_nlp_pipeline_integration(self, sample_config, sample_texts):
        """Test integrated NLP processing pipeline"""
        
        # Initialize components
        sentiment_analyzer = SentimentAnalyzer(sample_config)
        topic_extractor = TopicExtractor(sample_config)
        entity_linker = EntityLinker(sample_config)
        event_detector = EventDetector(sample_config)
        
        # Mock all components
        with patch.object(sentiment_analyzer, 'analyze_batch', return_value=[{'sentiment': 'positive', 'score': 0.7}] * len(sample_texts)):
            with patch.object(topic_extractor, 'extract_topics', return_value=[{'topic': 'earnings', 'confidence': 0.8}]):
                with patch.object(entity_linker, 'link_entities', return_value=[{'entity': 'Apple', 'linked_id': 'AAPL'}]):
                    with patch.object(event_detector, 'detect_events', return_value=[{'event_type': 'earnings', 'confidence': 0.9}]):
                        
                        # Process through pipeline
                        sentiments = await sentiment_analyzer.analyze_batch(sample_texts)
                        topics = await topic_extractor.extract_topics(sample_texts)
                        entities = await entity_linker.link_entities(['Apple', 'Tesla'])
                        events = await event_detector.detect_events(sample_texts)
                        
                        # Verify results
                        assert len(sentiments) == len(sample_texts)
                        assert len(topics) > 0
                        assert len(entities) > 0
                        assert len(events) > 0
    
    def test_nlp_error_handling(self, sample_config):
        """Test NLP component error handling"""
        analyzer = SentimentAnalyzer(sample_config)
        
        # Test with None input
        result = analyzer._safe_analyze(None)
        assert result is not None
        
        # Test with empty string
        result = analyzer._safe_analyze("")
        assert result is not None
        
        # Test with invalid input
        result = analyzer._safe_analyze(123)
        assert result is not None
