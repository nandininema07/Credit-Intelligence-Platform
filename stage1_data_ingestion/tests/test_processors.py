"""
Tests for data processing components in Stage 1.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

from ..data_processing.text_processor import TextProcessor
from ..data_processing.language_detector import LanguageDetector
from ..data_processing.entity_extractor import EntityExtractor
from ..data_processing.data_cleaner import DataCleaner

@pytest.fixture
def sample_text():
    return "Apple Inc. reported strong quarterly earnings with revenue of $123.5 billion, exceeding analyst expectations."

@pytest.fixture
def sample_article():
    return {
        'title': 'Apple Reports Strong Q4 Earnings',
        'content': 'Apple Inc. announced record-breaking quarterly results...',
        'url': 'https://example.com/apple-earnings',
        'source': 'Financial Times',
        'published_date': datetime.now(),
        'author': 'John Smith'
    }

class TestTextProcessor:
    """Test text processing functionality"""
    
    def test_text_processor_init(self):
        config = {'language_models': ['en', 'es', 'fr']}
        processor = TextProcessor(config)
        assert processor.config == config
    
    def test_clean_text(self, sample_text):
        processor = TextProcessor({})
        cleaned = processor.clean_text(sample_text)
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
    
    def test_extract_financial_terms(self, sample_text):
        processor = TextProcessor({})
        terms = processor.extract_financial_terms(sample_text)
        assert isinstance(terms, list)
        assert any('revenue' in term.lower() for term in terms)
    
    def test_tokenize_text(self, sample_text):
        processor = TextProcessor({})
        tokens = processor.tokenize_text(sample_text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
    
    def test_detect_events(self, sample_text):
        processor = TextProcessor({})
        events = processor.detect_events(sample_text)
        assert isinstance(events, list)

class TestLanguageDetector:
    """Test language detection functionality"""
    
    def test_language_detector_init(self):
        config = {'confidence_threshold': 0.8}
        detector = LanguageDetector(config)
        assert detector.confidence_threshold == 0.8
    
    def test_detect_language_english(self):
        detector = LanguageDetector({})
        text = "This is an English sentence about financial markets."
        result = detector.detect_language(text)
        assert result['language'] == 'en'
        assert result['confidence'] > 0.5
    
    def test_detect_language_spanish(self):
        detector = LanguageDetector({})
        text = "Esta es una oración en español sobre mercados financieros."
        result = detector.detect_language(text)
        assert result['language'] == 'es'
    
    def test_detect_mixed_language(self):
        detector = LanguageDetector({})
        text = "This is English and esto es español mixed together."
        result = detector.detect_mixed_languages(text)
        assert isinstance(result, list)
        assert len(result) >= 1

class TestEntityExtractor:
    """Test entity extraction functionality"""
    
    def test_entity_extractor_init(self):
        config = {'spacy_model': 'en_core_web_sm'}
        extractor = EntityExtractor(config)
        assert extractor.config == config
    
    def test_extract_companies(self, sample_text):
        extractor = EntityExtractor({})
        companies = extractor.extract_companies(sample_text)
        assert isinstance(companies, list)
        assert any('Apple' in company for company in companies)
    
    def test_extract_financial_metrics(self, sample_text):
        extractor = EntityExtractor({})
        metrics = extractor.extract_financial_metrics(sample_text)
        assert isinstance(metrics, list)
        assert len(metrics) > 0
    
    def test_extract_people(self):
        extractor = EntityExtractor({})
        text = "CEO Tim Cook announced the quarterly results."
        people = extractor.extract_people(text)
        assert isinstance(people, list)
    
    def test_extract_ticker_symbols(self):
        extractor = EntityExtractor({})
        text = "AAPL stock price increased while MSFT remained stable."
        tickers = extractor.extract_ticker_symbols(text)
        assert isinstance(tickers, list)
        assert 'AAPL' in tickers or 'MSFT' in tickers

class TestDataCleaner:
    """Test data cleaning functionality"""
    
    def test_data_cleaner_init(self):
        config = {'quality_threshold': 0.7}
        cleaner = DataCleaner(config)
        assert cleaner.quality_threshold == 0.7
    
    def test_clean_article(self, sample_article):
        cleaner = DataCleaner({})
        cleaned = cleaner.clean_article(sample_article)
        assert isinstance(cleaned, dict)
        assert 'title' in cleaned
        assert 'content' in cleaned
    
    def test_validate_article(self, sample_article):
        cleaner = DataCleaner({})
        is_valid = cleaner.validate_article(sample_article)
        assert isinstance(is_valid, bool)
        assert is_valid == True
    
    def test_calculate_quality_score(self, sample_article):
        cleaner = DataCleaner({})
        score = cleaner.calculate_quality_score(sample_article)
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_detect_duplicates(self):
        cleaner = DataCleaner({})
        articles = [
            {'title': 'Same Title', 'content': 'Same content'},
            {'title': 'Same Title', 'content': 'Same content'},
            {'title': 'Different Title', 'content': 'Different content'}
        ]
        duplicates = cleaner.detect_duplicates(articles)
        assert isinstance(duplicates, list)
        assert len(duplicates) >= 1
    
    def test_normalize_text(self):
        cleaner = DataCleaner({})
        text = "  This   has   extra   spaces  "
        normalized = cleaner.normalize_text(text)
        assert normalized.strip() == "This has extra spaces"
    
    def test_filter_spam(self, sample_article):
        cleaner = DataCleaner({})
        is_spam = cleaner.is_spam(sample_article)
        assert isinstance(is_spam, bool)
        assert is_spam == False

@pytest.mark.asyncio
async def test_processing_pipeline():
    """Test integrated processing pipeline"""
    config = {}
    
    # Initialize processors
    text_processor = TextProcessor(config)
    language_detector = LanguageDetector(config)
    entity_extractor = EntityExtractor(config)
    data_cleaner = DataCleaner(config)
    
    # Sample data
    article = {
        'title': 'Apple Inc. Reports Strong Quarterly Earnings',
        'content': 'Apple Inc. announced record quarterly revenue of $123.5 billion...',
        'url': 'https://example.com/apple-earnings',
        'source': 'Financial News',
        'published_date': datetime.now()
    }
    
    # Process through pipeline
    cleaned_article = data_cleaner.clean_article(article)
    assert cleaned_article is not None
    
    language_info = language_detector.detect_language(cleaned_article['content'])
    assert language_info['language'] is not None
    
    entities = entity_extractor.extract_companies(cleaned_article['content'])
    assert isinstance(entities, list)
    
    processed_text = text_processor.clean_text(cleaned_article['content'])
    assert isinstance(processed_text, str)

def test_error_handling():
    """Test error handling in processors"""
    processor = TextProcessor({})
    
    # Test with None input
    result = processor.clean_text(None)
    assert result == ""
    
    # Test with empty string
    result = processor.clean_text("")
    assert result == ""
    
    # Test with invalid input type
    result = processor.clean_text(123)
    assert isinstance(result, str)
