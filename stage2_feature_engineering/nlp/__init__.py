"""
NLP module for multilingual sentiment analysis and text processing.
"""

from .sentiment_analyzer import SentimentAnalyzer
from .topic_extractor import TopicExtractor
from .entity_linker import EntityLinker
from .event_detector import EventDetector
from .text_embeddings import TextEmbeddings

__all__ = [
    'SentimentAnalyzer',
    'TopicExtractor',
    'EntityLinker',
    'EventDetector',
    'TextEmbeddings'
]
