# Imports
import pytest
from ..data_processing.text_processor import DataProcessor
from ..data_processing.language_detector import LanguageDetector

# Test functions for data processing
@pytest.mark.asyncio
async def test_data_processor():
    processor = DataProcessor()
    text = "This is a test."
    processed = await processor.process_text(text)
    assert isinstance(processed, str)
    assert processed == "This is a test."

@pytest.mark.asyncio
async def test_language_detector():
    detector = LanguageDetector()
    text = "Ceci est un test."
    language = await detector.detect_language(text)
    assert language == "fr"