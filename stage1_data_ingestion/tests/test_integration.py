# Imports
import pytest
import asyncio

from stage1_data_ingestion.data_processing.data_models import DataPoint
from ..main_pipeline import DataIngestionPipeline

# Integration tests for full pipeline
@pytest.mark.asyncio
async def test_data_ingestion_pipeline():
    pipeline = DataIngestionPipeline()
    data = await pipeline.run_scraping_cycle()
    assert isinstance(data, list)
    assert all(isinstance(item, DataPoint) for item in data)

