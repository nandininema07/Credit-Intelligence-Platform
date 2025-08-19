# Imports
import re
from typing import List, Dict
from ..config.company_registry import CompanyRegistry
from .data_models import DataPoint
from .text_processor import DataProcessor

class EntityExtractor:
    @staticmethod
    # Function: extract_companies_from_content()
    def extract_companies_from_content(text: str, company_registry: CompanyRegistry) -> List[str]:
        return DataProcessor.extract_companies_from_text(text, company_registry)

    @staticmethod
    # Function: extract_financial_entities()
    def extract_financial_entities(text: str, company_registry: CompanyRegistry) -> List[str]:
        return DataProcessor.extract_financial_entities_from_text(text, company_registry)