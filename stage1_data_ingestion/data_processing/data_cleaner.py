# Imports
import re
from typing import Optional
from bs4 import BeautifulSoup

class DataCleaner:
    # Function: clean_html_content()
    @staticmethod
    def clean_html_content(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

# Function: normalize_text()
    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        return text

# Function: remove_duplicates()
    @staticmethod
    def remove_duplicates(items: list) -> list:
        return list(set(items))