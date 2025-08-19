# Imports
from langdetect import detect
from typing import Optional
import logging

class LanguageDetector:
    @staticmethod
    # Function: enhanced_language_detection()
    def enhanced_language_detection(text: str) -> Optional[str]:
        try:
            lang = detect(text)
            logging.info(f"Detected language: {lang}")
            return lang
        except Exception as e:
            logging.error(f"Error detecting language: {e}")
            return None
