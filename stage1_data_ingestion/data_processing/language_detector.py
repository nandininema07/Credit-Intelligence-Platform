"""
Language detection for multilingual financial content.
Supports detection of major languages in financial news and social media.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import re

logger = logging.getLogger(__name__)

# Set seed for consistent results
DetectorFactory.seed = 0

class LanguageDetector:
    """Advanced language detector for financial content"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.supported_languages = config.get('supported_languages', [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi'
        ])
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.financial_keywords = self._load_multilingual_keywords()
        
    async def initialize(self):
        """Initialize language detector"""
        logger.info("Language detector initialized successfully")
    
    def _load_multilingual_keywords(self) -> Dict[str, List[str]]:
        """Load financial keywords in different languages"""
        return {
            'en': ['stock', 'market', 'finance', 'investment', 'profit', 'loss', 'revenue', 'earnings'],
            'es': ['acción', 'mercado', 'finanzas', 'inversión', 'beneficio', 'pérdida', 'ingresos'],
            'fr': ['action', 'marché', 'finance', 'investissement', 'profit', 'perte', 'revenus'],
            'de': ['aktie', 'markt', 'finanzen', 'investition', 'gewinn', 'verlust', 'umsatz'],
            'it': ['azione', 'mercato', 'finanza', 'investimento', 'profitto', 'perdita', 'ricavi'],
            'pt': ['ação', 'mercado', 'finanças', 'investimento', 'lucro', 'perda', 'receita'],
            'zh': ['股票', '市场', '金融', '投资', '利润', '损失', '收入'],
            'ja': ['株式', '市場', '金融', '投資', '利益', '損失', '収益'],
            'ru': ['акция', 'рынок', 'финансы', 'инвестиции', 'прибыль', 'убыток', 'доходы'],
            'ar': ['سهم', 'سوق', 'مالية', 'استثمار', 'ربح', 'خسارة', 'إيرادات'],
            'hi': ['शेयर', 'बाजार', 'वित्त', 'निवेश', 'लाभ', 'हानि', 'आय']
        }
    
    async def detect_language(self, text: str) -> str:
        """Detect language of text with financial context awareness"""
        if not text or not text.strip():
            return 'en'  # Default to English
        
        # Clean text for better detection
        cleaned_text = self._clean_text_for_detection(text)
        
        if len(cleaned_text) < 10:
            return 'en'  # Default for very short text
        
        try:
            # Primary detection
            detected_lang = detect(cleaned_text)
            
            # Verify with confidence scores
            lang_probs = detect_langs(cleaned_text)
            confidence = max(prob.prob for prob in lang_probs if prob.lang == detected_lang)
            
            # If confidence is low, try financial keyword matching
            if confidence < self.confidence_threshold:
                keyword_lang = self._detect_by_keywords(cleaned_text)
                if keyword_lang:
                    detected_lang = keyword_lang
            
            # Ensure detected language is supported
            if detected_lang not in self.supported_languages:
                detected_lang = 'en'  # Fallback to English
                
            return detected_lang
            
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return 'en'  # Default fallback
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text to improve language detection accuracy"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers and special characters that might confuse detection
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _detect_by_keywords(self, text: str) -> Optional[str]:
        """Detect language using financial keywords"""
        text_lower = text.lower()
        keyword_scores = {}
        
        for lang, keywords in self.financial_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                keyword_scores[lang] = score
        
        if keyword_scores:
            # Return language with highest keyword score
            return max(keyword_scores, key=keyword_scores.get)
        
        return None
    
    async def detect_language_batch(self, texts: List[str]) -> List[str]:
        """Detect languages for multiple texts"""
        tasks = [self.detect_language(text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
    
    def get_language_confidence(self, text: str) -> Dict[str, float]:
        """Get confidence scores for all detected languages"""
        if not text or not text.strip():
            return {'en': 1.0}
        
        cleaned_text = self._clean_text_for_detection(text)
        
        try:
            lang_probs = detect_langs(cleaned_text)
            return {prob.lang: prob.prob for prob in lang_probs}
        except LangDetectException:
            return {'en': 1.0}
    
    def is_mixed_language(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text contains mixed languages"""
        confidences = self.get_language_confidence(text)
        
        # If multiple languages have confidence > threshold, it's mixed
        high_confidence_langs = [lang for lang, conf in confidences.items() if conf > threshold]
        return len(high_confidence_langs) > 1
    
    def detect_script_type(self, text: str) -> str:
        """Detect script type (Latin, Cyrillic, Arabic, etc.)"""
        # Count characters by script
        script_counts = {
            'latin': 0,
            'cyrillic': 0,
            'arabic': 0,
            'chinese': 0,
            'japanese': 0,
            'korean': 0,
            'devanagari': 0
        }
        
        for char in text:
            code = ord(char)
            if 0x0041 <= code <= 0x007A or 0x00C0 <= code <= 0x017F:  # Latin
                script_counts['latin'] += 1
            elif 0x0400 <= code <= 0x04FF:  # Cyrillic
                script_counts['cyrillic'] += 1
            elif 0x0600 <= code <= 0x06FF:  # Arabic
                script_counts['arabic'] += 1
            elif 0x4E00 <= code <= 0x9FFF:  # Chinese
                script_counts['chinese'] += 1
            elif 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF:  # Japanese
                script_counts['japanese'] += 1
            elif 0xAC00 <= code <= 0xD7AF:  # Korean
                script_counts['korean'] += 1
            elif 0x0900 <= code <= 0x097F:  # Devanagari (Hindi)
                script_counts['devanagari'] += 1
        
        # Return script with highest count
        if sum(script_counts.values()) == 0:
            return 'latin'  # Default
        
        return max(script_counts, key=script_counts.get)
    
    async def detect_with_fallback(self, text: str, fallback_lang: str = 'en') -> Dict[str, Any]:
        """Comprehensive language detection with fallback options"""
        result = {
            'detected_language': fallback_lang,
            'confidence': 0.0,
            'script_type': 'latin',
            'is_mixed': False,
            'all_probabilities': {},
            'method_used': 'fallback'
        }
        
        if not text or not text.strip():
            return result
        
        try:
            # Primary detection
            detected_lang = await self.detect_language(text)
            confidences = self.get_language_confidence(text)
            
            result.update({
                'detected_language': detected_lang,
                'confidence': confidences.get(detected_lang, 0.0),
                'script_type': self.detect_script_type(text),
                'is_mixed': self.is_mixed_language(text),
                'all_probabilities': confidences,
                'method_used': 'langdetect'
            })
            
            # If confidence is still low, try keyword method
            if result['confidence'] < self.confidence_threshold:
                keyword_lang = self._detect_by_keywords(text)
                if keyword_lang:
                    result['detected_language'] = keyword_lang
                    result['method_used'] = 'keywords'
                    result['confidence'] = 0.8  # Assign reasonable confidence for keyword detection
            
        except Exception as e:
            logger.error(f"Error in comprehensive language detection: {e}")
        
        return result
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'th': 'Thai',
            'vi': 'Vietnamese'
        }
        
        return language_names.get(lang_code, lang_code.upper())
    
    def is_financial_content(self, text: str, language: str) -> bool:
        """Check if text contains financial content based on keywords"""
        if language not in self.financial_keywords:
            language = 'en'  # Fallback to English keywords
        
        keywords = self.financial_keywords[language]
        text_lower = text.lower()
        
        # Count financial keywords
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Consider it financial if it has at least 2 financial keywords
        # or 1 keyword in short text
        word_count = len(text.split())
        if word_count < 50:
            return keyword_count >= 1
        else:
            return keyword_count >= 2
