"""
Text preprocessing for multilingual financial content.
Handles cleaning, normalization, and tokenization of text data.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import string

logger = logging.getLogger(__name__)

class TextProcessor:
    """Advanced text processor for multilingual financial content"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_models = {}
        self.lemmatizer = WordNetLemmatizer()
        self.financial_terms = self._load_financial_terms()
        
    async def initialize(self):
        """Initialize NLP models and download required resources"""
        try:
            # Download NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Load spaCy models for different languages
            models_to_load = self.config.get('spacy_models', {
                'en': 'en_core_web_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm',
                'de': 'de_core_news_sm',
                'zh': 'zh_core_web_sm'
            })
            
            for lang, model_name in models_to_load.items():
                try:
                    self.nlp_models[lang] = spacy.load(model_name)
                except OSError:
                    logger.warning(f"spaCy model {model_name} not found for language {lang}")
                    # Fallback to basic English model
                    if lang != 'en':
                        try:
                            self.nlp_models[lang] = spacy.load('en_core_web_sm')
                        except OSError:
                            logger.warning(f"Fallback English model also not available")
            
            logger.info("Text processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text processor: {e}")
            raise
    
    def _load_financial_terms(self) -> Dict[str, List[str]]:
        """Load financial terminology for domain-specific processing"""
        return {
            'positive_indicators': [
                'profit', 'growth', 'increase', 'gain', 'revenue', 'earnings',
                'bullish', 'upgrade', 'outperform', 'buy', 'strong', 'beat',
                'exceed', 'expansion', 'merger', 'acquisition', 'dividend',
                'rally', 'surge', 'boom', 'recovery', 'uptrend'
            ],
            'negative_indicators': [
                'loss', 'decline', 'decrease', 'fall', 'drop', 'bearish',
                'downgrade', 'underperform', 'sell', 'weak', 'miss',
                'bankruptcy', 'debt', 'lawsuit', 'scandal', 'recession',
                'layoffs', 'restructuring', 'warning', 'crash', 'plunge'
            ],
            'financial_metrics': [
                'revenue', 'ebitda', 'eps', 'pe ratio', 'debt ratio',
                'roe', 'roa', 'margin', 'cash flow', 'market cap',
                'book value', 'dividend yield', 'beta', 'volatility'
            ],
            'market_terms': [
                'stock', 'share', 'equity', 'bond', 'commodity',
                'futures', 'options', 'etf', 'mutual fund', 'index',
                'nasdaq', 'nyse', 's&p', 'dow jones', 'ftse'
            ]
        }
    
    async def process_text(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Process text with comprehensive NLP pipeline"""
        if not text or not text.strip():
            return {
                'original_text': text,
                'cleaned_text': '',
                'tokens': [],
                'sentences': [],
                'entities': [],
                'financial_terms': [],
                'language': language
            }
        
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Get appropriate NLP model
        nlp = self.nlp_models.get(language, self.nlp_models.get('en'))
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'language': language
        }
        
        if nlp:
            # Process with spaCy
            doc = nlp(cleaned_text)
            
            result.update({
                'tokens': [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct],
                'sentences': [sent.text.strip() for sent in doc.sents],
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'pos_tags': [(token.text, token.pos_) for token in doc],
                'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
            })
        else:
            # Fallback to NLTK
            result.update(await self._process_with_nltk(cleaned_text, language))
        
        # Extract financial terms
        result['financial_terms'] = self._extract_financial_terms(cleaned_text)
        
        # Calculate text statistics
        result['statistics'] = self._calculate_text_stats(cleaned_text)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Handle currency symbols and numbers
        text = re.sub(r'\$([0-9,]+\.?[0-9]*)', r'USD \1', text)
        text = re.sub(r'€([0-9,]+\.?[0-9]*)', r'EUR \1', text)
        text = re.sub(r'£([0-9,]+\.?[0-9]*)', r'GBP \1', text)
        
        return text.strip()
    
    async def _process_with_nltk(self, text: str, language: str) -> Dict[str, Any]:
        """Fallback processing with NLTK"""
        try:
            # Tokenization
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            
            # Remove stopwords and punctuation
            stop_words = set(stopwords.words('english'))  # Default to English
            tokens = [
                self.lemmatizer.lemmatize(word) 
                for word in words 
                if word not in stop_words and word not in string.punctuation
            ]
            
            # POS tagging
            pos_tags = nltk.pos_tag(words)
            
            return {
                'tokens': tokens,
                'sentences': sentences,
                'entities': [],  # NLTK doesn't provide NER out of the box
                'pos_tags': pos_tags,
                'noun_phrases': []
            }
            
        except Exception as e:
            logger.error(f"Error in NLTK processing: {e}")
            return {
                'tokens': text.split(),
                'sentences': [text],
                'entities': [],
                'pos_tags': [],
                'noun_phrases': []
            }
    
    def _extract_financial_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract financial terms from text"""
        text_lower = text.lower()
        found_terms = {}
        
        for category, terms in self.financial_terms.items():
            found_terms[category] = [
                term for term in terms 
                if term.lower() in text_lower
            ]
        
        return found_terms
    
    def _calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate text statistics"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'avg_chars_per_word': len(text) / max(len(words), 1)
        }
    
    async def process_batch(self, texts: List[str], language: str = 'en') -> List[Dict[str, Any]]:
        """Process multiple texts in batch"""
        tasks = [self.process_text(text, language) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key phrases using simple frequency analysis"""
        # This is a simplified implementation
        # In production, you might use more sophisticated methods like TF-IDF or TextRank
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def detect_financial_events(self, text: str) -> List[Dict[str, Any]]:
        """Detect potential financial events in text"""
        events = []
        text_lower = text.lower()
        
        # Define event patterns
        event_patterns = {
            'earnings_release': [
                r'earnings? (?:report|release|announcement)',
                r'quarterly (?:results|earnings)',
                r'q[1-4] (?:earnings|results)'
            ],
            'merger_acquisition': [
                r'merger? (?:with|and)',
                r'acquisition (?:of|by)',
                r'takeover (?:bid|offer)'
            ],
            'dividend': [
                r'dividend (?:payment|announcement|increase|cut)',
                r'(?:special|interim|final) dividend'
            ],
            'stock_split': [
                r'stock split',
                r'share split',
                r'(?:[0-9]+)-for-(?:[0-9]+) split'
            ],
            'ipo': [
                r'initial public offering',
                r'going public',
                r'ipo (?:filing|announcement)'
            ]
        }
        
        for event_type, patterns in event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    events.append({
                        'event_type': event_type,
                        'text_span': match.group(),
                        'start_pos': match.start(),
                        'end_pos': match.end()
                    })
        
        return events
    
    def normalize_company_names(self, text: str) -> str:
        """Normalize company name variations"""
        # Common company suffix variations
        replacements = {
            r'\b(corp|corporation)\b': 'Corporation',
            r'\b(inc|incorporated)\b': 'Inc.',
            r'\b(ltd|limited)\b': 'Ltd.',
            r'\b(llc)\b': 'LLC',
            r'\b(co)\b': 'Company'
        }
        
        normalized_text = text
        for pattern, replacement in replacements.items():
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    async def summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive text summarization"""
        if not text or not text.strip():
            return ""
        
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring based on sentence position and length
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position score (earlier sentences get higher scores)
            score += (len(sentences) - i) / len(sentences)
            
            # Length score (moderate length sentences preferred)
            words = len(sentence.split())
            if 10 <= words <= 30:
                score += 0.5
            
            # Financial terms score
            financial_term_count = sum(
                1 for term_list in self.financial_terms.values()
                for term in term_list
                if term.lower() in sentence.lower()
            )
            score += financial_term_count * 0.3
            
            sentence_scores.append((sentence, score))
        
        # Select top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in sentence_scores[:max_sentences]]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)
