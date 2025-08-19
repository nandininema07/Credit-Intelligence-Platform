"""
Entity extraction for financial content.
Extracts companies, people, organizations, and financial entities from text.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import spacy
from spacy.matcher import Matcher
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Advanced entity extractor for financial content"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp_models = {}
        self.company_patterns = []
        self.financial_entities = self._load_financial_entities()
        self.matchers = {}
        
    async def initialize(self):
        """Initialize entity extraction models"""
        try:
            # Load spaCy models for different languages
            models_to_load = self.config.get('spacy_models', {
                'en': 'en_core_web_sm',
                'es': 'es_core_news_sm',
                'fr': 'fr_core_news_sm',
                'de': 'de_core_news_sm'
            })
            
            for lang, model_name in models_to_load.items():
                try:
                    nlp = spacy.load(model_name)
                    self.nlp_models[lang] = nlp
                    
                    # Initialize matcher for this language
                    matcher = Matcher(nlp.vocab)
                    self._add_financial_patterns(matcher, lang)
                    self.matchers[lang] = matcher
                    
                except OSError:
                    logger.warning(f"spaCy model {model_name} not found for language {lang}")
            
            logger.info("Entity extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing entity extractor: {e}")
            raise
    
    def _load_financial_entities(self) -> Dict[str, List[str]]:
        """Load known financial entities and patterns"""
        return {
            'stock_exchanges': [
                'NYSE', 'NASDAQ', 'LSE', 'TSE', 'SSE', 'HKEX', 'BSE', 'NSE',
                'Euronext', 'Frankfurt Stock Exchange', 'Toronto Stock Exchange'
            ],
            'financial_institutions': [
                'Federal Reserve', 'ECB', 'Bank of England', 'Bank of Japan',
                'Goldman Sachs', 'JPMorgan', 'Morgan Stanley', 'Citigroup',
                'Wells Fargo', 'Bank of America', 'Deutsche Bank', 'UBS'
            ],
            'rating_agencies': [
                'Moody\\'s', 'S&P', 'Fitch', 'Standard & Poor\\'s',
                'Moody\\'s Investors Service', 'Fitch Ratings'
            ],
            'regulatory_bodies': [
                'SEC', 'FINRA', 'CFTC', 'FCA', 'BaFin', 'ASIC', 'SEBI',
                'Securities and Exchange Commission', 'Financial Conduct Authority'
            ]
        }
    
    def _add_financial_patterns(self, matcher: Matcher, language: str):
        """Add financial entity patterns to matcher"""
        if language == 'en':
            # Company patterns
            company_patterns = [
                [{"LOWER": {"IN": ["inc", "corp", "ltd", "llc", "plc"]}},
                 {"IS_ALPHA": True, "OP": "?"}],
                [{"IS_TITLE": True}, {"LOWER": {"IN": ["corporation", "incorporated", "limited", "company"]}},
                 {"IS_ALPHA": True, "OP": "?"}],
                [{"IS_TITLE": True}, {"IS_TITLE": True, "OP": "?"}, 
                 {"LOWER": {"IN": ["ag", "sa", "nv", "ab", "as"]}}]
            ]
            
            for i, pattern in enumerate(company_patterns):
                matcher.add(f"COMPANY_{i}", [pattern])
            
            # Financial metrics patterns
            financial_patterns = [
                [{"LOWER": {"IN": ["revenue", "earnings", "profit", "loss"]}},
                 {"LIKE_NUM": True, "OP": "?"}],
                [{"LOWER": {"IN": ["eps", "pe", "roe", "roa"]}},
                 {"LOWER": "ratio", "OP": "?"}],
                [{"LOWER": "market"}, {"LOWER": "cap"}],
                [{"LOWER": {"IN": ["dividend", "yield"]}},
                 {"LIKE_NUM": True, "OP": "?"}]
            ]
            
            for i, pattern in enumerate(financial_patterns):
                matcher.add(f"FINANCIAL_METRIC_{i}", [pattern])
    
    async def extract_entities(self, text: str, language: str = 'en') -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from text"""
        if not text or not text.strip():
            return {
                'companies': [],
                'people': [],
                'organizations': [],
                'financial_metrics': [],
                'locations': [],
                'dates': [],
                'money': []
            }
        
        # Get appropriate NLP model
        nlp = self.nlp_models.get(language, self.nlp_models.get('en'))
        if not nlp:
            return await self._extract_with_regex(text)
        
        try:
            doc = nlp(text)
            entities = {
                'companies': [],
                'people': [],
                'organizations': [],
                'financial_metrics': [],
                'locations': [],
                'dates': [],
                'money': []
            }
            
            # Extract named entities
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                }
                
                if ent.label_ in ['ORG']:
                    if self._is_likely_company(ent.text):
                        entities['companies'].append(entity_info)
                    else:
                        entities['organizations'].append(entity_info)
                elif ent.label_ in ['PERSON']:
                    entities['people'].append(entity_info)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['locations'].append(entity_info)
                elif ent.label_ in ['DATE', 'TIME']:
                    entities['dates'].append(entity_info)
                elif ent.label_ in ['MONEY', 'PERCENT']:
                    entities['money'].append(entity_info)
            
            # Extract financial metrics using matcher
            if language in self.matchers:
                matches = self.matchers[language](doc)
                for match_id, start, end in matches:
                    span = doc[start:end]
                    label = nlp.vocab.strings[match_id]
                    
                    if label.startswith('FINANCIAL_METRIC'):
                        entities['financial_metrics'].append({
                            'text': span.text,
                            'label': 'FINANCIAL_METRIC',
                            'start': span.start_char,
                            'end': span.end_char,
                            'confidence': 0.9
                        })
            
            # Additional company extraction
            additional_companies = self._extract_companies_regex(text)
            for company in additional_companies:
                if not any(comp['text'].lower() == company['text'].lower() 
                          for comp in entities['companies']):
                    entities['companies'].append(company)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return await self._extract_with_regex(text)
    
    def _is_likely_company(self, text: str) -> bool:
        """Check if an organization entity is likely a company"""
        company_indicators = [
            'inc', 'corp', 'ltd', 'llc', 'plc', 'ag', 'sa', 'nv', 'ab', 'as',
            'corporation', 'incorporated', 'limited', 'company', 'group',
            'holdings', 'enterprises', 'industries', 'systems', 'technologies'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in company_indicators)
    
    def _extract_companies_regex(self, text: str) -> List[Dict[str, Any]]:
        """Extract companies using regex patterns"""
        companies = []
        
        # Pattern for company names with suffixes
        company_pattern = r'\b([A-Z][a-zA-Z\s&]+(?:Inc\.?|Corp\.?|Ltd\.?|LLC|PLC|Corporation|Incorporated|Limited|Company|Group|Holdings))\b'
        
        matches = re.finditer(company_pattern, text)
        for match in matches:
            companies.append({
                'text': match.group(1).strip(),
                'label': 'COMPANY',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8,
                'method': 'regex'
            })
        
        # Pattern for stock tickers
        ticker_pattern = r'\b([A-Z]{2,5})\s*(?:\$|\(NYSE\)|\(NASDAQ\)|\(LSE\))'
        ticker_matches = re.finditer(ticker_pattern, text)
        for match in ticker_matches:
            companies.append({
                'text': match.group(1),
                'label': 'TICKER',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'method': 'regex'
            })
        
        return companies
    
    async def _extract_with_regex(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback entity extraction using regex patterns"""
        entities = {
            'companies': [],
            'people': [],
            'organizations': [],
            'financial_metrics': [],
            'locations': [],
            'dates': [],
            'money': []
        }
        
        # Extract companies
        entities['companies'] = self._extract_companies_regex(text)
        
        # Extract money amounts
        money_pattern = r'\$([0-9,]+\.?[0-9]*)\s*(million|billion|trillion|M|B|T)?'
        money_matches = re.finditer(money_pattern, text, re.IGNORECASE)
        for match in money_matches:
            entities['money'].append({
                'text': match.group(),
                'label': 'MONEY',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9,
                'method': 'regex'
            })
        
        # Extract percentages
        percent_pattern = r'([0-9]+\.?[0-9]*)\s*%'
        percent_matches = re.finditer(percent_pattern, text)
        for match in percent_matches:
            entities['financial_metrics'].append({
                'text': match.group(),
                'label': 'PERCENTAGE',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8,
                'method': 'regex'
            })
        
        return entities
    
    async def extract_company_mentions(self, text: str, known_companies: List[str] = None) -> List[Dict[str, Any]]:
        """Extract specific company mentions from text"""
        mentions = []
        
        if known_companies:
            for company in known_companies:
                # Case-insensitive search for company mentions
                pattern = re.compile(re.escape(company), re.IGNORECASE)
                matches = pattern.finditer(text)
                
                for match in matches:
                    mentions.append({
                        'company': company,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 1.0
                    })
        
        # Also extract using general patterns
        general_entities = await self.extract_entities(text)
        for company_entity in general_entities['companies']:
            mentions.append({
                'company': company_entity['text'],
                'text': company_entity['text'],
                'start': company_entity['start'],
                'end': company_entity['end'],
                'confidence': company_entity['confidence']
            })
        
        # Remove duplicates
        unique_mentions = []
        seen_companies = set()
        
        for mention in mentions:
            company_key = mention['company'].lower().strip()
            if company_key not in seen_companies:
                seen_companies.add(company_key)
                unique_mentions.append(mention)
        
        return unique_mentions
    
    def extract_financial_figures(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial figures and metrics from text"""
        figures = []
        
        # Revenue/Sales patterns
        revenue_pattern = r'(?:revenue|sales|turnover)(?:\s+of)?\s*\$?([0-9,]+\.?[0-9]*)\s*(million|billion|trillion|M|B|T)?'
        revenue_matches = re.finditer(revenue_pattern, text, re.IGNORECASE)
        for match in revenue_matches:
            figures.append({
                'type': 'revenue',
                'value': match.group(1),
                'unit': match.group(2) or '',
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Profit/Loss patterns
        profit_pattern = r'(?:profit|earnings|income|loss)(?:\s+of)?\s*\$?([0-9,]+\.?[0-9]*)\s*(million|billion|trillion|M|B|T)?'
        profit_matches = re.finditer(profit_pattern, text, re.IGNORECASE)
        for match in profit_matches:
            figures.append({
                'type': 'profit_loss',
                'value': match.group(1),
                'unit': match.group(2) or '',
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Stock price patterns
        price_pattern = r'(?:stock|share|price)(?:\s+(?:at|of))?\s*\$([0-9,]+\.?[0-9]*)'
        price_matches = re.finditer(price_pattern, text, re.IGNORECASE)
        for match in price_matches:
            figures.append({
                'type': 'stock_price',
                'value': match.group(1),
                'unit': 'USD',
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        # Market cap patterns
        mcap_pattern = r'market\s+cap(?:italization)?\s*(?:of)?\s*\$?([0-9,]+\.?[0-9]*)\s*(million|billion|trillion|M|B|T)?'
        mcap_matches = re.finditer(mcap_pattern, text, re.IGNORECASE)
        for match in mcap_matches:
            figures.append({
                'type': 'market_cap',
                'value': match.group(1),
                'unit': match.group(2) or '',
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
        
        return figures
    
    def _add_financial_patterns(self, matcher: Matcher, language: str):
        """Add financial entity patterns to matcher"""
        if language == 'en':
            # Financial ratio patterns
            ratio_patterns = [
                [{"LOWER": {"IN": ["pe", "p/e"]}}, {"LOWER": "ratio", "OP": "?"}],
                [{"LOWER": "price"}, {"LOWER": "to"}, {"LOWER": "earnings"}],
                [{"LOWER": {"IN": ["roe", "roa", "roi"]}}, {"LOWER": "ratio", "OP": "?"}],
                [{"LOWER": "debt"}, {"LOWER": "to"}, {"LOWER": "equity"}],
                [{"LOWER": "current"}, {"LOWER": "ratio"}],
                [{"LOWER": "quick"}, {"LOWER": "ratio"}]
            ]
            
            for i, pattern in enumerate(ratio_patterns):
                matcher.add(f"FINANCIAL_RATIO_{i}", [pattern])
            
            # Currency patterns
            currency_patterns = [
                [{"LOWER": {"IN": ["usd", "eur", "gbp", "jpy", "cny"]}},
                 {"LIKE_NUM": True, "OP": "?"}],
                [{"TEXT": {"IN": ["$", "€", "£", "¥"]}},
                 {"LIKE_NUM": True}]
            ]
            
            for i, pattern in enumerate(currency_patterns):
                matcher.add(f"CURRENCY_{i}", [pattern])
    
    async def extract_entities_batch(self, texts: List[str], language: str = 'en') -> List[Dict[str, List[Dict[str, Any]]]]:
        """Extract entities from multiple texts"""
        tasks = [self.extract_entities(text, language) for text in texts]
        results = await asyncio.gather(*tasks)
        return results
    
    def normalize_entity(self, entity_text: str, entity_type: str) -> str:
        """Normalize entity text for consistency"""
        normalized = entity_text.strip()
        
        if entity_type == 'COMPANY':
            # Normalize company names
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized = normalized.title()
            
            # Standardize suffixes
            suffix_map = {
                'Corp': 'Corporation',
                'Inc': 'Inc.',
                'Ltd': 'Limited',
                'Co': 'Company'
            }
            
            for old_suffix, new_suffix in suffix_map.items():
                if normalized.endswith(old_suffix):
                    normalized = normalized[:-len(old_suffix)] + new_suffix
        
        elif entity_type == 'PERSON':
            # Normalize person names
            normalized = ' '.join(word.capitalize() for word in normalized.split())
        
        return normalized
    
    def link_entities_to_companies(self, entities: Dict[str, List[Dict[str, Any]]], 
                                 company_registry: List[str]) -> Dict[str, List[str]]:
        """Link extracted entities to known companies"""
        linked_entities = {}
        
        for company in company_registry:
            company_lower = company.lower()
            linked_entities[company] = []
            
            # Check all entity types for mentions of this company
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_text_lower = entity['text'].lower()
                    
                    # Simple fuzzy matching
                    if (company_lower in entity_text_lower or 
                        entity_text_lower in company_lower or
                        self._calculate_similarity(company_lower, entity_text_lower) > 0.8):
                        
                        linked_entities[company].append({
                            'entity_type': entity_type,
                            'entity_text': entity['text'],
                            'confidence': entity.get('confidence', 0.5)
                        })
        
        return linked_entities
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        # Simple Jaccard similarity
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union
    
    def extract_ticker_symbols(self, text: str) -> List[Dict[str, Any]]:
        """Extract stock ticker symbols from text"""
        tickers = []
        
        # Pattern for tickers in parentheses or with exchange info
        ticker_patterns = [
            r'\(([A-Z]{2,5})\)',  # (AAPL)
            r'\b([A-Z]{2,5})\s*\$',  # AAPL$
            r'\$([A-Z]{2,5})\b',  # $AAPL
            r'\b([A-Z]{2,5})(?:\s*(?:NYSE|NASDAQ|LSE))',  # AAPL NYSE
            r'(?:NYSE|NASDAQ|LSE):\s*([A-Z]{2,5})'  # NYSE: AAPL
        ]
        
        for pattern in ticker_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                ticker = match.group(1)
                
                # Filter out common false positives
                if ticker not in ['USA', 'CEO', 'CFO', 'IPO', 'ETF', 'SEC', 'FDA']:
                    tickers.append({
                        'ticker': ticker,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9
                    })
        
        return tickers
    
    def extract_financial_events(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial events from text"""
        events = []
        
        event_patterns = {
            'earnings_announcement': [
                r'earnings? (?:announcement|release|report)',
                r'quarterly (?:results|earnings)',
                r'q[1-4] (?:earnings|results)'
            ],
            'dividend_announcement': [
                r'dividend (?:announcement|payment|increase|cut)',
                r'(?:special|interim|final) dividend'
            ],
            'merger_acquisition': [
                r'merger? (?:with|and|agreement)',
                r'acquisition (?:of|by|deal)',
                r'takeover (?:bid|offer|attempt)'
            ],
            'ipo': [
                r'(?:initial public offering|ipo)',
                r'going public',
                r'public listing'
            ],
            'stock_split': [
                r'stock split',
                r'share split',
                r'(?:[0-9]+)-for-(?:[0-9]+) split'
            ]
        }
        
        for event_type, patterns in event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    events.append({
                        'event_type': event_type,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    })
        
        return events
    
    async def extract_comprehensive(self, text: str, language: str = 'en',
                                  company_registry: List[str] = None) -> Dict[str, Any]:
        """Comprehensive entity extraction combining all methods"""
        # Basic entity extraction
        entities = await self.extract_entities(text, language)
        
        # Extract additional financial information
        financial_figures = self.extract_financial_figures(text)
        ticker_symbols = self.extract_ticker_symbols(text)
        financial_events = self.extract_financial_events(text)
        
        # Link to known companies if registry provided
        linked_entities = {}
        if company_registry:
            linked_entities = self.link_entities_to_companies(entities, company_registry)
        
        return {
            'entities': entities,
            'financial_figures': financial_figures,
            'ticker_symbols': ticker_symbols,
            'financial_events': financial_events,
            'linked_entities': linked_entities,
            'extraction_metadata': {
                'language': language,
                'timestamp': datetime.now(),
                'total_entities': sum(len(entity_list) for entity_list in entities.values())
            }
        }
