"""
Entity linking for company and financial entity recognition.
Links mentions in text to specific companies and financial entities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import re
import spacy
from fuzzywuzzy import fuzz, process
import pandas as pd
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class EntityLinker:
    """Entity linking for financial text analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = None
        self.company_database = {}
        self.financial_entities = self._load_financial_entities()
        self.initialize_nlp()
        
    def initialize_nlp(self):
        """Initialize NLP model for entity recognition"""
        try:
            # Load spaCy model
            model_name = self.config.get('spacy_model', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            
            # Add custom patterns for financial entities
            self._add_financial_patterns()
            
            logger.info("Entity linking NLP model initialized")
            
        except OSError:
            logger.warning(f"spaCy model not found, using basic entity recognition")
            self.nlp = None
    
    def _load_financial_entities(self) -> Dict[str, List[str]]:
        """Load financial entity types and patterns"""
        return {
            'stock_symbols': [
                r'\b[A-Z]{1,5}\b',  # Stock ticker patterns
                r'\$[A-Z]{1,5}\b'   # Stock symbols with $
            ],
            'currencies': [
                'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'CAD', 'AUD', 'CHF'
            ],
            'financial_instruments': [
                'bond', 'stock', 'option', 'future', 'derivative', 'etf',
                'mutual fund', 'reit', 'commodity', 'forex'
            ],
            'financial_metrics': [
                'revenue', 'profit', 'earnings', 'ebitda', 'eps', 'pe ratio',
                'market cap', 'book value', 'debt', 'cash flow', 'roe', 'roa'
            ],
            'market_indices': [
                'S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000', 'FTSE 100',
                'Nikkei', 'DAX', 'CAC 40', 'Hang Seng'
            ]
        }
    
    def _add_financial_patterns(self):
        """Add custom financial entity patterns to spaCy"""
        if not self.nlp:
            return
            
        # Add ruler for custom patterns
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        
        patterns = []
        
        # Stock symbol patterns
        patterns.extend([
            {"label": "STOCK_SYMBOL", "pattern": [{"TEXT": {"REGEX": r"^\$[A-Z]{1,5}$"}}]},
            {"label": "STOCK_SYMBOL", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}$"}, "POS": "PROPN"}]}
        ])
        
        # Currency patterns
        for currency in self.financial_entities['currencies']:
            patterns.append({"label": "CURRENCY", "pattern": currency})
        
        # Financial instrument patterns
        for instrument in self.financial_entities['financial_instruments']:
            patterns.append({"label": "FINANCIAL_INSTRUMENT", "pattern": instrument})
        
        ruler.add_patterns(patterns)
    
    async def link_entities(self, texts: List[str], target_company: str = None) -> Dict[str, Any]:
        """Link entities in texts to known companies and financial entities"""
        if not texts:
            return {'entities': [], 'company_mentions': [], 'financial_entities': []}
        
        all_entities = []
        company_mentions = []
        financial_entities = []
        
        for text in texts:
            # Extract entities from text
            text_entities = await self._extract_entities(text)
            
            # Link to companies
            company_links = await self._link_companies(text_entities, target_company)
            company_mentions.extend(company_links)
            
            # Extract financial entities
            fin_entities = self._extract_financial_entities(text)
            financial_entities.extend(fin_entities)
            
            all_entities.extend(text_entities)
        
        # Aggregate results
        entity_stats = self._aggregate_entity_stats(all_entities, company_mentions, financial_entities)
        
        return entity_stats
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        entities = []
        
        if self.nlp:
            # Use spaCy for entity extraction
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0  # spaCy doesn't provide confidence scores
                })
        else:
            # Fallback to regex-based extraction
            entities = self._regex_entity_extraction(text)
        
        return entities
    
    def _regex_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback regex-based entity extraction"""
        entities = []
        
        # Extract potential company names (capitalized words)
        company_pattern = r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|Ltd|LLC|Company|Co)\b'
        for match in re.finditer(company_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'ORG',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8
            })
        
        # Extract stock symbols
        stock_pattern = r'\$[A-Z]{1,5}\b|\b[A-Z]{2,5}\b(?=\s|$|[.,;:])'
        for match in re.finditer(stock_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'STOCK_SYMBOL',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # Extract monetary amounts
        money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|T))?'
        for match in re.finditer(money_pattern, text):
            entities.append({
                'text': match.group(),
                'label': 'MONEY',
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.95
            })
        
        return entities
    
    async def _link_companies(self, entities: List[Dict[str, Any]], target_company: str = None) -> List[Dict[str, Any]]:
        """Link extracted entities to known companies"""
        company_links = []
        
        # Filter for organization entities
        org_entities = [ent for ent in entities if ent['label'] in ['ORG', 'PERSON', 'STOCK_SYMBOL']]
        
        for entity in org_entities:
            entity_text = entity['text']
            
            # Check if it's the target company
            if target_company:
                similarity = fuzz.ratio(entity_text.lower(), target_company.lower())
                if similarity > 80:
                    company_links.append({
                        'entity': entity_text,
                        'linked_company': target_company,
                        'confidence': similarity / 100.0,
                        'link_type': 'target_match',
                        'position': entity['start']
                    })
                    continue
            
            # Try to match against known companies
            linked_company = await self._match_company_database(entity_text)
            if linked_company:
                company_links.append({
                    'entity': entity_text,
                    'linked_company': linked_company['name'],
                    'confidence': linked_company['confidence'],
                    'link_type': 'database_match',
                    'position': entity['start'],
                    'metadata': linked_company.get('metadata', {})
                })
        
        return company_links
    
    async def _match_company_database(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """Match entity against company database"""
        # This would typically query a real company database
        # For now, we'll use a simple hardcoded approach
        
        known_companies = {
            'Apple': ['Apple Inc', 'AAPL', 'Apple Computer'],
            'Microsoft': ['Microsoft Corp', 'MSFT', 'Microsoft Corporation'],
            'Google': ['Alphabet Inc', 'GOOGL', 'Google LLC', 'Alphabet'],
            'Amazon': ['Amazon.com Inc', 'AMZN', 'Amazon Web Services'],
            'Tesla': ['Tesla Inc', 'TSLA', 'Tesla Motors'],
            'Meta': ['Meta Platforms', 'META', 'Facebook Inc', 'Facebook'],
            'Netflix': ['Netflix Inc', 'NFLX'],
            'Nvidia': ['NVIDIA Corp', 'NVDA', 'Nvidia Corporation']
        }
        
        best_match = None
        best_score = 0
        
        for company, aliases in known_companies.items():
            for alias in aliases:
                score = fuzz.ratio(entity_text.lower(), alias.lower())
                if score > best_score and score > 70:
                    best_score = score
                    best_match = {
                        'name': company,
                        'confidence': score / 100.0,
                        'matched_alias': alias,
                        'metadata': {'ticker': aliases[1] if len(aliases) > 1 else None}
                    }
        
        return best_match
    
    def _extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial-specific entities"""
        financial_entities = []
        
        # Extract financial metrics mentions
        for metric in self.financial_entities['financial_metrics']:
            pattern = rf'\b{re.escape(metric)}\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                financial_entities.append({
                    'text': match.group(),
                    'type': 'financial_metric',
                    'category': 'metric',
                    'position': match.start(),
                    'confidence': 0.9
                })
        
        # Extract market indices
        for index in self.financial_entities['market_indices']:
            if index.lower() in text.lower():
                financial_entities.append({
                    'text': index,
                    'type': 'market_index',
                    'category': 'index',
                    'confidence': 0.95
                })
        
        # Extract percentage values
        percentage_pattern = r'\b\d+(?:\.\d+)?%'
        for match in re.finditer(percentage_pattern, text):
            financial_entities.append({
                'text': match.group(),
                'type': 'percentage',
                'category': 'metric',
                'position': match.start(),
                'confidence': 0.98
            })
        
        return financial_entities
    
    def _aggregate_entity_stats(self, all_entities: List[Dict[str, Any]], 
                              company_mentions: List[Dict[str, Any]], 
                              financial_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate entity extraction statistics"""
        
        # Count entity types
        entity_counts = {}
        for entity in all_entities:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        # Count company mentions
        company_counts = {}
        total_company_confidence = 0
        for mention in company_mentions:
            company = mention['linked_company']
            company_counts[company] = company_counts.get(company, 0) + 1
            total_company_confidence += mention['confidence']
        
        avg_company_confidence = (total_company_confidence / len(company_mentions) 
                                if company_mentions else 0.0)
        
        # Count financial entity types
        financial_counts = {}
        for entity in financial_entities:
            entity_type = entity['type']
            financial_counts[entity_type] = financial_counts.get(entity_type, 0) + 1
        
        return {
            'total_entities': len(all_entities),
            'entity_types': entity_counts,
            'company_mentions': company_mentions,
            'mention_count': len(company_mentions),
            'unique_companies': len(company_counts),
            'company_counts': company_counts,
            'avg_confidence': avg_company_confidence,
            'financial_entities': financial_entities,
            'financial_entity_counts': financial_counts,
            'financial_entity_total': len(financial_entities)
        }
    
    def update_company_database(self, companies: List[Dict[str, Any]]):
        """Update the company database with new companies"""
        for company in companies:
            name = company.get('name')
            if name:
                self.company_database[name.lower()] = company
        
        logger.info(f"Updated company database with {len(companies)} companies")
    
    def get_entity_context(self, text: str, entity: Dict[str, Any], context_window: int = 50) -> str:
        """Get surrounding context for an entity"""
        start = max(0, entity['start'] - context_window)
        end = min(len(text), entity['end'] + context_window)
        
        context = text[start:end]
        
        # Highlight the entity
        entity_start = entity['start'] - start
        entity_end = entity['end'] - start
        
        highlighted = (context[:entity_start] + 
                      f"**{context[entity_start:entity_end]}**" + 
                      context[entity_end:])
        
        return highlighted.strip()
