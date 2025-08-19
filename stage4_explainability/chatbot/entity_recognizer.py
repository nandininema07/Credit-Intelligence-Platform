"""
Entity recognizer for Stage 4 explainability chatbot.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re
import joblib
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entity data structure"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    metadata: Dict[str, Any] = None

class EntityRecognizer:
    """Named Entity Recognition for credit explanation chatbot"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entity_patterns = {}
        self.entity_types = [
            'CREDIT_SCORE',
            'LOAN_AMOUNT',
            'INCOME',
            'DEBT_AMOUNT',
            'PAYMENT_HISTORY',
            'CREDIT_UTILIZATION',
            'ACCOUNT_AGE',
            'CREDIT_INQUIRIES',
            'ACCOUNT_TYPE',
            'BANK_NAME',
            'DATE',
            'PERCENTAGE',
            'CURRENCY',
            'PERSON_NAME',
            'DOCUMENT_TYPE'
        ]
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize entity recognition patterns"""
        
        self.entity_patterns = {
            'CREDIT_SCORE': [
                (r'\b([3-8]\d{2})\b', 'credit_score_number'),
                (r'score.*?(\d{3})', 'credit_score_mention'),
                (r'fico.*?(\d{3})', 'fico_score'),
                (r'vantage.*?(\d{3})', 'vantage_score')
            ],
            'LOAN_AMOUNT': [
                (r'\$?([\d,]+(?:\.\d{2})?)\s*(?:loan|mortgage|credit)', 'loan_amount'),
                (r'(?:loan|mortgage|credit).*?\$?([\d,]+(?:\.\d{2})?)', 'loan_amount_after'),
                (r'borrow.*?\$?([\d,]+(?:\.\d{2})?)', 'borrow_amount')
            ],
            'INCOME': [
                (r'income.*?\$?([\d,]+(?:\.\d{2})?)', 'income_amount'),
                (r'salary.*?\$?([\d,]+(?:\.\d{2})?)', 'salary_amount'),
                (r'earn.*?\$?([\d,]+(?:\.\d{2})?)', 'earnings_amount'),
                (r'\$?([\d,]+(?:\.\d{2})?)\s*(?:income|salary|earnings)', 'income_before')
            ],
            'DEBT_AMOUNT': [
                (r'debt.*?\$?([\d,]+(?:\.\d{2})?)', 'debt_amount'),
                (r'owe.*?\$?([\d,]+(?:\.\d{2})?)', 'owed_amount'),
                (r'balance.*?\$?([\d,]+(?:\.\d{2})?)', 'balance_amount'),
                (r'\$?([\d,]+(?:\.\d{2})?)\s*(?:debt|owed|balance)', 'debt_before')
            ],
            'PAYMENT_HISTORY': [
                (r'(\d+)\s*(?:late|missed)\s*payment', 'late_payments'),
                (r'payment.*?(\d+)\s*(?:days|months)', 'payment_period'),
                (r'(\d+)%\s*on.time', 'on_time_percentage'),
                (r'never.*late', 'perfect_payment')
            ],
            'CREDIT_UTILIZATION': [
                (r'utilization.*?(\d+(?:\.\d+)?)%', 'utilization_percentage'),
                (r'(\d+(?:\.\d+)?)%.*?utilization', 'utilization_percentage_before'),
                (r'using.*?(\d+(?:\.\d+)?)%', 'usage_percentage')
            ],
            'ACCOUNT_AGE': [
                (r'(\d+)\s*(?:years?|yrs?)\s*(?:old|age)', 'account_years'),
                (r'(\d+)\s*(?:months?|mos?)\s*(?:old|age)', 'account_months'),
                (r'opened.*?(\d+)\s*(?:years?|months?)', 'opened_time'),
                (r'(\d{4})', 'year_opened')
            ],
            'CREDIT_INQUIRIES': [
                (r'(\d+)\s*(?:inquiries|inquiry)', 'inquiry_count'),
                (r'hard.*?(\d+)', 'hard_inquiries'),
                (r'soft.*?(\d+)', 'soft_inquiries')
            ],
            'ACCOUNT_TYPE': [
                (r'\b(credit card|mortgage|auto loan|personal loan|student loan|line of credit)\b', 'account_type'),
                (r'\b(checking|savings|investment)\s*account\b', 'bank_account_type'),
                (r'\b(visa|mastercard|amex|discover)\b', 'card_brand')
            ],
            'BANK_NAME': [
                (r'\b(chase|wells fargo|bank of america|citibank|capital one|discover|amex|american express)\b', 'major_bank'),
                (r'\b([A-Z][a-z]+\s+(?:bank|credit union|financial))\b', 'financial_institution')
            ],
            'DATE': [
                (r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', 'date_slash'),
                (r'\b(\d{1,2}-\d{1,2}-\d{2,4})\b', 'date_dash'),
                (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b', 'date_written'),
                (r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{4})\b', 'date_abbreviated')
            ],
            'PERCENTAGE': [
                (r'\b(\d+(?:\.\d+)?)%\b', 'percentage_value'),
                (r'\b(\d+(?:\.\d+)?)\s*percent\b', 'percent_written')
            ],
            'CURRENCY': [
                (r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', 'dollar_amount'),
                (r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*dollars?\b', 'dollar_written')
            ],
            'PERSON_NAME': [
                (r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b', 'full_name'),
                (r'\bMr\.?\s+([A-Z][a-z]+)\b', 'mr_name'),
                (r'\bMs\.?\s+([A-Z][a-z]+)\b', 'ms_name')
            ],
            'DOCUMENT_TYPE': [
                (r'\b(credit report|credit score|loan application|bank statement|pay stub|tax return|w2|1099)\b', 'financial_document'),
                (r'\b(driver.?s license|passport|ssn|social security)\b', 'identification_document')
            ]
        }
        
        # Compile patterns
        for entity_type, patterns in self.entity_patterns.items():
            compiled_patterns = []
            for pattern, label in patterns:
                compiled_patterns.append((re.compile(pattern, re.IGNORECASE), label))
            self.entity_patterns[entity_type] = compiled_patterns
    
    async def extract_entities(self, text: str) -> Dict[str, List[Entity]]:
        """Extract entities from text"""
        
        try:
            entities = {}
            
            for entity_type, patterns in self.entity_patterns.items():
                found_entities = []
                
                for pattern, label in patterns:
                    matches = pattern.finditer(text)
                    
                    for match in matches:
                        entity = Entity(
                            text=match.group(1) if match.groups() else match.group(0),
                            label=label,
                            start=match.start(),
                            end=match.end(),
                            confidence=self._calculate_confidence(entity_type, match.group(0)),
                            metadata={
                                'entity_type': entity_type,
                                'full_match': match.group(0),
                                'pattern_used': pattern.pattern
                            }
                        )
                        
                        # Validate entity
                        if self._validate_entity(entity):
                            found_entities.append(entity)
                
                if found_entities:
                    # Remove duplicates and sort by confidence
                    found_entities = self._deduplicate_entities(found_entities)
                    found_entities.sort(key=lambda x: x.confidence, reverse=True)
                    entities[entity_type] = found_entities
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}
    
    def _calculate_confidence(self, entity_type: str, match_text: str) -> float:
        """Calculate confidence score for entity"""
        
        try:
            base_confidence = 0.8
            
            # Adjust confidence based on entity type and match quality
            if entity_type == 'CREDIT_SCORE':
                score = int(re.search(r'\d{3}', match_text).group())
                if 300 <= score <= 850:
                    return 0.95
                else:
                    return 0.3
            
            elif entity_type == 'PERCENTAGE':
                percentage = float(re.search(r'\d+(?:\.\d+)?', match_text).group())
                if 0 <= percentage <= 100:
                    return 0.9
                else:
                    return 0.4
            
            elif entity_type == 'CURRENCY':
                # Higher confidence for reasonable amounts
                amount_str = re.search(r'[\d,]+(?:\.\d{2})?', match_text).group()
                amount = float(amount_str.replace(',', ''))
                if 1 <= amount <= 10000000:  # $1 to $10M seems reasonable
                    return 0.9
                else:
                    return 0.6
            
            elif entity_type == 'DATE':
                # Simple date validation
                if re.search(r'\d{4}', match_text):  # Has year
                    return 0.85
                else:
                    return 0.7
            
            return base_confidence
            
        except Exception:
            return 0.5
    
    def _validate_entity(self, entity: Entity) -> bool:
        """Validate extracted entity"""
        
        try:
            entity_type = entity.metadata['entity_type']
            
            if entity_type == 'CREDIT_SCORE':
                score = int(re.search(r'\d{3}', entity.text).group())
                return 300 <= score <= 850
            
            elif entity_type == 'PERCENTAGE':
                percentage = float(re.search(r'\d+(?:\.\d+)?', entity.text).group())
                return 0 <= percentage <= 100
            
            elif entity_type == 'CURRENCY':
                amount_str = re.search(r'[\d,]+(?:\.\d{2})?', entity.text).group()
                amount = float(amount_str.replace(',', ''))
                return amount > 0
            
            elif entity_type == 'ACCOUNT_AGE':
                age = int(re.search(r'\d+', entity.text).group())
                return 0 <= age <= 100  # Reasonable age range
            
            return True  # Default to valid
            
        except Exception:
            return False
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        
        unique_entities = []
        seen_texts = set()
        
        for entity in entities:
            # Create a key based on text and position
            key = f"{entity.text}_{entity.start}_{entity.end}"
            
            if key not in seen_texts:
                seen_texts.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def extract_credit_specific_entities(self, text: str) -> Dict[str, Any]:
        """Extract credit-specific entities with additional processing"""
        
        try:
            entities = await self.extract_entities(text)
            
            # Process credit-specific information
            credit_info = {
                'credit_scores': [],
                'financial_amounts': {},
                'time_references': [],
                'account_information': [],
                'payment_behavior': []
            }
            
            # Process credit scores
            if 'CREDIT_SCORE' in entities:
                for entity in entities['CREDIT_SCORE']:
                    score_value = int(re.search(r'\d{3}', entity.text).group())
                    credit_info['credit_scores'].append({
                        'value': score_value,
                        'range': self._get_credit_score_range(score_value),
                        'confidence': entity.confidence
                    })
            
            # Process financial amounts
            for amount_type in ['LOAN_AMOUNT', 'INCOME', 'DEBT_AMOUNT']:
                if amount_type in entities:
                    amounts = []
                    for entity in entities[amount_type]:
                        amount_str = re.search(r'[\d,]+(?:\.\d{2})?', entity.text).group()
                        amount_value = float(amount_str.replace(',', ''))
                        amounts.append({
                            'value': amount_value,
                            'formatted': f"${amount_value:,.2f}",
                            'confidence': entity.confidence
                        })
                    credit_info['financial_amounts'][amount_type.lower()] = amounts
            
            # Process time references
            for time_type in ['ACCOUNT_AGE', 'DATE']:
                if time_type in entities:
                    for entity in entities[time_type]:
                        credit_info['time_references'].append({
                            'type': time_type.lower(),
                            'value': entity.text,
                            'confidence': entity.confidence
                        })
            
            # Process account information
            for account_type in ['ACCOUNT_TYPE', 'BANK_NAME']:
                if account_type in entities:
                    for entity in entities[account_type]:
                        credit_info['account_information'].append({
                            'type': account_type.lower(),
                            'value': entity.text,
                            'confidence': entity.confidence
                        })
            
            # Process payment behavior
            if 'PAYMENT_HISTORY' in entities:
                for entity in entities['PAYMENT_HISTORY']:
                    credit_info['payment_behavior'].append({
                        'description': entity.text,
                        'label': entity.label,
                        'confidence': entity.confidence
                    })
            
            return credit_info
            
        except Exception as e:
            logger.error(f"Error extracting credit-specific entities: {e}")
            return {}
    
    def _get_credit_score_range(self, score: int) -> str:
        """Get credit score range category"""
        
        if score >= 800:
            return "Excellent"
        elif score >= 740:
            return "Very Good"
        elif score >= 670:
            return "Good"
        elif score >= 580:
            return "Fair"
        else:
            return "Poor"
    
    async def get_entity_context(self, entities: Dict[str, List[Entity]], 
                               text: str) -> Dict[str, Any]:
        """Get contextual information about extracted entities"""
        
        try:
            context = {
                'entity_count': sum(len(entity_list) for entity_list in entities.values()),
                'entity_types': list(entities.keys()),
                'high_confidence_entities': [],
                'relationships': [],
                'text_coverage': 0.0
            }
            
            # Find high confidence entities
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    if entity.confidence > 0.8:
                        context['high_confidence_entities'].append({
                            'type': entity_type,
                            'text': entity.text,
                            'confidence': entity.confidence
                        })
            
            # Calculate text coverage
            total_chars = len(text)
            covered_chars = 0
            
            for entity_list in entities.values():
                for entity in entity_list:
                    covered_chars += entity.end - entity.start
            
            context['text_coverage'] = covered_chars / total_chars if total_chars > 0 else 0.0
            
            # Find entity relationships
            context['relationships'] = self._find_entity_relationships(entities, text)
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting entity context: {e}")
            return {}
    
    def _find_entity_relationships(self, entities: Dict[str, List[Entity]], 
                                 text: str) -> List[Dict[str, Any]]:
        """Find relationships between entities"""
        
        relationships = []
        
        try:
            # Look for credit score and utilization relationships
            credit_scores = entities.get('CREDIT_SCORE', [])
            utilization = entities.get('CREDIT_UTILIZATION', [])
            
            if credit_scores and utilization:
                relationships.append({
                    'type': 'credit_score_utilization',
                    'entities': [credit_scores[0].text, utilization[0].text],
                    'description': 'Credit score mentioned with utilization rate'
                })
            
            # Look for loan amount and income relationships
            loan_amounts = entities.get('LOAN_AMOUNT', [])
            income = entities.get('INCOME', [])
            
            if loan_amounts and income:
                relationships.append({
                    'type': 'loan_income_ratio',
                    'entities': [loan_amounts[0].text, income[0].text],
                    'description': 'Loan amount mentioned with income'
                })
            
            # Look for debt and income relationships
            debt_amounts = entities.get('DEBT_AMOUNT', [])
            
            if debt_amounts and income:
                relationships.append({
                    'type': 'debt_income_ratio',
                    'entities': [debt_amounts[0].text, income[0].text],
                    'description': 'Debt amount mentioned with income'
                })
            
        except Exception as e:
            logger.error(f"Error finding entity relationships: {e}")
        
        return relationships
    
    def add_custom_pattern(self, entity_type: str, pattern: str, label: str):
        """Add custom entity pattern"""
        
        try:
            if entity_type not in self.entity_patterns:
                self.entity_patterns[entity_type] = []
            
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self.entity_patterns[entity_type].append((compiled_pattern, label))
            
            logger.info(f"Added custom pattern for {entity_type}: {pattern}")
            
        except Exception as e:
            logger.error(f"Error adding custom pattern: {e}")
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get entity recognition statistics"""
        
        pattern_counts = {}
        for entity_type, patterns in self.entity_patterns.items():
            pattern_counts[entity_type] = len(patterns)
        
        return {
            'supported_entity_types': self.entity_types,
            'pattern_counts': pattern_counts,
            'total_patterns': sum(pattern_counts.values()),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_recognizer(self, filepath: str):
        """Save entity recognizer to file"""
        
        recognizer_data = {
            'entity_patterns': self.entity_patterns,
            'entity_types': self.entity_types,
            'config': self.config
        }
        
        joblib.dump(recognizer_data, filepath)
        logger.info(f"Entity recognizer saved to {filepath}")
    
    def load_recognizer(self, filepath: str):
        """Load entity recognizer from file"""
        
        recognizer_data = joblib.load(filepath)
        
        self.entity_patterns = recognizer_data.get('entity_patterns', {})
        self.entity_types = recognizer_data.get('entity_types', [])
        self.config = recognizer_data.get('config', {})
        
        logger.info(f"Entity recognizer loaded from {filepath}")
