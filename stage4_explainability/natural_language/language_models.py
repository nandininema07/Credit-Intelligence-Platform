"""
Language model interface for Stage 4 explainability.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
import asyncio
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Language model types"""
    RULE_BASED = "rule_based"
    STATISTICAL = "statistical"
    NEURAL = "neural"
    HYBRID = "hybrid"

class LanguageTask(Enum):
    """Language generation tasks"""
    EXPLANATION = "explanation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    TEXT_CLASSIFICATION = "text_classification"
    ENTITY_EXTRACTION = "entity_extraction"

class LanguageModelInterface(ABC):
    """Abstract interface for language models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_type = ModelType.RULE_BASED
        self.supported_tasks = []
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the language model"""
        pass
    
    @abstractmethod
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None,
                          max_length: int = 200) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text into categories"""
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        pass
    
    @abstractmethod
    async def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize text"""
        pass

class RuleBasedLanguageModel(LanguageModelInterface):
    """Rule-based language model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = ModelType.RULE_BASED
        self.supported_tasks = [
            LanguageTask.EXPLANATION,
            LanguageTask.SUMMARIZATION,
            LanguageTask.TEXT_CLASSIFICATION
        ]
        self.rules = {}
        self.patterns = {}
        self.vocabulary = {}
        
    async def initialize(self) -> bool:
        """Initialize rule-based model"""
        
        try:
            # Initialize rules and patterns
            await self._load_rules()
            await self._load_patterns()
            await self._load_vocabulary()
            
            self.is_initialized = True
            logger.info("Rule-based language model initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing rule-based model: {e}")
            return False
    
    async def _load_rules(self):
        """Load generation rules"""
        
        self.rules = {
            'explanation_rules': {
                'high_importance': [
                    "This factor has a significant impact on your credit score",
                    "This is one of the most important factors in your profile",
                    "This factor strongly influences your credit decision"
                ],
                'medium_importance': [
                    "This factor has a moderate impact on your credit score",
                    "This factor plays an important role in your profile",
                    "This factor contributes meaningfully to your credit assessment"
                ],
                'low_importance': [
                    "This factor has a minor impact on your credit score",
                    "This factor has limited influence on your profile",
                    "This factor contributes slightly to your credit assessment"
                ]
            },
            'improvement_rules': {
                'payment_history': [
                    "Make all payments on time to improve this factor",
                    "Set up automatic payments to ensure timely payments",
                    "Focus on consistent payment behavior"
                ],
                'credit_utilization': [
                    "Reduce your credit card balances to improve utilization",
                    "Keep credit utilization below 30% of available credit",
                    "Pay down existing balances before making new purchases"
                ],
                'credit_length': [
                    "Keep older accounts open to maintain credit history length",
                    "Avoid closing your oldest credit accounts",
                    "This factor improves naturally over time"
                ]
            },
            'risk_rules': {
                'high_risk': [
                    "This requires immediate attention",
                    "This poses a significant risk to your credit profile",
                    "Address this issue as soon as possible"
                ],
                'medium_risk': [
                    "This should be addressed in the near term",
                    "This could impact your credit profile if not addressed",
                    "Consider taking action on this factor"
                ],
                'low_risk': [
                    "This is not an immediate concern",
                    "Monitor this factor for any changes",
                    "This has minimal impact on your current profile"
                ]
            }
        }
    
    async def _load_patterns(self):
        """Load text patterns"""
        
        self.patterns = {
            'score_patterns': {
                'excellent': r'(excellent|outstanding|exceptional|superior)',
                'good': r'(good|solid|strong|positive)',
                'fair': r'(fair|average|moderate|acceptable)',
                'poor': r'(poor|weak|concerning|problematic)'
            },
            'trend_patterns': {
                'improving': r'(improv|increas|better|upward|positive)',
                'declining': r'(declin|decreas|worse|downward|negative)',
                'stable': r'(stable|steady|consistent|unchanged)'
            },
            'urgency_patterns': {
                'urgent': r'(urgent|immediate|critical|asap)',
                'soon': r'(soon|near.term|short.term|quickly)',
                'eventual': r'(eventual|long.term|over.time|gradual)'
            }
        }
    
    async def _load_vocabulary(self):
        """Load vocabulary mappings"""
        
        self.vocabulary = {
            'credit_terms': {
                'utilization': ['credit usage', 'balance ratio', 'credit card usage'],
                'payment_history': ['payment record', 'payment behavior', 'payment track record'],
                'credit_length': ['credit history length', 'account age', 'credit experience'],
                'credit_mix': ['credit variety', 'account types', 'credit diversity'],
                'inquiries': ['credit checks', 'credit applications', 'hard pulls']
            },
            'impact_terms': {
                'high': ['significant', 'major', 'substantial', 'considerable'],
                'medium': ['moderate', 'meaningful', 'notable', 'important'],
                'low': ['minor', 'slight', 'limited', 'small']
            },
            'direction_terms': {
                'positive': ['beneficial', 'helpful', 'favorable', 'advantageous'],
                'negative': ['harmful', 'detrimental', 'unfavorable', 'damaging'],
                'neutral': ['neutral', 'balanced', 'stable', 'unchanged']
            }
        }
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None,
                          max_length: int = 200) -> str:
        """Generate text using rules"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Parse context for relevant information
            importance_level = self._extract_importance_level(context)
            factor_type = self._extract_factor_type(context)
            risk_level = self._extract_risk_level(context)
            
            # Generate text based on rules
            generated_parts = []
            
            # Add importance-based text
            if importance_level and importance_level in self.rules['explanation_rules']:
                importance_text = np.random.choice(self.rules['explanation_rules'][importance_level])
                generated_parts.append(importance_text)
            
            # Add factor-specific advice
            if factor_type and factor_type in self.rules['improvement_rules']:
                advice_text = np.random.choice(self.rules['improvement_rules'][factor_type])
                generated_parts.append(advice_text)
            
            # Add risk-based guidance
            if risk_level and risk_level in self.rules['risk_rules']:
                risk_text = np.random.choice(self.rules['risk_rules'][risk_level])
                generated_parts.append(risk_text)
            
            # Combine and format
            if generated_parts:
                full_text = '. '.join(generated_parts) + '.'
            else:
                full_text = "Based on the analysis, this factor affects your credit profile."
            
            # Truncate if necessary
            if len(full_text) > max_length:
                full_text = full_text[:max_length-3] + '...'
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return "Unable to generate explanation text."
    
    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text using pattern matching"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            text_lower = text.lower()
            scores = {}
            
            for category in categories:
                score = 0.0
                
                # Check for category-specific patterns
                if category.lower() in self.patterns.get('score_patterns', {}):
                    pattern = self.patterns['score_patterns'][category.lower()]
                    import re
                    if re.search(pattern, text_lower):
                        score += 0.8
                
                # Check for trend patterns
                for trend, pattern in self.patterns.get('trend_patterns', {}).items():
                    import re
                    if re.search(pattern, text_lower):
                        if trend in category.lower():
                            score += 0.6
                
                # Check for urgency patterns
                for urgency, pattern in self.patterns.get('urgency_patterns', {}).items():
                    import re
                    if re.search(pattern, text_lower):
                        if urgency in category.lower():
                            score += 0.7
                
                # Basic keyword matching
                category_words = category.lower().split()
                for word in category_words:
                    if word in text_lower:
                        score += 0.3
                
                scores[category] = min(1.0, score)
            
            # Normalize scores
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {k: v/total_score for k, v in scores.items()}
            
            return scores
            
        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return {category: 0.0 for category in categories}
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using pattern matching"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            entities = []
            text_lower = text.lower()
            
            # Credit-specific entity patterns
            entity_patterns = {
                'CREDIT_SCORE': r'\b(\d{3})\b',
                'PERCENTAGE': r'\b(\d+(?:\.\d+)?%)\b',
                'CURRENCY': r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
                'CREDIT_FACTOR': r'\b(payment history|credit utilization|credit length|credit mix|inquiries)\b',
                'TIME_PERIOD': r'\b(\d+\s*(?:month|year|day)s?)\b'
            }
            
            import re
            for entity_type, pattern in entity_patterns.items():
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize text using extractive methods"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Simple extractive summarization
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return text[:max_length]
            
            # Score sentences based on key terms
            scored_sentences = []
            
            key_terms = [
                'credit', 'score', 'factor', 'important', 'impact', 'improve',
                'payment', 'utilization', 'history', 'risk', 'recommendation'
            ]
            
            for sentence in sentences:
                score = 0
                sentence_lower = sentence.lower()
                
                # Score based on key terms
                for term in key_terms:
                    if term in sentence_lower:
                        score += 1
                
                # Prefer shorter sentences for summaries
                if len(sentence) < 100:
                    score += 0.5
                
                # Prefer sentences with numbers/percentages
                import re
                if re.search(r'\d+', sentence):
                    score += 0.5
                
                scored_sentences.append((sentence, score))
            
            # Sort by score and select top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            summary_sentences = []
            current_length = 0
            
            for sentence, score in scored_sentences:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            if summary_sentences:
                return '. '.join(summary_sentences) + '.'
            else:
                return text[:max_length-3] + '...'
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length]
    
    def _extract_importance_level(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract importance level from context"""
        
        if not context:
            return None
        
        # Check for importance scores
        if 'importance' in context:
            importance = context['importance']
            if isinstance(importance, (int, float)):
                if importance > 0.6:
                    return 'high_importance'
                elif importance > 0.3:
                    return 'medium_importance'
                else:
                    return 'low_importance'
        
        # Check for explicit importance level
        if 'importance_level' in context:
            level = context['importance_level'].lower()
            if 'high' in level:
                return 'high_importance'
            elif 'medium' in level or 'moderate' in level:
                return 'medium_importance'
            elif 'low' in level:
                return 'low_importance'
        
        return None
    
    def _extract_factor_type(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract factor type from context"""
        
        if not context:
            return None
        
        # Check for factor name
        if 'factor' in context:
            factor = context['factor'].lower()
            
            if 'payment' in factor:
                return 'payment_history'
            elif 'utilization' in factor:
                return 'credit_utilization'
            elif 'length' in factor or 'age' in factor:
                return 'credit_length'
        
        # Check for feature name
        if 'feature' in context:
            feature = context['feature'].lower()
            
            if 'payment' in feature:
                return 'payment_history'
            elif 'utilization' in feature:
                return 'credit_utilization'
            elif 'length' in feature or 'age' in feature:
                return 'credit_length'
        
        return None
    
    def _extract_risk_level(self, context: Dict[str, Any]) -> Optional[str]:
        """Extract risk level from context"""
        
        if not context:
            return None
        
        # Check for risk score
        if 'risk' in context:
            risk = context['risk']
            if isinstance(risk, (int, float)):
                if risk > 0.7:
                    return 'high_risk'
                elif risk > 0.4:
                    return 'medium_risk'
                else:
                    return 'low_risk'
        
        # Check for explicit risk level
        if 'risk_level' in context:
            level = context['risk_level'].lower()
            if 'high' in level:
                return 'high_risk'
            elif 'medium' in level or 'moderate' in level:
                return 'medium_risk'
            elif 'low' in level:
                return 'low_risk'
        
        return None

class StatisticalLanguageModel(LanguageModelInterface):
    """Statistical language model implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = ModelType.STATISTICAL
        self.supported_tasks = [
            LanguageTask.TEXT_CLASSIFICATION,
            LanguageTask.SUMMARIZATION
        ]
        self.ngram_models = {}
        self.word_frequencies = {}
        
    async def initialize(self) -> bool:
        """Initialize statistical model"""
        
        try:
            # Initialize n-gram models and word frequencies
            await self._build_ngram_models()
            await self._build_word_frequencies()
            
            self.is_initialized = True
            logger.info("Statistical language model initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing statistical model: {e}")
            return False
    
    async def _build_ngram_models(self):
        """Build n-gram models from training data"""
        
        # Placeholder for n-gram model building
        # In a real implementation, this would train on credit explanation text
        self.ngram_models = {
            'unigram': {},
            'bigram': {},
            'trigram': {}
        }
    
    async def _build_word_frequencies(self):
        """Build word frequency models"""
        
        # Placeholder for word frequency calculation
        self.word_frequencies = {}
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None,
                          max_length: int = 200) -> str:
        """Generate text using statistical methods"""
        
        # Placeholder implementation
        return "Statistical text generation not fully implemented."
    
    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text using statistical methods"""
        
        # Placeholder implementation using basic frequency analysis
        scores = {}
        for category in categories:
            scores[category] = np.random.random()  # Placeholder
        
        return scores
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using statistical methods"""
        
        # Placeholder implementation
        return []
    
    async def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize text using statistical methods"""
        
        # Placeholder implementation
        return text[:max_length]

class HybridLanguageModel(LanguageModelInterface):
    """Hybrid language model combining multiple approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = ModelType.HYBRID
        self.supported_tasks = [
            LanguageTask.EXPLANATION,
            LanguageTask.SUMMARIZATION,
            LanguageTask.TEXT_CLASSIFICATION,
            LanguageTask.ENTITY_EXTRACTION
        ]
        self.rule_based_model = None
        self.statistical_model = None
        self.ensemble_weights = {}
        
    async def initialize(self) -> bool:
        """Initialize hybrid model"""
        
        try:
            # Initialize component models
            self.rule_based_model = RuleBasedLanguageModel(self.config)
            self.statistical_model = StatisticalLanguageModel(self.config)
            
            # Initialize component models
            rule_init = await self.rule_based_model.initialize()
            stat_init = await self.statistical_model.initialize()
            
            if not (rule_init and stat_init):
                logger.warning("Some component models failed to initialize")
            
            # Set ensemble weights
            self.ensemble_weights = {
                'rule_based': 0.7,
                'statistical': 0.3
            }
            
            self.is_initialized = True
            logger.info("Hybrid language model initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing hybrid model: {e}")
            return False
    
    async def generate_text(self, prompt: str, context: Dict[str, Any] = None,
                          max_length: int = 200) -> str:
        """Generate text using hybrid approach"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get outputs from component models
            rule_output = await self.rule_based_model.generate_text(prompt, context, max_length)
            
            # For now, primarily use rule-based output
            # In a full implementation, this would combine outputs intelligently
            return rule_output
            
        except Exception as e:
            logger.error(f"Error generating hybrid text: {e}")
            return "Error generating text with hybrid model."
    
    async def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """Classify text using hybrid approach"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get classifications from component models
            rule_scores = await self.rule_based_model.classify_text(text, categories)
            stat_scores = await self.statistical_model.classify_text(text, categories)
            
            # Combine scores using ensemble weights
            combined_scores = {}
            for category in categories:
                rule_weight = self.ensemble_weights['rule_based']
                stat_weight = self.ensemble_weights['statistical']
                
                combined_score = (
                    rule_scores.get(category, 0.0) * rule_weight +
                    stat_scores.get(category, 0.0) * stat_weight
                )
                combined_scores[category] = combined_score
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"Error classifying with hybrid model: {e}")
            return {category: 0.0 for category in categories}
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using hybrid approach"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get entities from rule-based model
            entities = await self.rule_based_model.extract_entities(text)
            
            # In a full implementation, this would combine with statistical methods
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities with hybrid model: {e}")
            return []
    
    async def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize text using hybrid approach"""
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Get summaries from component models
            rule_summary = await self.rule_based_model.summarize_text(text, max_length)
            
            # For now, use rule-based summary
            # In a full implementation, this would intelligently combine summaries
            return rule_summary
            
        except Exception as e:
            logger.error(f"Error summarizing with hybrid model: {e}")
            return text[:max_length]

class LanguageModelFactory:
    """Factory for creating language models"""
    
    @staticmethod
    def create_model(model_type: ModelType, config: Dict[str, Any]) -> LanguageModelInterface:
        """Create language model of specified type"""
        
        if model_type == ModelType.RULE_BASED:
            return RuleBasedLanguageModel(config)
        elif model_type == ModelType.STATISTICAL:
            return StatisticalLanguageModel(config)
        elif model_type == ModelType.HYBRID:
            return HybridLanguageModel(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[ModelType]:
        """Get list of available model types"""
        return list(ModelType)
    
    @staticmethod
    def get_model_capabilities(model_type: ModelType) -> List[LanguageTask]:
        """Get capabilities of model type"""
        
        if model_type == ModelType.RULE_BASED:
            return [LanguageTask.EXPLANATION, LanguageTask.SUMMARIZATION, LanguageTask.TEXT_CLASSIFICATION]
        elif model_type == ModelType.STATISTICAL:
            return [LanguageTask.TEXT_CLASSIFICATION, LanguageTask.SUMMARIZATION]
        elif model_type == ModelType.HYBRID:
            return list(LanguageTask)
        else:
            return []

async def test_language_models():
    """Test language model implementations"""
    
    try:
        config = {'test': True}
        
        # Test rule-based model
        rule_model = RuleBasedLanguageModel(config)
        await rule_model.initialize()
        
        test_context = {
            'importance': 0.8,
            'factor': 'payment_history',
            'risk': 0.3
        }
        
        generated_text = await rule_model.generate_text(
            "Explain this factor", 
            test_context, 
            max_length=150
        )
        
        print(f"Generated text: {generated_text}")
        
        # Test classification
        classification = await rule_model.classify_text(
            "This is an excellent credit score with strong payment history",
            ['excellent', 'good', 'fair', 'poor']
        )
        
        print(f"Classification: {classification}")
        
        # Test entity extraction
        entities = await rule_model.extract_entities(
            "Your credit score is 750 with 15% utilization and $5,000 balance"
        )
        
        print(f"Entities: {entities}")
        
        logger.info("Language model tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error testing language models: {e}")

if __name__ == "__main__":
    asyncio.run(test_language_models())
