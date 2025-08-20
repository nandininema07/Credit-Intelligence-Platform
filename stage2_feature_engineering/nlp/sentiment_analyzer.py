"""
Multi-language sentiment analysis for financial text.
Supports BERT-based models and financial domain-specific sentiment analysis.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from textblob import TextBlob
import torch
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Multi-language sentiment analyzer for financial text"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.pipelines = {}
        self.financial_keywords = self._load_financial_keywords()
        self.initialize_models()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("SentimentAnalyzer initialized successfully")
        return True
        
    def initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Financial BERT model
            if self.config.get('use_finbert', True):
                self.models['finbert'] = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
                
            # Multilingual BERT
            if self.config.get('use_multilingual', True):
                self.models['multilingual'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
                )
                
            # VADER for quick analysis
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.models['vader'] = SentimentIntensityAnalyzer()
            except ImportError:
                logger.warning("VADER not available, skipping")
                
            logger.info("Sentiment analysis models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
            # Fallback to TextBlob
            self.models['textblob'] = True
    
    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """Load financial sentiment keywords"""
        return {
            'positive': [
                'profit', 'growth', 'increase', 'gain', 'revenue', 'earnings',
                'bullish', 'upgrade', 'outperform', 'buy', 'strong', 'beat',
                'exceed', 'expansion', 'merger', 'acquisition', 'dividend'
            ],
            'negative': [
                'loss', 'decline', 'decrease', 'fall', 'drop', 'bearish',
                'downgrade', 'underperform', 'sell', 'weak', 'miss',
                'bankruptcy', 'debt', 'lawsuit', 'scandal', 'recession',
                'layoffs', 'restructuring', 'warning'
            ],
            'uncertainty': [
                'volatile', 'uncertain', 'risk', 'concern', 'caution',
                'investigation', 'pending', 'unclear', 'speculation'
            ]
        }
    
    async def analyze_text(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Analyze sentiment of a single text"""
        if not text or not text.strip():
            return {'sentiment': 'neutral', 'confidence': 0.0, 'scores': {}}
            
        results = {}
        
        # Clean text
        cleaned_text = self._preprocess_text(text)
        
        # Financial keyword analysis
        keyword_sentiment = self._analyze_keywords(cleaned_text)
        results['keyword_sentiment'] = keyword_sentiment
        
        # Model-based analysis
        if 'finbert' in self.models and language == 'en':
            finbert_result = await self._analyze_with_finbert(cleaned_text)
            results['finbert'] = finbert_result
            
        if 'multilingual' in self.models:
            multilingual_result = await self._analyze_with_multilingual(cleaned_text)
            results['multilingual'] = multilingual_result
            
        if 'vader' in self.models and language == 'en':
            vader_result = self._analyze_with_vader(cleaned_text)
            results['vader'] = vader_result
            
        # Fallback to TextBlob
        if not results or 'textblob' in self.models:
            textblob_result = self._analyze_with_textblob(cleaned_text)
            results['textblob'] = textblob_result
        
        # Combine results
        combined_result = self._combine_sentiment_results(results)
        
        return combined_result
    
    async def analyze_batch(self, texts: List[str], language: str = 'en') -> Dict[str, Any]:
        """Analyze sentiment for multiple texts"""
        if not texts:
            return {'mean_sentiment': 0.0, 'sentiment_std': 0.0, 'sentiments': []}
            
        # Process texts in batches to avoid memory issues
        batch_size = self.config.get('batch_size', 32)
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_tasks = [self.analyze_text(text, language) for text in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            all_results.extend(batch_results)
        
        # Aggregate results
        sentiments = [result.get('sentiment_score', 0.0) for result in all_results]
        sentiment_labels = [result.get('sentiment', 'neutral') for result in all_results]
        
        # Calculate statistics
        mean_sentiment = np.mean(sentiments) if sentiments else 0.0
        sentiment_std = np.std(sentiments) if len(sentiments) > 1 else 0.0
        
        # Calculate ratios
        total_count = len(sentiment_labels)
        positive_count = sum(1 for s in sentiment_labels if s == 'positive')
        negative_count = sum(1 for s in sentiment_labels if s == 'negative')
        neutral_count = total_count - positive_count - negative_count
        
        return {
            'mean_sentiment': float(mean_sentiment),
            'sentiment_std': float(sentiment_std),
            'positive_ratio': positive_count / total_count if total_count > 0 else 0.0,
            'negative_ratio': negative_count / total_count if total_count > 0 else 0.0,
            'neutral_ratio': neutral_count / total_count if total_count > 0 else 0.0,
            'sentiments': all_results,
            'total_texts': total_count
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Limit length for model processing
        max_length = self.config.get('max_text_length', 512)
        if len(text) > max_length:
            text = text[:max_length]
            
        return text.strip()
    
    def _analyze_keywords(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment based on financial keywords"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.financial_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.financial_keywords['negative'] if word in text_lower)
        uncertainty_count = sum(1 for word in self.financial_keywords['uncertainty'] if word in text_lower)
        
        total_keywords = positive_count + negative_count + uncertainty_count
        
        if total_keywords == 0:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        # Calculate weighted sentiment
        sentiment_score = (positive_count - negative_count - 0.5 * uncertainty_count) / total_keywords
        
        if sentiment_score > 0.1:
            sentiment = 'positive'
        elif sentiment_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        confidence = min(total_keywords / 10.0, 1.0)  # Max confidence at 10+ keywords
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': confidence,
            'keyword_counts': {
                'positive': positive_count,
                'negative': negative_count,
                'uncertainty': uncertainty_count
            }
        }
    
    async def _analyze_with_finbert(self, text: str) -> Dict[str, Any]:
        """Analyze with FinBERT model"""
        try:
            result = self.models['finbert'](text)[0]
            
            # Map FinBERT labels to standard format
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral'
            }
            
            sentiment = label_mapping.get(result['label'].lower(), 'neutral')
            confidence = result['score']
            
            # Convert to sentiment score (-1 to 1)
            if sentiment == 'positive':
                sentiment_score = confidence
            elif sentiment == 'negative':
                sentiment_score = -confidence
            else:
                sentiment_score = 0.0
                
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': confidence,
                'model': 'finbert'
            }
            
        except Exception as e:
            logger.error(f"Error with FinBERT analysis: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    async def _analyze_with_multilingual(self, text: str) -> Dict[str, Any]:
        """Analyze with multilingual model"""
        try:
            result = self.models['multilingual'](text)[0]
            
            # Map labels
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            sentiment = label_mapping.get(result['label'], 'neutral')
            confidence = result['score']
            
            # Convert to sentiment score
            if sentiment == 'positive':
                sentiment_score = confidence
            elif sentiment == 'negative':
                sentiment_score = -confidence
            else:
                sentiment_score = 0.0
                
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': confidence,
                'model': 'multilingual'
            }
            
        except Exception as e:
            logger.error(f"Error with multilingual analysis: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def _analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Analyze with VADER sentiment analyzer"""
        try:
            scores = self.models['vader'].polarity_scores(text)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                sentiment = 'positive'
            elif compound_score <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            confidence = abs(compound_score)
            
            return {
                'sentiment': sentiment,
                'score': compound_score,
                'confidence': confidence,
                'model': 'vader',
                'detailed_scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error with VADER analysis: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def _analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze with TextBlob (fallback)"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            confidence = abs(polarity)
            
            return {
                'sentiment': sentiment,
                'score': polarity,
                'confidence': confidence,
                'model': 'textblob'
            }
            
        except Exception as e:
            logger.error(f"Error with TextBlob analysis: {e}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
    
    def _combine_sentiment_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple sentiment analysis results"""
        if not results:
            return {'sentiment': 'neutral', 'sentiment_score': 0.0, 'confidence': 0.0}
        
        # Weight different models
        model_weights = {
            'finbert': 0.4,
            'multilingual': 0.3,
            'keyword_sentiment': 0.2,
            'vader': 0.1,
            'textblob': 0.05
        }
        
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for model_name, result in results.items():
            if model_name in model_weights and 'score' in result:
                weight = model_weights[model_name]
                weighted_score += result['score'] * weight
                weighted_confidence += result.get('confidence', 0.0) * weight
                total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        else:
            final_score = 0.0
            final_confidence = 0.0
        
        # Determine final sentiment
        if final_score > 0.1:
            final_sentiment = 'positive'
        elif final_score < -0.1:
            final_sentiment = 'negative'
        else:
            final_sentiment = 'neutral'
        
        return {
            'sentiment': final_sentiment,
            'sentiment_score': final_score,
            'confidence': final_confidence,
            'model_results': results,
            'combined': True
        }
