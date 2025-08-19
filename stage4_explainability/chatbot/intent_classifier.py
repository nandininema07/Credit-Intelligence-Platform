"""
Intent classifier for Stage 4 explainability chatbot.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Intent classification for credit explanation chatbot"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.vectorizer = None
        self.intent_patterns = {}
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.intents = [
            'credit_score_inquiry',
            'loan_decision_explanation',
            'feature_importance',
            'what_if_scenario',
            'counterfactual_analysis',
            'improvement_suggestions',
            'general_explanation',
            'greeting',
            'goodbye',
            'help',
            'complaint',
            'clarification',
            'follow_up',
            'unknown'
        ]
        
    def initialize(self, training_data: List[Dict[str, str]] = None):
        """Initialize intent classifier"""
        
        if training_data:
            self._train_model(training_data)
        else:
            self._initialize_rule_based_classifier()
        
        logger.info("Intent classifier initialized")
    
    def _initialize_rule_based_classifier(self):
        """Initialize rule-based classifier with patterns"""
        
        self.intent_patterns = {
            'credit_score_inquiry': [
                r'credit score',
                r'my score',
                r'score.*\d+',
                r'what.*score',
                r'credit rating',
                r'creditworthiness'
            ],
            'loan_decision_explanation': [
                r'loan.*decision',
                r'why.*approved',
                r'why.*denied',
                r'why.*rejected',
                r'loan.*application',
                r'decision.*process',
                r'approval.*reason'
            ],
            'feature_importance': [
                r'important.*factor',
                r'key.*factor',
                r'feature.*importance',
                r'main.*driver',
                r'primary.*reason',
                r'biggest.*impact',
                r'most.*significant'
            ],
            'what_if_scenario': [
                r'what.*if',
                r'scenario',
                r'hypothetical',
                r'suppose.*i',
                r'if.*i.*change',
                r'what.*would.*happen',
                r'simulation'
            ],
            'counterfactual_analysis': [
                r'counterfactual',
                r'alternative.*outcome',
                r'different.*result',
                r'change.*to.*get',
                r'minimum.*change',
                r'how.*to.*achieve'
            ],
            'improvement_suggestions': [
                r'how.*improve',
                r'increase.*score',
                r'better.*credit',
                r'recommendation',
                r'suggestion',
                r'advice',
                r'tips.*improve'
            ],
            'general_explanation': [
                r'explain',
                r'how.*work',
                r'understand',
                r'clarify',
                r'tell.*me.*about',
                r'what.*mean',
                r'definition'
            ],
            'greeting': [
                r'hello',
                r'hi',
                r'hey',
                r'good.*morning',
                r'good.*afternoon',
                r'good.*evening',
                r'greetings'
            ],
            'goodbye': [
                r'bye',
                r'goodbye',
                r'see.*you',
                r'farewell',
                r'thanks.*bye',
                r'that.*all'
            ],
            'help': [
                r'help',
                r'assist',
                r'support',
                r'guide',
                r'how.*use',
                r'what.*can.*do',
                r'options'
            ],
            'complaint': [
                r'wrong',
                r'incorrect',
                r'error',
                r'mistake',
                r'problem',
                r'issue',
                r'not.*working',
                r'frustrated'
            ],
            'clarification': [
                r'what.*mean',
                r'clarify',
                r'confused',
                r'understand',
                r'explain.*more',
                r'elaborate',
                r'detail'
            ],
            'follow_up': [
                r'also',
                r'additionally',
                r'furthermore',
                r'what.*else',
                r'more.*about',
                r'tell.*me.*more',
                r'continue'
            ]
        }
        
        # Compile patterns
        for intent, patterns in self.intent_patterns.items():
            self.intent_patterns[intent] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    async def classify_intent(self, message: str) -> Dict[str, Any]:
        """Classify intent of user message"""
        
        try:
            if self.model:
                return await self._classify_with_model(message)
            else:
                return await self._classify_with_rules(message)
                
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'alternatives': [],
                'timestamp': datetime.now().isoformat()
            }
    
    async def _classify_with_model(self, message: str) -> Dict[str, Any]:
        """Classify using trained ML model"""
        
        try:
            # Preprocess message
            processed_message = self._preprocess_message(message)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba([processed_message])[0]
            
            # Get top predictions
            intent_probs = list(zip(self.model.classes_, probabilities))
            intent_probs.sort(key=lambda x: x[1], reverse=True)
            
            primary_intent = intent_probs[0][0]
            primary_confidence = intent_probs[0][1]
            
            # Get alternatives
            alternatives = [
                {'intent': intent, 'confidence': float(conf)}
                for intent, conf in intent_probs[1:4]
                if conf > 0.1
            ]
            
            return {
                'intent': primary_intent,
                'confidence': float(primary_confidence),
                'alternatives': alternatives,
                'method': 'ml_model',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in model-based classification: {e}")
            return await self._classify_with_rules(message)
    
    async def _classify_with_rules(self, message: str) -> Dict[str, Any]:
        """Classify using rule-based approach"""
        
        try:
            processed_message = self._preprocess_message(message)
            intent_scores = {}
            
            # Score each intent based on pattern matches
            for intent, patterns in self.intent_patterns.items():
                score = 0
                matches = []
                
                for pattern in patterns:
                    match = pattern.search(processed_message)
                    if match:
                        score += 1
                        matches.append(match.group())
                
                if score > 0:
                    # Normalize score by number of patterns
                    intent_scores[intent] = {
                        'score': score / len(patterns),
                        'matches': matches
                    }
            
            if not intent_scores:
                return {
                    'intent': 'unknown',
                    'confidence': 0.0,
                    'alternatives': [],
                    'method': 'rule_based',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get best intent
            best_intent = max(intent_scores.items(), key=lambda x: x[1]['score'])
            primary_intent = best_intent[0]
            primary_score = best_intent[1]['score']
            
            # Convert score to confidence (simple mapping)
            confidence = min(1.0, primary_score * 2)
            
            # Get alternatives
            alternatives = []
            for intent, data in intent_scores.items():
                if intent != primary_intent and data['score'] > 0.1:
                    alternatives.append({
                        'intent': intent,
                        'confidence': min(1.0, data['score'] * 2),
                        'matches': data['matches']
                    })
            
            alternatives.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'intent': primary_intent,
                'confidence': confidence,
                'alternatives': alternatives[:3],
                'matches': best_intent[1]['matches'],
                'method': 'rule_based',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based classification: {e}")
            return {
                'intent': 'unknown',
                'confidence': 0.0,
                'alternatives': [],
                'timestamp': datetime.now().isoformat()
            }
    
    def _preprocess_message(self, message: str) -> str:
        """Preprocess message for classification"""
        
        # Convert to lowercase
        processed = message.lower().strip()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove special characters but keep alphanumeric and spaces
        processed = re.sub(r'[^a-zA-Z0-9\s]', ' ', processed)
        
        return processed
    
    def _train_model(self, training_data: List[Dict[str, str]]):
        """Train ML model for intent classification"""
        
        try:
            # Prepare training data
            texts = [item['text'] for item in training_data]
            labels = [item['intent'] for item in training_data]
            
            # Preprocess texts
            processed_texts = [self._preprocess_message(text) for text in texts]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                processed_texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Create pipeline
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )),
                ('classifier', LogisticRegression(random_state=42))
            ])
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Intent classifier trained with accuracy: {accuracy:.3f}")
            
            # Print classification report
            report = classification_report(y_test, y_pred)
            logger.info(f"Classification report:\n{report}")
            
        except Exception as e:
            logger.error(f"Error training intent classifier: {e}")
            self._initialize_rule_based_classifier()
    
    def add_training_example(self, text: str, intent: str):
        """Add training example for online learning"""
        
        try:
            if self.model and hasattr(self.model, 'partial_fit'):
                processed_text = self._preprocess_message(text)
                self.model.partial_fit([processed_text], [intent])
                logger.info(f"Added training example: {intent}")
            else:
                logger.warning("Model does not support online learning")
                
        except Exception as e:
            logger.error(f"Error adding training example: {e}")
    
    def get_intent_confidence_distribution(self, message: str) -> Dict[str, float]:
        """Get confidence distribution across all intents"""
        
        try:
            if self.model:
                processed_message = self._preprocess_message(message)
                probabilities = self.model.predict_proba([processed_message])[0]
                
                return dict(zip(self.model.classes_, probabilities.astype(float)))
            else:
                # Rule-based confidence distribution
                processed_message = self._preprocess_message(message)
                intent_scores = {}
                
                for intent, patterns in self.intent_patterns.items():
                    score = 0
                    for pattern in patterns:
                        if pattern.search(processed_message):
                            score += 1
                    
                    intent_scores[intent] = score / len(patterns) if patterns else 0
                
                # Normalize to sum to 1
                total_score = sum(intent_scores.values())
                if total_score > 0:
                    intent_scores = {k: v / total_score for k, v in intent_scores.items()}
                
                return intent_scores
                
        except Exception as e:
            logger.error(f"Error getting confidence distribution: {e}")
            return {intent: 0.0 for intent in self.intents}
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold"""
        
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Updated confidence threshold to {new_threshold}")
        else:
            logger.warning("Invalid confidence threshold. Must be between 0 and 1")
    
    def get_intent_statistics(self) -> Dict[str, Any]:
        """Get intent classification statistics"""
        
        return {
            'supported_intents': self.intents,
            'confidence_threshold': self.confidence_threshold,
            'model_type': 'ml_model' if self.model else 'rule_based',
            'pattern_count': sum(len(patterns) for patterns in self.intent_patterns.values()),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_classifier(self, filepath: str):
        """Save classifier to file"""
        
        classifier_data = {
            'model': self.model,
            'intent_patterns': self.intent_patterns,
            'intents': self.intents,
            'confidence_threshold': self.confidence_threshold,
            'config': self.config
        }
        
        joblib.dump(classifier_data, filepath)
        logger.info(f"Intent classifier saved to {filepath}")
    
    def load_classifier(self, filepath: str):
        """Load classifier from file"""
        
        classifier_data = joblib.load(filepath)
        
        self.model = classifier_data.get('model')
        self.intent_patterns = classifier_data.get('intent_patterns', {})
        self.intents = classifier_data.get('intents', [])
        self.confidence_threshold = classifier_data.get('confidence_threshold', 0.6)
        self.config = classifier_data.get('config', {})
        
        logger.info(f"Intent classifier loaded from {filepath}")
