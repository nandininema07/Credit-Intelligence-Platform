"""
Topic modeling and extraction for financial text analysis.
Uses LDA, BERT-based topic modeling, and financial domain-specific topics.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime
import re
from collections import Counter

logger = logging.getLogger(__name__)

class TopicExtractor:
    """Topic extraction and modeling for financial text"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.n_topics = config.get('n_topics', 10)
        self.models = {}
        self.financial_topics = self._define_financial_topics()
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize topic modeling components"""
        try:
            # TF-IDF Vectorizer
            self.tfidf = TfidfVectorizer(
                max_features=self.config.get('max_features', 1000),
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # LDA Model
            self.lda = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            
            # BERT model for embeddings
            if self.config.get('use_bert', True):
                try:
                    self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
                    self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
                    self.models['bert'] = True
                except Exception as e:
                    logger.warning(f"Could not load BERT model: {e}")
                    self.models['bert'] = False
            
            logger.info("Topic extraction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing topic models: {e}")
            
    def _define_financial_topics(self) -> Dict[str, List[str]]:
        """Define predefined financial topic categories"""
        return {
            'earnings': [
                'earnings', 'revenue', 'profit', 'income', 'quarterly', 'annual',
                'eps', 'guidance', 'forecast', 'results'
            ],
            'market_performance': [
                'stock', 'price', 'market', 'trading', 'volume', 'shares',
                'performance', 'returns', 'volatility', 'index'
            ],
            'corporate_actions': [
                'merger', 'acquisition', 'dividend', 'split', 'buyback',
                'spinoff', 'restructuring', 'divestiture'
            ],
            'regulatory': [
                'sec', 'filing', 'compliance', 'regulation', 'investigation',
                'lawsuit', 'settlement', 'fine', 'penalty'
            ],
            'management': [
                'ceo', 'cfo', 'management', 'board', 'executive', 'leadership',
                'appointment', 'resignation', 'strategy'
            ],
            'financial_health': [
                'debt', 'cash', 'liquidity', 'credit', 'rating', 'balance',
                'assets', 'liabilities', 'solvency', 'bankruptcy'
            ],
            'growth': [
                'expansion', 'growth', 'investment', 'capex', 'r&d',
                'innovation', 'product', 'launch', 'market share'
            ],
            'risk': [
                'risk', 'uncertainty', 'volatility', 'exposure', 'hedge',
                'insurance', 'contingency', 'crisis', 'downturn'
            ]
        }
    
    async def extract_topics(self, texts: List[str]) -> Dict[str, Any]:
        """Extract topics from a collection of texts"""
        if not texts:
            return {'topics': [], 'topic_weights': [], 'topic_distribution': {}}
        
        # Clean and preprocess texts
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        cleaned_texts = [text for text in cleaned_texts if text.strip()]
        
        if not cleaned_texts:
            return {'topics': [], 'topic_weights': [], 'topic_distribution': {}}
        
        results = {}
        
        # Rule-based topic classification
        predefined_topics = self._classify_predefined_topics(cleaned_texts)
        results['predefined_topics'] = predefined_topics
        
        # LDA topic modeling
        if len(cleaned_texts) >= 3:  # Need minimum texts for LDA
            lda_topics = await self._extract_lda_topics(cleaned_texts)
            results['lda_topics'] = lda_topics
        
        # BERT-based topic clustering
        if self.models.get('bert') and len(cleaned_texts) >= 5:
            bert_topics = await self._extract_bert_topics(cleaned_texts)
            results['bert_topics'] = bert_topics
        
        # Combine and rank topics
        combined_topics = self._combine_topic_results(results)
        
        return combined_topics
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for topic extraction"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters but keep financial symbols
        text = re.sub(r'[^\w\s$%]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _classify_predefined_topics(self, texts: List[str]) -> Dict[str, Any]:
        """Classify texts into predefined financial topics"""
        topic_scores = {topic: 0 for topic in self.financial_topics.keys()}
        topic_matches = {topic: [] for topic in self.financial_topics.keys()}
        
        for text in texts:
            text_lower = text.lower()
            
            for topic, keywords in self.financial_topics.items():
                matches = [keyword for keyword in keywords if keyword in text_lower]
                if matches:
                    topic_scores[topic] += len(matches)
                    topic_matches[topic].extend(matches)
        
        # Normalize scores
        total_matches = sum(topic_scores.values())
        if total_matches > 0:
            topic_weights = {topic: score / total_matches for topic, score in topic_scores.items()}
        else:
            topic_weights = {topic: 0.0 for topic in topic_scores.keys()}
        
        # Get top topics
        sorted_topics = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'topic_weights': topic_weights,
            'top_topics': sorted_topics[:5],
            'topic_matches': topic_matches,
            'total_matches': total_matches
        }
    
    async def _extract_lda_topics(self, texts: List[str]) -> Dict[str, Any]:
        """Extract topics using Latent Dirichlet Allocation"""
        try:
            # Vectorize texts
            tfidf_matrix = self.tfidf.fit_transform(texts)
            
            # Fit LDA model
            lda_model = self.lda.fit(tfidf_matrix)
            
            # Get topic-word distributions
            feature_names = self.tfidf.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topic_weight = topic[top_words_idx].sum()
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weight': float(topic_weight),
                    'coherence': self._calculate_topic_coherence(top_words, texts)
                })
            
            # Get document-topic distributions
            doc_topic_dist = lda_model.transform(tfidf_matrix)
            
            return {
                'topics': topics,
                'document_topic_distribution': doc_topic_dist.tolist(),
                'perplexity': lda_model.perplexity(tfidf_matrix),
                'log_likelihood': lda_model.score(tfidf_matrix)
            }
            
        except Exception as e:
            logger.error(f"Error in LDA topic extraction: {e}")
            return {'topics': [], 'document_topic_distribution': []}
    
    async def _extract_bert_topics(self, texts: List[str]) -> Dict[str, Any]:
        """Extract topics using BERT embeddings and clustering"""
        try:
            # Generate BERT embeddings
            embeddings = []
            
            for text in texts:
                # Tokenize and encode
                inputs = self.bert_tokenizer(
                    text, 
                    return_tensors='pt', 
                    max_length=512, 
                    truncation=True, 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].numpy()
                    embeddings.append(embedding[0])
            
            embeddings = np.array(embeddings)
            
            # Cluster embeddings
            n_clusters = min(self.n_topics, len(texts) // 2)
            if n_clusters < 2:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Extract representative words for each cluster
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_texts = [texts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if cluster_texts:
                    # Get most common words in cluster
                    all_words = []
                    for text in cluster_texts:
                        words = text.split()
                        all_words.extend(words)
                    
                    word_counts = Counter(all_words)
                    top_words = [word for word, count in word_counts.most_common(10)]
                    
                    clusters.append({
                        'cluster_id': cluster_id,
                        'size': len(cluster_texts),
                        'words': top_words,
                        'representative_texts': cluster_texts[:3]
                    })
            
            return {
                'clusters': clusters,
                'cluster_labels': cluster_labels.tolist(),
                'embeddings_shape': embeddings.shape,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error in BERT topic extraction: {e}")
            return {'clusters': [], 'cluster_labels': []}
    
    def _calculate_topic_coherence(self, words: List[str], texts: List[str]) -> float:
        """Calculate topic coherence score"""
        try:
            # Simple coherence based on word co-occurrence
            coherence_scores = []
            
            for i, word1 in enumerate(words[:5]):  # Top 5 words
                for word2 in words[i+1:6]:
                    # Count co-occurrence
                    co_occurrence = sum(1 for text in texts if word1 in text and word2 in text)
                    individual_occurrence = sum(1 for text in texts if word1 in text or word2 in text)
                    
                    if individual_occurrence > 0:
                        coherence = co_occurrence / individual_occurrence
                        coherence_scores.append(coherence)
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating topic coherence: {e}")
            return 0.0
    
    def _combine_topic_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different topic extraction methods"""
        combined = {
            'topics': [],
            'topic_weights': [],
            'topic_distribution': {},
            'method_results': results
        }
        
        # Start with predefined topics
        if 'predefined_topics' in results:
            predefined = results['predefined_topics']
            combined['topic_distribution'].update(predefined['topic_weights'])
            combined['topic_weights'] = list(predefined['topic_weights'].values())
        
        # Add LDA topics if available
        if 'lda_topics' in results and results['lda_topics'].get('topics'):
            lda_topics = results['lda_topics']['topics']
            for topic in lda_topics:
                combined['topics'].append({
                    'method': 'lda',
                    'words': topic['words'],
                    'weight': topic['weight'],
                    'coherence': topic.get('coherence', 0.0)
                })
        
        # Add BERT clusters if available
        if 'bert_topics' in results and results['bert_topics'].get('clusters'):
            bert_clusters = results['bert_topics']['clusters']
            for cluster in bert_clusters:
                combined['topics'].append({
                    'method': 'bert',
                    'words': cluster['words'],
                    'size': cluster['size'],
                    'weight': cluster['size'] / len(results['bert_topics'].get('cluster_labels', [1]))
                })
        
        # Sort topics by weight/importance
        combined['topics'].sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        return combined
    
    async def get_topic_trends(self, historical_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topic trends over time"""
        if not historical_topics:
            return {'trends': {}, 'emerging_topics': [], 'declining_topics': []}
        
        # Group topics by time period
        topic_timeline = {}
        
        for topic_data in historical_topics:
            timestamp = topic_data.get('timestamp', datetime.now())
            date_key = timestamp.strftime('%Y-%m-%d')
            
            if date_key not in topic_timeline:
                topic_timeline[date_key] = {}
            
            # Aggregate topic weights
            for topic, weight in topic_data.get('topic_distribution', {}).items():
                if topic not in topic_timeline[date_key]:
                    topic_timeline[date_key][topic] = 0
                topic_timeline[date_key][topic] += weight
        
        # Calculate trends
        trends = {}
        for topic in self.financial_topics.keys():
            topic_values = []
            dates = sorted(topic_timeline.keys())
            
            for date in dates:
                value = topic_timeline[date].get(topic, 0)
                topic_values.append(value)
            
            if len(topic_values) >= 2:
                # Simple trend calculation
                trend = (topic_values[-1] - topic_values[0]) / len(topic_values)
                trends[topic] = {
                    'trend': trend,
                    'current_value': topic_values[-1],
                    'change': topic_values[-1] - topic_values[0] if len(topic_values) > 1 else 0
                }
        
        # Identify emerging and declining topics
        sorted_trends = sorted(trends.items(), key=lambda x: x[1]['trend'], reverse=True)
        emerging_topics = [topic for topic, data in sorted_trends[:3] if data['trend'] > 0]
        declining_topics = [topic for topic, data in sorted_trends[-3:] if data['trend'] < 0]
        
        return {
            'trends': trends,
            'emerging_topics': emerging_topics,
            'declining_topics': declining_topics,
            'timeline': topic_timeline
        }
