"""
Text embeddings generation using BERT, FinBERT, and other transformer models.
Creates dense vector representations for financial text analysis.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
try:
    import torch
except ImportError:
    torch = None
try:
    from transformers import AutoTokenizer, AutoModel, AutoConfig
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    AutoConfig = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
import pickle
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class TextEmbeddings:
    """Text embeddings generator for financial text"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device('cuda' if torch and torch.cuda.is_available() else 'cpu') if torch else None
        self.initialize_models()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("TextEmbeddings initialized successfully")
        return True
        
    def initialize_models(self):
        """Initialize embedding models"""
        try:
            if not AutoModel or not AutoTokenizer:
                logger.warning("Transformers not available, using basic embeddings")
                return
                
            # FinBERT for financial text
            if self.config.get('use_finbert', True):
                try:
                    self.models['finbert'] = AutoModel.from_pretrained('ProsusAI/finbert')
                    self.tokenizers['finbert'] = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                    self.models['finbert'].to(self.device)
                    logger.info("FinBERT model loaded")
                except Exception as e:
                    logger.warning(f"Could not load FinBERT: {e}")
            
            # Sentence-BERT for general embeddings
            if self.config.get('use_sentence_bert', True):
                try:
                    model_name = self.config.get('sentence_bert_model', 'all-MiniLM-L6-v2')
                    self.models['sentence_bert'] = SentenceTransformer(model_name)
                    logger.info("Sentence-BERT model loaded")
                except Exception as e:
                    logger.warning(f"Could not load Sentence-BERT: {e}")
            
            # Multilingual BERT
            if self.config.get('use_multilingual', True):
                try:
                    self.models['multilingual'] = AutoModel.from_pretrained('bert-base-multilingual-cased')
                    self.tokenizers['multilingual'] = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
                    self.models['multilingual'].to(self.device)
                    logger.info("Multilingual BERT model loaded")
                except Exception as e:
                    logger.warning(f"Could not load Multilingual BERT: {e}")
            
            logger.info("Text embedding models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing embedding models: {e}")
    
    async def generate_embeddings(self, texts: List[str], model_name: str = 'auto') -> np.ndarray:
        """Generate embeddings for a list of texts"""
        if not texts:
            return np.array([])
        
        # Choose model
        if model_name == 'auto':
            if 'sentence_bert' in self.models:
                model_name = 'sentence_bert'
            elif 'finbert' in self.models:
                model_name = 'finbert'
            elif 'multilingual' in self.models:
                model_name = 'multilingual'
            else:
                logger.error("No embedding models available")
                return np.array([])
        
        if model_name == 'sentence_bert':
            return await self._generate_sentence_bert_embeddings(texts)
        elif model_name in ['finbert', 'multilingual']:
            return await self._generate_transformer_embeddings(texts, model_name)
        else:
            logger.error(f"Unknown model: {model_name}")
            return np.array([])
    
    async def _generate_sentence_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Sentence-BERT"""
        try:
            model = self.models['sentence_bert']
            
            # Process in batches to avoid memory issues
            batch_size = self.config.get('batch_size', 32)
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch, convert_to_numpy=True)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            logger.info(f"Generated Sentence-BERT embeddings: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Sentence-BERT embeddings: {e}")
            return np.array([])
    
    async def _generate_transformer_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings using transformer models"""
        try:
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            embeddings = []
            batch_size = self.config.get('batch_size', 16)  # Smaller batch for transformers
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self._process_batch(batch, model, tokenizer)
                embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(embeddings)
            logger.info(f"Generated {model_name} embeddings: {embeddings_array.shape}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error generating {model_name} embeddings: {e}")
            return np.array([])
    
    async def _process_batch(self, texts: List[str], model, tokenizer) -> List[np.ndarray]:
        """Process a batch of texts through transformer model"""
        embeddings = []
        
        for text in texts:
            try:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                    # Use CLS token embedding (first token)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(embedding[0])
                    
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                # Add zero embedding as fallback
                embeddings.append(np.zeros(768))  # Standard BERT embedding size
        
        return embeddings
    
    async def calculate_similarity(self, text1: str, text2: str, model_name: str = 'auto') -> float:
        """Calculate semantic similarity between two texts"""
        try:
            embeddings = await self.generate_embeddings([text1, text2], model_name)
            
            if embeddings.shape[0] == 2:
                # Calculate cosine similarity
                dot_product = np.dot(embeddings[0], embeddings[1])
                norm1 = np.linalg.norm(embeddings[0])
                norm2 = np.linalg.norm(embeddings[1])
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    return float(similarity)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def find_similar_texts(self, query_text: str, candidate_texts: List[str], 
                               top_k: int = 5, model_name: str = 'auto') -> List[Dict[str, Any]]:
        """Find most similar texts to a query"""
        if not candidate_texts:
            return []
        
        try:
            # Generate embeddings for all texts
            all_texts = [query_text] + candidate_texts
            embeddings = await self.generate_embeddings(all_texts, model_name)
            
            if embeddings.shape[0] < 2:
                return []
            
            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            
            # Calculate similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = np.dot(query_embedding, candidate_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
                )
                similarities.append({
                    'text': candidate_texts[i],
                    'similarity': float(similarity),
                    'index': i
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar texts: {e}")
            return []
    
    async def cluster_texts(self, texts: List[str], n_clusters: int = 5, 
                          model_name: str = 'auto') -> Dict[str, Any]:
        """Cluster texts based on semantic similarity"""
        if len(texts) < n_clusters:
            n_clusters = max(1, len(texts) // 2)
        
        try:
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts, model_name)
            
            if embeddings.shape[0] == 0:
                return {'clusters': [], 'labels': [], 'centroids': []}
            
            # Perform clustering
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Organize results
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'text': texts[i],
                    'index': i
                })
            
            # Calculate cluster centroids
            centroids = kmeans.cluster_centers_
            
            return {
                'clusters': [clusters[i] for i in range(n_clusters)],
                'labels': cluster_labels.tolist(),
                'centroids': centroids.tolist(),
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            logger.error(f"Error clustering texts: {e}")
            return {'clusters': [], 'labels': [], 'centroids': []}
    
    async def generate_document_embeddings(self, documents: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for documents with metadata"""
        document_embeddings = {}
        
        for doc in documents:
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            content = doc.get('content', '')
            
            if content:
                embedding = await self.generate_embeddings([content])
                if embedding.shape[0] > 0:
                    document_embeddings[doc_id] = embedding[0]
        
        logger.info(f"Generated embeddings for {len(document_embeddings)} documents")
        return document_embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filepath: str):
        """Save embeddings to disk"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load embeddings from disk"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings from {filepath}")
                return embeddings
            else:
                logger.warning(f"Embeddings file not found: {filepath}")
                return {}
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return {}
    
    def reduce_dimensionality(self, embeddings: np.ndarray, target_dim: int = 50) -> np.ndarray:
        """Reduce embedding dimensionality using PCA"""
        try:
            from sklearn.decomposition import PCA
            
            if embeddings.shape[1] <= target_dim:
                return embeddings
            
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            logger.info(f"Reduced embeddings from {embeddings.shape[1]} to {target_dim} dimensions")
            return reduced_embeddings
            
        except Exception as e:
            logger.error(f"Error reducing dimensionality: {e}")
            return embeddings
