"""
Data cleaning utilities for financial content.
Handles deduplication, validation, and normalization of scraped data.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
import hashlib
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)

class DataCleaner:
    """Advanced data cleaner for financial content"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.duplicate_threshold = config.get('duplicate_threshold', 0.8)
        self.min_content_length = config.get('min_content_length', 50)
        self.max_content_length = config.get('max_content_length', 10000)
        self.spam_keywords = self._load_spam_keywords()
        self.quality_filters = self._load_quality_filters()
        
    def _load_spam_keywords(self) -> Set[str]:
        """Load spam and low-quality content keywords"""
        return {
            'click here', 'buy now', 'limited time', 'act now', 'free trial',
            'guaranteed', 'no risk', 'make money fast', 'get rich quick',
            'investment opportunity', 'hot stock tip', 'insider information',
            'pump and dump', 'penny stock alert', 'guaranteed returns'
        }
    
    def _load_quality_filters(self) -> Dict[str, Any]:
        """Load quality filters for content validation"""
        return {
            'min_words': 10,
            'max_repeated_chars': 5,
            'min_sentence_length': 5,
            'max_caps_ratio': 0.5,
            'blocked_domains': {
                'spam.com', 'fake-news.com', 'clickbait.net'
            }
        }
    
    async def clean_article_data(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate article data"""
        if not article_data:
            return {}
        
        cleaned_data = article_data.copy()
        
        # Clean title
        if 'title' in cleaned_data:
            cleaned_data['title'] = self._clean_text_field(cleaned_data['title'])
        
        # Clean content
        if 'content' in cleaned_data:
            cleaned_data['content'] = self._clean_text_field(cleaned_data['content'])
            
            # Validate content quality
            if not self._is_quality_content(cleaned_data['content']):
                cleaned_data['quality_score'] = 0.0
                cleaned_data['quality_issues'] = self._identify_quality_issues(cleaned_data['content'])
            else:
                cleaned_data['quality_score'] = self._calculate_quality_score(cleaned_data['content'])
                cleaned_data['quality_issues'] = []
        
        # Clean URL
        if 'url' in cleaned_data:
            cleaned_data['url'] = self._clean_url(cleaned_data['url'])
        
        # Validate and clean timestamp
        if 'published_date' in cleaned_data:
            cleaned_data['published_date'] = self._clean_timestamp(cleaned_data['published_date'])
        
        # Clean source
        if 'source' in cleaned_data:
            cleaned_data['source'] = self._clean_source_name(cleaned_data['source'])
        
        # Add cleaning metadata
        cleaned_data['cleaning_metadata'] = {
            'cleaned_at': datetime.now(),
            'original_length': len(str(article_data.get('content', ''))),
            'cleaned_length': len(str(cleaned_data.get('content', ''))),
            'quality_score': cleaned_data.get('quality_score', 0.0)
        }
        
        return cleaned_data
    
    async def clean_social_data(self, social_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate social media data"""
        if not social_data:
            return {}
        
        cleaned_data = social_data.copy()
        
        # Clean text content
        if 'text' in cleaned_data:
            cleaned_data['text'] = self._clean_social_text(cleaned_data['text'])
            cleaned_data['quality_score'] = self._calculate_social_quality_score(cleaned_data['text'])
        
        # Clean author
        if 'author' in cleaned_data:
            cleaned_data['author'] = self._clean_username(cleaned_data['author'])
        
        # Validate engagement metrics
        if 'engagement' in cleaned_data:
            cleaned_data['engagement'] = self._clean_engagement_data(cleaned_data['engagement'])
        
        # Clean hashtags
        if 'hashtags' in cleaned_data:
            cleaned_data['hashtags'] = self._clean_hashtags(cleaned_data['hashtags'])
        
        # Clean mentions
        if 'mentions' in cleaned_data:
            cleaned_data['mentions'] = self._clean_mentions(cleaned_data['mentions'])
        
        # Add cleaning metadata
        cleaned_data['cleaning_metadata'] = {
            'cleaned_at': datetime.now(),
            'platform': cleaned_data.get('platform', 'unknown'),
            'quality_score': cleaned_data.get('quality_score', 0.0)
        }
        
        return cleaned_data
    
    def _clean_text_field(self, text: str) -> str:
        """Clean text field with comprehensive cleaning"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Fix encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Remove URLs (optional, based on config)
        if self.config.get('remove_urls', False):
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def _clean_social_text(self, text: str) -> str:
        """Clean social media text with platform-specific handling"""
        if not text:
            return ""
        
        # Basic cleaning
        text = self._clean_text_field(text)
        
        # Handle Twitter-specific elements
        # Keep hashtags and mentions but clean them
        text = re.sub(r'#(\w+)', r'#\1', text)  # Normalize hashtags
        text = re.sub(r'@(\w+)', r'@\1', text)  # Normalize mentions
        
        # Remove excessive emoji repetition
        text = re.sub(r'([\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF])\1{3,}', r'\1\1\1', text)
        
        return text.strip()
    
    def _clean_url(self, url: str) -> str:
        """Clean and validate URL"""
        if not url:
            return ""
        
        # Remove tracking parameters
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            # Remove common tracking parameters
            query_params = parsed.query
            if query_params:
                # Remove utm_, fbclid, gclid parameters
                clean_params = []
                for param in query_params.split('&'):
                    if not any(param.startswith(track) for track in ['utm_', 'fbclid', 'gclid', '_ga']):
                        clean_params.append(param)
                
                query_string = '&'.join(clean_params)
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if query_string:
                    url += f"?{query_string}"
            else:
                url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        return url
    
    def _clean_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Clean and validate timestamp"""
        if not timestamp:
            return None
        
        if isinstance(timestamp, datetime):
            # Check if timestamp is reasonable (not too far in future/past)
            now = datetime.now()
            if timestamp > now + timedelta(days=1):
                return now  # Cap future dates
            if timestamp < now - timedelta(days=365 * 10):
                return None  # Reject very old dates
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                # Try to parse ISO format
                if 'T' in timestamp:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                else:
                    dt = datetime.fromisoformat(timestamp)
                return self._clean_timestamp(dt)
            except ValueError:
                logger.warning(f"Could not parse timestamp: {timestamp}")
                return None
        
        return None
    
    def _clean_source_name(self, source: str) -> str:
        """Clean source name"""
        if not source:
            return "Unknown"
        
        # Remove common prefixes/suffixes
        source = re.sub(r'^(www\.)', '', source, flags=re.IGNORECASE)
        source = re.sub(r'(\.com|\.org|\.net|\.co\.uk)$', '', source, flags=re.IGNORECASE)
        
        # Capitalize properly
        source = ' '.join(word.capitalize() for word in source.split())
        
        return source.strip()
    
    def _clean_username(self, username: str) -> str:
        """Clean social media username"""
        if not username:
            return "anonymous"
        
        # Remove @ symbol if present
        username = username.lstrip('@')
        
        # Remove special characters except underscore and hyphen
        username = re.sub(r'[^\w\-_]', '', username)
        
        return username.lower().strip()
    
    def _clean_engagement_data(self, engagement: Dict[str, Any]) -> Dict[str, int]:
        """Clean engagement metrics"""
        cleaned_engagement = {}
        
        for key, value in engagement.items():
            try:
                # Convert to integer and ensure non-negative
                cleaned_value = max(0, int(value))
                cleaned_engagement[key] = cleaned_value
            except (ValueError, TypeError):
                cleaned_engagement[key] = 0
        
        return cleaned_engagement
    
    def _clean_hashtags(self, hashtags: List[str]) -> List[str]:
        """Clean hashtag list"""
        if not hashtags:
            return []
        
        cleaned_hashtags = []
        for hashtag in hashtags:
            if isinstance(hashtag, str):
                # Remove # symbol and clean
                clean_tag = hashtag.lstrip('#').strip().lower()
                if clean_tag and len(clean_tag) > 1:
                    cleaned_hashtags.append(clean_tag)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_hashtags = []
        for tag in cleaned_hashtags:
            if tag not in seen:
                seen.add(tag)
                unique_hashtags.append(tag)
        
        return unique_hashtags
    
    def _clean_mentions(self, mentions: List[str]) -> List[str]:
        """Clean mentions list"""
        if not mentions:
            return []
        
        cleaned_mentions = []
        for mention in mentions:
            if isinstance(mention, str):
                clean_mention = self._clean_username(mention)
                if clean_mention and clean_mention != "anonymous":
                    cleaned_mentions.append(clean_mention)
        
        # Remove duplicates
        return list(set(cleaned_mentions))
    
    def _is_quality_content(self, content: str) -> bool:
        """Check if content meets quality standards"""
        if not content or len(content) < self.min_content_length:
            return False
        
        if len(content) > self.max_content_length:
            return False
        
        # Check for spam keywords
        content_lower = content.lower()
        spam_count = sum(1 for keyword in self.spam_keywords if keyword in content_lower)
        if spam_count > 2:
            return False
        
        # Check caps ratio
        if len(content) > 0:
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > self.quality_filters['max_caps_ratio']:
                return False
        
        # Check for excessive repetition
        words = content.split()
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:  # Less than 30% unique words
                return False
        
        return True
    
    def _identify_quality_issues(self, content: str) -> List[str]:
        """Identify specific quality issues with content"""
        issues = []
        
        if len(content) < self.min_content_length:
            issues.append("content_too_short")
        
        if len(content) > self.max_content_length:
            issues.append("content_too_long")
        
        # Check for spam
        content_lower = content.lower()
        spam_count = sum(1 for keyword in self.spam_keywords if keyword in content_lower)
        if spam_count > 0:
            issues.append("contains_spam_keywords")
        
        # Check caps ratio
        if len(content) > 0:
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > self.quality_filters['max_caps_ratio']:
                issues.append("excessive_caps")
        
        # Check for repetition
        words = content.split()
        if len(words) > 10:
            unique_words = len(set(words))
            if unique_words / len(words) < 0.3:
                issues.append("excessive_repetition")
        
        return issues
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate quality score for content"""
        if not content:
            return 0.0
        
        score = 1.0
        
        # Length score
        length = len(content)
        if length < self.min_content_length:
            score *= 0.3
        elif length > self.max_content_length:
            score *= 0.7
        else:
            # Optimal length range
            optimal_min = 100
            optimal_max = 2000
            if optimal_min <= length <= optimal_max:
                score *= 1.0
            else:
                # Gradual penalty for non-optimal length
                if length < optimal_min:
                    score *= length / optimal_min
                else:
                    score *= optimal_max / length
        
        # Spam penalty
        content_lower = content.lower()
        spam_count = sum(1 for keyword in self.spam_keywords if keyword in content_lower)
        score *= max(0.1, 1.0 - (spam_count * 0.3))
        
        # Caps penalty
        if len(content) > 0:
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > 0.3:
                score *= max(0.3, 1.0 - caps_ratio)
        
        # Diversity bonus
        words = content.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            score *= min(1.2, 0.5 + unique_ratio)
        
        return min(1.0, max(0.0, score))
    
    def _calculate_social_quality_score(self, text: str) -> float:
        """Calculate quality score for social media content"""
        if not text:
            return 0.0
        
        score = 1.0
        
        # Length considerations (social media is typically shorter)
        length = len(text)
        if length < 10:
            score *= 0.3
        elif length > 500:
            score *= 0.8
        
        # Hashtag and mention balance
        hashtag_count = len(re.findall(r'#\w+', text))
        mention_count = len(re.findall(r'@\w+', text))
        
        # Penalty for excessive hashtags/mentions
        if hashtag_count > 5:
            score *= 0.7
        if mention_count > 3:
            score *= 0.8
        
        # Emoji consideration
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
        if emoji_count > 10:
            score *= 0.8
        
        return min(1.0, max(0.0, score))
    
    async def deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on content similarity"""
        if not articles:
            return []
        
        unique_articles = []
        seen_hashes = set()
        
        for article in articles:
            # Create content hash
            content = article.get('content', '') + article.get('title', '')
            content_hash = self._create_content_hash(content)
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_articles.append(article)
            else:
                # Check for exact duplicates vs similar content
                is_exact_duplicate = any(
                    self._calculate_similarity(content, existing.get('content', '') + existing.get('title', '')) > 0.95
                    for existing in unique_articles
                )
                
                if not is_exact_duplicate:
                    unique_articles.append(article)
        
        logger.info(f"Deduplicated {len(articles)} articles to {len(unique_articles)}")
        return unique_articles
    
    def _create_content_hash(self, content: str) -> str:
        """Create hash for content deduplication"""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Create hash of first 200 characters (for similarity detection)
        hash_content = normalized[:200]
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity for deduplication"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def validate_data_integrity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data integrity and completeness"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'completeness_score': 0.0
        }
        
        required_fields = ['content', 'source', 'timestamp']
        optional_fields = ['title', 'author', 'url', 'language']
        
        # Check required fields
        missing_required = [field for field in required_fields if not data.get(field)]
        if missing_required:
            validation_result['is_valid'] = False
            validation_result['errors'].extend([f"Missing required field: {field}" for field in missing_required])
        
        # Check optional fields
        missing_optional = [field for field in optional_fields if not data.get(field)]
        validation_result['warnings'].extend([f"Missing optional field: {field}" for field in missing_optional])
        
        # Calculate completeness score
        total_fields = len(required_fields) + len(optional_fields)
        present_fields = total_fields - len(missing_required) - len(missing_optional)
        validation_result['completeness_score'] = present_fields / total_fields
        
        # Validate field types and formats
        if 'timestamp' in data and data['timestamp']:
            if not isinstance(data['timestamp'], datetime):
                validation_result['errors'].append("Invalid timestamp format")
        
        if 'url' in data and data['url']:
            parsed_url = urlparse(data['url'])
            if not parsed_url.scheme or not parsed_url.netloc:
                validation_result['warnings'].append("Invalid URL format")
        
        return validation_result
    
    async def clean_batch(self, data_list: List[Dict[str, Any]], data_type: str = 'article') -> List[Dict[str, Any]]:
        """Clean a batch of data items"""
        if data_type == 'article':
            tasks = [self.clean_article_data(item) for item in data_list]
        elif data_type == 'social':
            tasks = [self.clean_social_data(item) for item in data_list]
        else:
            # Generic cleaning
            tasks = [self.clean_article_data(item) for item in data_list]
        
        cleaned_data = await asyncio.gather(*tasks)
        
        # Filter out low-quality items
        quality_threshold = self.config.get('quality_threshold', 0.3)
        filtered_data = [
            item for item in cleaned_data 
            if item.get('quality_score', 0) >= quality_threshold
        ]
        
        logger.info(f"Cleaned {len(data_list)} items, {len(filtered_data)} passed quality filter")
        return filtered_data
    
    def get_cleaning_stats(self, original_data: List[Dict[str, Any]], 
                          cleaned_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the cleaning process"""
        return {
            'original_count': len(original_data),
            'cleaned_count': len(cleaned_data),
            'removed_count': len(original_data) - len(cleaned_data),
            'removal_rate': (len(original_data) - len(cleaned_data)) / max(len(original_data), 1),
            'avg_quality_score': sum(item.get('quality_score', 0) for item in cleaned_data) / max(len(cleaned_data), 1),
            'timestamp': datetime.now()
        }
