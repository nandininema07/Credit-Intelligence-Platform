"""
Tests for chatbot module components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from stage4_explainability.chatbot.chat_engine import ChatEngine
from stage4_explainability.chatbot.intent_classifier import IntentClassifier
from stage4_explainability.chatbot.entity_recognizer import EntityRecognizer
from stage4_explainability.chatbot.response_generator import ResponseGenerator
from stage4_explainability.chatbot.context_manager import ContextManager
from stage4_explainability.chatbot.knowledge_base import KnowledgeBase

class TestChatEngine:
    """Test cases for ChatEngine"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'session_timeout': 3600,
            'max_context_length': 10,
            'enable_analytics': True
        }
    
    @pytest.fixture
    def mock_components(self):
        return {
            'intent_classifier': Mock(),
            'entity_recognizer': Mock(),
            'response_generator': Mock(),
            'context_manager': Mock(),
            'knowledge_base': Mock()
        }
    
    def test_chat_engine_initialization(self, sample_config):
        engine = ChatEngine(sample_config)
        assert engine.config == sample_config
        assert hasattr(engine, 'sessions')
        assert hasattr(engine, 'analytics')
    
    @pytest.mark.asyncio
    async def test_process_message(self, sample_config, mock_components):
        engine = ChatEngine(sample_config)
        
        # Mock component responses
        mock_components['intent_classifier'].classify_intent = AsyncMock(return_value={
            'intent': 'credit_score_inquiry',
            'confidence': 0.9
        })
        
        mock_components['entity_recognizer'].extract_entities = AsyncMock(return_value=[
            {'text': '750', 'type': 'CREDIT_SCORE', 'confidence': 0.95}
        ])
        
        mock_components['response_generator'].generate_response = AsyncMock(return_value={
            'text': 'Your credit score of 750 is excellent!',
            'confidence': 0.9,
            'suggestions': []
        })
        
        mock_components['context_manager'].update_context = AsyncMock()
        mock_components['context_manager'].get_context = AsyncMock(return_value={})
        
        # Set components
        for name, component in mock_components.items():
            setattr(engine, name, component)
        
        result = await engine.process_message(
            user_id="test_user",
            message="What does a credit score of 750 mean?",
            session_id="test_session"
        )
        
        assert 'response' in result
        assert 'intent' in result
        assert 'entities' in result
        assert result['response']['text'] == 'Your credit score of 750 is excellent!'

class TestIntentClassifier:
    """Test cases for IntentClassifier"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'confidence_threshold': 0.7,
            'enable_online_learning': True,
            'model_path': None
        }
    
    def test_intent_classifier_initialization(self, sample_config):
        classifier = IntentClassifier(sample_config)
        assert classifier.config == sample_config
        assert hasattr(classifier, 'rule_patterns')
        assert hasattr(classifier, 'intent_stats')
    
    @pytest.mark.asyncio
    async def test_classify_intent_rule_based(self, sample_config):
        classifier = IntentClassifier(sample_config)
        
        # Test credit score inquiry
        result = await classifier.classify_intent("What is my credit score?")
        assert result['intent'] == 'credit_score_inquiry'
        assert result['confidence'] > 0.5
        
        # Test improvement advice
        result = await classifier.classify_intent("How can I improve my credit?")
        assert result['intent'] == 'improvement_advice'
        assert result['confidence'] > 0.5
    
    @pytest.mark.asyncio
    async def test_classify_intent_unknown(self, sample_config):
        classifier = IntentClassifier(sample_config)
        
        result = await classifier.classify_intent("Random unrelated text")
        assert result['intent'] == 'unknown'
        assert result['confidence'] < 0.7

class TestEntityRecognizer:
    """Test cases for EntityRecognizer"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'confidence_threshold': 0.8,
            'enable_validation': True,
            'custom_patterns': {}
        }
    
    def test_entity_recognizer_initialization(self, sample_config):
        recognizer = EntityRecognizer(sample_config)
        assert recognizer.config == sample_config
        assert hasattr(recognizer, 'entity_patterns')
        assert hasattr(recognizer, 'entity_stats')
    
    @pytest.mark.asyncio
    async def test_extract_credit_score(self, sample_config):
        recognizer = EntityRecognizer(sample_config)
        
        entities = await recognizer.extract_entities("My credit score is 750")
        
        score_entities = [e for e in entities if e['type'] == 'CREDIT_SCORE']
        assert len(score_entities) > 0
        assert score_entities[0]['text'] == '750'
    
    @pytest.mark.asyncio
    async def test_extract_percentage(self, sample_config):
        recognizer = EntityRecognizer(sample_config)
        
        entities = await recognizer.extract_entities("My utilization is 30%")
        
        percentage_entities = [e for e in entities if e['type'] == 'PERCENTAGE']
        assert len(percentage_entities) > 0
        assert '30%' in percentage_entities[0]['text']
    
    @pytest.mark.asyncio
    async def test_extract_currency(self, sample_config):
        recognizer = EntityRecognizer(sample_config)
        
        entities = await recognizer.extract_entities("I owe $5,000 on my credit card")
        
        currency_entities = [e for e in entities if e['type'] == 'CURRENCY']
        assert len(currency_entities) > 0
        assert '$5,000' in currency_entities[0]['text']

class TestResponseGenerator:
    """Test cases for ResponseGenerator"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'default_confidence_threshold': 0.7,
            'max_suggestions': 3,
            'enable_personalization': True
        }
    
    def test_response_generator_initialization(self, sample_config):
        generator = ResponseGenerator(sample_config)
        assert generator.config == sample_config
        assert hasattr(generator, 'response_templates')
        assert hasattr(generator, 'response_stats')
    
    @pytest.mark.asyncio
    async def test_generate_response_credit_score_inquiry(self, sample_config):
        generator = ResponseGenerator(sample_config)
        
        context = {
            'intent': 'credit_score_inquiry',
            'entities': [{'text': '750', 'type': 'CREDIT_SCORE'}],
            'user_context': {}
        }
        
        response = await generator.generate_response(context)
        
        assert 'text' in response
        assert 'confidence' in response
        assert '750' in response['text'] or 'credit score' in response['text'].lower()
    
    @pytest.mark.asyncio
    async def test_generate_fallback_response(self, sample_config):
        generator = ResponseGenerator(sample_config)
        
        context = {
            'intent': 'unknown',
            'entities': [],
            'user_context': {}
        }
        
        response = await generator.generate_response(context)
        
        assert 'text' in response
        assert response['confidence'] < 0.7
        assert len(response.get('suggestions', [])) > 0

class TestContextManager:
    """Test cases for ContextManager"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'max_history_length': 10,
            'context_timeout': 3600,
            'enable_persistence': False
        }
    
    def test_context_manager_initialization(self, sample_config):
        manager = ContextManager(sample_config)
        assert manager.config == sample_config
        assert hasattr(manager, 'user_contexts')
        assert hasattr(manager, 'session_contexts')
    
    @pytest.mark.asyncio
    async def test_update_context(self, sample_config):
        manager = ContextManager(sample_config)
        
        await manager.update_context(
            user_id="test_user",
            session_id="test_session",
            message="What is my credit score?",
            intent="credit_score_inquiry",
            entities=[{'text': 'credit score', 'type': 'CREDIT_TERM'}],
            response="Your credit score is important for loan approvals."
        )
        
        context = await manager.get_context("test_user", "test_session")
        
        assert context is not None
        assert 'conversation_history' in context
        assert len(context['conversation_history']) > 0
    
    @pytest.mark.asyncio
    async def test_extract_credit_profile(self, sample_config):
        manager = ContextManager(sample_config)
        
        # Simulate conversation with credit information
        await manager.update_context(
            user_id="test_user",
            session_id="test_session",
            message="My credit score is 750",
            intent="credit_score_inquiry",
            entities=[{'text': '750', 'type': 'CREDIT_SCORE'}],
            response="That's an excellent score!"
        )
        
        profile = await manager.extract_credit_profile("test_user")
        
        assert 'credit_score' in profile
        assert profile['credit_score'] == 750

class TestKnowledgeBase:
    """Test cases for KnowledgeBase"""
    
    @pytest.fixture
    def sample_config(self):
        return {
            'enable_caching': True,
            'cache_ttl': 3600,
            'enable_updates': True
        }
    
    def test_knowledge_base_initialization(self, sample_config):
        kb = KnowledgeBase(sample_config)
        assert kb.config == sample_config
        assert hasattr(kb, 'factor_definitions')
        assert hasattr(kb, 'improvement_strategies')
        assert hasattr(kb, 'faq_database')
    
    @pytest.mark.asyncio
    async def test_get_factor_explanation(self, sample_config):
        kb = KnowledgeBase(sample_config)
        
        explanation = await kb.get_factor_explanation('payment_history')
        
        assert explanation is not None
        assert 'definition' in explanation
        assert 'importance' in explanation
        assert 'payment' in explanation['definition'].lower()
    
    @pytest.mark.asyncio
    async def test_get_improvement_strategies(self, sample_config):
        kb = KnowledgeBase(sample_config)
        
        strategies = await kb.get_improvement_strategies('credit_utilization')
        
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert all('strategy' in s and 'impact' in s for s in strategies)
    
    @pytest.mark.asyncio
    async def test_search_faq(self, sample_config):
        kb = KnowledgeBase(sample_config)
        
        results = await kb.search_faq("credit score")
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all('question' in r and 'answer' in r for r in results)

@pytest.mark.asyncio
async def test_chatbot_integration():
    """Integration test for chatbot components"""
    
    config = {
        'session_timeout': 3600,
        'confidence_threshold': 0.7,
        'max_suggestions': 3
    }
    
    # Initialize components
    engine = ChatEngine(config)
    intent_classifier = IntentClassifier(config)
    entity_recognizer = EntityRecognizer(config)
    response_generator = ResponseGenerator(config)
    context_manager = ContextManager(config)
    knowledge_base = KnowledgeBase(config)
    
    # Set components in engine
    engine.intent_classifier = intent_classifier
    engine.entity_recognizer = entity_recognizer
    engine.response_generator = response_generator
    engine.context_manager = context_manager
    engine.knowledge_base = knowledge_base
    
    # Test conversation flow
    result = await engine.process_message(
        user_id="integration_test_user",
        message="What does a credit score of 720 mean?",
        session_id="integration_test_session"
    )
    
    assert 'response' in result
    assert 'intent' in result
    assert 'entities' in result
    assert result['intent']['intent'] in ['credit_score_inquiry', 'explanation_request']
    
    # Test follow-up message
    result2 = await engine.process_message(
        user_id="integration_test_user",
        message="How can I improve it?",
        session_id="integration_test_session"
    )
    
    assert 'response' in result2
    assert result2['intent']['intent'] in ['improvement_advice', 'how_to_question']

if __name__ == "__main__":
    pytest.main([__file__])
