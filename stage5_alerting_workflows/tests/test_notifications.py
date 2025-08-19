"""
Tests for Stage 5 notification components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from stage5_alerting_workflows.notifications import (
    EmailNotifier, SMSNotifier, WebhookNotifier, SlackIntegration, TeamsIntegration
)

class TestEmailNotifier:
    """Tests for EmailNotifier"""
    
    @pytest.fixture
    def email_notifier(self, notification_config):
        return EmailNotifier(notification_config['email'])
    
    @pytest.mark.asyncio
    async def test_send_alert_email(self, email_notifier, sample_alert_data):
        """Test sending alert email"""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            
            result = await email_notifier.send_alert_email(
                sample_alert_data,
                ['test@example.com']
            )
            
            assert result is True
            mock_smtp.assert_called_once()
            mock_server.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_resolution_email(self, email_notifier, sample_alert_data):
        """Test sending resolution email"""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            
            result = await email_notifier.send_resolution_email(
                sample_alert_data,
                ['test@example.com']
            )
            
            assert result is True
            mock_server.send_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_summary_email(self, email_notifier, sample_summary_data):
        """Test sending summary email"""
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = Mock()
            mock_smtp.return_value = mock_server
            
            result = await email_notifier.send_summary_email(
                sample_summary_data,
                ['test@example.com']
            )
            
            assert result is True
            mock_server.send_message.assert_called_once()

class TestSMSNotifier:
    """Tests for SMSNotifier"""
    
    @pytest.fixture
    def sms_notifier(self, notification_config):
        return SMSNotifier(notification_config['sms'])
    
    @pytest.mark.asyncio
    async def test_send_alert_sms(self, sms_notifier, sample_alert_data):
        """Test sending alert SMS"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'status': 'sent'})
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await sms_notifier.send_alert_sms(
                sample_alert_data,
                ['+1234567890']
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_format_alert_message(self, sms_notifier, sample_alert_data):
        """Test SMS message formatting"""
        message = sms_notifier._format_alert_message(sample_alert_data)
        
        assert isinstance(message, str)
        assert sample_alert_data['company_id'] in message
        assert sample_alert_data['severity'].upper() in message
        assert len(message) <= 160  # SMS length limit

class TestWebhookNotifier:
    """Tests for WebhookNotifier"""
    
    @pytest.fixture
    def webhook_notifier(self, notification_config):
        return WebhookNotifier(notification_config['webhook'])
    
    @pytest.mark.asyncio
    async def test_send_alert_webhook(self, webhook_notifier, sample_alert_data):
        """Test sending alert webhook"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await webhook_notifier.send_alert_webhook(
                sample_alert_data,
                ['https://example.com/webhook']
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_webhook_retry_logic(self, webhook_notifier, sample_alert_data):
        """Test webhook retry logic on failure"""
        with patch('aiohttp.ClientSession') as mock_session:
            # First attempt fails, second succeeds
            mock_response_fail = Mock()
            mock_response_fail.status = 500
            
            mock_response_success = Mock()
            mock_response_success.status = 200
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.side_effect = [
                mock_response_fail, mock_response_success
            ]
            
            result = await webhook_notifier.send_alert_webhook(
                sample_alert_data,
                ['https://example.com/webhook']
            )
            
            assert result is True

class TestSlackIntegration:
    """Tests for SlackIntegration"""
    
    @pytest.fixture
    def slack_integration(self, notification_config):
        return SlackIntegration(notification_config['slack'])
    
    @pytest.mark.asyncio
    async def test_send_alert_notification(self, slack_integration, sample_alert_data):
        """Test sending Slack alert notification"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'ok': True})
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await slack_integration.send_alert_notification(
                sample_alert_data,
                ['#alerts']
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_format_alert_message(self, slack_integration, sample_alert_data):
        """Test Slack message formatting"""
        message = slack_integration._format_alert_message(sample_alert_data)
        
        assert isinstance(message, dict)
        assert 'text' in message
        assert 'attachments' in message
        assert len(message['attachments']) > 0
    
    @pytest.mark.asyncio
    async def test_send_summary_message(self, slack_integration, sample_summary_data):
        """Test sending Slack summary message"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={'ok': True})
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await slack_integration.send_summary_message(
                sample_summary_data,
                ['#alerts']
            )
            
            assert result is True

class TestTeamsIntegration:
    """Tests for TeamsIntegration"""
    
    @pytest.fixture
    def teams_integration(self, notification_config):
        return TeamsIntegration(notification_config['teams'])
    
    @pytest.mark.asyncio
    async def test_send_alert_notification(self, teams_integration, sample_alert_data):
        """Test sending Teams alert notification"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await teams_integration.send_alert_notification(
                sample_alert_data,
                ['general']
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_format_alert_message(self, teams_integration, sample_alert_data):
        """Test Teams message formatting"""
        message = teams_integration._format_alert_message(sample_alert_data)
        
        assert isinstance(message, dict)
        assert '@type' in message
        assert message['@type'] == 'MessageCard'
        assert 'sections' in message
    
    @pytest.mark.asyncio
    async def test_get_theme_color(self, teams_integration):
        """Test theme color mapping"""
        assert teams_integration._get_theme_color('CRITICAL') == 'dc3545'
        assert teams_integration._get_theme_color('HIGH') == 'fd7e14'
        assert teams_integration._get_theme_color('MEDIUM') == 'ffc107'
        assert teams_integration._get_theme_color('LOW') == '28a745'
