"""
SMS notification service for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SMSNotifier:
    """SMS notification service using multiple providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = {}
        self.statistics = {
            'sms_sent': 0,
            'sms_failed': 0,
            'providers_used': {}
        }
        self._initialize_providers()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("SMSNotifier initialized successfully")
        return True
    
    def _initialize_providers(self):
        """Initialize SMS providers"""
        
        # Twilio provider
        twilio_config = self.config.get('twilio', {})
        if twilio_config.get('account_sid') and twilio_config.get('auth_token'):
            self.providers['twilio'] = {
                'account_sid': twilio_config['account_sid'],
                'auth_token': twilio_config['auth_token'],
                'from_number': twilio_config.get('from_number', ''),
                'url': f"https://api.twilio.com/2010-04-01/Accounts/{twilio_config['account_sid']}/Messages.json"
            }
        
        # AWS SNS provider
        aws_config = self.config.get('aws_sns', {})
        if aws_config.get('access_key') and aws_config.get('secret_key'):
            self.providers['aws_sns'] = {
                'access_key': aws_config['access_key'],
                'secret_key': aws_config['secret_key'],
                'region': aws_config.get('region', 'us-east-1')
            }
        
        # Generic webhook provider
        webhook_config = self.config.get('webhook', {})
        if webhook_config.get('url'):
            self.providers['webhook'] = {
                'url': webhook_config['url'],
                'headers': webhook_config.get('headers', {}),
                'method': webhook_config.get('method', 'POST')
            }
        
        self.default_provider = self.config.get('default_provider', 'twilio')
    
    async def send_alert_sms(self, alert_data: Dict[str, Any], 
                           phone_numbers: List[str]) -> bool:
        """Send SMS alert notification"""
        
        try:
            message = self._format_alert_message(alert_data)
            
            return await self.send_sms(
                phone_numbers=phone_numbers,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error sending alert SMS: {e}")
            return False
    
    async def send_escalation_sms(self, alert_data: Dict[str, Any],
                                phone_numbers: List[str]) -> bool:
        """Send SMS escalation notification"""
        
        try:
            message = self._format_escalation_message(alert_data)
            
            return await self.send_sms(
                phone_numbers=phone_numbers,
                message=message
            )
            
        except Exception as e:
            logger.error(f"Error sending escalation SMS: {e}")
            return False
    
    async def send_sms(self, phone_numbers: List[str], message: str,
                      provider: str = None) -> bool:
        """Send SMS using specified or default provider"""
        
        try:
            provider_name = provider or self.default_provider
            
            if provider_name not in self.providers:
                logger.error(f"SMS provider {provider_name} not configured")
                return False
            
            success_count = 0
            
            for phone_number in phone_numbers:
                try:
                    success = await self._send_single_sms(
                        phone_number, message, provider_name
                    )
                    
                    if success:
                        success_count += 1
                        self.statistics['sms_sent'] += 1
                    else:
                        self.statistics['sms_failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error sending SMS to {phone_number}: {e}")
                    self.statistics['sms_failed'] += 1
            
            # Update provider statistics
            provider_count = self.statistics['providers_used'].get(provider_name, 0)
            self.statistics['providers_used'][provider_name] = provider_count + success_count
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    async def _send_single_sms(self, phone_number: str, message: str, 
                             provider: str) -> bool:
        """Send SMS to single recipient"""
        
        try:
            if provider == 'twilio':
                return await self._send_twilio_sms(phone_number, message)
            elif provider == 'aws_sns':
                return await self._send_aws_sns_sms(phone_number, message)
            elif provider == 'webhook':
                return await self._send_webhook_sms(phone_number, message)
            else:
                logger.error(f"Unknown SMS provider: {provider}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending single SMS: {e}")
            return False
    
    async def _send_twilio_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS via Twilio"""
        
        try:
            provider_config = self.providers['twilio']
            
            auth = aiohttp.BasicAuth(
                provider_config['account_sid'],
                provider_config['auth_token']
            )
            
            data = {
                'From': provider_config['from_number'],
                'To': phone_number,
                'Body': message
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    provider_config['url'],
                    auth=auth,
                    data=data
                ) as response:
                    
                    if response.status == 201:
                        logger.info(f"Twilio SMS sent successfully to {phone_number}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Twilio SMS failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending Twilio SMS: {e}")
            return False
    
    async def _send_aws_sns_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS via AWS SNS"""
        
        try:
            # This would typically use boto3 for AWS SNS
            # For demonstration, simulate the API call
            
            logger.info(f"AWS SNS SMS would be sent to {phone_number}")
            logger.info(f"Message: {message}")
            
            # Simulate success
            return True
            
        except Exception as e:
            logger.error(f"Error sending AWS SNS SMS: {e}")
            return False
    
    async def _send_webhook_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS via webhook"""
        
        try:
            provider_config = self.providers['webhook']
            
            payload = {
                'phone_number': phone_number,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            headers = provider_config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    provider_config['method'],
                    provider_config['url'],
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if 200 <= response.status < 300:
                        logger.info(f"Webhook SMS sent successfully to {phone_number}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Webhook SMS failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending webhook SMS: {e}")
            return False
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert data into SMS message"""
        
        try:
            severity = alert_data.get('severity', 'MEDIUM').upper()
            company_id = alert_data.get('company_id', 'Unknown')
            factor = alert_data.get('factor', 'Unknown')
            title = alert_data.get('title', 'Credit Alert')
            
            # Keep message under 160 characters for standard SMS
            message = f"🚨 {severity} ALERT\n{company_id} - {factor}\n{title}"
            
            if len(message) > 160:
                # Truncate if too long
                message = message[:157] + "..."
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return "Credit Alert - Check system for details"
    
    def _format_escalation_message(self, alert_data: Dict[str, Any]) -> str:
        """Format escalation data into SMS message"""
        
        try:
            alert_id = alert_data.get('alert_id', 'Unknown')
            age_minutes = alert_data.get('age_minutes', 0)
            priority = alert_data.get('current_priority', 'MEDIUM')
            
            message = f"⚠️ ESCALATION\nAlert {alert_id}\nAge: {int(age_minutes)}min\nPriority: {priority}"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting escalation message: {e}")
            return "Alert Escalation - Check system for details"
    
    async def validate_phone_number(self, phone_number: str) -> bool:
        """Validate phone number format"""
        
        try:
            # Basic validation - should start with + and contain only digits
            if not phone_number.startswith('+'):
                return False
            
            # Remove + and check if remaining characters are digits
            digits_only = phone_number[1:]
            if not digits_only.isdigit():
                return False
            
            # Check length (international numbers are typically 10-15 digits)
            if len(digits_only) < 10 or len(digits_only) > 15:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating phone number: {e}")
            return False
    
    async def test_provider(self, provider: str, test_number: str) -> bool:
        """Test SMS provider with a test message"""
        
        try:
            if provider not in self.providers:
                logger.error(f"Provider {provider} not configured")
                return False
            
            if not await self.validate_phone_number(test_number):
                logger.error(f"Invalid test phone number: {test_number}")
                return False
            
            test_message = f"Test message from Credit Intelligence Platform - {datetime.now().strftime('%H:%M:%S')}"
            
            success = await self._send_single_sms(test_number, test_message, provider)
            
            if success:
                logger.info(f"SMS provider {provider} test successful")
            else:
                logger.error(f"SMS provider {provider} test failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing SMS provider: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get SMS notification statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['sms_sent'] + stats['sms_failed']
            success_rate = (stats['sms_sent'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'configured_providers': list(self.providers.keys()),
                'default_provider': self.default_provider
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
