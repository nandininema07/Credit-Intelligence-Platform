"""
Notification service for Stage 5 alerting workflows.
Handles multi-channel notifications (Email, Slack, Teams, SMS).
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class NotificationChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    SMS = "sms"
    WEBHOOK = "webhook"

@dataclass
class NotificationTemplate:
    """Notification template configuration"""
    channel: NotificationChannel
    subject_template: str
    body_template: str
    priority_mapping: Dict[str, str]

class NotificationService:
    """Multi-channel notification service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.templates = {}
        self.channels = {}
        self.delivery_stats = {
            'sent': 0,
            'failed': 0,
            'by_channel': {}
        }
        self.setup_channels()
        self.setup_templates()
    
    def setup_channels(self):
        """Setup notification channels"""
        # Email configuration
        if self.config.get('email', {}).get('enabled', False):
            self.channels[NotificationChannel.EMAIL] = {
                'smtp_server': self.config['email'].get('smtp_server', 'smtp.gmail.com'),
                'smtp_port': self.config['email'].get('smtp_port', 587),
                'username': self.config['email'].get('username'),
                'password': self.config['email'].get('password'),
                'from_address': self.config['email'].get('from_address')
            }
        
        # Slack configuration
        if self.config.get('slack', {}).get('enabled', False):
            self.channels[NotificationChannel.SLACK] = {
                'webhook_url': self.config['slack'].get('webhook_url'),
                'token': self.config['slack'].get('token'),
                'default_channel': self.config['slack'].get('default_channel', '#alerts')
            }
        
        # Teams configuration
        if self.config.get('teams', {}).get('enabled', False):
            self.channels[NotificationChannel.TEAMS] = {
                'webhook_url': self.config['teams'].get('webhook_url')
            }
        
        # SMS configuration
        if self.config.get('sms', {}).get('enabled', False):
            self.channels[NotificationChannel.SMS] = {
                'provider': self.config['sms'].get('provider', 'twilio'),
                'account_sid': self.config['sms'].get('account_sid'),
                'auth_token': self.config['sms'].get('auth_token'),
                'from_number': self.config['sms'].get('from_number')
            }
    
    def setup_templates(self):
        """Setup notification templates"""
        default_templates = {
            NotificationChannel.EMAIL: NotificationTemplate(
                channel=NotificationChannel.EMAIL,
                subject_template="Credit Alert: {alert_type} for {company_name}",
                body_template="""
                Alert Details:
                Company: {company_name}
                Alert Type: {alert_type}
                Severity: {severity}
                Score: {score}
                Description: {description}
                Timestamp: {timestamp}
                
                Please review and take appropriate action.
                """,
                priority_mapping={'high': 'URGENT', 'medium': 'NORMAL', 'low': 'LOW'}
            ),
            NotificationChannel.SLACK: NotificationTemplate(
                channel=NotificationChannel.SLACK,
                subject_template="🚨 Credit Alert",
                body_template="""
                *Alert: {alert_type}*
                Company: {company_name}
                Severity: {severity}
                Score: {score}
                Description: {description}
                """,
                priority_mapping={'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            )
        }
        
        self.templates.update(default_templates)
    
    async def send_notification(self, alert_data: Dict[str, Any], 
                              channels: List[NotificationChannel] = None,
                              recipients: List[str] = None) -> Dict[str, bool]:
        """Send notification across specified channels"""
        if channels is None:
            channels = list(self.channels.keys())
        
        results = {}
        
        for channel in channels:
            try:
                if channel in self.channels:
                    success = await self._send_to_channel(channel, alert_data, recipients)
                    results[channel.value] = success
                    
                    # Update stats
                    if success:
                        self.delivery_stats['sent'] += 1
                    else:
                        self.delivery_stats['failed'] += 1
                    
                    if channel.value not in self.delivery_stats['by_channel']:
                        self.delivery_stats['by_channel'][channel.value] = {'sent': 0, 'failed': 0}
                    
                    if success:
                        self.delivery_stats['by_channel'][channel.value]['sent'] += 1
                    else:
                        self.delivery_stats['by_channel'][channel.value]['failed'] += 1
                
            except Exception as e:
                logger.error(f"Error sending notification to {channel.value}: {e}")
                results[channel.value] = False
                self.delivery_stats['failed'] += 1
        
        return results
    
    async def _send_to_channel(self, channel: NotificationChannel, 
                             alert_data: Dict[str, Any], 
                             recipients: List[str] = None) -> bool:
        """Send notification to specific channel"""
        try:
            if channel == NotificationChannel.EMAIL:
                return await self._send_email(alert_data, recipients)
            elif channel == NotificationChannel.SLACK:
                return await self._send_slack(alert_data)
            elif channel == NotificationChannel.TEAMS:
                return await self._send_teams(alert_data)
            elif channel == NotificationChannel.SMS:
                return await self._send_sms(alert_data, recipients)
            else:
                logger.warning(f"Unsupported channel: {channel}")
                return False
                
        except Exception as e:
            logger.error(f"Error in _send_to_channel for {channel.value}: {e}")
            return False
    
    async def _send_email(self, alert_data: Dict[str, Any], recipients: List[str]) -> bool:
        """Send email notification"""
        try:
            if NotificationChannel.EMAIL not in self.channels:
                return False
            
            config = self.channels[NotificationChannel.EMAIL]
            template = self.templates[NotificationChannel.EMAIL]
            
            # Format message
            subject = template.subject_template.format(**alert_data)
            body = template.body_template.format(**alert_data)
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_address']
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send to each recipient
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            
            for recipient in recipients or []:
                msg['To'] = recipient
                server.send_message(msg)
                del msg['To']
            
            server.quit()
            logger.info(f"Email sent successfully to {len(recipients or [])} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_slack(self, alert_data: Dict[str, Any]) -> bool:
        """Send Slack notification"""
        try:
            if NotificationChannel.SLACK not in self.channels:
                return False
            
            config = self.channels[NotificationChannel.SLACK]
            template = self.templates[NotificationChannel.SLACK]
            
            # Format message
            text = template.body_template.format(**alert_data)
            
            payload = {
                "text": text,
                "channel": config['default_channel'],
                "username": "Credit Alert Bot"
            }
            
            response = requests.post(config['webhook_url'], json=payload)
            
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
                return True
            else:
                logger.error(f"Slack notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_teams(self, alert_data: Dict[str, Any]) -> bool:
        """Send Teams notification"""
        try:
            if NotificationChannel.TEAMS not in self.channels:
                return False
            
            config = self.channels[NotificationChannel.TEAMS]
            
            # Format Teams card
            card = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "summary": f"Credit Alert: {alert_data.get('alert_type', 'Unknown')}",
                "themeColor": "FF0000" if alert_data.get('severity') == 'high' else "FFA500",
                "sections": [{
                    "activityTitle": f"Credit Alert: {alert_data.get('alert_type', 'Unknown')}",
                    "facts": [
                        {"name": "Company", "value": alert_data.get('company_name', 'Unknown')},
                        {"name": "Severity", "value": alert_data.get('severity', 'Unknown')},
                        {"name": "Score", "value": str(alert_data.get('score', 'N/A'))},
                        {"name": "Description", "value": alert_data.get('description', 'No description')}
                    ]
                }]
            }
            
            response = requests.post(config['webhook_url'], json=card)
            
            if response.status_code == 200:
                logger.info("Teams notification sent successfully")
                return True
            else:
                logger.error(f"Teams notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Teams notification: {e}")
            return False
    
    async def _send_sms(self, alert_data: Dict[str, Any], recipients: List[str]) -> bool:
        """Send SMS notification"""
        try:
            if NotificationChannel.SMS not in self.channels:
                return False
            
            config = self.channels[NotificationChannel.SMS]
            
            # Format SMS message
            message = f"Credit Alert: {alert_data.get('alert_type')} for {alert_data.get('company_name')}. Severity: {alert_data.get('severity')}. Score: {alert_data.get('score')}"
            
            # For now, just log the SMS (would need actual SMS provider integration)
            logger.info(f"SMS would be sent to {recipients}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get notification delivery statistics"""
        return self.delivery_stats.copy()
    
    def reset_stats(self):
        """Reset delivery statistics"""
        self.delivery_stats = {
            'sent': 0,
            'failed': 0,
            'by_channel': {}
        }
