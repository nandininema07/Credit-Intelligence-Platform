"""
Slack integration for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SlackIntegration:
    """Slack integration for alert notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_token = config.get('bot_token', '')
        self.webhook_urls = config.get('webhook_urls', {})
        self.default_channel = config.get('default_channel', '#alerts')
        self.statistics = {
            'messages_sent': 0,
            'messages_failed': 0,
            'channels_used': {}
        }
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("SlackIntegration initialized successfully")
        return True
    
    async def send_alert_notification(self, alert_data: Dict[str, Any], 
                                    channels: List[str] = None) -> bool:
        """Send alert notification to Slack"""
        
        try:
            message = self._format_alert_message(alert_data)
            
            return await self.send_message(
                message=message,
                channels=channels or [self.default_channel]
            )
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False
    
    async def send_resolution_notification(self, alert_data: Dict[str, Any],
                                         channels: List[str] = None) -> bool:
        """Send resolution notification to Slack"""
        
        try:
            message = self._format_resolution_message(alert_data)
            
            return await self.send_message(
                message=message,
                channels=channels or [self.default_channel]
            )
            
        except Exception as e:
            logger.error(f"Error sending Slack resolution: {e}")
            return False
    
    async def send_message(self, message: Dict[str, Any], 
                         channels: List[str]) -> bool:
        """Send message to Slack channels"""
        
        try:
            success_count = 0
            
            for channel in channels:
                try:
                    if self.bot_token:
                        success = await self._send_via_api(channel, message)
                    elif channel in self.webhook_urls:
                        success = await self._send_via_webhook(channel, message)
                    else:
                        logger.warning(f"No configuration for Slack channel: {channel}")
                        continue
                    
                    if success:
                        success_count += 1
                        self.statistics['messages_sent'] += 1
                        
                        # Update channel statistics
                        channel_count = self.statistics['channels_used'].get(channel, 0)
                        self.statistics['channels_used'][channel] = channel_count + 1
                    else:
                        self.statistics['messages_failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error sending to Slack channel {channel}: {e}")
                    self.statistics['messages_failed'] += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False
    
    async def _send_via_api(self, channel: str, message: Dict[str, Any]) -> bool:
        """Send message via Slack Web API"""
        
        try:
            url = "https://slack.com/api/chat.postMessage"
            
            headers = {
                'Authorization': f'Bearer {self.bot_token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'channel': channel,
                **message
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get('ok'):
                            logger.info(f"Slack message sent successfully to {channel}")
                            return True
                        else:
                            logger.error(f"Slack API error: {result.get('error', 'Unknown error')}")
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Slack API failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending via Slack API: {e}")
            return False
    
    async def _send_via_webhook(self, channel: str, message: Dict[str, Any]) -> bool:
        """Send message via Slack webhook"""
        
        try:
            webhook_url = self.webhook_urls.get(channel)
            
            if not webhook_url:
                logger.error(f"No webhook URL configured for channel: {channel}")
                return False
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    
                    if response.status == 200:
                        logger.info(f"Slack webhook message sent successfully to {channel}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Slack webhook failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending via Slack webhook: {e}")
            return False
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert data into Slack message"""
        
        try:
            severity = alert_data.get('severity', 'medium').upper()
            company_id = alert_data.get('company_id', 'Unknown')
            factor = alert_data.get('factor', 'Unknown')
            title = alert_data.get('title', 'Credit Alert')
            description = alert_data.get('description', 'No description available')
            current_value = alert_data.get('current_value', 'N/A')
            threshold_value = alert_data.get('threshold_value', 'N/A')
            
            # Color coding for severity
            color_map = {
                'LOW': '#28a745',
                'MEDIUM': '#ffc107',
                'HIGH': '#fd7e14',
                'CRITICAL': '#dc3545'
            }
            
            # Emoji for severity
            emoji_map = {
                'LOW': ':information_source:',
                'MEDIUM': ':warning:',
                'HIGH': ':exclamation:',
                'CRITICAL': ':rotating_light:'
            }
            
            message = {
                "text": f"{emoji_map.get(severity, ':warning:')} Credit Alert: {title}",
                "attachments": [
                    {
                        "color": color_map.get(severity, '#6c757d'),
                        "fields": [
                            {
                                "title": "Company",
                                "value": company_id,
                                "short": True
                            },
                            {
                                "title": "Severity",
                                "value": severity,
                                "short": True
                            },
                            {
                                "title": "Factor",
                                "value": factor,
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": str(current_value),
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": str(threshold_value),
                                "short": True
                            },
                            {
                                "title": "Description",
                                "value": description,
                                "short": False
                            }
                        ],
                        "footer": "Credit Intelligence Platform",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return {
                "text": "Credit Alert - Error formatting message",
                "attachments": []
            }
    
    def _format_resolution_message(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format resolution data into Slack message"""
        
        try:
            title = alert_data.get('title', 'Credit Alert')
            company_id = alert_data.get('company_id', 'Unknown')
            resolved_by = alert_data.get('resolved_by', 'System')
            resolved_at = alert_data.get('resolved_at', datetime.now())
            
            if isinstance(resolved_at, str):
                resolved_at = datetime.fromisoformat(resolved_at.replace('Z', '+00:00'))
            
            message = {
                "text": ":white_check_mark: Alert Resolved",
                "attachments": [
                    {
                        "color": "#28a745",
                        "fields": [
                            {
                                "title": "Alert",
                                "value": title,
                                "short": False
                            },
                            {
                                "title": "Company",
                                "value": company_id,
                                "short": True
                            },
                            {
                                "title": "Resolved By",
                                "value": resolved_by,
                                "short": True
                            },
                            {
                                "title": "Resolved At",
                                "value": resolved_at.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "Credit Intelligence Platform",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting resolution message: {e}")
            return {
                "text": "Alert Resolved - Error formatting message",
                "attachments": []
            }
    
    async def send_summary_message(self, summary_data: Dict[str, Any],
                                 channels: List[str] = None) -> bool:
        """Send daily summary to Slack"""
        
        try:
            message = self._format_summary_message(summary_data)
            
            return await self.send_message(
                message=message,
                channels=channels or [self.default_channel]
            )
            
        except Exception as e:
            logger.error(f"Error sending Slack summary: {e}")
            return False
    
    def _format_summary_message(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format summary data into Slack message"""
        
        try:
            date = summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            total_alerts = summary_data.get('total_alerts', 0)
            critical_alerts = summary_data.get('critical_alerts', 0)
            high_alerts = summary_data.get('high_alerts', 0)
            medium_alerts = summary_data.get('medium_alerts', 0)
            low_alerts = summary_data.get('low_alerts', 0)
            resolved_alerts = summary_data.get('resolved_alerts', 0)
            
            top_companies = summary_data.get('top_companies', [])
            
            # Format top companies
            companies_text = ""
            for i, company in enumerate(top_companies[:5], 1):
                companies_text += f"{i}. {company.get('company_id', 'Unknown')}: {company.get('alert_count', 0)} alerts\n"
            
            if not companies_text:
                companies_text = "No alerts today"
            
            message = {
                "text": f":chart_with_upwards_trend: Daily Alert Summary - {date}",
                "attachments": [
                    {
                        "color": "#36a64f",
                        "fields": [
                            {
                                "title": "Total Alerts",
                                "value": str(total_alerts),
                                "short": True
                            },
                            {
                                "title": "Resolved",
                                "value": str(resolved_alerts),
                                "short": True
                            },
                            {
                                "title": "Critical",
                                "value": str(critical_alerts),
                                "short": True
                            },
                            {
                                "title": "High",
                                "value": str(high_alerts),
                                "short": True
                            },
                            {
                                "title": "Medium",
                                "value": str(medium_alerts),
                                "short": True
                            },
                            {
                                "title": "Low",
                                "value": str(low_alerts),
                                "short": True
                            },
                            {
                                "title": "Top Companies by Alert Count",
                                "value": companies_text.strip(),
                                "short": False
                            }
                        ],
                        "footer": "Credit Intelligence Platform",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting summary message: {e}")
            return {
                "text": "Daily Summary - Error formatting message",
                "attachments": []
            }
    
    async def test_connection(self, channel: str = None) -> bool:
        """Test Slack connection"""
        
        try:
            test_channel = channel or self.default_channel
            
            test_message = {
                "text": ":white_check_mark: Test message from Credit Intelligence Platform",
                "attachments": [
                    {
                        "color": "#36a64f",
                        "fields": [
                            {
                                "title": "Status",
                                "value": "Connection successful",
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "Credit Intelligence Platform Test"
                    }
                ]
            }
            
            success = await self.send_message(test_message, [test_channel])
            
            if success:
                logger.info(f"Slack connection test successful for channel: {test_channel}")
            else:
                logger.error(f"Slack connection test failed for channel: {test_channel}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing Slack connection: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get Slack integration statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['messages_sent'] + stats['messages_failed']
            success_rate = (stats['messages_sent'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'configured_webhooks': len(self.webhook_urls),
                'has_bot_token': bool(self.bot_token),
                'default_channel': self.default_channel
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
