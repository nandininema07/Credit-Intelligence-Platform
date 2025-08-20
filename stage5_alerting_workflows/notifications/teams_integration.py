"""
Microsoft Teams integration for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TeamsIntegration:
    """Microsoft Teams integration for alert notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhook_urls = config.get('webhook_urls', {})
        self.default_channel = config.get('default_channel', 'general')
        self.statistics = {
            'messages_sent': 0,
            'messages_failed': 0,
            'channels_used': {}
        }
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("TeamsIntegration initialized successfully")
        return True
    
    async def send_alert_notification(self, alert_data: Dict[str, Any], 
                                    channels: List[str] = None) -> bool:
        """Send alert notification to Teams"""
        
        try:
            message = self._format_alert_message(alert_data)
            
            return await self.send_message(
                message=message,
                channels=channels or [self.default_channel]
            )
            
        except Exception as e:
            logger.error(f"Error sending Teams alert: {e}")
            return False
    
    async def send_resolution_notification(self, alert_data: Dict[str, Any],
                                         channels: List[str] = None) -> bool:
        """Send resolution notification to Teams"""
        
        try:
            message = self._format_resolution_message(alert_data)
            
            return await self.send_message(
                message=message,
                channels=channels or [self.default_channel]
            )
            
        except Exception as e:
            logger.error(f"Error sending Teams resolution: {e}")
            return False
    
    async def send_message(self, message: Dict[str, Any], 
                         channels: List[str]) -> bool:
        """Send message to Teams channels"""
        
        try:
            success_count = 0
            
            for channel in channels:
                try:
                    if channel not in self.webhook_urls:
                        logger.warning(f"No webhook URL configured for Teams channel: {channel}")
                        continue
                    
                    success = await self._send_via_webhook(channel, message)
                    
                    if success:
                        success_count += 1
                        self.statistics['messages_sent'] += 1
                        
                        # Update channel statistics
                        channel_count = self.statistics['channels_used'].get(channel, 0)
                        self.statistics['channels_used'][channel] = channel_count + 1
                    else:
                        self.statistics['messages_failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error sending to Teams channel {channel}: {e}")
                    self.statistics['messages_failed'] += 1
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending Teams message: {e}")
            return False
    
    async def _send_via_webhook(self, channel: str, message: Dict[str, Any]) -> bool:
        """Send message via Teams webhook"""
        
        try:
            webhook_url = self.webhook_urls.get(channel)
            
            if not webhook_url:
                logger.error(f"No webhook URL configured for channel: {channel}")
                return False
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message, headers=headers) as response:
                    
                    if response.status == 200:
                        logger.info(f"Teams message sent successfully to {channel}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Teams webhook failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending via Teams webhook: {e}")
            return False
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert data into Teams message"""
        
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
                'LOW': 'Good',
                'MEDIUM': 'Warning',
                'HIGH': 'Attention',
                'CRITICAL': 'Attention'
            }
            
            message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": f"Credit Alert: {title}",
                "themeColor": self._get_theme_color(severity),
                "sections": [
                    {
                        "activityTitle": f"🚨 Credit Alert: {title}",
                        "activitySubtitle": f"Severity: {severity}",
                        "facts": [
                            {
                                "name": "Company",
                                "value": company_id
                            },
                            {
                                "name": "Factor",
                                "value": factor
                            },
                            {
                                "name": "Current Value",
                                "value": str(current_value)
                            },
                            {
                                "name": "Threshold",
                                "value": str(threshold_value)
                            },
                            {
                                "name": "Description",
                                "value": description
                            },
                            {
                                "name": "Created",
                                "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                        ],
                        "markdown": True
                    }
                ],
                "potentialAction": [
                    {
                        "@type": "OpenUri",
                        "name": "View Alert Details",
                        "targets": [
                            {
                                "os": "default",
                                "uri": f"https://credtech.platform/alerts/{alert_data.get('id', '')}"
                            }
                        ]
                    }
                ]
            }
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting alert message: {e}")
            return {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Credit Alert - Error formatting message",
                "text": "There was an error formatting the alert message."
            }
    
    def _format_resolution_message(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format resolution data into Teams message"""
        
        try:
            title = alert_data.get('title', 'Credit Alert')
            company_id = alert_data.get('company_id', 'Unknown')
            resolved_by = alert_data.get('resolved_by', 'System')
            resolved_at = alert_data.get('resolved_at', datetime.now())
            
            if isinstance(resolved_at, str):
                resolved_at = datetime.fromisoformat(resolved_at.replace('Z', '+00:00'))
            
            message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Alert Resolved",
                "themeColor": "28a745",
                "sections": [
                    {
                        "activityTitle": "✅ Alert Resolved",
                        "activitySubtitle": title,
                        "facts": [
                            {
                                "name": "Company",
                                "value": company_id
                            },
                            {
                                "name": "Resolved By",
                                "value": resolved_by
                            },
                            {
                                "name": "Resolved At",
                                "value": resolved_at.strftime('%Y-%m-%d %H:%M:%S')
                            }
                        ],
                        "markdown": True
                    }
                ]
            }
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting resolution message: {e}")
            return {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Alert Resolved - Error formatting message",
                "text": "There was an error formatting the resolution message."
            }
    
    def _get_theme_color(self, severity: str) -> str:
        """Get theme color for severity"""
        
        color_map = {
            'LOW': '28a745',
            'MEDIUM': 'ffc107',
            'HIGH': 'fd7e14',
            'CRITICAL': 'dc3545'
        }
        
        return color_map.get(severity, '6c757d')
    
    async def send_summary_message(self, summary_data: Dict[str, Any],
                                 channels: List[str] = None) -> bool:
        """Send daily summary to Teams"""
        
        try:
            message = self._format_summary_message(summary_data)
            
            return await self.send_message(
                message=message,
                channels=channels or [self.default_channel]
            )
            
        except Exception as e:
            logger.error(f"Error sending Teams summary: {e}")
            return False
    
    def _format_summary_message(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format summary data into Teams message"""
        
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
            companies_facts = []
            for i, company in enumerate(top_companies[:5], 1):
                companies_facts.append({
                    "name": f"#{i}",
                    "value": f"{company.get('company_id', 'Unknown')}: {company.get('alert_count', 0)} alerts"
                })
            
            if not companies_facts:
                companies_facts = [{"name": "Status", "value": "No alerts today"}]
            
            message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": f"Daily Alert Summary - {date}",
                "themeColor": "36a64f",
                "sections": [
                    {
                        "activityTitle": f"📊 Daily Alert Summary - {date}",
                        "facts": [
                            {
                                "name": "Total Alerts",
                                "value": str(total_alerts)
                            },
                            {
                                "name": "Resolved",
                                "value": str(resolved_alerts)
                            },
                            {
                                "name": "Critical",
                                "value": str(critical_alerts)
                            },
                            {
                                "name": "High",
                                "value": str(high_alerts)
                            },
                            {
                                "name": "Medium",
                                "value": str(medium_alerts)
                            },
                            {
                                "name": "Low",
                                "value": str(low_alerts)
                            }
                        ],
                        "markdown": True
                    },
                    {
                        "activityTitle": "🏢 Top Companies by Alert Count",
                        "facts": companies_facts,
                        "markdown": True
                    }
                ]
            }
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting summary message: {e}")
            return {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Daily Summary - Error formatting message",
                "text": "There was an error formatting the summary message."
            }
    
    async def test_connection(self, channel: str = None) -> bool:
        """Test Teams connection"""
        
        try:
            test_channel = channel or self.default_channel
            
            test_message = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "summary": "Test message from Credit Intelligence Platform",
                "themeColor": "36a64f",
                "sections": [
                    {
                        "activityTitle": "✅ Connection Test",
                        "activitySubtitle": "Credit Intelligence Platform",
                        "facts": [
                            {
                                "name": "Status",
                                "value": "Connection successful"
                            },
                            {
                                "name": "Timestamp",
                                "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                        ],
                        "markdown": True
                    }
                ]
            }
            
            success = await self.send_message(test_message, [test_channel])
            
            if success:
                logger.info(f"Teams connection test successful for channel: {test_channel}")
            else:
                logger.error(f"Teams connection test failed for channel: {test_channel}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing Teams connection: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get Teams integration statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['messages_sent'] + stats['messages_failed']
            success_rate = (stats['messages_sent'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'configured_webhooks': len(self.webhook_urls),
                'default_channel': self.default_channel
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
