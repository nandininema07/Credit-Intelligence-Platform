"""
Jira integration for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
import base64
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class JiraIntegration:
    """Jira integration for creating tickets from alerts"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get('base_url', '').rstrip('/')
        self.username = config.get('username', '')
        self.api_token = config.get('api_token', '')
        self.project_key = config.get('project_key', 'CRED')
        self.default_issue_type = config.get('default_issue_type', 'Task')
        self.default_priority = config.get('default_priority', 'Medium')
        self.statistics = {
            'tickets_created': 0,
            'tickets_updated': 0,
            'tickets_failed': 0,
            'projects_used': {}
        }
        
        # Create auth header
        self.auth_header = self._create_auth_header()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("JiraIntegration initialized successfully")
        return True
    
    def _create_auth_header(self) -> str:
        """Create basic auth header for Jira API"""
        if not self.username or not self.api_token:
            return ""
        
        credentials = f"{self.username}:{self.api_token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"
    
    async def create_ticket_from_alert(self, alert_data: Dict[str, Any]) -> Optional[str]:
        """Create Jira ticket from alert data"""
        
        try:
            if not self.auth_header:
                logger.error("Jira authentication not configured")
                return None
            
            ticket_data = self._format_alert_ticket(alert_data)
            
            url = f"{self.base_url}/rest/api/3/issue"
            headers = {
                'Authorization': self.auth_header,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=ticket_data, headers=headers) as response:
                    
                    if response.status == 201:
                        result = await response.json()
                        ticket_key = result.get('key')
                        
                        logger.info(f"Created Jira ticket: {ticket_key}")
                        self.statistics['tickets_created'] += 1
                        
                        # Update project statistics
                        project_count = self.statistics['projects_used'].get(self.project_key, 0)
                        self.statistics['projects_used'][self.project_key] = project_count + 1
                        
                        return ticket_key
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to create Jira ticket: {response.status} - {error_text}")
                        self.statistics['tickets_failed'] += 1
                        return None
                        
        except Exception as e:
            logger.error(f"Error creating Jira ticket: {e}")
            self.statistics['tickets_failed'] += 1
            return None
    
    def _format_alert_ticket(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert data into Jira ticket"""
        
        try:
            severity = alert_data.get('severity', 'medium').upper()
            company_id = alert_data.get('company_id', 'Unknown')
            factor = alert_data.get('factor', 'Unknown')
            title = alert_data.get('title', 'Credit Alert')
            description = alert_data.get('description', 'No description available')
            current_value = alert_data.get('current_value', 'N/A')
            threshold_value = alert_data.get('threshold_value', 'N/A')
            
            # Map severity to Jira priority
            priority_map = {
                'LOW': 'Low',
                'MEDIUM': 'Medium',
                'HIGH': 'High',
                'CRITICAL': 'Highest'
            }
            
            # Create detailed description
            detailed_description = {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "heading",
                        "attrs": {"level": 2},
                        "content": [{"type": "text", "text": "Alert Details"}]
                    },
                    {
                        "type": "table",
                        "attrs": {"isNumberColumnEnabled": False, "layout": "default"},
                        "content": [
                            {
                                "type": "tableRow",
                                "content": [
                                    {"type": "tableHeader", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Field"}]}]},
                                    {"type": "tableHeader", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Value"}]}]}
                                ]
                            },
                            {
                                "type": "tableRow",
                                "content": [
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Company ID"}]}]},
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": str(company_id)}]}]}
                                ]
                            },
                            {
                                "type": "tableRow",
                                "content": [
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Factor"}]}]},
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": str(factor)}]}]}
                                ]
                            },
                            {
                                "type": "tableRow",
                                "content": [
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Current Value"}]}]},
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": str(current_value)}]}]}
                                ]
                            },
                            {
                                "type": "tableRow",
                                "content": [
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Threshold"}]}]},
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": str(threshold_value)}]}]}
                                ]
                            },
                            {
                                "type": "tableRow",
                                "content": [
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Severity"}]}]},
                                    {"type": "tableCell", "content": [{"type": "paragraph", "content": [{"type": "text", "text": severity}]}]}
                                ]
                            }
                        ]
                    },
                    {
                        "type": "heading",
                        "attrs": {"level": 2},
                        "content": [{"type": "text", "text": "Description"}]
                    },
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    },
                    {
                        "type": "heading",
                        "attrs": {"level": 2},
                        "content": [{"type": "text", "text": "Recommended Actions"}]
                    },
                    {
                        "type": "bulletList",
                        "content": [
                            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Review company financial data"}]}]},
                            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Analyze trend patterns"}]}]},
                            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Consider risk mitigation strategies"}]}]},
                            {"type": "listItem", "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Update credit assessment if necessary"}]}]}
                        ]
                    }
                ]
            }
            
            ticket_data = {
                "fields": {
                    "project": {
                        "key": self.project_key
                    },
                    "summary": f"[{severity}] {title} - {company_id}",
                    "description": detailed_description,
                    "issuetype": {
                        "name": self.default_issue_type
                    },
                    "priority": {
                        "name": priority_map.get(severity, self.default_priority)
                    },
                    "labels": [
                        "credit-alert",
                        f"severity-{severity.lower()}",
                        f"factor-{factor.lower().replace(' ', '-')}",
                        "automated"
                    ]
                }
            }
            
            # Add custom fields if configured
            custom_fields = self.config.get('custom_fields', {})
            for field_id, field_value in custom_fields.items():
                ticket_data["fields"][field_id] = field_value
            
            return ticket_data
            
        except Exception as e:
            logger.error(f"Error formatting Jira ticket: {e}")
            return {
                "fields": {
                    "project": {"key": self.project_key},
                    "summary": f"Credit Alert - {company_id}",
                    "description": {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": [{"type": "text", "text": "Error formatting alert data"}]}]},
                    "issuetype": {"name": self.default_issue_type}
                }
            }
    
    async def update_ticket(self, ticket_key: str, update_data: Dict[str, Any]) -> bool:
        """Update existing Jira ticket"""
        
        try:
            if not self.auth_header:
                logger.error("Jira authentication not configured")
                return False
            
            url = f"{self.base_url}/rest/api/3/issue/{ticket_key}"
            headers = {
                'Authorization': self.auth_header,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.put(url, json=update_data, headers=headers) as response:
                    
                    if response.status == 204:
                        logger.info(f"Updated Jira ticket: {ticket_key}")
                        self.statistics['tickets_updated'] += 1
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to update Jira ticket: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error updating Jira ticket: {e}")
            return False
    
    async def add_comment(self, ticket_key: str, comment: str) -> bool:
        """Add comment to Jira ticket"""
        
        try:
            if not self.auth_header:
                logger.error("Jira authentication not configured")
                return False
            
            comment_data = {
                "body": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {
                                    "type": "text",
                                    "text": comment
                                }
                            ]
                        }
                    ]
                }
            }
            
            url = f"{self.base_url}/rest/api/3/issue/{ticket_key}/comment"
            headers = {
                'Authorization': self.auth_header,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=comment_data, headers=headers) as response:
                    
                    if response.status == 201:
                        logger.info(f"Added comment to Jira ticket: {ticket_key}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to add comment: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error adding comment to Jira ticket: {e}")
            return False
    
    async def resolve_ticket(self, ticket_key: str, resolution: str = "Done") -> bool:
        """Resolve Jira ticket"""
        
        try:
            # Get available transitions
            transitions = await self.get_ticket_transitions(ticket_key)
            
            if not transitions:
                logger.error(f"Could not get transitions for ticket: {ticket_key}")
                return False
            
            # Find resolve transition
            resolve_transition = None
            for transition in transitions:
                if 'resolve' in transition.get('name', '').lower() or 'done' in transition.get('name', '').lower():
                    resolve_transition = transition
                    break
            
            if not resolve_transition:
                logger.warning(f"No resolve transition found for ticket: {ticket_key}")
                return False
            
            # Execute transition
            transition_data = {
                "transition": {
                    "id": resolve_transition['id']
                },
                "fields": {
                    "resolution": {
                        "name": resolution
                    }
                }
            }
            
            url = f"{self.base_url}/rest/api/3/issue/{ticket_key}/transitions"
            headers = {
                'Authorization': self.auth_header,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=transition_data, headers=headers) as response:
                    
                    if response.status == 204:
                        logger.info(f"Resolved Jira ticket: {ticket_key}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to resolve ticket: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error resolving Jira ticket: {e}")
            return False
    
    async def get_ticket_transitions(self, ticket_key: str) -> List[Dict[str, Any]]:
        """Get available transitions for ticket"""
        
        try:
            if not self.auth_header:
                return []
            
            url = f"{self.base_url}/rest/api/3/issue/{ticket_key}/transitions"
            headers = {
                'Authorization': self.auth_header,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result.get('transitions', [])
                    else:
                        logger.error(f"Failed to get transitions: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error getting ticket transitions: {e}")
            return []
    
    async def test_connection(self) -> bool:
        """Test Jira connection"""
        
        try:
            if not self.auth_header:
                logger.error("Jira authentication not configured")
                return False
            
            url = f"{self.base_url}/rest/api/3/myself"
            headers = {
                'Authorization': self.auth_header,
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    
                    if response.status == 200:
                        user_info = await response.json()
                        logger.info(f"Jira connection successful. User: {user_info.get('displayName', 'Unknown')}")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Jira connection failed: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error testing Jira connection: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get Jira integration statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['tickets_created'] + stats['tickets_failed']
            success_rate = (stats['tickets_created'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'base_url': self.base_url,
                'project_key': self.project_key,
                'default_issue_type': self.default_issue_type,
                'authentication_configured': bool(self.auth_header)
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
