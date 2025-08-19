"""
Webhook notification service for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import aiohttp
from datetime import datetime
import json
import hashlib
import hmac

logger = logging.getLogger(__name__)

class WebhookNotifier:
    """Webhook notification service for external integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.webhooks = {}
        self.statistics = {
            'webhooks_sent': 0,
            'webhooks_failed': 0,
            'webhooks_by_endpoint': {}
        }
        self._initialize_webhooks()
    
    def _initialize_webhooks(self):
        """Initialize webhook configurations"""
        
        # Load webhook endpoints from config
        webhook_configs = self.config.get('webhooks', {})
        
        for webhook_id, webhook_config in webhook_configs.items():
            self.webhooks[webhook_id] = {
                'url': webhook_config['url'],
                'method': webhook_config.get('method', 'POST'),
                'headers': webhook_config.get('headers', {}),
                'secret': webhook_config.get('secret'),
                'timeout': webhook_config.get('timeout', 30),
                'retry_count': webhook_config.get('retry_count', 3),
                'retry_delay': webhook_config.get('retry_delay', 5),
                'enabled': webhook_config.get('enabled', True)
            }
    
    async def send_alert_webhook(self, alert_data: Dict[str, Any], 
                               webhook_ids: List[str] = None) -> bool:
        """Send alert notification via webhook"""
        
        try:
            payload = self._format_alert_payload(alert_data)
            
            return await self.send_webhook(
                payload=payload,
                event_type='alert_created',
                webhook_ids=webhook_ids
            )
            
        except Exception as e:
            logger.error(f"Error sending alert webhook: {e}")
            return False
    
    async def send_resolution_webhook(self, alert_data: Dict[str, Any],
                                    webhook_ids: List[str] = None) -> bool:
        """Send alert resolution webhook"""
        
        try:
            payload = self._format_resolution_payload(alert_data)
            
            return await self.send_webhook(
                payload=payload,
                event_type='alert_resolved',
                webhook_ids=webhook_ids
            )
            
        except Exception as e:
            logger.error(f"Error sending resolution webhook: {e}")
            return False
    
    async def send_webhook(self, payload: Dict[str, Any], event_type: str,
                         webhook_ids: List[str] = None) -> bool:
        """Send webhook to specified endpoints"""
        
        try:
            target_webhooks = webhook_ids or list(self.webhooks.keys())
            success_count = 0
            
            for webhook_id in target_webhooks:
                if webhook_id not in self.webhooks:
                    logger.warning(f"Webhook {webhook_id} not found")
                    continue
                
                webhook_config = self.webhooks[webhook_id]
                
                if not webhook_config['enabled']:
                    logger.info(f"Webhook {webhook_id} is disabled")
                    continue
                
                success = await self._send_single_webhook(
                    webhook_id, webhook_config, payload, event_type
                )
                
                if success:
                    success_count += 1
                    self.statistics['webhooks_sent'] += 1
                else:
                    self.statistics['webhooks_failed'] += 1
                
                # Update endpoint statistics
                endpoint_stats = self.statistics['webhooks_by_endpoint'].get(webhook_id, {'sent': 0, 'failed': 0})
                if success:
                    endpoint_stats['sent'] += 1
                else:
                    endpoint_stats['failed'] += 1
                self.statistics['webhooks_by_endpoint'][webhook_id] = endpoint_stats
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            return False
    
    async def _send_single_webhook(self, webhook_id: str, webhook_config: Dict[str, Any],
                                 payload: Dict[str, Any], event_type: str) -> bool:
        """Send webhook to single endpoint with retry logic"""
        
        try:
            # Prepare webhook payload
            webhook_payload = {
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'webhook_id': webhook_id,
                'data': payload
            }
            
            # Prepare headers
            headers = webhook_config['headers'].copy()
            headers['Content-Type'] = 'application/json'
            headers['User-Agent'] = 'CredTech-Webhook/1.0'
            
            # Add signature if secret is configured
            if webhook_config.get('secret'):
                signature = self._generate_signature(webhook_payload, webhook_config['secret'])
                headers['X-Webhook-Signature'] = signature
            
            # Retry logic
            for attempt in range(webhook_config['retry_count']):
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=webhook_config['timeout'])) as session:
                        async with session.request(
                            webhook_config['method'],
                            webhook_config['url'],
                            json=webhook_payload,
                            headers=headers
                        ) as response:
                            
                            if 200 <= response.status < 300:
                                logger.info(f"Webhook {webhook_id} sent successfully (attempt {attempt + 1})")
                                return True
                            else:
                                error_text = await response.text()
                                logger.warning(f"Webhook {webhook_id} failed: {response.status} - {error_text} (attempt {attempt + 1})")
                                
                                if attempt < webhook_config['retry_count'] - 1:
                                    await asyncio.sleep(webhook_config['retry_delay'])
                
                except asyncio.TimeoutError:
                    logger.warning(f"Webhook {webhook_id} timeout (attempt {attempt + 1})")
                    if attempt < webhook_config['retry_count'] - 1:
                        await asyncio.sleep(webhook_config['retry_delay'])
                
                except Exception as e:
                    logger.warning(f"Webhook {webhook_id} error: {e} (attempt {attempt + 1})")
                    if attempt < webhook_config['retry_count'] - 1:
                        await asyncio.sleep(webhook_config['retry_delay'])
            
            logger.error(f"Webhook {webhook_id} failed after {webhook_config['retry_count']} attempts")
            return False
            
        except Exception as e:
            logger.error(f"Error sending single webhook: {e}")
            return False
    
    def _generate_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        
        try:
            payload_string = json.dumps(payload, sort_keys=True, separators=(',', ':'))
            signature = hmac.new(
                secret.encode('utf-8'),
                payload_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return f"sha256={signature}"
            
        except Exception as e:
            logger.error(f"Error generating signature: {e}")
            return ""
    
    def _format_alert_payload(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert data for webhook payload"""
        
        try:
            return {
                'alert': {
                    'id': alert_data.get('id'),
                    'title': alert_data.get('title'),
                    'description': alert_data.get('description'),
                    'severity': alert_data.get('severity'),
                    'status': alert_data.get('status'),
                    'company_id': alert_data.get('company_id'),
                    'factor': alert_data.get('factor'),
                    'current_value': alert_data.get('current_value'),
                    'threshold_value': alert_data.get('threshold_value'),
                    'created_at': alert_data.get('created_at', datetime.now()).isoformat() if isinstance(alert_data.get('created_at'), datetime) else alert_data.get('created_at'),
                    'tags': alert_data.get('tags', []),
                    'metadata': alert_data.get('metadata', {})
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting alert payload: {e}")
            return {}
    
    def _format_resolution_payload(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format resolution data for webhook payload"""
        
        try:
            return {
                'alert': {
                    'id': alert_data.get('id'),
                    'title': alert_data.get('title'),
                    'company_id': alert_data.get('company_id'),
                    'resolved_by': alert_data.get('resolved_by'),
                    'resolved_at': alert_data.get('resolved_at', datetime.now()).isoformat() if isinstance(alert_data.get('resolved_at'), datetime) else alert_data.get('resolved_at'),
                    'resolution_time_minutes': alert_data.get('resolution_time_minutes', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting resolution payload: {e}")
            return {}
    
    async def add_webhook(self, webhook_id: str, url: str, method: str = 'POST',
                        headers: Dict[str, str] = None, secret: str = None,
                        timeout: int = 30, retry_count: int = 3) -> bool:
        """Add new webhook endpoint"""
        
        try:
            self.webhooks[webhook_id] = {
                'url': url,
                'method': method,
                'headers': headers or {},
                'secret': secret,
                'timeout': timeout,
                'retry_count': retry_count,
                'retry_delay': 5,
                'enabled': True
            }
            
            logger.info(f"Added webhook: {webhook_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding webhook: {e}")
            return False
    
    async def remove_webhook(self, webhook_id: str) -> bool:
        """Remove webhook endpoint"""
        
        try:
            if webhook_id in self.webhooks:
                del self.webhooks[webhook_id]
                logger.info(f"Removed webhook: {webhook_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing webhook: {e}")
            return False
    
    async def enable_webhook(self, webhook_id: str) -> bool:
        """Enable webhook endpoint"""
        
        try:
            if webhook_id in self.webhooks:
                self.webhooks[webhook_id]['enabled'] = True
                logger.info(f"Enabled webhook: {webhook_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error enabling webhook: {e}")
            return False
    
    async def disable_webhook(self, webhook_id: str) -> bool:
        """Disable webhook endpoint"""
        
        try:
            if webhook_id in self.webhooks:
                self.webhooks[webhook_id]['enabled'] = False
                logger.info(f"Disabled webhook: {webhook_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error disabling webhook: {e}")
            return False
    
    async def test_webhook(self, webhook_id: str) -> bool:
        """Test webhook endpoint"""
        
        try:
            if webhook_id not in self.webhooks:
                logger.error(f"Webhook {webhook_id} not found")
                return False
            
            test_payload = {
                'test': True,
                'message': 'Test webhook from Credit Intelligence Platform',
                'timestamp': datetime.now().isoformat()
            }
            
            success = await self.send_webhook(
                payload=test_payload,
                event_type='test',
                webhook_ids=[webhook_id]
            )
            
            if success:
                logger.info(f"Webhook {webhook_id} test successful")
            else:
                logger.error(f"Webhook {webhook_id} test failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing webhook: {e}")
            return False
    
    async def get_webhooks(self) -> Dict[str, Dict[str, Any]]:
        """Get all webhook configurations (without secrets)"""
        
        try:
            safe_webhooks = {}
            
            for webhook_id, config in self.webhooks.items():
                safe_config = config.copy()
                # Remove secret from response for security
                if 'secret' in safe_config:
                    safe_config['secret'] = '***' if safe_config['secret'] else None
                
                safe_webhooks[webhook_id] = safe_config
            
            return safe_webhooks
            
        except Exception as e:
            logger.error(f"Error getting webhooks: {e}")
            return {}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get webhook notification statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['webhooks_sent'] + stats['webhooks_failed']
            success_rate = (stats['webhooks_sent'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'configured_webhooks': len(self.webhooks),
                'enabled_webhooks': sum(1 for w in self.webhooks.values() if w['enabled'])
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
