"""
Email notification service for Stage 5 alerting workflows.
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import json
import ssl

logger = logging.getLogger(__name__)

class EmailNotifier:
    """Email notification service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.smtp_config = config.get('smtp', {})
        self.templates = {}
        self.statistics = {
            'emails_sent': 0,
            'emails_failed': 0,
            'templates_used': {}
        }
        self._initialize_notifier()
    
    async def initialize(self):
        """Async initialize method required by pipeline"""
        logger.info("EmailNotifier initialized successfully")
        return True
    
    def _initialize_notifier(self):
        """Initialize email notifier"""
        
        # SMTP configuration
        self.smtp_server = self.smtp_config.get('server', 'smtp.gmail.com')
        self.smtp_port = self.smtp_config.get('port', 587)
        self.smtp_username = self.smtp_config.get('username', '')
        self.smtp_password = self.smtp_config.get('password', '')
        self.use_tls = self.smtp_config.get('use_tls', True)
        
        # Default sender
        self.default_sender = self.config.get('default_sender', 'alerts@credtech.com')
        
        # Load email templates
        self._load_templates()
    
    def _load_templates(self):
        """Load email templates"""
        
        self.templates = {
            'alert_created': {
                'subject': 'Credit Alert: {severity} - {title}',
                'html': '''
                <html>
                <body>
                    <h2 style="color: {color};">Credit Alert: {title}</h2>
                    <p><strong>Company:</strong> {company_id}</p>
                    <p><strong>Severity:</strong> {severity}</p>
                    <p><strong>Factor:</strong> {factor}</p>
                    <p><strong>Current Value:</strong> {current_value}</p>
                    <p><strong>Threshold:</strong> {threshold_value}</p>
                    <p><strong>Description:</strong> {description}</p>
                    <p><strong>Created:</strong> {created_at}</p>
                    <hr>
                    <p><em>This is an automated alert from the Credit Intelligence Platform.</em></p>
                </body>
                </html>
                ''',
                'text': '''
                Credit Alert: {title}
                
                Company: {company_id}
                Severity: {severity}
                Factor: {factor}
                Current Value: {current_value}
                Threshold: {threshold_value}
                Description: {description}
                Created: {created_at}
                
                This is an automated alert from the Credit Intelligence Platform.
                '''
            },
            'alert_resolved': {
                'subject': 'Alert Resolved: {title}',
                'html': '''
                <html>
                <body>
                    <h2 style="color: green;">Alert Resolved: {title}</h2>
                    <p><strong>Company:</strong> {company_id}</p>
                    <p><strong>Resolved By:</strong> {resolved_by}</p>
                    <p><strong>Resolved At:</strong> {resolved_at}</p>
                    <p><strong>Duration:</strong> {duration_minutes} minutes</p>
                    <hr>
                    <p><em>This is an automated notification from the Credit Intelligence Platform.</em></p>
                </body>
                </html>
                ''',
                'text': '''
                Alert Resolved: {title}
                
                Company: {company_id}
                Resolved By: {resolved_by}
                Resolved At: {resolved_at}
                Duration: {duration_minutes} minutes
                
                This is an automated notification from the Credit Intelligence Platform.
                '''
            },
            'daily_summary': {
                'subject': 'Daily Credit Alerts Summary - {date}',
                'html': '''
                <html>
                <body>
                    <h2>Daily Credit Alerts Summary</h2>
                    <p><strong>Date:</strong> {date}</p>
                    <p><strong>Total Alerts:</strong> {total_alerts}</p>
                    <p><strong>Critical:</strong> {critical_alerts}</p>
                    <p><strong>High:</strong> {high_alerts}</p>
                    <p><strong>Medium:</strong> {medium_alerts}</p>
                    <p><strong>Low:</strong> {low_alerts}</p>
                    <p><strong>Resolved:</strong> {resolved_alerts}</p>
                    
                    <h3>Top Companies by Alert Count:</h3>
                    <ul>
                    {top_companies}
                    </ul>
                    
                    <hr>
                    <p><em>This is an automated summary from the Credit Intelligence Platform.</em></p>
                </body>
                </html>
                ''',
                'text': '''
                Daily Credit Alerts Summary
                
                Date: {date}
                Total Alerts: {total_alerts}
                Critical: {critical_alerts}
                High: {high_alerts}
                Medium: {medium_alerts}
                Low: {low_alerts}
                Resolved: {resolved_alerts}
                
                Top Companies by Alert Count:
                {top_companies_text}
                
                This is an automated summary from the Credit Intelligence Platform.
                '''
            }
        }
    
    async def send_alert_notification(self, alert: Dict[str, Any]) -> bool:
        """Send alert notification email"""
        try:
            # Get alert template data
            template_data = {
                'title': alert.get('title', 'Credit Alert'),
                'company_id': alert.get('company_id', 'Unknown'),
                'severity': alert.get('severity', 'Medium'),
                'factor': alert.get('factor', 'Unknown'),
                'current_value': alert.get('current_value', 'N/A'),
                'threshold_value': alert.get('threshold_value', 'N/A'),
                'description': alert.get('description', 'No description available'),
                'created_at': alert.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'color': self._get_severity_color(alert.get('severity', 'Medium'))
            }
            
            # Get recipients from config or use default
            recipients = self.config.get('alert_recipients', ['admin@credtech.com'])
            
            # Send email
            success = await self.send_email(
                recipients=recipients,
                template='alert_created',
                template_data=template_data
            )
            
            if success:
                logger.info(f"Alert notification sent for {alert.get('company_id', 'Unknown')}")
            else:
                logger.error(f"Failed to send alert notification for {alert.get('company_id', 'Unknown')}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
            return False
    
    async def send_resolution_notification(self, alert_data: Dict[str, Any],
                                         recipients: List[str]) -> bool:
        """Send alert resolution notification"""
        
        try:
            template_data = self._prepare_resolution_template_data(alert_data)
            
            return await self.send_email(
                recipients=recipients,
                template='alert_resolved',
                template_data=template_data
            )
            
        except Exception as e:
            logger.error(f"Error sending resolution notification: {e}")
            return False
    
    async def send_daily_summary(self, summary_data: Dict[str, Any],
                               recipients: List[str]) -> bool:
        """Send daily summary email"""
        
        try:
            template_data = self._prepare_summary_template_data(summary_data)
            
            return await self.send_email(
                recipients=recipients,
                template='daily_summary',
                template_data=template_data
            )
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
    
    async def send_email(self, recipients: List[str], template: str,
                        template_data: Dict[str, Any], sender: str = None,
                        attachments: List[Dict[str, Any]] = None) -> bool:
        """Send email using template"""
        
        try:
            if template not in self.templates:
                logger.error(f"Template {template} not found")
                return False
            
            template_config = self.templates[template]
            
            # Format subject and content
            subject = template_config['subject'].format(**template_data)
            html_content = template_config['html'].format(**template_data)
            text_content = template_config['text'].format(**template_data)
            
            # Send email
            success = await self._send_smtp_email(
                recipients=recipients,
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                sender=sender or self.default_sender,
                attachments=attachments
            )
            
            # Update statistics
            if success:
                self.statistics['emails_sent'] += 1
                template_count = self.statistics['templates_used'].get(template, 0)
                self.statistics['templates_used'][template] = template_count + 1
            else:
                self.statistics['emails_failed'] += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            self.statistics['emails_failed'] += 1
            return False
    
    async def _send_smtp_email(self, recipients: List[str], subject: str,
                              html_content: str, text_content: str,
                              sender: str, attachments: List[Dict[str, Any]] = None) -> bool:
        """Send email via SMTP"""
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = sender
            msg['To'] = ', '.join(recipients)
            
            # Add text and HTML parts
            text_part = MIMEText(text_content, 'plain')
            html_part = MIMEText(html_content, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Add attachments if provided
            if attachments:
                for attachment in attachments:
                    await self._add_attachment(msg, attachment)
            
            # Send email
            await self._send_via_smtp(msg, recipients)
            
            logger.info(f"Email sent successfully to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMTP email: {e}")
            return False
    
    async def _add_attachment(self, msg: MIMEMultipart, attachment: Dict[str, Any]):
        """Add attachment to email"""
        
        try:
            filename = attachment.get('filename', 'attachment')
            content = attachment.get('content', b'')
            content_type = attachment.get('content_type', 'application/octet-stream')
            
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(content)
            encoders.encode_base64(part)
            
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            
            msg.attach(part)
            
        except Exception as e:
            logger.error(f"Error adding attachment: {e}")
    
    async def _send_via_smtp(self, msg: MIMEMultipart, recipients: List[str]):
        """Send message via SMTP server"""
        
        try:
            # Create SMTP connection
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls(context=ssl.create_default_context())
            else:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            
            # Login if credentials provided
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            
            # Send email
            server.send_message(msg, to_addrs=recipients)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending via SMTP: {e}")
            raise
    
    def _prepare_alert_template_data(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template data for alert notifications"""
        
        try:
            severity = alert_data.get('severity', 'medium').upper()
            
            # Color coding for severity
            color_map = {
                'LOW': '#28a745',
                'MEDIUM': '#ffc107',
                'HIGH': '#fd7e14',
                'CRITICAL': '#dc3545'
            }
            
            template_data = {
                'title': alert_data.get('title', 'Credit Alert'),
                'company_id': alert_data.get('company_id', 'Unknown'),
                'severity': severity,
                'color': color_map.get(severity, '#6c757d'),
                'factor': alert_data.get('factor', 'Unknown'),
                'current_value': alert_data.get('current_value', 'N/A'),
                'threshold_value': alert_data.get('threshold_value', 'N/A'),
                'description': alert_data.get('description', 'No description available'),
                'created_at': alert_data.get('created_at', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return template_data
            
        except Exception as e:
            logger.error(f"Error preparing alert template data: {e}")
            return {}
    
    def _prepare_resolution_template_data(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template data for resolution notifications"""
        
        try:
            created_at = alert_data.get('created_at', datetime.now())
            resolved_at = alert_data.get('resolved_at', datetime.now())
            
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            if isinstance(resolved_at, str):
                resolved_at = datetime.fromisoformat(resolved_at.replace('Z', '+00:00'))
            
            duration = (resolved_at - created_at).total_seconds() / 60
            
            template_data = {
                'title': alert_data.get('title', 'Credit Alert'),
                'company_id': alert_data.get('company_id', 'Unknown'),
                'resolved_by': alert_data.get('resolved_by', 'System'),
                'resolved_at': resolved_at.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_minutes': int(duration)
            }
            
            return template_data
            
        except Exception as e:
            logger.error(f"Error preparing resolution template data: {e}")
            return {}
    
    def _prepare_summary_template_data(self, summary_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template data for summary notifications"""
        
        try:
            top_companies = summary_data.get('top_companies', [])
            
            # Format top companies for HTML
            html_companies = []
            text_companies = []
            
            for company in top_companies[:5]:  # Top 5
                company_id = company.get('company_id', 'Unknown')
                alert_count = company.get('alert_count', 0)
                
                html_companies.append(f'<li>{company_id}: {alert_count} alerts</li>')
                text_companies.append(f'- {company_id}: {alert_count} alerts')
            
            template_data = {
                'date': summary_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                'total_alerts': summary_data.get('total_alerts', 0),
                'critical_alerts': summary_data.get('critical_alerts', 0),
                'high_alerts': summary_data.get('high_alerts', 0),
                'medium_alerts': summary_data.get('medium_alerts', 0),
                'low_alerts': summary_data.get('low_alerts', 0),
                'resolved_alerts': summary_data.get('resolved_alerts', 0),
                'top_companies': '\n'.join(html_companies),
                'top_companies_text': '\n'.join(text_companies)
            }
            
            return template_data
            
        except Exception as e:
            logger.error(f"Error preparing summary template data: {e}")
            return {}
    
    async def add_template(self, template_id: str, subject: str, 
                          html_content: str, text_content: str):
        """Add custom email template"""
        
        try:
            self.templates[template_id] = {
                'subject': subject,
                'html': html_content,
                'text': text_content
            }
            
            logger.info(f"Added email template: {template_id}")
            
        except Exception as e:
            logger.error(f"Error adding template: {e}")
    
    async def test_connection(self) -> bool:
        """Test SMTP connection"""
        
        try:
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls(context=ssl.create_default_context())
            else:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            
            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)
            
            server.quit()
            
            logger.info("SMTP connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"SMTP connection test failed: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get email notification statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['emails_sent'] + stats['emails_failed']
            success_rate = (stats['emails_sent'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'available_templates': list(self.templates.keys()),
                'smtp_server': self.smtp_server,
                'smtp_port': self.smtp_port
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}

    async def process_queue(self):
        """Process email notification queue - placeholder for compatibility"""
        # This method is called by AlertingEngine but EmailNotifier doesn't use a queue
        # All emails are sent immediately when send_email is called
        logger.debug("EmailNotifier process_queue called - no queue processing needed")
        return True
