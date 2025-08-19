"""
PDF report generator for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta
import json
import os
from io import BytesIO
import base64

# For PDF generation - using reportlab
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.piecharts import Pie
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)

class PDFGenerator:
    """PDF report generator for credit intelligence reports"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_directory = config.get('output_directory', './reports')
        self.company_logo = config.get('company_logo', '')
        self.statistics = {
            'reports_generated': 0,
            'reports_failed': 0,
            'total_pages_generated': 0,
            'report_types': {}
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. PDF generation will be limited.")
    
    async def generate_alert_report(self, alert_data: Dict[str, Any]) -> Optional[str]:
        """Generate PDF report for a single alert"""
        
        try:
            if not REPORTLAB_AVAILABLE:
                return await self._generate_simple_report(alert_data, 'alert')
            
            company_id = alert_data.get('company_id', 'Unknown')
            alert_id = alert_data.get('id', 'unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"alert_report_{company_id}_{alert_id}_{timestamp}.pdf"
            filepath = os.path.join(self.output_directory, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            story.append(Paragraph(f"Credit Alert Report", title_style))
            story.append(Spacer(1, 12))
            
            # Alert summary table
            alert_summary = [
                ['Field', 'Value'],
                ['Alert ID', str(alert_data.get('id', 'N/A'))],
                ['Company ID', str(alert_data.get('company_id', 'N/A'))],
                ['Severity', str(alert_data.get('severity', 'N/A')).upper()],
                ['Factor', str(alert_data.get('factor', 'N/A'))],
                ['Current Value', str(alert_data.get('current_value', 'N/A'))],
                ['Threshold Value', str(alert_data.get('threshold_value', 'N/A'))],
                ['Created At', str(alert_data.get('created_at', 'N/A'))],
                ['Status', str(alert_data.get('status', 'N/A'))]
            ]
            
            alert_table = Table(alert_summary, colWidths=[2*inch, 3*inch])
            alert_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Alert Details", styles['Heading2']))
            story.append(alert_table)
            story.append(Spacer(1, 20))
            
            # Description
            description = alert_data.get('description', 'No description available')
            story.append(Paragraph("Description", styles['Heading2']))
            story.append(Paragraph(description, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Recommended Actions", styles['Heading2']))
            recommendations = [
                "• Review the company's recent financial statements",
                "• Analyze market conditions affecting the company",
                "• Consider adjusting credit limits or terms",
                "• Monitor for additional risk factors",
                "• Schedule follow-up assessment"
            ]
            
            for rec in recommendations:
                story.append(Paragraph(rec, styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Footer
            footer_text = f"Generated by Credit Intelligence Platform on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.statistics['reports_generated'] += 1
            self.statistics['total_pages_generated'] += 1
            
            # Update report type statistics
            report_count = self.statistics['report_types'].get('alert', 0)
            self.statistics['report_types']['alert'] = report_count + 1
            
            logger.info(f"Generated alert report: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating alert report: {e}")
            self.statistics['reports_failed'] += 1
            return None
    
    async def generate_summary_report(self, summary_data: Dict[str, Any]) -> Optional[str]:
        """Generate PDF summary report"""
        
        try:
            if not REPORTLAB_AVAILABLE:
                return await self._generate_simple_report(summary_data, 'summary')
            
            date = summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"summary_report_{date}_{timestamp}.pdf"
            filepath = os.path.join(self.output_directory, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            story.append(Paragraph(f"Daily Credit Alert Summary - {date}", title_style))
            story.append(Spacer(1, 12))
            
            # Executive Summary
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            
            total_alerts = summary_data.get('total_alerts', 0)
            critical_alerts = summary_data.get('critical_alerts', 0)
            resolved_alerts = summary_data.get('resolved_alerts', 0)
            
            exec_summary = f"""
            On {date}, the Credit Intelligence Platform processed {total_alerts} alerts across all monitored companies.
            Of these, {critical_alerts} were classified as critical severity requiring immediate attention.
            {resolved_alerts} alerts were successfully resolved during the reporting period.
            """
            
            story.append(Paragraph(exec_summary, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Alert Statistics Table
            alert_stats = [
                ['Severity', 'Count', 'Percentage'],
                ['Critical', str(summary_data.get('critical_alerts', 0)), f"{(summary_data.get('critical_alerts', 0) / max(total_alerts, 1) * 100):.1f}%"],
                ['High', str(summary_data.get('high_alerts', 0)), f"{(summary_data.get('high_alerts', 0) / max(total_alerts, 1) * 100):.1f}%"],
                ['Medium', str(summary_data.get('medium_alerts', 0)), f"{(summary_data.get('medium_alerts', 0) / max(total_alerts, 1) * 100):.1f}%"],
                ['Low', str(summary_data.get('low_alerts', 0)), f"{(summary_data.get('low_alerts', 0) / max(total_alerts, 1) * 100):.1f}%"],
                ['Total', str(total_alerts), '100.0%']
            ]
            
            stats_table = Table(alert_stats, colWidths=[2*inch, 1*inch, 1*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(Paragraph("Alert Statistics", styles['Heading2']))
            story.append(stats_table)
            story.append(Spacer(1, 20))
            
            # Top Companies
            top_companies = summary_data.get('top_companies', [])
            if top_companies:
                story.append(Paragraph("Top Companies by Alert Count", styles['Heading2']))
                
                company_data = [['Rank', 'Company ID', 'Alert Count', 'Primary Risk Factor']]
                for i, company in enumerate(top_companies[:10], 1):
                    company_data.append([
                        str(i),
                        str(company.get('company_id', 'Unknown')),
                        str(company.get('alert_count', 0)),
                        str(company.get('primary_risk_factor', 'N/A'))
                    ])
                
                company_table = Table(company_data, colWidths=[0.7*inch, 2*inch, 1*inch, 2*inch])
                company_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(company_table)
                story.append(Spacer(1, 20))
            
            # Trend Analysis
            story.append(Paragraph("Trend Analysis", styles['Heading2']))
            trend_text = f"""
            Compared to the previous day, alert volume has {'increased' if summary_data.get('trend', 0) > 0 else 'decreased'} 
            by {abs(summary_data.get('trend', 0))}%. The most common risk factors identified were related to 
            {', '.join(summary_data.get('common_factors', ['market volatility', 'financial ratios']))}.
            """
            story.append(Paragraph(trend_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Footer
            footer_text = f"Generated by Credit Intelligence Platform on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.statistics['reports_generated'] += 1
            self.statistics['total_pages_generated'] += 2  # Summary reports are typically 2 pages
            
            # Update report type statistics
            report_count = self.statistics['report_types'].get('summary', 0)
            self.statistics['report_types']['summary'] = report_count + 1
            
            logger.info(f"Generated summary report: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            self.statistics['reports_failed'] += 1
            return None
    
    async def generate_company_report(self, company_data: Dict[str, Any]) -> Optional[str]:
        """Generate comprehensive company credit report"""
        
        try:
            if not REPORTLAB_AVAILABLE:
                return await self._generate_simple_report(company_data, 'company')
            
            company_id = company_data.get('company_id', 'Unknown')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"company_report_{company_id}_{timestamp}.pdf"
            filepath = os.path.join(self.output_directory, filename)
            
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            
            story.append(Paragraph(f"Credit Assessment Report - {company_id}", title_style))
            story.append(Spacer(1, 12))
            
            # Company Overview
            story.append(Paragraph("Company Overview", styles['Heading2']))
            
            overview_data = [
                ['Field', 'Value'],
                ['Company ID', str(company_data.get('company_id', 'N/A'))],
                ['Company Name', str(company_data.get('company_name', 'N/A'))],
                ['Industry', str(company_data.get('industry', 'N/A'))],
                ['Credit Score', str(company_data.get('credit_score', 'N/A'))],
                ['Risk Category', str(company_data.get('risk_category', 'N/A'))],
                ['Last Updated', str(company_data.get('last_updated', 'N/A'))]
            ]
            
            overview_table = Table(overview_data, colWidths=[2*inch, 3*inch])
            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(overview_table)
            story.append(Spacer(1, 20))
            
            # Financial Metrics
            story.append(Paragraph("Key Financial Metrics", styles['Heading2']))
            
            financial_metrics = company_data.get('financial_metrics', {})
            metrics_data = [['Metric', 'Value', 'Industry Average', 'Status']]
            
            for metric, value in financial_metrics.items():
                status = "Good" if isinstance(value, (int, float)) and value > 0 else "Review"
                metrics_data.append([
                    metric.replace('_', ' ').title(),
                    str(value),
                    'N/A',
                    status
                ])
            
            if len(metrics_data) > 1:
                metrics_table = Table(metrics_data, colWidths=[2*inch, 1*inch, 1.2*inch, 1*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(metrics_table)
            else:
                story.append(Paragraph("No financial metrics available", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Risk Assessment
            story.append(Paragraph("Risk Assessment", styles['Heading2']))
            risk_factors = company_data.get('risk_factors', [])
            
            if risk_factors:
                for factor in risk_factors:
                    story.append(Paragraph(f"• {factor}", styles['Normal']))
            else:
                story.append(Paragraph("No specific risk factors identified", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Recommendations
            story.append(Paragraph("Recommendations", styles['Heading2']))
            recommendations = company_data.get('recommendations', [
                "Continue monitoring financial performance",
                "Review credit terms quarterly",
                "Monitor industry trends",
                "Assess market conditions impact"
            ])
            
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Footer
            footer_text = f"Generated by Credit Intelligence Platform on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(footer_text, styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.statistics['reports_generated'] += 1
            self.statistics['total_pages_generated'] += 3  # Company reports are typically 3 pages
            
            # Update report type statistics
            report_count = self.statistics['report_types'].get('company', 0)
            self.statistics['report_types']['company'] = report_count + 1
            
            logger.info(f"Generated company report: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating company report: {e}")
            self.statistics['reports_failed'] += 1
            return None
    
    async def _generate_simple_report(self, data: Dict[str, Any], report_type: str) -> Optional[str]:
        """Generate simple text-based report when ReportLab is not available"""
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{report_type}_report_{timestamp}.txt"
            filepath = os.path.join(self.output_directory, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Credit Intelligence Platform - {report_type.title()} Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write data in a readable format
                for key, value in data.items():
                    if isinstance(value, dict):
                        f.write(f"{key.replace('_', ' ').title()}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key.replace('_', ' ').title()}: {sub_value}\n")
                        f.write("\n")
                    elif isinstance(value, list):
                        f.write(f"{key.replace('_', ' ').title()}:\n")
                        for item in value:
                            f.write(f"  - {item}\n")
                        f.write("\n")
                    else:
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("End of Report\n")
            
            self.statistics['reports_generated'] += 1
            self.statistics['total_pages_generated'] += 1
            
            # Update report type statistics
            report_count = self.statistics['report_types'].get(report_type, 0)
            self.statistics['report_types'][report_type] = report_count + 1
            
            logger.info(f"Generated simple {report_type} report: {filename}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating simple report: {e}")
            self.statistics['reports_failed'] += 1
            return None
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get PDF generation statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['reports_generated'] + stats['reports_failed']
            success_rate = (stats['reports_generated'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'output_directory': self.output_directory,
                'reportlab_available': REPORTLAB_AVAILABLE,
                'average_pages_per_report': round(stats['total_pages_generated'] / max(stats['reports_generated'], 1), 1)
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
