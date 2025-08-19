"""
Export manager for Stage 5 alerting workflows.
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime
import os
import pandas as pd

logger = logging.getLogger(__name__)

class ExportManager:
    """Manager for exporting alert and report data in various formats"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_directory = config.get('output_directory', './exports')
        self.supported_formats = ['json', 'csv', 'xml', 'excel', 'parquet']
        self.statistics = {
            'exports_completed': 0,
            'exports_failed': 0,
            'formats_used': {},
            'total_records_exported': 0
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)
    
    async def export_alerts(self, alerts: List[Dict[str, Any]], 
                          export_format: str = 'json',
                          filename: str = None) -> Optional[str]:
        """Export alerts data in specified format"""
        
        try:
            if export_format not in self.supported_formats:
                logger.error(f"Unsupported export format: {export_format}")
                return None
            
            if not alerts:
                logger.warning("No alerts to export")
                return None
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"alerts_export_{timestamp}.{export_format}"
            
            filepath = os.path.join(self.output_directory, filename)
            
            # Export based on format
            if export_format == 'json':
                success = await self._export_json(alerts, filepath)
            elif export_format == 'csv':
                success = await self._export_csv(alerts, filepath)
            elif export_format == 'xml':
                success = await self._export_xml(alerts, filepath, 'alerts')
            elif export_format == 'excel':
                success = await self._export_excel(alerts, filepath)
            elif export_format == 'parquet':
                success = await self._export_parquet(alerts, filepath)
            else:
                success = False
            
            if success:
                self.statistics['exports_completed'] += 1
                self.statistics['total_records_exported'] += len(alerts)
                
                # Update format statistics
                format_count = self.statistics['formats_used'].get(export_format, 0)
                self.statistics['formats_used'][export_format] = format_count + 1
                
                logger.info(f"Exported {len(alerts)} alerts to {filename}")
                return filepath
            else:
                self.statistics['exports_failed'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error exporting alerts: {e}")
            self.statistics['exports_failed'] += 1
            return None
    
    async def export_summary_data(self, summary_data: Dict[str, Any],
                                export_format: str = 'json',
                                filename: str = None) -> Optional[str]:
        """Export summary data in specified format"""
        
        try:
            if export_format not in self.supported_formats:
                logger.error(f"Unsupported export format: {export_format}")
                return None
            
            # Generate filename if not provided
            if not filename:
                date = summary_data.get('date', datetime.now().strftime('%Y-%m-%d'))
                filename = f"summary_export_{date}.{export_format}"
            
            filepath = os.path.join(self.output_directory, filename)
            
            # Convert summary data to list format for consistent handling
            summary_list = [summary_data]
            
            # Export based on format
            if export_format == 'json':
                success = await self._export_json(summary_list, filepath)
            elif export_format == 'csv':
                success = await self._export_csv(summary_list, filepath)
            elif export_format == 'xml':
                success = await self._export_xml(summary_list, filepath, 'summary')
            elif export_format == 'excel':
                success = await self._export_excel(summary_list, filepath)
            elif export_format == 'parquet':
                success = await self._export_parquet(summary_list, filepath)
            else:
                success = False
            
            if success:
                self.statistics['exports_completed'] += 1
                self.statistics['total_records_exported'] += 1
                
                # Update format statistics
                format_count = self.statistics['formats_used'].get(export_format, 0)
                self.statistics['formats_used'][export_format] = format_count + 1
                
                logger.info(f"Exported summary data to {filename}")
                return filepath
            else:
                self.statistics['exports_failed'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error exporting summary data: {e}")
            self.statistics['exports_failed'] += 1
            return None
    
    async def export_company_data(self, companies: List[Dict[str, Any]],
                                export_format: str = 'json',
                                filename: str = None) -> Optional[str]:
        """Export company data in specified format"""
        
        try:
            if export_format not in self.supported_formats:
                logger.error(f"Unsupported export format: {export_format}")
                return None
            
            if not companies:
                logger.warning("No company data to export")
                return None
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"companies_export_{timestamp}.{export_format}"
            
            filepath = os.path.join(self.output_directory, filename)
            
            # Export based on format
            if export_format == 'json':
                success = await self._export_json(companies, filepath)
            elif export_format == 'csv':
                success = await self._export_csv(companies, filepath)
            elif export_format == 'xml':
                success = await self._export_xml(companies, filepath, 'companies')
            elif export_format == 'excel':
                success = await self._export_excel(companies, filepath)
            elif export_format == 'parquet':
                success = await self._export_parquet(companies, filepath)
            else:
                success = False
            
            if success:
                self.statistics['exports_completed'] += 1
                self.statistics['total_records_exported'] += len(companies)
                
                # Update format statistics
                format_count = self.statistics['formats_used'].get(export_format, 0)
                self.statistics['formats_used'][export_format] = format_count + 1
                
                logger.info(f"Exported {len(companies)} companies to {filename}")
                return filepath
            else:
                self.statistics['exports_failed'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error exporting company data: {e}")
            self.statistics['exports_failed'] += 1
            return None
    
    async def _export_json(self, data: List[Dict[str, Any]], filepath: str) -> bool:
        """Export data as JSON"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            return True
            
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            return False
    
    async def _export_csv(self, data: List[Dict[str, Any]], filepath: str) -> bool:
        """Export data as CSV"""
        
        try:
            if not data:
                return False
            
            # Flatten nested dictionaries and lists for CSV
            flattened_data = []
            for record in data:
                flat_record = self._flatten_dict(record)
                flattened_data.append(flat_record)
            
            # Get all unique keys for CSV headers
            all_keys = set()
            for record in flattened_data:
                all_keys.update(record.keys())
            
            fieldnames = sorted(list(all_keys))
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting CSV: {e}")
            return False
    
    async def _export_xml(self, data: List[Dict[str, Any]], filepath: str, root_name: str) -> bool:
        """Export data as XML"""
        
        try:
            root = ET.Element(root_name)
            
            for i, record in enumerate(data):
                record_element = ET.SubElement(root, f"{root_name[:-1]}")  # Remove 's' from plural
                record_element.set('id', str(i))
                
                self._dict_to_xml(record, record_element)
            
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting XML: {e}")
            return False
    
    async def _export_excel(self, data: List[Dict[str, Any]], filepath: str) -> bool:
        """Export data as Excel"""
        
        try:
            # Flatten data for Excel
            flattened_data = []
            for record in data:
                flat_record = self._flatten_dict(record)
                flattened_data.append(flat_record)
            
            df = pd.DataFrame(flattened_data)
            
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
                
                # Add metadata sheet
                metadata = pd.DataFrame([
                    {'Field': 'Export Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                    {'Field': 'Record Count', 'Value': len(data)},
                    {'Field': 'Generated By', 'Value': 'Credit Intelligence Platform'}
                ])
                metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Excel: {e}")
            return False
    
    async def _export_parquet(self, data: List[Dict[str, Any]], filepath: str) -> bool:
        """Export data as Parquet"""
        
        try:
            # Flatten data for Parquet
            flattened_data = []
            for record in data:
                flat_record = self._flatten_dict(record)
                flattened_data.append(flat_record)
            
            df = pd.DataFrame(flattened_data)
            df.to_parquet(filepath, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Parquet: {e}")
            return False
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        
        return dict(items)
    
    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element):
        """Convert dictionary to XML elements"""
        
        for key, value in data.items():
            # Clean key name for XML
            clean_key = str(key).replace(' ', '_').replace('-', '_')
            
            if isinstance(value, dict):
                child = ET.SubElement(parent, clean_key)
                self._dict_to_xml(value, child)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    child = ET.SubElement(parent, clean_key)
                    child.set('index', str(i))
                    if isinstance(item, dict):
                        self._dict_to_xml(item, child)
                    else:
                        child.text = str(item)
            else:
                child = ET.SubElement(parent, clean_key)
                child.text = str(value) if value is not None else ''
    
    async def create_data_package(self, package_data: Dict[str, Any],
                                package_name: str = None) -> Optional[str]:
        """Create a comprehensive data package with multiple formats"""
        
        try:
            if not package_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                package_name = f"data_package_{timestamp}"
            
            package_dir = os.path.join(self.output_directory, package_name)
            os.makedirs(package_dir, exist_ok=True)
            
            exported_files = []
            
            # Export each data type in multiple formats
            for data_type, data_content in package_data.items():
                if not data_content:
                    continue
                
                # Ensure data_content is a list
                if isinstance(data_content, dict):
                    data_content = [data_content]
                
                # Export in JSON and CSV formats
                for export_format in ['json', 'csv']:
                    filename = f"{data_type}.{export_format}"
                    filepath = os.path.join(package_dir, filename)
                    
                    if export_format == 'json':
                        success = await self._export_json(data_content, filepath)
                    else:
                        success = await self._export_csv(data_content, filepath)
                    
                    if success:
                        exported_files.append(filename)
            
            # Create package manifest
            manifest = {
                'package_name': package_name,
                'created_at': datetime.now().isoformat(),
                'data_types': list(package_data.keys()),
                'exported_files': exported_files,
                'total_records': sum(len(data) if isinstance(data, list) else 1 
                                   for data in package_data.values() if data),
                'generated_by': 'Credit Intelligence Platform'
            }
            
            manifest_path = os.path.join(package_dir, 'manifest.json')
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            self.statistics['exports_completed'] += 1
            logger.info(f"Created data package: {package_name}")
            
            return package_dir
            
        except Exception as e:
            logger.error(f"Error creating data package: {e}")
            self.statistics['exports_failed'] += 1
            return None
    
    async def schedule_export(self, data: List[Dict[str, Any]],
                            export_format: str,
                            schedule_time: datetime,
                            filename: str = None) -> str:
        """Schedule an export for future execution"""
        
        try:
            # Calculate delay
            delay = (schedule_time - datetime.now()).total_seconds()
            
            if delay <= 0:
                logger.warning("Scheduled time is in the past, exporting immediately")
                return await self.export_alerts(data, export_format, filename)
            
            # Schedule the export
            async def delayed_export():
                await asyncio.sleep(delay)
                return await self.export_alerts(data, export_format, filename)
            
            # Start the delayed export task
            task = asyncio.create_task(delayed_export())
            
            logger.info(f"Scheduled export for {schedule_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return f"Export scheduled for {schedule_time.strftime('%Y-%m-%d %H:%M:%S')}"
            
        except Exception as e:
            logger.error(f"Error scheduling export: {e}")
            return None
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return self.supported_formats.copy()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get export manager statistics"""
        
        try:
            stats = self.statistics.copy()
            
            # Calculate success rate
            total_attempts = stats['exports_completed'] + stats['exports_failed']
            success_rate = (stats['exports_completed'] / total_attempts * 100) if total_attempts > 0 else 0
            
            stats.update({
                'total_attempts': total_attempts,
                'success_rate': round(success_rate, 2),
                'output_directory': self.output_directory,
                'supported_formats': self.supported_formats,
                'average_records_per_export': round(stats['total_records_exported'] / max(stats['exports_completed'], 1), 1)
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}
