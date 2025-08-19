"""
Model metadata management for Stage 3.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

class ModelMetadataManager:
    """Manage comprehensive model metadata"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata_store = {}
        
    def create_training_metadata(self, model_id: str, training_config: Dict[str, Any],
                               training_data_info: Dict[str, Any],
                               performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive training metadata"""
        
        metadata = {
            'model_id': model_id,
            'training': {
                'timestamp': datetime.now().isoformat(),
                'config': training_config,
                'data_info': training_data_info,
                'performance': performance_metrics,
                'training_duration': training_config.get('training_duration'),
                'hyperparameters': training_config.get('hyperparameters', {}),
                'cross_validation': training_config.get('cross_validation', {}),
                'feature_selection': training_config.get('feature_selection', {})
            },
            'data': {
                'training_samples': training_data_info.get('n_samples', 0),
                'features': training_data_info.get('n_features', 0),
                'target_distribution': training_data_info.get('target_distribution', {}),
                'data_quality': training_data_info.get('data_quality', {}),
                'feature_names': training_data_info.get('feature_names', []),
                'data_sources': training_data_info.get('data_sources', []),
                'data_version': training_data_info.get('data_version'),
                'preprocessing_steps': training_data_info.get('preprocessing_steps', [])
            },
            'performance': {
                'validation_metrics': performance_metrics,
                'training_metrics': performance_metrics.get('training_metrics', {}),
                'cross_validation_scores': performance_metrics.get('cv_scores', {}),
                'feature_importance': performance_metrics.get('feature_importance', {}),
                'model_complexity': performance_metrics.get('model_complexity', {})
            }
        }
        
        self.metadata_store[model_id] = metadata
        return metadata
    
    def add_deployment_metadata(self, model_id: str, deployment_info: Dict[str, Any]):
        """Add deployment metadata"""
        
        if model_id not in self.metadata_store:
            self.metadata_store[model_id] = {}
        
        deployment_metadata = {
            'deployment_timestamp': datetime.now().isoformat(),
            'environment': deployment_info.get('environment'),
            'deployment_config': deployment_info.get('config', {}),
            'infrastructure': deployment_info.get('infrastructure', {}),
            'scaling_config': deployment_info.get('scaling', {}),
            'monitoring_config': deployment_info.get('monitoring', {}),
            'deployment_version': deployment_info.get('version'),
            'deployed_by': deployment_info.get('deployed_by', 'system')
        }
        
        self.metadata_store[model_id]['deployment'] = deployment_metadata
        
    def add_monitoring_metadata(self, model_id: str, monitoring_data: Dict[str, Any]):
        """Add monitoring and performance metadata"""
        
        if model_id not in self.metadata_store:
            self.metadata_store[model_id] = {}
        
        if 'monitoring' not in self.metadata_store[model_id]:
            self.metadata_store[model_id]['monitoring'] = []
        
        monitoring_entry = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': monitoring_data.get('performance', {}),
            'data_drift_metrics': monitoring_data.get('data_drift', {}),
            'prediction_drift_metrics': monitoring_data.get('prediction_drift', {}),
            'system_metrics': monitoring_data.get('system_metrics', {}),
            'alerts': monitoring_data.get('alerts', []),
            'prediction_volume': monitoring_data.get('prediction_volume', 0),
            'latency_metrics': monitoring_data.get('latency', {}),
            'error_rate': monitoring_data.get('error_rate', 0)
        }
        
        self.metadata_store[model_id]['monitoring'].append(monitoring_entry)
        
    def add_business_metadata(self, model_id: str, business_info: Dict[str, Any]):
        """Add business context metadata"""
        
        if model_id not in self.metadata_store:
            self.metadata_store[model_id] = {}
        
        business_metadata = {
            'use_case': business_info.get('use_case'),
            'business_owner': business_info.get('owner'),
            'stakeholders': business_info.get('stakeholders', []),
            'business_metrics': business_info.get('business_metrics', {}),
            'risk_tolerance': business_info.get('risk_tolerance'),
            'regulatory_requirements': business_info.get('regulatory', []),
            'approval_status': business_info.get('approval_status'),
            'business_impact': business_info.get('business_impact', {}),
            'cost_benefit_analysis': business_info.get('cost_benefit', {}),
            'success_criteria': business_info.get('success_criteria', {})
        }
        
        self.metadata_store[model_id]['business'] = business_metadata
        
    def add_compliance_metadata(self, model_id: str, compliance_info: Dict[str, Any]):
        """Add compliance and governance metadata"""
        
        if model_id not in self.metadata_store:
            self.metadata_store[model_id] = {}
        
        compliance_metadata = {
            'regulatory_framework': compliance_info.get('framework'),
            'compliance_checks': compliance_info.get('checks', []),
            'audit_trail': compliance_info.get('audit_trail', []),
            'data_privacy': compliance_info.get('data_privacy', {}),
            'model_explainability': compliance_info.get('explainability', {}),
            'bias_assessment': compliance_info.get('bias_assessment', {}),
            'fairness_metrics': compliance_info.get('fairness_metrics', {}),
            'documentation_status': compliance_info.get('documentation_status'),
            'approval_workflow': compliance_info.get('approval_workflow', []),
            'risk_assessment': compliance_info.get('risk_assessment', {})
        }
        
        self.metadata_store[model_id]['compliance'] = compliance_metadata
        
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get complete metadata for a model"""
        
        if model_id not in self.metadata_store:
            return {}
        
        return self.metadata_store[model_id]
    
    def search_models(self, criteria: Dict[str, Any]) -> List[str]:
        """Search models based on metadata criteria"""
        
        matching_models = []
        
        for model_id, metadata in self.metadata_store.items():
            if self._matches_criteria(metadata, criteria):
                matching_models.append(model_id)
        
        return matching_models
    
    def _matches_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches search criteria"""
        
        for key, value in criteria.items():
            if '.' in key:
                # Nested key search
                keys = key.split('.')
                current = metadata
                
                for k in keys:
                    if isinstance(current, dict) and k in current:
                        current = current[k]
                    else:
                        return False
                
                if current != value:
                    return False
            else:
                # Top-level key search
                if key not in metadata or metadata[key] != value:
                    return False
        
        return True
    
    def get_performance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get performance summary for a model"""
        
        metadata = self.get_model_metadata(model_id)
        
        if not metadata:
            return {}
        
        summary = {
            'model_id': model_id,
            'training_performance': metadata.get('performance', {}).get('validation_metrics', {}),
            'latest_monitoring': None,
            'performance_trend': None
        }
        
        # Get latest monitoring data
        monitoring_data = metadata.get('monitoring', [])
        if monitoring_data:
            latest_monitoring = monitoring_data[-1]
            summary['latest_monitoring'] = latest_monitoring.get('performance_metrics', {})
            
            # Calculate performance trend
            if len(monitoring_data) > 1:
                summary['performance_trend'] = self._calculate_performance_trend(monitoring_data)
        
        return summary
    
    def _calculate_performance_trend(self, monitoring_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trend from monitoring data"""
        
        if len(monitoring_data) < 2:
            return {}
        
        # Extract key metrics over time
        timestamps = []
        auc_scores = []
        accuracy_scores = []
        
        for entry in monitoring_data:
            timestamps.append(entry['timestamp'])
            perf_metrics = entry.get('performance_metrics', {})
            
            if 'roc_auc' in perf_metrics:
                auc_scores.append(perf_metrics['roc_auc'])
            if 'accuracy' in perf_metrics:
                accuracy_scores.append(perf_metrics['accuracy'])
        
        trend = {}
        
        # Calculate trends
        if len(auc_scores) > 1:
            auc_trend = np.polyfit(range(len(auc_scores)), auc_scores, 1)[0]
            trend['auc_trend'] = {
                'slope': auc_trend,
                'direction': 'improving' if auc_trend > 0 else 'degrading',
                'current_value': auc_scores[-1],
                'change_from_first': auc_scores[-1] - auc_scores[0]
            }
        
        if len(accuracy_scores) > 1:
            acc_trend = np.polyfit(range(len(accuracy_scores)), accuracy_scores, 1)[0]
            trend['accuracy_trend'] = {
                'slope': acc_trend,
                'direction': 'improving' if acc_trend > 0 else 'degrading',
                'current_value': accuracy_scores[-1],
                'change_from_first': accuracy_scores[-1] - accuracy_scores[0]
            }
        
        return trend
    
    def generate_model_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive model report"""
        
        metadata = self.get_model_metadata(model_id)
        
        if not metadata:
            return {'error': 'Model not found'}
        
        report = {
            'model_id': model_id,
            'report_timestamp': datetime.now().isoformat(),
            'summary': self._generate_model_summary(metadata),
            'training_details': metadata.get('training', {}),
            'data_details': metadata.get('data', {}),
            'performance_analysis': self.get_performance_summary(model_id),
            'deployment_status': metadata.get('deployment', {}),
            'business_context': metadata.get('business', {}),
            'compliance_status': metadata.get('compliance', {}),
            'recommendations': self._generate_recommendations(metadata)
        }
        
        return report
    
    def _generate_model_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model summary"""
        
        training_info = metadata.get('training', {})
        data_info = metadata.get('data', {})
        performance_info = metadata.get('performance', {})
        
        summary = {
            'model_type': training_info.get('config', {}).get('model_type', 'Unknown'),
            'training_date': training_info.get('timestamp'),
            'data_samples': data_info.get('training_samples', 0),
            'feature_count': data_info.get('features', 0),
            'primary_metric': self._get_primary_metric(performance_info),
            'deployment_status': metadata.get('deployment', {}).get('environment', 'Not deployed'),
            'last_monitored': self._get_last_monitoring_date(metadata)
        }
        
        return summary
    
    def _get_primary_metric(self, performance_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get primary performance metric"""
        
        validation_metrics = performance_info.get('validation_metrics', {})
        
        if 'roc_auc' in validation_metrics:
            return {
                'metric': 'ROC AUC',
                'value': validation_metrics['roc_auc']
            }
        elif 'accuracy' in validation_metrics:
            return {
                'metric': 'Accuracy',
                'value': validation_metrics['accuracy']
            }
        else:
            return {
                'metric': 'Unknown',
                'value': None
            }
    
    def _get_last_monitoring_date(self, metadata: Dict[str, Any]) -> str:
        """Get last monitoring date"""
        
        monitoring_data = metadata.get('monitoring', [])
        
        if monitoring_data:
            return monitoring_data[-1]['timestamp']
        else:
            return 'Never monitored'
    
    def _generate_recommendations(self, metadata: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metadata"""
        
        recommendations = []
        
        # Check monitoring frequency
        monitoring_data = metadata.get('monitoring', [])
        if not monitoring_data:
            recommendations.append("Set up model monitoring to track performance")
        elif len(monitoring_data) == 1:
            recommendations.append("Increase monitoring frequency for better trend analysis")
        
        # Check performance trends
        if len(monitoring_data) > 1:
            trend = self._calculate_performance_trend(monitoring_data)
            
            if trend.get('auc_trend', {}).get('direction') == 'degrading':
                recommendations.append("Model performance is degrading - consider retraining")
            
            if trend.get('accuracy_trend', {}).get('direction') == 'degrading':
                recommendations.append("Accuracy is declining - investigate data drift")
        
        # Check compliance
        compliance_info = metadata.get('compliance', {})
        if not compliance_info:
            recommendations.append("Complete compliance assessment for regulatory requirements")
        
        # Check business context
        business_info = metadata.get('business', {})
        if not business_info:
            recommendations.append("Document business context and success criteria")
        
        # Check deployment
        deployment_info = metadata.get('deployment', {})
        if not deployment_info:
            recommendations.append("Consider deploying model to staging environment for testing")
        
        return recommendations
    
    def export_metadata(self, model_id: str, filepath: str):
        """Export metadata to file"""
        
        metadata = self.get_model_metadata(model_id)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Exported metadata for {model_id} to {filepath}")
    
    def import_metadata(self, filepath: str) -> str:
        """Import metadata from file"""
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        model_id = metadata.get('model_id')
        if not model_id:
            raise ValueError("No model_id found in metadata")
        
        self.metadata_store[model_id] = metadata
        
        logger.info(f"Imported metadata for {model_id} from {filepath}")
        return model_id
    
    def update_metadata(self, model_id: str, updates: Dict[str, Any]):
        """Update specific metadata fields"""
        
        if model_id not in self.metadata_store:
            self.metadata_store[model_id] = {}
        
        # Deep merge updates
        self._deep_merge(self.metadata_store[model_id], updates)
        
        logger.info(f"Updated metadata for {model_id}")
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge two dictionaries"""
        
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def get_metadata_schema(self) -> Dict[str, Any]:
        """Get metadata schema definition"""
        
        schema = {
            'training': {
                'timestamp': 'ISO datetime string',
                'config': 'Training configuration dictionary',
                'data_info': 'Training data information',
                'performance': 'Performance metrics',
                'hyperparameters': 'Model hyperparameters'
            },
            'data': {
                'training_samples': 'Number of training samples',
                'features': 'Number of features',
                'target_distribution': 'Target variable distribution',
                'feature_names': 'List of feature names',
                'data_sources': 'List of data sources'
            },
            'performance': {
                'validation_metrics': 'Validation performance metrics',
                'training_metrics': 'Training performance metrics',
                'cross_validation_scores': 'CV scores',
                'feature_importance': 'Feature importance scores'
            },
            'deployment': {
                'deployment_timestamp': 'Deployment datetime',
                'environment': 'Deployment environment',
                'deployment_config': 'Deployment configuration',
                'infrastructure': 'Infrastructure details'
            },
            'monitoring': {
                'timestamp': 'Monitoring timestamp',
                'performance_metrics': 'Current performance metrics',
                'data_drift_metrics': 'Data drift indicators',
                'system_metrics': 'System performance metrics'
            },
            'business': {
                'use_case': 'Business use case description',
                'business_owner': 'Business owner',
                'stakeholders': 'List of stakeholders',
                'business_metrics': 'Business KPIs'
            },
            'compliance': {
                'regulatory_framework': 'Applicable regulations',
                'compliance_checks': 'Compliance verification results',
                'audit_trail': 'Audit trail information',
                'bias_assessment': 'Model bias assessment'
            }
        }
        
        return schema
