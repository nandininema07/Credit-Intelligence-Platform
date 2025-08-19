"""
Tests for Stage 5 workflow components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os

from stage5_alerting_workflows.workflows import (
    WorkflowEngine, WorkflowStatus, TaskStatus, JiraIntegration, 
    PDFGenerator, ExportManager
)

class TestWorkflowEngine:
    """Tests for WorkflowEngine"""
    
    @pytest.fixture
    def workflow_engine(self, workflow_config):
        return WorkflowEngine(workflow_config['workflow_engine'])
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, workflow_engine, sample_workflow_tasks):
        """Test creating a workflow"""
        workflow_id = await workflow_engine.create_workflow(
            name='Test Workflow',
            description='Test workflow description',
            tasks=sample_workflow_tasks
        )
        
        assert workflow_id is not None
        assert workflow_id in workflow_engine.active_workflows
        
        workflow = workflow_engine.active_workflows[workflow_id]
        assert workflow.name == 'Test Workflow'
        assert len(workflow.tasks) == len(sample_workflow_tasks)
        assert workflow.status == WorkflowStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self, workflow_engine, sample_workflow_tasks):
        """Test workflow execution"""
        workflow_id = await workflow_engine.create_workflow(
            name='Test Workflow',
            description='Test workflow',
            tasks=sample_workflow_tasks
        )
        
        result = await workflow_engine.execute_workflow(workflow_id)
        
        assert result is True
        workflow = workflow_engine.active_workflows[workflow_id]
        assert workflow.status == WorkflowStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_task_dependencies(self, workflow_engine):
        """Test workflow task dependency handling"""
        tasks = [
            {
                'task_id': 'task_1',
                'name': 'First Task',
                'task_type': 'data_ingestion',
                'parameters': {},
                'dependencies': []
            },
            {
                'task_id': 'task_2',
                'name': 'Second Task',
                'task_type': 'feature_engineering',
                'parameters': {},
                'dependencies': ['task_1']
            }
        ]
        
        workflow_id = await workflow_engine.create_workflow(
            name='Dependency Test',
            description='Test dependencies',
            tasks=tasks
        )
        
        result = await workflow_engine.execute_workflow(workflow_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cancel_workflow(self, workflow_engine, sample_workflow_tasks):
        """Test workflow cancellation"""
        workflow_id = await workflow_engine.create_workflow(
            name='Test Workflow',
            description='Test workflow',
            tasks=sample_workflow_tasks
        )
        
        # Start workflow execution in background
        execution_task = asyncio.create_task(
            workflow_engine.execute_workflow(workflow_id)
        )
        
        # Cancel workflow
        result = workflow_engine.cancel_workflow(workflow_id)
        assert result is True
        
        workflow = workflow_engine.active_workflows[workflow_id]
        assert workflow.status == WorkflowStatus.CANCELLED

class TestJiraIntegration:
    """Tests for JiraIntegration"""
    
    @pytest.fixture
    def jira_integration(self, workflow_config):
        return JiraIntegration(workflow_config['jira'])
    
    @pytest.mark.asyncio
    async def test_create_ticket_from_alert(self, jira_integration, sample_alert_data):
        """Test creating Jira ticket from alert"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 201
            mock_response.json = AsyncMock(return_value={'key': 'CRED-123'})
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            ticket_key = await jira_integration.create_ticket_from_alert(sample_alert_data)
            
            assert ticket_key == 'CRED-123'
    
    @pytest.mark.asyncio
    async def test_format_alert_ticket(self, jira_integration, sample_alert_data):
        """Test formatting alert data for Jira ticket"""
        ticket_data = jira_integration._format_alert_ticket(sample_alert_data)
        
        assert isinstance(ticket_data, dict)
        assert 'fields' in ticket_data
        assert 'project' in ticket_data['fields']
        assert 'summary' in ticket_data['fields']
        assert 'description' in ticket_data['fields']
    
    @pytest.mark.asyncio
    async def test_add_comment(self, jira_integration):
        """Test adding comment to Jira ticket"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 201
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await jira_integration.add_comment(
                'CRED-123',
                'Test comment'
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_resolve_ticket(self, jira_integration):
        """Test resolving Jira ticket"""
        with patch.object(jira_integration, 'get_ticket_transitions') as mock_transitions:
            mock_transitions.return_value = [
                {'id': '31', 'name': 'Done'}
            ]
            
            with patch('aiohttp.ClientSession') as mock_session:
                mock_response = Mock()
                mock_response.status = 204
                
                mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
                
                result = await jira_integration.resolve_ticket('CRED-123')
                assert result is True

class TestPDFGenerator:
    """Tests for PDFGenerator"""
    
    @pytest.fixture
    def pdf_generator(self, workflow_config):
        return PDFGenerator(workflow_config['pdf_generator'])
    
    @pytest.mark.asyncio
    async def test_generate_alert_report(self, pdf_generator, sample_alert_data):
        """Test generating PDF alert report"""
        filepath = await pdf_generator.generate_alert_report(sample_alert_data)
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith('.pdf') or filepath.endswith('.txt')
    
    @pytest.mark.asyncio
    async def test_generate_summary_report(self, pdf_generator, sample_summary_data):
        """Test generating PDF summary report"""
        filepath = await pdf_generator.generate_summary_report(sample_summary_data)
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith('.pdf') or filepath.endswith('.txt')
    
    @pytest.mark.asyncio
    async def test_generate_company_report(self, pdf_generator, sample_company_data):
        """Test generating PDF company report"""
        filepath = await pdf_generator.generate_company_report(sample_company_data)
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith('.pdf') or filepath.endswith('.txt')

class TestExportManager:
    """Tests for ExportManager"""
    
    @pytest.fixture
    def export_manager(self, workflow_config):
        return ExportManager(workflow_config['export_manager'])
    
    @pytest.mark.asyncio
    async def test_export_alerts_json(self, export_manager):
        """Test exporting alerts to JSON"""
        from conftest import create_test_alerts
        alerts = create_test_alerts(3)
        
        filepath = await export_manager.export_alerts(alerts, 'json')
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
        
        # Verify file content
        import json
        with open(filepath, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 3
        assert exported_data[0]['id'] == 'alert_1'
    
    @pytest.mark.asyncio
    async def test_export_alerts_csv(self, export_manager):
        """Test exporting alerts to CSV"""
        from conftest import create_test_alerts
        alerts = create_test_alerts(3)
        
        filepath = await export_manager.export_alerts(alerts, 'csv')
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith('.csv')
        
        # Verify file content
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3
    
    @pytest.mark.asyncio
    async def test_export_summary_data(self, export_manager, sample_summary_data):
        """Test exporting summary data"""
        filepath = await export_manager.export_summary_data(sample_summary_data, 'json')
        
        assert filepath is not None
        assert os.path.exists(filepath)
        assert filepath.endswith('.json')
    
    @pytest.mark.asyncio
    async def test_create_data_package(self, export_manager):
        """Test creating comprehensive data package"""
        from conftest import create_test_alerts, create_test_companies
        
        package_data = {
            'alerts': create_test_alerts(3),
            'companies': create_test_companies(2)
        }
        
        package_dir = await export_manager.create_data_package(package_data)
        
        assert package_dir is not None
        assert os.path.exists(package_dir)
        assert os.path.exists(os.path.join(package_dir, 'manifest.json'))
        assert os.path.exists(os.path.join(package_dir, 'alerts.json'))
        assert os.path.exists(os.path.join(package_dir, 'companies.json'))
    
    def test_flatten_dict(self, export_manager):
        """Test dictionary flattening utility"""
        nested_dict = {
            'level1': {
                'level2': {
                    'value': 'test'
                },
                'simple': 'value'
            },
            'list_field': [1, 2, 3]
        }
        
        flattened = export_manager._flatten_dict(nested_dict)
        
        assert 'level1_level2_value' in flattened
        assert flattened['level1_level2_value'] == 'test'
        assert 'level1_simple' in flattened
        assert flattened['level1_simple'] == 'value'
