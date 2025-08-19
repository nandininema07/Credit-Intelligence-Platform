"""
Workflow engine for automated credit risk workflows and business processes.
Handles workflow execution, task orchestration, and integration with external systems.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStatus(Enum):
    """Individual task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowTask:
    """Individual workflow task"""
    task_id: str
    name: str
    task_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    timeout_minutes: int = 30
    retry_count: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = None
    start_time: datetime = None
    end_time: datetime = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class Workflow:
    """Workflow definition"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class WorkflowEngine:
    """Workflow execution engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_workflows = {}
        self.workflow_history = []
        self.task_handlers = {}
        self.max_concurrent_workflows = config.get('max_concurrent_workflows', 5)
        
        # Register default task handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default task handlers"""
        self.task_handlers.update({
            'data_ingestion': self._handle_data_ingestion,
            'feature_engineering': self._handle_feature_engineering,
            'model_scoring': self._handle_model_scoring,
            'alert_check': self._handle_alert_check,
            'notification': self._handle_notification,
            'data_validation': self._handle_data_validation,
            'report_generation': self._handle_report_generation,
            'api_call': self._handle_api_call,
            'email_notification': self._handle_email_notification,
            'slack_notification': self._handle_slack_notification
        })
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register custom task handler"""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    async def create_workflow(self, name: str, description: str, 
                            tasks: List[Dict[str, Any]]) -> str:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Convert task dictionaries to WorkflowTask objects
        workflow_tasks = []
        for i, task_data in enumerate(tasks):
            task = WorkflowTask(
                task_id=task_data.get('task_id', f"task_{i}"),
                name=task_data.get('name', f"Task {i+1}"),
                task_type=task_data.get('task_type', 'generic'),
                parameters=task_data.get('parameters', {}),
                dependencies=task_data.get('dependencies', []),
                timeout_minutes=task_data.get('timeout_minutes', 30),
                retry_count=task_data.get('retry_count', 3)
            )
            workflow_tasks.append(task)
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            tasks=workflow_tasks
        )
        
        self.active_workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name} ({workflow_id})")
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """Execute a workflow"""
        if workflow_id not in self.active_workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.PENDING:
            logger.warning(f"Workflow {workflow_id} is not in pending status")
            return False
        
        logger.info(f"Starting workflow execution: {workflow.name}")
        
        try:
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            
            # Execute tasks based on dependencies
            completed_tasks = set()
            
            while len(completed_tasks) < len(workflow.tasks):
                # Find tasks ready to execute
                ready_tasks = []
                for task in workflow.tasks:
                    if (task.status == TaskStatus.PENDING and 
                        all(dep in completed_tasks for dep in task.dependencies)):
                        ready_tasks.append(task)
                
                if not ready_tasks:
                    # Check if we're stuck
                    pending_tasks = [t for t in workflow.tasks if t.status == TaskStatus.PENDING]
                    if pending_tasks:
                        logger.error(f"Workflow {workflow_id} stuck - circular dependencies or failed dependencies")
                        workflow.status = WorkflowStatus.FAILED
                        return False
                    break
                
                # Execute ready tasks concurrently
                tasks_to_execute = ready_tasks[:3]  # Limit concurrent tasks
                await asyncio.gather(*[self._execute_task(task) for task in tasks_to_execute])
                
                # Update completed tasks
                for task in tasks_to_execute:
                    if task.status == TaskStatus.COMPLETED:
                        completed_tasks.add(task.task_id)
                    elif task.status == TaskStatus.FAILED:
                        logger.error(f"Task {task.task_id} failed, workflow may be incomplete")
            
            # Check overall workflow status
            failed_tasks = [t for t in workflow.tasks if t.status == TaskStatus.FAILED]
            if failed_tasks:
                workflow.status = WorkflowStatus.FAILED
                logger.error(f"Workflow {workflow_id} failed - {len(failed_tasks)} tasks failed")
            else:
                workflow.status = WorkflowStatus.COMPLETED
                logger.info(f"Workflow {workflow_id} completed successfully")
            
            workflow.completed_at = datetime.now()
            return workflow.status == WorkflowStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            return False
    
    async def _execute_task(self, task: WorkflowTask):
        """Execute individual task"""
        logger.info(f"Executing task: {task.name}")
        
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        retry_count = 0
        
        while retry_count <= task.retry_count:
            try:
                # Get task handler
                handler = self.task_handlers.get(task.task_type)
                if not handler:
                    raise ValueError(f"No handler for task type: {task.task_type}")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(task.parameters),
                    timeout=task.timeout_minutes * 60
                )
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.end_time = datetime.now()
                
                logger.info(f"Task {task.task_id} completed successfully")
                return
                
            except asyncio.TimeoutError:
                logger.error(f"Task {task.task_id} timed out")
                task.error = "Task timed out"
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {str(e)}")
                task.error = str(e)
            
            retry_count += 1
            if retry_count <= task.retry_count:
                logger.info(f"Retrying task {task.task_id} (attempt {retry_count + 1})")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
        
        task.status = TaskStatus.FAILED
        task.end_time = datetime.now()
    
    # Default task handlers
    async def _handle_data_ingestion(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data ingestion task"""
        company_id = parameters.get('company_id')
        data_sources = parameters.get('data_sources', ['news', 'financial'])
        
        # Simulate data ingestion
        await asyncio.sleep(1)
        
        return {
            'status': 'success',
            'company_id': company_id,
            'sources_processed': data_sources,
            'records_ingested': 150
        }
    
    async def _handle_feature_engineering(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature engineering task"""
        company_id = parameters.get('company_id')
        
        # Simulate feature engineering
        await asyncio.sleep(2)
        
        return {
            'status': 'success',
            'company_id': company_id,
            'features_generated': 45,
            'feature_quality_score': 0.87
        }
    
    async def _handle_model_scoring(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model scoring task"""
        company_id = parameters.get('company_id')
        model_name = parameters.get('model_name', 'xgboost')
        
        # Simulate scoring
        await asyncio.sleep(1)
        
        return {
            'status': 'success',
            'company_id': company_id,
            'credit_score': 675,
            'risk_category': 'Medium Risk',
            'model_used': model_name
        }
    
    async def _handle_alert_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle alert checking task"""
        company_id = parameters.get('company_id')
        
        # Simulate alert checking
        await asyncio.sleep(0.5)
        
        return {
            'status': 'success',
            'company_id': company_id,
            'alerts_triggered': 0,
            'checks_performed': 5
        }
    
    async def _handle_notification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle notification task"""
        recipient = parameters.get('recipient')
        message = parameters.get('message')
        channel = parameters.get('channel', 'email')
        
        # Simulate notification
        await asyncio.sleep(0.3)
        
        return {
            'status': 'success',
            'recipient': recipient,
            'channel': channel,
            'delivered': True
        }
    
    async def _handle_data_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation task"""
        data_source = parameters.get('data_source')
        
        # Simulate validation
        await asyncio.sleep(1)
        
        return {
            'status': 'success',
            'data_source': data_source,
            'validation_passed': True,
            'quality_score': 0.92
        }
    
    async def _handle_report_generation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report generation task"""
        report_type = parameters.get('report_type', 'credit_summary')
        company_id = parameters.get('company_id')
        
        # Simulate report generation
        await asyncio.sleep(3)
        
        return {
            'status': 'success',
            'report_type': report_type,
            'company_id': company_id,
            'report_url': f'/reports/{company_id}_{report_type}_{datetime.now().strftime("%Y%m%d")}.pdf'
        }
    
    async def _handle_api_call(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle external API call task"""
        url = parameters.get('url')
        method = parameters.get('method', 'GET')
        
        # Simulate API call
        await asyncio.sleep(1)
        
        return {
            'status': 'success',
            'url': url,
            'method': method,
            'response_code': 200
        }
    
    async def _handle_email_notification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email notification task"""
        recipient = parameters.get('recipient')
        subject = parameters.get('subject')
        
        # Simulate email sending
        await asyncio.sleep(0.5)
        
        return {
            'status': 'success',
            'recipient': recipient,
            'subject': subject,
            'delivered': True
        }
    
    async def _handle_slack_notification(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Slack notification task"""
        channel = parameters.get('channel')
        message = parameters.get('message')
        
        # Simulate Slack notification
        await asyncio.sleep(0.3)
        
        return {
            'status': 'success',
            'channel': channel,
            'message_sent': True
        }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        task_statuses = {}
        for task in workflow.tasks:
            task_statuses[task.task_id] = {
                'name': task.name,
                'status': task.status.value,
                'start_time': task.start_time.isoformat() if task.start_time else None,
                'end_time': task.end_time.isoformat() if task.end_time else None,
                'error': task.error
            }
        
        return {
            'workflow_id': workflow_id,
            'name': workflow.name,
            'status': workflow.status.value,
            'created_at': workflow.created_at.isoformat(),
            'started_at': workflow.started_at.isoformat() if workflow.started_at else None,
            'completed_at': workflow.completed_at.isoformat() if workflow.completed_at else None,
            'task_count': len(workflow.tasks),
            'completed_tasks': len([t for t in workflow.tasks if t.status == TaskStatus.COMPLETED]),
            'failed_tasks': len([t for t in workflow.tasks if t.status == TaskStatus.FAILED]),
            'tasks': task_statuses
        }
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id not in self.active_workflows:
            return False
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            
            # Cancel running tasks
            for task in workflow.tasks:
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.SKIPPED
                    task.end_time = datetime.now()
            
            logger.info(f"Cancelled workflow: {workflow.name}")
            return True
        
        return False
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        all_workflows = list(self.active_workflows.values()) + self.workflow_history
        
        if not all_workflows:
            return {
                'total_workflows': 0,
                'active_workflows': 0,
                'completed_workflows': 0,
                'failed_workflows': 0
            }
        
        status_counts = {}
        for status in WorkflowStatus:
            count = len([w for w in all_workflows if w.status == status])
            status_counts[status.value] = count
        
        # Calculate average execution time for completed workflows
        completed_workflows = [w for w in all_workflows if w.status == WorkflowStatus.COMPLETED]
        avg_execution_time = 0
        
        if completed_workflows:
            execution_times = []
            for w in completed_workflows:
                if w.started_at and w.completed_at:
                    duration = (w.completed_at - w.started_at).total_seconds()
                    execution_times.append(duration)
            
            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            'total_workflows': len(all_workflows),
            'active_workflows': len(self.active_workflows),
            'status_breakdown': status_counts,
            'average_execution_time_seconds': avg_execution_time,
            'task_handlers_registered': len(self.task_handlers)
        }
