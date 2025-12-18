#!/usr/bin/env python3
"""
Workflow Templates for AI Agent Ecosystem
Pre-built workflow templates for common AI tasks and processes
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepType(Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    CODE_GENERATION = "code_generation"
    RESEARCH = "research"
    DATA_PROCESSING = "data_processing"
    SUMMARIZATION = "summarization"
    VALIDATION = "validation"
    NOTIFICATION = "notification"

@dataclass
class WorkflowStep:
    id: str
    name: str
    step_type: StepType
    agent_type: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 2
    condition: Optional[str] = None  # Conditional execution
    on_success: Optional[str] = None  # Next step on success
    on_failure: Optional[str] = None  # Next step on failure

@dataclass
class WorkflowTemplate:
    id: str
    name: str
    description: str
    category: str
    steps: List[WorkflowStep]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    estimated_duration: int = 0  # seconds
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class WorkflowInstance:
    id: str
    template_id: str
    status: WorkflowStatus
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class WorkflowEngine:
    """Workflow execution engine"""
    
    def __init__(self, task_manager, agent_manager):
        self.task_manager = task_manager
        self.agent_manager = agent_manager
        self.templates = {}
        self.active_workflows = {}
        self.completed_workflows = {}
        
        # Load built-in templates
        self._load_builtin_templates()
    
    def register_template(self, template: WorkflowTemplate):
        """Register a workflow template"""
        self.templates[template.id] = template
        logger.info(f"Registered workflow template: {template.name}")
    
    async def start_workflow(self, template_id: str, 
                           input_data: Dict[str, Any],
                           workflow_id: str = None) -> str:
        """Start a workflow instance"""
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template {template_id} not found")
            
            template = self.templates[template_id]
            
            # Validate input data against schema
            if not self._validate_input(input_data, template.input_schema):
                raise ValueError("Input data does not match template schema")
            
            # Create workflow instance
            instance_id = workflow_id or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(input_data)) % 1000:03d}"
            
            instance = WorkflowInstance(
                id=instance_id,
                template_id=template_id,
                status=WorkflowStatus.PENDING,
                input_data=input_data
            )
            
            self.active_workflows[instance_id] = instance
            
            # Start execution
            asyncio.create_task(self._execute_workflow(instance))
            
            logger.info(f"Started workflow {instance_id} from template {template_id}")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            raise
    
    async def _execute_workflow(self, instance: WorkflowInstance):
        """Execute a workflow instance"""
        try:
            template = self.templates[instance.template_id]
            instance.status = WorkflowStatus.RUNNING
            instance.started_at = datetime.now()
            
            # Execute steps in dependency order
            remaining_steps = template.steps.copy()
            
            while remaining_steps and instance.status == WorkflowStatus.RUNNING:
                # Find steps that can be executed (dependencies satisfied)
                executable_steps = []
                
                for step in remaining_steps:
                    if self._dependencies_satisfied(step, instance.completed_steps):
                        # Check condition if specified
                        if not step.condition or self._evaluate_condition(step.condition, instance):
                            executable_steps.append(step)
                
                if not executable_steps:
                    # No more executable steps - check if we're done
                    if not remaining_steps:
                        break
                    else:
                        # Deadlock - some steps can't execute
                        raise Exception("Workflow deadlock: remaining steps have unsatisfied dependencies")
                
                # Execute steps (can be parallel if no conflicts)
                execution_tasks = []
                for step in executable_steps:
                    task = asyncio.create_task(self._execute_step(step, instance))
                    execution_tasks.append((step, task))
                
                # Wait for step completions
                for step, task in execution_tasks:
                    try:
                        result = await task
                        instance.step_results[step.id] = result
                        instance.completed_steps.append(step.id)
                        remaining_steps.remove(step)
                        
                        logger.info(f"Workflow {instance.id} completed step {step.name}")
                        
                    except Exception as e:
                        logger.error(f"Workflow {instance.id} step {step.name} failed: {e}")
                        instance.failed_steps.append(step.id)
                        remaining_steps.remove(step)
                        
                        # Handle failure based on step configuration
                        if step.on_failure:
                            # Continue to failure handler step
                            continue
                        else:
                            # Fail the entire workflow
                            raise e
            
            # Workflow completed successfully
            instance.status = WorkflowStatus.COMPLETED
            instance.completed_at = datetime.now()
            
            # Generate output data
            instance.output_data = self._generate_output(instance, template)
            
            # Move to completed workflows
            self.completed_workflows[instance.id] = instance
            del self.active_workflows[instance.id]
            
            logger.info(f"Workflow {instance.id} completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow {instance.id} failed: {e}")
            instance.status = WorkflowStatus.FAILED
            instance.error_message = str(e)
            instance.completed_at = datetime.now()
            
            # Move to completed workflows
            self.completed_workflows[instance.id] = instance
            if instance.id in self.active_workflows:
                del self.active_workflows[instance.id]
    
    async def _execute_step(self, step: WorkflowStep, instance: WorkflowInstance) -> Any:
        """Execute a single workflow step"""
        try:
            instance.current_step = step.id
            
            # Prepare step input data
            step_input = self._prepare_step_input(step, instance)
            
            # Create and submit task
            task_id = self.task_manager.create_task(
                agent_type=step.agent_type,
                description=f"Workflow {instance.id} - {step.name}",
                input_data=step_input,
                timeout_seconds=step.timeout_seconds
            )
            
            # Wait for task completion
            while True:
                task_status = self.task_manager.get_task_status(task_id)
                if not task_status:
                    raise Exception(f"Task {task_id} not found")
                
                if task_status['status'] == 'completed':
                    return task_status['result']
                elif task_status['status'] == 'failed':
                    error_msg = task_status.get('error_message', 'Unknown error')
                    
                    # Retry if possible
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        logger.warning(f"Retrying step {step.name} (attempt {step.retry_count})")
                        return await self._execute_step(step, instance)
                    else:
                        raise Exception(f"Step failed: {error_msg}")
                
                await asyncio.sleep(1)  # Poll every second
                
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            raise
        finally:
            instance.current_step = None
    
    def _dependencies_satisfied(self, step: WorkflowStep, completed_steps: List[str]) -> bool:
        """Check if step dependencies are satisfied"""
        return all(dep in completed_steps for dep in step.dependencies)
    
    def _evaluate_condition(self, condition: str, instance: WorkflowInstance) -> bool:
        """Evaluate step condition"""
        try:
            # Simple condition evaluation - can be enhanced with proper parser
            # For now, support basic comparisons like "step1.success == true"
            context = {
                'input': instance.input_data,
                'results': instance.step_results,
                'completed': instance.completed_steps,
                'failed': instance.failed_steps
            }
            
            # Replace step references with actual values
            eval_condition = condition
            for step_id, result in instance.step_results.items():
                eval_condition = eval_condition.replace(f"{step_id}.result", str(result))
                eval_condition = eval_condition.replace(f"{step_id}.success", str(step_id in instance.completed_steps))
            
            # Simple evaluation (in production, use a proper expression parser)
            return eval(eval_condition, {"__builtins__": {}}, context)
            
        except Exception as e:
            logger.error(f"Condition evaluation failed: {e}")
            return False
    
    def _prepare_step_input(self, step: WorkflowStep, instance: WorkflowInstance) -> Dict[str, Any]:
        """Prepare input data for step execution"""
        step_input = step.parameters.copy()
        
        # Add workflow context
        step_input['workflow_id'] = instance.id
        step_input['workflow_input'] = instance.input_data
        step_input['previous_results'] = instance.step_results
        
        # Replace placeholders with actual values
        step_input = self._replace_placeholders(step_input, instance)
        
        return step_input
    
    def _replace_placeholders(self, data: Any, instance: WorkflowInstance) -> Any:
        """Replace placeholders in step parameters"""
        if isinstance(data, dict):
            return {k: self._replace_placeholders(v, instance) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_placeholders(item, instance) for item in data]
        elif isinstance(data, str):
            # Replace placeholders like ${input.field} or ${step1.result}
            result = data
            
            # Replace input placeholders
            for key, value in instance.input_data.items():
                result = result.replace(f"${{input.{key}}}", str(value))
            
            # Replace step result placeholders
            for step_id, step_result in instance.step_results.items():
                if isinstance(step_result, dict):
                    for res_key, res_value in step_result.items():
                        result = result.replace(f"${{steps.{step_id}.{res_key}}}", str(res_value))
                else:
                    result = result.replace(f"${{steps.{step_id}}}", str(step_result))
            
            return result
        else:
            return data
    
    def _generate_output(self, instance: WorkflowInstance, template: WorkflowTemplate) -> Dict[str, Any]:
        """Generate workflow output data"""
        output = {}
        
        # Map step results to output schema
        for key, schema_def in template.output_schema.items():
            if 'source_step' in schema_def:
                step_id = schema_def['source_step']
                if step_id in instance.step_results:
                    if 'field' in schema_def:
                        result = instance.step_results[step_id]
                        if isinstance(result, dict) and schema_def['field'] in result:
                            output[key] = result[schema_def['field']]
                    else:
                        output[key] = instance.step_results[step_id]
        
        return output
    
    def _validate_input(self, input_data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate input data against schema"""
        try:
            for field, field_schema in schema.items():
                if field_schema.get('required', False) and field not in input_data:
                    return False
                
                if field in input_data:
                    value = input_data[field]
                    expected_type = field_schema.get('type')
                    
                    if expected_type == 'string' and not isinstance(value, str):
                        return False
                    elif expected_type == 'number' and not isinstance(value, (int, float)):
                        return False
                    elif expected_type == 'boolean' and not isinstance(value, bool):
                        return False
                    elif expected_type == 'array' and not isinstance(value, list):
                        return False
                    elif expected_type == 'object' and not isinstance(value, dict):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        # Check active workflows
        if workflow_id in self.active_workflows:
            instance = self.active_workflows[workflow_id]
        elif workflow_id in self.completed_workflows:
            instance = self.completed_workflows[workflow_id]
        else:
            return None
        
        return {
            'id': instance.id,
            'template_id': instance.template_id,
            'status': instance.status.value,
            'current_step': instance.current_step,
            'completed_steps': instance.completed_steps,
            'failed_steps': instance.failed_steps,
            'progress': len(instance.completed_steps) / len(self.templates[instance.template_id].steps),
            'created_at': instance.created_at.isoformat(),
            'started_at': instance.started_at.isoformat() if instance.started_at else None,
            'completed_at': instance.completed_at.isoformat() if instance.completed_at else None,
            'error_message': instance.error_message,
            'output_data': instance.output_data
        }
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available workflow templates"""
        return [
            {
                'id': template.id,
                'name': template.name,
                'description': template.description,
                'category': template.category,
                'steps_count': len(template.steps),
                'estimated_duration': template.estimated_duration,
                'tags': template.tags
            }
            for template in self.templates.values()
        ]
    
    def _load_builtin_templates(self):
        """Load built-in workflow templates"""
        
        # Document Analysis Workflow
        doc_analysis_template = WorkflowTemplate(
            id="document_analysis",
            name="Document Analysis",
            description="Comprehensive document analysis including summarization, key points extraction, and Q&A generation",
            category="Document Processing",
            steps=[
                WorkflowStep(
                    id="upload_document",
                    name="Upload Document",
                    step_type=StepType.DOCUMENT_ANALYSIS,
                    agent_type="document_analyzer",
                    parameters={
                        "action": "upload",
                        "file_path": "${input.file_path}"
                    }
                ),
                WorkflowStep(
                    id="extract_text",
                    name="Extract Text",
                    step_type=StepType.DOCUMENT_ANALYSIS,
                    agent_type="document_analyzer",
                    parameters={
                        "action": "extract_text",
                        "document_id": "${steps.upload_document.document_id}"
                    },
                    dependencies=["upload_document"]
                ),
                WorkflowStep(
                    id="generate_summary",
                    name="Generate Summary",
                    step_type=StepType.SUMMARIZATION,
                    agent_type="research_assistant",
                    parameters={
                        "action": "summarize",
                        "text": "${steps.extract_text.content}",
                        "summary_type": "comprehensive"
                    },
                    dependencies=["extract_text"]
                ),
                WorkflowStep(
                    id="extract_key_points",
                    name="Extract Key Points",
                    step_type=StepType.DOCUMENT_ANALYSIS,
                    agent_type="research_assistant",
                    parameters={
                        "action": "extract_key_points",
                        "text": "${steps.extract_text.content}"
                    },
                    dependencies=["extract_text"]
                ),
                WorkflowStep(
                    id="generate_questions",
                    name="Generate Questions",
                    step_type=StepType.RESEARCH,
                    agent_type="research_assistant",
                    parameters={
                        "action": "generate_questions",
                        "text": "${steps.extract_text.content}",
                        "question_count": 5
                    },
                    dependencies=["extract_text"]
                )
            ],
            input_schema={
                "file_path": {"type": "string", "required": True, "description": "Path to document file"}
            },
            output_schema={
                "summary": {"source_step": "generate_summary", "field": "summary"},
                "key_points": {"source_step": "extract_key_points", "field": "key_points"},
                "questions": {"source_step": "generate_questions", "field": "questions"},
                "document_id": {"source_step": "upload_document", "field": "document_id"}
            },
            estimated_duration=180,
            tags=["document", "analysis", "nlp"]
        )
        
        # Research Workflow
        research_template = WorkflowTemplate(
            id="research_project",
            name="Research Project",
            description="Comprehensive research workflow with multiple sources and synthesis",
            category="Research",
            steps=[
                WorkflowStep(
                    id="initial_search",
                    name="Initial Search",
                    step_type=StepType.RESEARCH,
                    agent_type="research_assistant",
                    parameters={
                        "action": "search",
                        "query": "${input.research_query}",
                        "max_results": 10
                    }
                ),
                WorkflowStep(
                    id="analyze_sources",
                    name="Analyze Sources",
                    step_type=StepType.DOCUMENT_ANALYSIS,
                    agent_type="document_analyzer",
                    parameters={
                        "action": "batch_analyze",
                        "sources": "${steps.initial_search.results}"
                    },
                    dependencies=["initial_search"]
                ),
                WorkflowStep(
                    id="synthesize_findings",
                    name="Synthesize Findings",
                    step_type=StepType.RESEARCH,
                    agent_type="research_assistant",
                    parameters={
                        "action": "synthesize",
                        "analyses": "${steps.analyze_sources.analyses}",
                        "research_question": "${input.research_query}"
                    },
                    dependencies=["analyze_sources"]
                ),
                WorkflowStep(
                    id="generate_report",
                    name="Generate Report",
                    step_type=StepType.DOCUMENT_ANALYSIS,
                    agent_type="research_assistant",
                    parameters={
                        "action": "generate_report",
                        "synthesis": "${steps.synthesize_findings.synthesis}",
                        "format": "${input.report_format}"
                    },
                    dependencies=["synthesize_findings"]
                )
            ],
            input_schema={
                "research_query": {"type": "string", "required": True},
                "report_format": {"type": "string", "required": False, "default": "markdown"}
            },
            output_schema={
                "report": {"source_step": "generate_report", "field": "report"},
                "sources": {"source_step": "initial_search", "field": "results"},
                "synthesis": {"source_step": "synthesize_findings", "field": "synthesis"}
            },
            estimated_duration=300,
            tags=["research", "analysis", "report"]
        )
        
        # Code Generation Workflow
        code_gen_template = WorkflowTemplate(
            id="code_generation",
            name="Code Generation",
            description="Generate, review, and test code based on requirements",
            category="Development",
            steps=[
                WorkflowStep(
                    id="analyze_requirements",
                    name="Analyze Requirements",
                    step_type=StepType.DOCUMENT_ANALYSIS,
                    agent_type="code_generator",
                    parameters={
                        "action": "analyze_requirements",
                        "requirements": "${input.requirements}",
                        "language": "${input.language}"
                    }
                ),
                WorkflowStep(
                    id="generate_code",
                    name="Generate Code",
                    step_type=StepType.CODE_GENERATION,
                    agent_type="code_generator",
                    parameters={
                        "action": "generate",
                        "specification": "${steps.analyze_requirements.specification}",
                        "language": "${input.language}",
                        "style": "${input.coding_style}"
                    },
                    dependencies=["analyze_requirements"]
                ),
                WorkflowStep(
                    id="review_code",
                    name="Review Code",
                    step_type=StepType.VALIDATION,
                    agent_type="code_generator",
                    parameters={
                        "action": "review",
                        "code": "${steps.generate_code.code}",
                        "language": "${input.language}"
                    },
                    dependencies=["generate_code"]
                ),
                WorkflowStep(
                    id="generate_tests",
                    name="Generate Tests",
                    step_type=StepType.CODE_GENERATION,
                    agent_type="code_generator",
                    parameters={
                        "action": "generate_tests",
                        "code": "${steps.generate_code.code}",
                        "language": "${input.language}"
                    },
                    dependencies=["generate_code"]
                )
            ],
            input_schema={
                "requirements": {"type": "string", "required": True},
                "language": {"type": "string", "required": True},
                "coding_style": {"type": "string", "required": False, "default": "standard"}
            },
            output_schema={
                "code": {"source_step": "generate_code", "field": "code"},
                "tests": {"source_step": "generate_tests", "field": "tests"},
                "review": {"source_step": "review_code", "field": "review"},
                "specification": {"source_step": "analyze_requirements", "field": "specification"}
            },
            estimated_duration=240,
            tags=["code", "development", "testing"]
        )
        
        # Register templates
        self.register_template(doc_analysis_template)
        self.register_template(research_template)
        self.register_template(code_gen_template)