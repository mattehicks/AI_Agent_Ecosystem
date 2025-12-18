#!/usr/bin/env python3
"""
Task Coordinator Agent
Specialized agent for coordinating complex multi-step tasks
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_agent import BaseAgent

class TaskCoordinatorAgent(BaseAgent):
    """Agent for coordinating complex multi-step tasks"""
    
    def __init__(self, orchestrator):
        super().__init__(orchestrator)
        
        # Coordination strategies
        self.coordination_strategies = {
            'sequential': self._execute_sequential,
            'parallel': self._execute_parallel,
            'conditional': self._execute_conditional,
            'pipeline': self._execute_pipeline,
            'workflow': self._execute_workflow
        }
        
        # Workflow templates
        self.workflow_templates = {
            'document_analysis_pipeline': self._document_analysis_workflow,
            'code_development_cycle': self._code_development_workflow,
            'research_and_report': self._research_report_workflow,
            'data_processing_pipeline': self._data_processing_workflow
        }
        
        self.logger.info("TaskCoordinatorAgent initialized")

    async def process_task(self, task) -> Dict[str, Any]:
        """Process task coordination request"""
        self.logger.info(f"Processing coordination task: {task.description}")
        
        try:
            input_data = task.input_data
            coordination_type = input_data.get('coordination_type', 'sequential')
            
            # Check if it's a workflow template
            if coordination_type in self.workflow_templates:
                result = await self.workflow_templates[coordination_type](input_data)
            elif coordination_type in self.coordination_strategies:
                result = await self.coordination_strategies[coordination_type](input_data)
            else:
                raise ValueError(f"Unsupported coordination type: {coordination_type}")
            
            return self.format_response(
                content=result,
                metadata={
                    'coordination_type': coordination_type,
                    'coordinated_at': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            return self.handle_error(e, "Task coordination failed")

    async def _execute_sequential(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtasks sequentially"""
        subtasks = input_data.get('subtasks', [])
        
        if not subtasks:
            raise ValueError("subtasks required for sequential execution")
        
        results = []
        execution_log = []
        
        for i, subtask in enumerate(subtasks):
            try:
                self.logger.info(f"Executing subtask {i+1}/{len(subtasks)}: {subtask.get('description', 'Unknown')}")
                
                # Create and execute subtask
                task_id = await self._create_subtask(subtask)
                result = await self._wait_for_task_completion(task_id)
                
                results.append({
                    'subtask_index': i,
                    'subtask_id': task_id,
                    'description': subtask.get('description', ''),
                    'result': result,
                    'status': 'completed'
                })
                
                execution_log.append(f"Subtask {i+1} completed successfully")
                
            except Exception as e:
                error_msg = f"Subtask {i+1} failed: {str(e)}"
                execution_log.append(error_msg)
                
                results.append({
                    'subtask_index': i,
                    'description': subtask.get('description', ''),
                    'error': str(e),
                    'status': 'failed'
                })
                
                # Decide whether to continue or stop
                if input_data.get('stop_on_error', True):
                    break
        
        return {
            'coordination_strategy': 'sequential',
            'total_subtasks': len(subtasks),
            'completed_subtasks': len([r for r in results if r.get('status') == 'completed']),
            'results': results,
            'execution_log': execution_log
        }

    async def _execute_parallel(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtasks in parallel"""
        subtasks = input_data.get('subtasks', [])
        max_concurrent = input_data.get('max_concurrent', 3)
        
        if not subtasks:
            raise ValueError("subtasks required for parallel execution")
        
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_subtask_with_semaphore(subtask, index):
            async with semaphore:
                try:
                    task_id = await self._create_subtask(subtask)
                    result = await self._wait_for_task_completion(task_id)
                    return {
                        'subtask_index': index,
                        'subtask_id': task_id,
                        'description': subtask.get('description', ''),
                        'result': result,
                        'status': 'completed'
                    }
                except Exception as e:
                    return {
                        'subtask_index': index,
                        'description': subtask.get('description', ''),
                        'error': str(e),
                        'status': 'failed'
                    }
        
        # Execute all subtasks in parallel
        tasks = [
            execute_subtask_with_semaphore(subtask, i) 
            for i, subtask in enumerate(subtasks)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        execution_log = []
        
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'error': str(result),
                    'status': 'failed'
                })
                execution_log.append(f"Subtask failed with exception: {result}")
            else:
                processed_results.append(result)
                status = result.get('status', 'unknown')
                execution_log.append(f"Subtask {result.get('subtask_index', '?')} {status}")
        
        return {
            'coordination_strategy': 'parallel',
            'total_subtasks': len(subtasks),
            'completed_subtasks': len([r for r in processed_results if r.get('status') == 'completed']),
            'max_concurrent': max_concurrent,
            'results': processed_results,
            'execution_log': execution_log
        }

    async def _execute_conditional(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtasks based on conditions"""
        subtasks = input_data.get('subtasks', [])
        conditions = input_data.get('conditions', {})
        
        if not subtasks:
            raise ValueError("subtasks required for conditional execution")
        
        results = []
        execution_log = []
        context = {}  # Store results for condition evaluation
        
        for i, subtask in enumerate(subtasks):
            # Evaluate condition if present
            condition = subtask.get('condition')
            should_execute = True
            
            if condition:
                should_execute = await self._evaluate_condition(condition, context, conditions)
                execution_log.append(f"Subtask {i+1} condition evaluated: {should_execute}")
            
            if should_execute:
                try:
                    task_id = await self._create_subtask(subtask)
                    result = await self._wait_for_task_completion(task_id)
                    
                    # Store result in context for future conditions
                    context[f"subtask_{i}"] = result
                    
                    results.append({
                        'subtask_index': i,
                        'subtask_id': task_id,
                        'description': subtask.get('description', ''),
                        'result': result,
                        'status': 'completed',
                        'condition_met': True
                    })
                    
                    execution_log.append(f"Subtask {i+1} completed successfully")
                    
                except Exception as e:
                    results.append({
                        'subtask_index': i,
                        'description': subtask.get('description', ''),
                        'error': str(e),
                        'status': 'failed',
                        'condition_met': True
                    })
                    
                    execution_log.append(f"Subtask {i+1} failed: {str(e)}")
            else:
                results.append({
                    'subtask_index': i,
                    'description': subtask.get('description', ''),
                    'status': 'skipped',
                    'condition_met': False
                })
                
                execution_log.append(f"Subtask {i+1} skipped (condition not met)")
        
        return {
            'coordination_strategy': 'conditional',
            'total_subtasks': len(subtasks),
            'executed_subtasks': len([r for r in results if r.get('status') in ['completed', 'failed']]),
            'skipped_subtasks': len([r for r in results if r.get('status') == 'skipped']),
            'results': results,
            'execution_log': execution_log,
            'final_context': context
        }

    async def _execute_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtasks as a data pipeline"""
        subtasks = input_data.get('subtasks', [])
        initial_data = input_data.get('initial_data', {})
        
        if not subtasks:
            raise ValueError("subtasks required for pipeline execution")
        
        results = []
        execution_log = []
        pipeline_data = initial_data.copy()
        
        for i, subtask in enumerate(subtasks):
            try:
                # Add pipeline data to subtask input
                subtask_input = subtask.get('input_data', {}).copy()
                subtask_input['pipeline_data'] = pipeline_data
                
                # Create modified subtask
                pipeline_subtask = subtask.copy()
                pipeline_subtask['input_data'] = subtask_input
                
                task_id = await self._create_subtask(pipeline_subtask)
                result = await self._wait_for_task_completion(task_id)
                
                # Extract output data for next stage
                if isinstance(result, dict) and 'content' in result:
                    content = result['content']
                    if isinstance(content, dict) and 'output_data' in content:
                        pipeline_data = content['output_data']
                    elif isinstance(content, dict):
                        pipeline_data.update(content)
                
                results.append({
                    'subtask_index': i,
                    'subtask_id': task_id,
                    'description': subtask.get('description', ''),
                    'result': result,
                    'status': 'completed',
                    'pipeline_data': pipeline_data.copy()
                })
                
                execution_log.append(f"Pipeline stage {i+1} completed")
                
            except Exception as e:
                results.append({
                    'subtask_index': i,
                    'description': subtask.get('description', ''),
                    'error': str(e),
                    'status': 'failed'
                })
                
                execution_log.append(f"Pipeline stage {i+1} failed: {str(e)}")
                break  # Stop pipeline on failure
        
        return {
            'coordination_strategy': 'pipeline',
            'total_stages': len(subtasks),
            'completed_stages': len([r for r in results if r.get('status') == 'completed']),
            'initial_data': initial_data,
            'final_data': pipeline_data,
            'results': results,
            'execution_log': execution_log
        }

    async def _execute_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complex workflow with dependencies"""
        workflow_definition = input_data.get('workflow', {})
        workflow_data = input_data.get('workflow_data', {})
        
        if not workflow_definition:
            raise ValueError("workflow definition required")
        
        # Parse workflow definition
        nodes = workflow_definition.get('nodes', {})
        dependencies = workflow_definition.get('dependencies', {})
        
        # Execute workflow using topological sort
        execution_order = self._topological_sort(nodes, dependencies)
        
        results = {}
        execution_log = []
        
        for node_id in execution_order:
            node = nodes[node_id]
            
            try:
                # Prepare node input data
                node_input = node.get('input_data', {}).copy()
                node_input['workflow_data'] = workflow_data
                
                # Add dependency results
                for dep_id in dependencies.get(node_id, []):
                    if dep_id in results:
                        node_input[f"dependency_{dep_id}"] = results[dep_id]
                
                # Create and execute node task
                node_task = {
                    'agent_type': node.get('agent_type', 'document_analyzer'),
                    'description': node.get('description', f'Workflow node {node_id}'),
                    'input_data': node_input
                }
                
                task_id = await self._create_subtask(node_task)
                result = await self._wait_for_task_completion(task_id)
                
                results[node_id] = {
                    'task_id': task_id,
                    'result': result,
                    'status': 'completed'
                }
                
                execution_log.append(f"Workflow node {node_id} completed")
                
            except Exception as e:
                results[node_id] = {
                    'error': str(e),
                    'status': 'failed'
                }
                
                execution_log.append(f"Workflow node {node_id} failed: {str(e)}")
                
                # Check if this is a critical node
                if node.get('critical', False):
                    execution_log.append("Critical node failed, stopping workflow")
                    break
        
        return {
            'coordination_strategy': 'workflow',
            'workflow_definition': workflow_definition,
            'execution_order': execution_order,
            'node_results': results,
            'execution_log': execution_log
        }

    # Workflow templates

    async def _document_analysis_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-defined workflow for document analysis"""
        document_path = input_data.get('document_path')
        analysis_depth = input_data.get('analysis_depth', 'comprehensive')
        
        if not document_path:
            raise ValueError("document_path required for document analysis workflow")
        
        # Define workflow steps
        workflow_steps = [
            {
                'agent_type': 'document_analyzer',
                'description': 'Extract document structure',
                'input_data': {
                    'document_path': document_path,
                    'analysis_type': 'structure'
                }
            },
            {
                'agent_type': 'document_analyzer',
                'description': 'Generate document summary',
                'input_data': {
                    'document_path': document_path,
                    'analysis_type': 'summary'
                }
            },
            {
                'agent_type': 'document_analyzer',
                'description': 'Extract keywords',
                'input_data': {
                    'document_path': document_path,
                    'analysis_type': 'extract_keywords'
                }
            }
        ]
        
        if analysis_depth == 'comprehensive':
            workflow_steps.extend([
                {
                    'agent_type': 'document_analyzer',
                    'description': 'Extract entities',
                    'input_data': {
                        'document_path': document_path,
                        'analysis_type': 'extract_entities'
                    }
                },
                {
                    'agent_type': 'document_analyzer',
                    'description': 'Analyze sentiment',
                    'input_data': {
                        'document_path': document_path,
                        'analysis_type': 'sentiment'
                    }
                }
            ])
        
        # Execute workflow
        result = await self._execute_sequential({
            'subtasks': workflow_steps,
            'stop_on_error': False
        })
        
        return {
            'workflow_type': 'document_analysis_pipeline',
            'document_path': document_path,
            'analysis_depth': analysis_depth,
            'workflow_result': result
        }

    async def _code_development_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-defined workflow for code development"""
        requirements = input_data.get('requirements')
        language = input_data.get('language', 'python')
        
        if not requirements:
            raise ValueError("requirements required for code development workflow")
        
        # Define development workflow
        workflow_steps = [
            {
                'agent_type': 'code_generator',
                'description': 'Generate initial code',
                'input_data': {
                    'requirements': requirements,
                    'language': language,
                    'generation_type': 'function'
                }
            },
            {
                'agent_type': 'code_generator',
                'description': 'Generate tests',
                'input_data': {
                    'language': language,
                    'generation_type': 'test',
                    'code_to_test': '${previous_result}'  # Placeholder for pipeline data
                }
            },
            {
                'agent_type': 'code_generator',
                'description': 'Generate documentation',
                'input_data': {
                    'language': language,
                    'generation_type': 'documentation',
                    'code': '${initial_code}'  # Placeholder
                }
            }
        ]
        
        # Execute as pipeline
        result = await self._execute_pipeline({
            'subtasks': workflow_steps,
            'initial_data': {
                'requirements': requirements,
                'language': language
            }
        })
        
        return {
            'workflow_type': 'code_development_cycle',
            'requirements': requirements,
            'language': language,
            'workflow_result': result
        }

    async def _research_report_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-defined workflow for research and reporting"""
        research_topic = input_data.get('topic')
        report_type = input_data.get('report_type', 'comprehensive')
        
        if not research_topic:
            raise ValueError("topic required for research report workflow")
        
        # Define research workflow
        workflow_steps = [
            {
                'agent_type': 'research_assistant',
                'description': 'Conduct initial research',
                'input_data': {
                    'query': research_topic,
                    'research_type': 'comprehensive',
                    'sources': ['documents', 'knowledge_base', 'privateGPT']
                }
            },
            {
                'agent_type': 'document_analyzer',
                'description': 'Analyze research findings',
                'input_data': {
                    'analysis_type': 'summary',
                    'content': '${research_results}'  # Placeholder
                }
            }
        ]
        
        # Execute workflow
        result = await self._execute_sequential({
            'subtasks': workflow_steps
        })
        
        return {
            'workflow_type': 'research_and_report',
            'research_topic': research_topic,
            'report_type': report_type,
            'workflow_result': result
        }

    async def _data_processing_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-defined workflow for data processing"""
        data_path = input_data.get('data_path')
        processing_goals = input_data.get('goals', ['clean', 'analyze'])
        
        if not data_path:
            raise ValueError("data_path required for data processing workflow")
        
        # Define processing workflow based on goals
        workflow_steps = []
        
        if 'clean' in processing_goals:
            workflow_steps.append({
                'agent_type': 'data_processor',
                'description': 'Clean data',
                'input_data': {
                    'data_path': data_path,
                    'processing_type': 'clean',
                    'operations': ['remove_duplicates', 'handle_missing', 'trim_whitespace']
                }
            })
        
        if 'analyze' in processing_goals:
            workflow_steps.append({
                'agent_type': 'data_processor',
                'description': 'Analyze data',
                'input_data': {
                    'data_path': data_path,
                    'processing_type': 'analyze',
                    'analysis_type': 'statistical'
                }
            })
        
        if 'visualize' in processing_goals:
            workflow_steps.append({
                'agent_type': 'data_processor',
                'description': 'Generate visualizations',
                'input_data': {
                    'data_path': data_path,
                    'processing_type': 'visualize'
                }
            })
        
        # Execute as pipeline
        result = await self._execute_pipeline({
            'subtasks': workflow_steps,
            'initial_data': {'data_path': data_path}
        })
        
        return {
            'workflow_type': 'data_processing_pipeline',
            'data_path': data_path,
            'processing_goals': processing_goals,
            'workflow_result': result
        }

    # Helper methods

    async def _create_subtask(self, subtask: Dict[str, Any]) -> str:
        """Create a subtask and return its ID"""
        from orchestrator.orchestrator import AgentType, TaskPriority
        
        agent_type = AgentType(subtask.get('agent_type', 'document_analyzer'))
        description = subtask.get('description', 'Coordinated subtask')
        input_data = subtask.get('input_data', {})
        priority = TaskPriority.HIGH  # Coordinated tasks get high priority
        
        task_id = self.orchestrator.create_task(
            agent_type=agent_type,
            description=description,
            input_data=input_data,
            priority=priority
        )
        
        return task_id

    async def _wait_for_task_completion(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for a task to complete and return its result"""
        start_time = datetime.now()
        
        while True:
            task_status = self.orchestrator.get_task_status(task_id)
            
            if not task_status:
                raise Exception(f"Task {task_id} not found")
            
            status = task_status.get('status')
            
            if status == 'completed':
                return task_status.get('result', {})
            elif status == 'failed':
                error_msg = task_status.get('error_message', 'Unknown error')
                raise Exception(f"Task {task_id} failed: {error_msg}")
            elif status == 'cancelled':
                raise Exception(f"Task {task_id} was cancelled")
            
            # Check timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > timeout:
                raise Exception(f"Task {task_id} timed out after {timeout} seconds")
            
            # Wait before checking again
            await asyncio.sleep(2)

    async def _evaluate_condition(self, condition: str, context: Dict[str, Any], 
                                 global_conditions: Dict[str, Any]) -> bool:
        """Evaluate a condition string"""
        # Simple condition evaluation (could be expanded)
        try:
            # Replace context variables
            for key, value in context.items():
                condition = condition.replace(f"${{{key}}}", str(value))
            
            # Replace global condition variables
            for key, value in global_conditions.items():
                condition = condition.replace(f"${{{key}}}", str(value))
            
            # Evaluate simple conditions
            if 'success' in condition.lower():
                return 'success' in str(context).lower()
            elif 'error' in condition.lower():
                return 'error' in str(context).lower()
            else:
                # Default to True for now
                return True
                
        except Exception as e:
            self.logger.warning(f"Error evaluating condition '{condition}': {e}")
            return True

    def _topological_sort(self, nodes: Dict[str, Any], dependencies: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on workflow nodes"""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node_id):
            if node_id in temp_visited:
                raise Exception(f"Circular dependency detected involving node {node_id}")
            if node_id in visited:
                return
            
            temp_visited.add(node_id)
            
            # Visit dependencies first
            for dep in dependencies.get(node_id, []):
                visit(dep)
            
            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)
        
        # Visit all nodes
        for node_id in nodes:
            if node_id not in visited:
                visit(node_id)
        
        return result