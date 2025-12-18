#!/usr/bin/env python3
"""
Advanced Task Manager for AI Agent Ecosystem
Handles task scheduling, prioritization, and queue management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import heapq
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

@dataclass
class Task:
    id: str
    agent_type: str
    description: str
    input_data: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    dependencies: List[str] = field(default_factory=list)
    callback: Optional[Callable] = None

class TaskScheduler:
    """Advanced task scheduler with priority queues and dependency management"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.priority_queues = {agent_type: [] for agent_type in ['document_analyzer', 'code_generator', 'research_assistant', 'data_processor', 'task_coordinator']}
        self.active_tasks = {}
        self.completed_tasks = {}
        self.task_dependencies = {}
        self.task_callbacks = {}
        
    def add_task(self, task: Task) -> str:
        """Add task to appropriate priority queue"""
        try:
            # Store in database
            self._save_task_to_db(task)
            
            # Add to priority queue (negative priority for min-heap behavior)
            heapq.heappush(
                self.priority_queues[task.agent_type],
                (-task.priority.value, task.created_at.timestamp(), task.id, task)
            )
            
            task.status = TaskStatus.QUEUED
            logger.info(f"Task {task.id} queued for {task.agent_type}")
            return task.id
            
        except Exception as e:
            logger.error(f"Failed to add task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            return task.id
    
    def get_next_task(self, agent_type: str) -> Optional[Task]:
        """Get next highest priority task for agent type"""
        queue = self.priority_queues.get(agent_type, [])
        
        while queue:
            try:
                _, _, task_id, task = heapq.heappop(queue)
                
                # Check if task dependencies are satisfied
                if self._dependencies_satisfied(task):
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    self.active_tasks[task_id] = task
                    self._update_task_in_db(task)
                    return task
                else:
                    # Re-queue if dependencies not satisfied
                    heapq.heappush(queue, (-task.priority.value, task.created_at.timestamp(), task_id, task))
                    
            except Exception as e:
                logger.error(f"Error getting next task: {e}")
                
        return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark task as completed with results"""
        try:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                
                # Move to completed tasks
                self.completed_tasks[task_id] = task
                del self.active_tasks[task_id]
                
                # Update database
                self._update_task_in_db(task)
                
                # Execute callback if exists
                if task.callback:
                    try:
                        task.callback(task)
                    except Exception as e:
                        logger.error(f"Task callback failed: {e}")
                
                logger.info(f"Task {task_id} completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            
        return False
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """Mark task as failed and handle retries"""
        try:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.retry_count += 1
                task.error_message = error_message
                
                if task.retry_count < task.max_retries:
                    # Retry the task
                    task.status = TaskStatus.RETRYING
                    del self.active_tasks[task_id]
                    
                    # Re-add to queue with delay
                    asyncio.create_task(self._retry_task_delayed(task, delay_seconds=2 ** task.retry_count))
                    logger.info(f"Task {task_id} will retry in {2 ** task.retry_count} seconds")
                else:
                    # Max retries reached
                    task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    self.completed_tasks[task_id] = task
                    del self.active_tasks[task_id]
                    logger.error(f"Task {task_id} failed after {task.retry_count} retries")
                
                self._update_task_in_db(task)
                return True
                
        except Exception as e:
            logger.error(f"Failed to handle task failure {task_id}: {e}")
            
        return False
    
    async def _retry_task_delayed(self, task: Task, delay_seconds: int):
        """Retry task after delay"""
        await asyncio.sleep(delay_seconds)
        self.add_task(task)
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
            
        for dep_task_id in task.dependencies:
            if dep_task_id not in self.completed_tasks:
                return False
            if self.completed_tasks[dep_task_id].status != TaskStatus.COMPLETED:
                return False
                
        return True
    
    def _save_task_to_db(self, task: Task):
        """Save task to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tasks 
                (id, agent_type, description, input_data, priority, status, result, 
                 created_at, started_at, completed_at, error_message, retry_count, cache_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id, task.agent_type, task.description,
                json.dumps(task.input_data), task.priority.value, task.status.value,
                json.dumps(task.result) if task.result else None,
                task.created_at.isoformat(),
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
                task.error_message, task.retry_count, None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save task to database: {e}")
    
    def _update_task_in_db(self, task: Task):
        """Update existing task in database"""
        self._save_task_to_db(task)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about task queues"""
        stats = {}
        
        for agent_type, queue in self.priority_queues.items():
            stats[agent_type] = {
                'queued': len(queue),
                'active': len([t for t in self.active_tasks.values() if t.agent_type == agent_type]),
                'completed_today': len([t for t in self.completed_tasks.values() 
                                     if t.agent_type == agent_type and 
                                     t.completed_at and 
                                     t.completed_at.date() == datetime.now().date()])
            }
            
        return stats
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        try:
            # Check active tasks first
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                self.completed_tasks[task_id] = task
                del self.active_tasks[task_id]
                self._update_task_in_db(task)
                return True
            
            # Check queued tasks
            for agent_type, queue in self.priority_queues.items():
                for i, (_, _, tid, task) in enumerate(queue):
                    if tid == task_id:
                        task.status = TaskStatus.CANCELLED
                        task.completed_at = datetime.now()
                        self.completed_tasks[task_id] = task
                        del queue[i]
                        heapq.heapify(queue)
                        self._update_task_in_db(task)
                        return True
                        
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            
        return False

class TaskManager:
    """Main task management system"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.scheduler = TaskScheduler(db_path)
        self.task_timeout_monitor = None
        self.cleanup_interval = 3600  # 1 hour
        
    async def start(self):
        """Start task management services"""
        logger.info("Starting Task Manager")
        
        # Start timeout monitor
        self.task_timeout_monitor = asyncio.create_task(self._monitor_task_timeouts())
        
        # Start cleanup service
        asyncio.create_task(self._periodic_cleanup())
    
    async def stop(self):
        """Stop task management services"""
        logger.info("Stopping Task Manager")
        
        if self.task_timeout_monitor:
            self.task_timeout_monitor.cancel()
    
    def create_task(self, agent_type: str, description: str, input_data: Dict[str, Any], 
                   priority: TaskPriority = TaskPriority.NORMAL, 
                   dependencies: List[str] = None,
                   timeout_seconds: int = 300) -> str:
        """Create and queue a new task"""
        
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(input_data)) % 10000:04d}"
        
        task = Task(
            id=task_id,
            agent_type=agent_type,
            description=description,
            input_data=input_data,
            priority=priority,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds
        )
        
        return self.scheduler.add_task(task)
    
    def get_next_task(self, agent_type: str) -> Optional[Task]:
        """Get next task for agent"""
        return self.scheduler.get_next_task(agent_type)
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Complete a task with results"""
        return self.scheduler.complete_task(task_id, result)
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """Fail a task with error message"""
        return self.scheduler.fail_task(task_id, error_message)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        return self.scheduler.cancel_task(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed task status"""
        # Check active tasks
        if task_id in self.scheduler.active_tasks:
            task = self.scheduler.active_tasks[task_id]
            return self._task_to_dict(task)
        
        # Check completed tasks
        if task_id in self.scheduler.completed_tasks:
            task = self.scheduler.completed_tasks[task_id]
            return self._task_to_dict(task)
        
        # Check database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'agent_type': row[1],
                    'description': row[2],
                    'input_data': json.loads(row[3]),
                    'priority': row[4],
                    'status': row[5],
                    'result': json.loads(row[6]) if row[6] else None,
                    'created_at': row[7],
                    'started_at': row[8],
                    'completed_at': row[9],
                    'error_message': row[10],
                    'retry_count': row[11]
                }
        except Exception as e:
            logger.error(f"Failed to get task status from database: {e}")
        
        return None
    
    def list_tasks(self, agent_type: str = None, status: TaskStatus = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List tasks with optional filtering"""
        tasks = []
        
        # Get from active tasks
        for task in self.scheduler.active_tasks.values():
            if (not agent_type or task.agent_type == agent_type) and (not status or task.status == status):
                tasks.append(self._task_to_dict(task))
        
        # Get from completed tasks
        for task in self.scheduler.completed_tasks.values():
            if (not agent_type or task.agent_type == agent_type) and (not status or task.status == status):
                tasks.append(self._task_to_dict(task))
        
        # Get from database if needed
        if len(tasks) < limit:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = 'SELECT * FROM tasks'
                params = []
                
                conditions = []
                if agent_type:
                    conditions.append('agent_type = ?')
                    params.append(agent_type)
                if status:
                    conditions.append('status = ?')
                    params.append(status.value)
                
                if conditions:
                    query += ' WHERE ' + ' AND '.join(conditions)
                
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                conn.close()
                
                for row in rows:
                    task_dict = {
                        'id': row[0],
                        'agent_type': row[1],
                        'description': row[2],
                        'input_data': json.loads(row[3]),
                        'priority': row[4],
                        'status': row[5],
                        'result': json.loads(row[6]) if row[6] else None,
                        'created_at': row[7],
                        'started_at': row[8],
                        'completed_at': row[9],
                        'error_message': row[10],
                        'retry_count': row[11]
                    }
                    
                    # Avoid duplicates
                    if not any(t['id'] == task_dict['id'] for t in tasks):
                        tasks.append(task_dict)
                        
            except Exception as e:
                logger.error(f"Failed to list tasks from database: {e}")
        
        return tasks[:limit]
    
    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        return self.scheduler.get_queue_stats()
    
    async def _monitor_task_timeouts(self):
        """Monitor and handle task timeouts"""
        while True:
            try:
                current_time = datetime.now()
                timed_out_tasks = []
                
                for task_id, task in self.scheduler.active_tasks.items():
                    if task.started_at:
                        elapsed = (current_time - task.started_at).total_seconds()
                        if elapsed > task.timeout_seconds:
                            timed_out_tasks.append((task_id, task))
                
                for task_id, task in timed_out_tasks:
                    logger.warning(f"Task {task_id} timed out after {task.timeout_seconds} seconds")
                    self.fail_task(task_id, f"Task timed out after {task.timeout_seconds} seconds")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in timeout monitor: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old completed tasks"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                cutoff_time = datetime.now() - timedelta(hours=24)
                cleaned_count = 0
                
                # Clean completed tasks older than 24 hours
                to_remove = []
                for task_id, task in self.scheduler.completed_tasks.items():
                    if task.completed_at and task.completed_at < cutoff_time:
                        to_remove.append(task_id)
                
                for task_id in to_remove:
                    del self.scheduler.completed_tasks[task_id]
                    cleaned_count += 1
                
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old completed tasks")
                    
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        """Convert task object to dictionary"""
        return {
            'id': task.id,
            'agent_type': task.agent_type,
            'description': task.description,
            'input_data': task.input_data,
            'priority': task.priority.value,
            'status': task.status.value,
            'result': task.result,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'error_message': task.error_message,
            'retry_count': task.retry_count,
            'max_retries': task.max_retries,
            'timeout_seconds': task.timeout_seconds,
            'dependencies': task.dependencies
        }