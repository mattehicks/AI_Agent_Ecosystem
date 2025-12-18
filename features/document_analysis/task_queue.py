#!/usr/bin/env python3
"""
Task Queue System for Document Analysis
"""

import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory queue")

from .config import config
from .models import TaskStatus, TaskUpdate, AnalysisType
from .database import db_manager, get_db
from .models import TaskRecord, DocumentRecord

logger = logging.getLogger(__name__)

class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

class TaskQueue:
    """Task queue management with Redis backend or in-memory fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.in_memory_queue = []
        self.task_callbacks = {}
        self.running_tasks = {}
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize Redis or fallback to in-memory queue"""
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(
                    config.redis_url,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis task queue")
                
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Using in-memory queue.")
                self.redis_client = None
        else:
            logger.info("Using in-memory task queue")
    
    async def enqueue_task(
        self,
        task_type: str,
        document_id: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Add task to queue"""
        
        task_id = str(uuid.uuid4())
        
        task_data = {
            'id': task_id,
            'type': task_type,
            'document_id': document_id,
            'parameters': parameters,
            'priority': priority.value,
            'status': TaskStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'retry_count': 0
        }
        
        # Store in database
        with db_manager.get_session() as session:
            task_record = TaskRecord(
                id=task_id,
                document_id=document_id,
                task_type=task_type,
                status=TaskStatus.PENDING.value,
                parameters=parameters
            )
            session.add(task_record)
            session.commit()
        
        # Add to queue
        if self.redis_client:
            await self._enqueue_redis(task_data, priority)
        else:
            await self._enqueue_memory(task_data, priority)
        
        logger.info(f"Enqueued task {task_id}: {task_type} for document {document_id}")
        return task_id
    
    async def _enqueue_redis(self, task_data: Dict[str, Any], priority: TaskPriority):
        """Add task to Redis queue"""
        
        queue_name = f"tasks:priority:{priority.value}"
        
        try:
            self.redis_client.lpush(queue_name, json.dumps(task_data))
            
            # Set task expiration
            task_key = f"task:{task_data['id']}"
            self.redis_client.setex(
                task_key,
                timedelta(hours=24),  # Tasks expire after 24 hours
                json.dumps(task_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to enqueue task to Redis: {e}")
            # Fallback to in-memory queue
            await self._enqueue_memory(task_data, priority)
    
    async def _enqueue_memory(self, task_data: Dict[str, Any], priority: TaskPriority):
        """Add task to in-memory queue"""
        
        # Insert task based on priority (higher priority first)
        inserted = False
        for i, existing_task in enumerate(self.in_memory_queue):
            if existing_task['priority'] < priority.value:
                self.in_memory_queue.insert(i, task_data)
                inserted = True
                break
        
        if not inserted:
            self.in_memory_queue.append(task_data)
    
    async def dequeue_task(self) -> Optional[Dict[str, Any]]:
        """Get next task from queue"""
        
        if self.redis_client:
            return await self._dequeue_redis()
        else:
            return await self._dequeue_memory()
    
    async def _dequeue_redis(self) -> Optional[Dict[str, Any]]:
        """Get next task from Redis queue"""
        
        # Check queues in priority order
        for priority in [TaskPriority.URGENT, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue_name = f"tasks:priority:{priority.value}"
            
            try:
                task_json = self.redis_client.rpop(queue_name)
                if task_json:
                    task_data = json.loads(task_json)
                    return task_data
                    
            except Exception as e:
                logger.error(f"Failed to dequeue from Redis: {e}")
                break
        
        return None
    
    async def _dequeue_memory(self) -> Optional[Dict[str, Any]]:
        """Get next task from in-memory queue"""
        
        if self.in_memory_queue:
            return self.in_memory_queue.pop(0)
        
        return None
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float = 0.0,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ):
        """Update task status"""
        
        update_data = {
            'status': status.value,
            'progress': progress,
            'updated_at': datetime.now().isoformat()
        }
        
        if result:
            update_data['result'] = result
        
        if error_message:
            update_data['error_message'] = error_message
        
        # Update database
        with db_manager.get_session() as session:
            task_record = session.query(TaskRecord).filter(TaskRecord.id == task_id).first()
            if task_record:
                task_record.status = status.value
                task_record.progress = progress
                
                if status == TaskStatus.PROCESSING and not task_record.started_at:
                    task_record.started_at = datetime.now()
                
                if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task_record.completed_at = datetime.now()
                    if task_record.started_at:
                        processing_time = (task_record.completed_at - task_record.started_at).total_seconds()
                        task_record.processing_time = processing_time
                
                if result:
                    task_record.result = result
                
                if error_message:
                    task_record.error_message = error_message
                
                session.commit()
        
        # Update Redis cache if available
        if self.redis_client:
            try:
                task_key = f"task:{task_id}"
                existing_data = self.redis_client.get(task_key)
                if existing_data:
                    task_data = json.loads(existing_data)
                    task_data.update(update_data)
                    self.redis_client.setex(task_key, timedelta(hours=24), json.dumps(task_data))
            except Exception as e:
                logger.warning(f"Failed to update task in Redis: {e}")
        
        # Notify callbacks
        await self._notify_task_update(task_id, status, progress, result, error_message)
        
        logger.info(f"Updated task {task_id}: {status.value} ({progress:.1f}%)")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current task status"""
        
        # Try Redis first
        if self.redis_client:
            try:
                task_key = f"task:{task_id}"
                task_data = self.redis_client.get(task_key)
                if task_data:
                    return json.loads(task_data)
            except Exception as e:
                logger.warning(f"Failed to get task from Redis: {e}")
        
        # Fallback to database
        with db_manager.get_session() as session:
            task_record = session.query(TaskRecord).filter(TaskRecord.id == task_id).first()
            if task_record:
                return {
                    'id': task_record.id,
                    'document_id': task_record.document_id,
                    'type': task_record.task_type,
                    'status': task_record.status,
                    'progress': task_record.progress,
                    'result': task_record.result,
                    'error_message': task_record.error_message,
                    'created_at': task_record.created_at.isoformat(),
                    'started_at': task_record.started_at.isoformat() if task_record.started_at else None,
                    'completed_at': task_record.completed_at.isoformat() if task_record.completed_at else None,
                    'processing_time': task_record.processing_time,
                    'retry_count': task_record.retry_count
                }
        
        return None
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        
        stats = {
            'pending_tasks': 0,
            'processing_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'queue_sizes': {},
            'backend': 'redis' if self.redis_client else 'memory'
        }
        
        if self.redis_client:
            try:
                for priority in TaskPriority:
                    queue_name = f"tasks:priority:{priority.value}"
                    size = self.redis_client.llen(queue_name)
                    stats['queue_sizes'][priority.name.lower()] = size
                    stats['pending_tasks'] += size
            except Exception as e:
                logger.warning(f"Failed to get Redis queue stats: {e}")
        else:
            stats['queue_sizes']['memory'] = len(self.in_memory_queue)
            stats['pending_tasks'] = len(self.in_memory_queue)
        
        # Get database stats
        with db_manager.get_session() as session:
            for status in TaskStatus:
                count = session.query(TaskRecord).filter(TaskRecord.status == status.value).count()
                stats[f'{status.value}_tasks'] = count
        
        return stats
    
    def register_task_callback(self, callback: Callable[[str, TaskStatus, float, Optional[Dict], Optional[str]], None]):
        """Register callback for task updates"""
        callback_id = str(uuid.uuid4())
        self.task_callbacks[callback_id] = callback
        return callback_id
    
    def unregister_task_callback(self, callback_id: str):
        """Unregister task callback"""
        self.task_callbacks.pop(callback_id, None)
    
    async def _notify_task_update(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float,
        result: Optional[Dict[str, Any]],
        error_message: Optional[str]
    ):
        """Notify registered callbacks of task updates"""
        
        for callback in self.task_callbacks.values():
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task_id, status, progress, result, error_message)
                else:
                    callback(task_id, status, progress, result, error_message)
            except Exception as e:
                logger.error(f"Task callback error: {e}")
    
    async def retry_failed_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        
        task_data = await self.get_task_status(task_id)
        if not task_data or task_data['status'] != TaskStatus.FAILED.value:
            return False
        
        if task_data['retry_count'] >= config.max_retries:
            logger.warning(f"Task {task_id} has exceeded max retries")
            return False
        
        # Reset task status and re-enqueue
        task_data['status'] = TaskStatus.PENDING.value
        task_data['progress'] = 0.0
        task_data['retry_count'] += 1
        task_data['error_message'] = None
        
        # Update database
        with db_manager.get_session() as session:
            task_record = session.query(TaskRecord).filter(TaskRecord.id == task_id).first()
            if task_record:
                task_record.status = TaskStatus.PENDING.value
                task_record.progress = 0.0
                task_record.retry_count += 1
                task_record.error_message = None
                task_record.started_at = None
                task_record.completed_at = None
                session.commit()
        
        # Re-enqueue task
        priority = TaskPriority(task_data.get('priority', TaskPriority.NORMAL.value))
        if self.redis_client:
            await self._enqueue_redis(task_data, priority)
        else:
            await self._enqueue_memory(task_data, priority)
        
        logger.info(f"Retrying task {task_id} (attempt {task_data['retry_count']})")
        return True
    
    async def cleanup_old_tasks(self, max_age_days: int = 7) -> int:
        """Clean up old completed/failed tasks"""
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cleaned_count = 0
        
        with db_manager.get_session() as session:
            old_tasks = session.query(TaskRecord).filter(
                TaskRecord.completed_at < cutoff_date,
                TaskRecord.status.in_([TaskStatus.COMPLETED.value, TaskStatus.FAILED.value])
            ).all()
            
            for task in old_tasks:
                session.delete(task)
                cleaned_count += 1
            
            session.commit()
        
        logger.info(f"Cleaned up {cleaned_count} old tasks")
        return cleaned_count

# Global task queue instance
task_queue = TaskQueue()