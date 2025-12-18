#!/usr/bin/env python3
"""
AI Agent Orchestrator
Main coordination system for the AI Agent Ecosystem
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging - detect environment
import os
if os.name == 'nt' or os.path.exists("X:/"):
    # Windows environment
    base_project_path = Path("X:/AI_Agent_Ecosystem")
else:
    # Linux environment
    base_project_path = Path("/mnt/llm/AI_Agent_Ecosystem")

log_dir = base_project_path / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    DOCUMENT_ANALYZER = "document_analyzer"
    CODE_GENERATOR = "code_generator"
    RESEARCH_ASSISTANT = "research_assistant"
    DATA_PROCESSOR = "data_processor"
    TASK_COORDINATOR = "task_coordinator"

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    id: str
    agent_type: AgentType
    description: str
    input_data: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class AgentInstance:
    def __init__(self, agent_type: AgentType, instance_id: str):
        self.agent_type = agent_type
        self.instance_id = instance_id
        self.status = "idle"
        self.current_task_id = None
        self.last_activity = datetime.now()
        self.total_tasks = 0
        self.successful_tasks = 0

class AgentOrchestrator:
    """Main orchestrator for managing AI agents and tasks"""
    
    def __init__(self, config_path: str = None):
        # Detect environment and set paths accordingly
        if os.name == 'nt' or os.path.exists("X:/"):
            # Windows environment
            self.base_path = Path("X:/AI_Agent_Ecosystem")
            default_config_path = "X:/AI_Agent_Ecosystem/config"
        else:
            # Linux environment  
            self.base_path = Path("/mnt/llm/AI_Agent_Ecosystem")
            default_config_path = "/mnt/llm/AI_Agent_Ecosystem/config"
            
        if config_path is None:
            config_path = default_config_path
        self.config_path = Path(config_path)
        self.db_path = self.base_path / "data" / "orchestrator.db"
        
        # Ensure directories exist
        (self.base_path / "data").mkdir(exist_ok=True)
        (self.base_path / "logs").mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize database
        self.database_available = False
        self._init_database()
        
        # Task management
        self.task_queues = {agent_type: asyncio.Queue() for agent_type in AgentType}
        self.active_tasks = {}
        self.agent_instances = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # Performance tracking
        self.metrics = {
            'tasks_created': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'start_time': datetime.now()
        }
        
        # Cache for results
        self.result_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        
        logger.info("AgentOrchestrator initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        config_file = self.config_path / "system.yaml"
        
        # Default configuration
        default_config = {
            'max_workers': 4,
            'task_timeout': 300,
            'cache_ttl': 3600,
            'max_concurrent_tasks': 10,
            'agent_configs': {
                AgentType.DOCUMENT_ANALYZER: {'max_instances': 2, 'timeout': 120},
                AgentType.CODE_GENERATOR: {'max_instances': 1, 'timeout': 300},
                AgentType.RESEARCH_ASSISTANT: {'max_instances': 2, 'timeout': 180},
                AgentType.DATA_PROCESSOR: {'max_instances': 1, 'timeout': 240},
                AgentType.TASK_COORDINATOR: {'max_instances': 1, 'timeout': 60}
            }
        }
        
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using defaults.")
        
        return default_config

    def _init_database(self):
        """Initialize SQLite database for task and agent tracking"""
        try:
            # Ensure parent directories exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    cache_key TEXT
                )
            ''')
            
            # Agent instances table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_instances (
                    instance_id TEXT PRIMARY KEY,
                    agent_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_task_id TEXT,
                    last_activity TEXT NOT NULL,
                    total_tasks INTEGER DEFAULT 0,
                    successful_tasks INTEGER DEFAULT 0
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    metric_name TEXT PRIMARY KEY,
                    metric_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Result cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS result_cache (
                    cache_key TEXT PRIMARY KEY,
                    result TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized successfully at {self.db_path}")
            self.database_available = True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            logger.warning("Continuing without database - some features may be limited")
            self.database_available = False

    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def generate_cache_key(self, agent_type: AgentType, input_data: Dict[str, Any]) -> str:
        """Generate cache key for task results"""
        cache_input = {
            'agent_type': agent_type.value,
            'input_data': input_data
        }
        return hashlib.md5(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()

    def create_task(self, 
                   agent_type: AgentType, 
                   description: str, 
                   input_data: Dict[str, Any],
                   priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Create a new task and add it to the appropriate queue"""
        
        task_id = self.generate_task_id()
        cache_key = self.generate_cache_key(agent_type, input_data)
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Returning cached result for task {task_id}")
            # Create completed task with cached result
            task = Task(
                id=task_id,
                agent_type=agent_type,
                description=description,
                input_data=input_data,
                priority=priority,
                status=TaskStatus.COMPLETED,
                result=cached_result,
                completed_at=datetime.now()
            )
        else:
            task = Task(
                id=task_id,
                agent_type=agent_type,
                description=description,
                input_data=input_data,
                priority=priority
            )
        
        # Store in database
        self._save_task(task, cache_key)
        
        # Add to appropriate queue if not cached
        if not cached_result:
            asyncio.create_task(self._add_task_to_queue(task))
        
        # Update metrics
        self.metrics['tasks_created'] += 1
        
        logger.info(f"Created task {task_id}: {description}")
        return task_id

    async def _add_task_to_queue(self, task: Task):
        """Add task to the appropriate agent queue"""
        await self.task_queues[task.agent_type].put(task)
        logger.debug(f"Task {task.id} added to {task.agent_type.value} queue")

    def _save_task(self, task: Task, cache_key: str = None):
        """Save task to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO tasks 
            (id, agent_type, description, input_data, priority, status, result, 
             created_at, started_at, completed_at, error_message, retry_count, cache_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.id,
            task.agent_type.value,
            task.description,
            json.dumps(task.input_data),
            task.priority.value,
            task.status.value,
            json.dumps(task.result) if task.result else None,
            task.created_at.isoformat(),
            task.started_at.isoformat() if task.started_at else None,
            task.completed_at.isoformat() if task.completed_at else None,
            task.error_message,
            task.retry_count,
            cache_key
        ))
        
        conn.commit()
        conn.close()

    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT result, expires_at FROM result_cache 
            WHERE cache_key = ? AND expires_at > ?
        ''', (cache_key, datetime.now().isoformat()))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache task result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        expires_at = datetime.now().timestamp() + self.cache_ttl
        
        cursor.execute('''
            INSERT OR REPLACE INTO result_cache 
            (cache_key, result, created_at, expires_at)
            VALUES (?, ?, ?, ?)
        ''', (
            cache_key,
            json.dumps(result),
            datetime.now().isoformat(),
            datetime.fromtimestamp(expires_at).isoformat()
        ))
        
        conn.commit()
        conn.close()

    async def start_agent_workers(self):
        """Start agent worker processes"""
        logger.info("Starting agent workers...")
        
        # Import agent classes
        from agents.document_analyzer import DocumentAnalyzerAgent
        from agents.code_generator import CodeGeneratorAgent
        from agents.research_assistant import ResearchAssistantAgent
        from agents.data_processor import DataProcessorAgent
        from agents.task_coordinator import TaskCoordinatorAgent
        
        agent_classes = {
            AgentType.DOCUMENT_ANALYZER: DocumentAnalyzerAgent,
            AgentType.CODE_GENERATOR: CodeGeneratorAgent,
            AgentType.RESEARCH_ASSISTANT: ResearchAssistantAgent,
            AgentType.DATA_PROCESSOR: DataProcessorAgent,
            AgentType.TASK_COORDINATOR: TaskCoordinatorAgent
        }
        
        # Start workers for each agent type
        for agent_type in AgentType:
            max_instances = self.config['agent_configs'][agent_type]['max_instances']
            
            for i in range(max_instances):
                instance_id = f"{agent_type.value}_{i}"
                agent_class = agent_classes[agent_type]
                
                # Create agent instance
                agent_instance = AgentInstance(agent_type, instance_id)
                self.agent_instances[instance_id] = agent_instance
                
                # Start worker task
                asyncio.create_task(
                    self._agent_worker(agent_type, instance_id, agent_class)
                )
                
                logger.info(f"Started agent worker: {instance_id}")

    async def _agent_worker(self, agent_type: AgentType, instance_id: str, agent_class):
        """Worker process for a specific agent instance"""
        logger.info(f"Agent worker {instance_id} started")
        
        agent = agent_class(self)
        agent_instance = self.agent_instances[instance_id]
        
        while True:
            try:
                # Get task from queue
                task = await self.task_queues[agent_type].get()
                
                # Update agent status
                agent_instance.status = "busy"
                agent_instance.current_task_id = task.id
                agent_instance.last_activity = datetime.now()
                
                # Update task status
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()
                self._save_task(task)
                self.active_tasks[task.id] = task
                
                logger.info(f"Agent {instance_id} processing task {task.id}")
                
                try:
                    # Process task with timeout
                    timeout = self.config['agent_configs'][agent_type]['timeout']
                    result = await asyncio.wait_for(
                        agent.process_task(task),
                        timeout=timeout
                    )
                    
                    # Task completed successfully
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    
                    # Cache result
                    cache_key = self.generate_cache_key(agent_type, task.input_data)
                    self._cache_result(cache_key, result)
                    
                    # Update metrics
                    agent_instance.successful_tasks += 1
                    self.metrics['tasks_completed'] += 1
                    
                    logger.info(f"Task {task.id} completed successfully by {instance_id}")
                    
                except asyncio.TimeoutError:
                    task.status = TaskStatus.FAILED
                    task.error_message = f"Task timed out after {timeout} seconds"
                    task.completed_at = datetime.now()
                    
                    self.metrics['tasks_failed'] += 1
                    logger.error(f"Task {task.id} timed out")
                    
                except Exception as e:
                    # Handle task failure
                    task.retry_count += 1
                    
                    if task.retry_count <= task.max_retries:
                        # Retry task
                        task.status = TaskStatus.PENDING
                        task.error_message = f"Attempt {task.retry_count}: {str(e)}"
                        
                        # Add back to queue with delay
                        await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                        await self.task_queues[agent_type].put(task)
                        
                        logger.warning(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
                    else:
                        # Max retries exceeded
                        task.status = TaskStatus.FAILED
                        task.error_message = f"Max retries exceeded. Last error: {str(e)}"
                        task.completed_at = datetime.now()
                        
                        self.metrics['tasks_failed'] += 1
                        logger.error(f"Task {task.id} failed permanently: {e}")
                
                # Save final task state
                self._save_task(task)
                
                # Remove from active tasks
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                
                # Update agent status
                agent_instance.status = "idle"
                agent_instance.current_task_id = None
                agent_instance.total_tasks += 1
                agent_instance.last_activity = datetime.now()
                
                # Mark queue task as done
                self.task_queues[agent_type].task_done()
                
            except Exception as e:
                logger.error(f"Error in agent worker {instance_id}: {e}")
                await asyncio.sleep(1)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if not self.database_available:
            # Check active tasks in memory
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    'task_id': task.id,
                    'agent_type': task.agent_type.value,
                    'description': task.description,
                    'status': task.status.value,
                    'result': task.result,
                    'created_at': task.created_at.isoformat(),
                    'started_at': task.started_at.isoformat() if task.started_at else None,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                    'error_message': task.error_message,
                    'retry_count': task.retry_count
                }
            return None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'task_id': row[0],
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
            logger.warning(f"Could not retrieve task status from database: {e}")
            
        return None

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        # Default metrics structure
        task_stats = {}
        agent_stats = {}
        
        # Try to get database metrics if available
        if self.database_available:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Task statistics
                cursor.execute('''
                    SELECT status, COUNT(*) FROM tasks GROUP BY status
                ''')
                task_stats = dict(cursor.fetchall())
                
                # Agent statistics
                for agent_type in AgentType:
                    cursor.execute('''
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                            AVG(CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
                                THEN (julianday(completed_at) - julianday(started_at)) * 24 * 60 * 60 
                                ELSE NULL END) as avg_duration
                        FROM tasks WHERE agent_type = ?
                    ''', (agent_type.value,))
                    
                    row = cursor.fetchone()
                    if row and row[0] > 0:
                        total, completed, avg_duration = row
                        success_rate = completed / total if total > 0 else 0
                        
                        agent_stats[agent_type.value] = {
                            'total_tasks': total,
                            'completed_tasks': completed,
                            'success_rate': success_rate,
                            'avg_duration_seconds': avg_duration or 0
                        }
                
                conn.close()
                
            except Exception as e:
                logger.warning(f"Could not retrieve database metrics: {e}")
                # Fall back to in-memory metrics
                task_stats = {
                    'created': self.metrics.get('tasks_created', 0),
                    'completed': self.metrics.get('tasks_completed', 0),
                    'failed': self.metrics.get('tasks_failed', 0)
                }
        else:
            # Use in-memory metrics when database is unavailable
            task_stats = {
                'created': self.metrics.get('tasks_created', 0),
                'completed': self.metrics.get('tasks_completed', 0),
                'failed': self.metrics.get('tasks_failed', 0)
            }
        
        # Current system state
        uptime = datetime.now() - self.metrics['start_time']
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'task_stats': task_stats,
            'agent_stats': agent_stats,
            'active_tasks': len(self.active_tasks),
            'agent_instances': {
                instance_id: {
                    'agent_type': instance.agent_type.value,
                    'status': instance.status,
                    'current_task_id': instance.current_task_id,
                    'total_tasks': instance.total_tasks,
                    'successful_tasks': instance.successful_tasks,
                    'success_rate': instance.successful_tasks / instance.total_tasks if instance.total_tasks > 0 else 0
                }
                for instance_id, instance in self.agent_instances.items()
            },
            'queue_sizes': {
                agent_type.value: self.task_queues[agent_type].qsize()
                for agent_type in AgentType
            },
            'database_available': self.database_available
        }

    async def shutdown(self):
        """Graceful shutdown of the orchestrator"""
        logger.info("Shutting down orchestrator...")
        
        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.status = TaskStatus.CANCELLED
            self._save_task(task)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Orchestrator shutdown complete")

# Main execution
async def main():
    """Main entry point for the orchestrator"""
    orchestrator = AgentOrchestrator()
    
    try:
        # Start agent workers
        await orchestrator.start_agent_workers()
        
        logger.info("Orchestrator is running. Press Ctrl+C to stop.")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())