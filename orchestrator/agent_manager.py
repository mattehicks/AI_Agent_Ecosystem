#!/usr/bin/env python3
"""
Agent Manager for AI Agent Ecosystem
Manages agent lifecycle, health, and resource allocation
"""

import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path
import importlib
import sys

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    STARTING = "starting"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"

@dataclass
class AgentInstance:
    instance_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.STARTING
    current_task_id: Optional[str] = None
    last_activity: datetime = field(default_factory=datetime.now)
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_task_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    agent_instance: Any = None
    process_id: Optional[int] = None

class AgentPool:
    """Manages a pool of agent instances for a specific agent type"""
    
    def __init__(self, agent_type: str, max_instances: int = 3):
        self.agent_type = agent_type
        self.max_instances = max_instances
        self.instances = {}
        self.next_instance_id = 0
        
    def create_instance(self) -> Optional[AgentInstance]:
        """Create a new agent instance"""
        if len(self.instances) >= self.max_instances:
            return None
            
        instance_id = f"{self.agent_type}_{self.next_instance_id}"
        self.next_instance_id += 1
        
        try:
            # Dynamically import and create agent
            module_path = f"agents.{self.agent_type}"
            agent_module = importlib.import_module(module_path)
            
            # Get agent class (capitalize first letter of each word)
            class_name = ''.join(word.capitalize() for word in self.agent_type.split('_')) + 'Agent'
            agent_class = getattr(agent_module, class_name)
            
            # Create agent instance
            agent_instance = agent_class()
            
            instance = AgentInstance(
                instance_id=instance_id,
                agent_type=self.agent_type,
                agent_instance=agent_instance,
                process_id=psutil.Process().pid
            )
            
            self.instances[instance_id] = instance
            logger.info(f"Created agent instance: {instance_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create agent instance {instance_id}: {e}")
            return None
    
    def get_available_instance(self) -> Optional[AgentInstance]:
        """Get an available agent instance"""
        for instance in self.instances.values():
            if instance.status == AgentStatus.IDLE:
                return instance
        return None
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """Get statistics for this agent pool"""
        total_instances = len(self.instances)
        idle_instances = len([i for i in self.instances.values() if i.status == AgentStatus.IDLE])
        busy_instances = len([i for i in self.instances.values() if i.status == AgentStatus.BUSY])
        error_instances = len([i for i in self.instances.values() if i.status == AgentStatus.ERROR])
        
        total_tasks = sum(i.total_tasks for i in self.instances.values())
        successful_tasks = sum(i.successful_tasks for i in self.instances.values())
        failed_tasks = sum(i.failed_tasks for i in self.instances.values())
        
        avg_memory = sum(i.memory_usage_mb for i in self.instances.values()) / max(total_instances, 1)
        avg_cpu = sum(i.cpu_usage_percent for i in self.instances.values()) / max(total_instances, 1)
        
        return {
            'agent_type': self.agent_type,
            'total_instances': total_instances,
            'idle_instances': idle_instances,
            'busy_instances': busy_instances,
            'error_instances': error_instances,
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'success_rate': successful_tasks / max(total_tasks, 1),
            'average_memory_mb': avg_memory,
            'average_cpu_percent': avg_cpu
        }

class AgentManager:
    """Main agent management system"""
    
    def __init__(self, db_path: Path, config: Dict[str, Any]):
        self.db_path = db_path
        self.config = config
        self.agent_pools = {}
        self.monitoring_task = None
        self.auto_scaling_enabled = True
        self.performance_history = {}
        
        # Initialize agent pools
        agent_configs = config.get('agent_configs', {})
        for agent_type, agent_config in agent_configs.items():
            if isinstance(agent_type, Enum):
                agent_type = agent_type.value
            max_instances = agent_config.get('max_instances', 2)
            self.agent_pools[agent_type] = AgentPool(agent_type, max_instances)
    
    async def start(self):
        """Start agent management services"""
        logger.info("Starting Agent Manager")
        
        # Create initial agent instances
        for agent_type, pool in self.agent_pools.items():
            for _ in range(1):  # Start with 1 instance per type
                instance = pool.create_instance()
                if instance:
                    instance.status = AgentStatus.IDLE
                    await self._register_instance_in_db(instance)
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitor_agents())
        
        # Start auto-scaling
        asyncio.create_task(self._auto_scale_agents())
    
    async def stop(self):
        """Stop all agents and management services"""
        logger.info("Stopping Agent Manager")
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Stop all agent instances
        for pool in self.agent_pools.values():
            for instance in pool.instances.values():
                instance.status = AgentStatus.STOPPING
                # Graceful shutdown logic here
                instance.status = AgentStatus.STOPPED
    
    def get_available_agent(self, agent_type: str) -> Optional[AgentInstance]:
        """Get available agent instance for task assignment"""
        pool = self.agent_pools.get(agent_type)
        if not pool:
            return None
        
        instance = pool.get_available_instance()
        if instance:
            instance.status = AgentStatus.BUSY
            instance.last_activity = datetime.now()
            asyncio.create_task(self._update_instance_in_db(instance))
        
        return instance
    
    def release_agent(self, instance_id: str, task_success: bool = True):
        """Release agent back to idle state"""
        for pool in self.agent_pools.values():
            if instance_id in pool.instances:
                instance = pool.instances[instance_id]
                instance.status = AgentStatus.IDLE
                instance.current_task_id = None
                instance.last_activity = datetime.now()
                instance.total_tasks += 1
                
                if task_success:
                    instance.successful_tasks += 1
                else:
                    instance.failed_tasks += 1
                
                asyncio.create_task(self._update_instance_in_db(instance))
                break
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        stats = {}
        
        for agent_type, pool in self.agent_pools.items():
            stats[agent_type] = pool.get_instance_stats()
        
        # Overall system stats
        total_instances = sum(len(pool.instances) for pool in self.agent_pools.values())
        total_idle = sum(len([i for i in pool.instances.values() if i.status == AgentStatus.IDLE]) 
                        for pool in self.agent_pools.values())
        total_busy = sum(len([i for i in pool.instances.values() if i.status == AgentStatus.BUSY]) 
                        for pool in self.agent_pools.values())
        
        stats['system_overview'] = {
            'total_instances': total_instances,
            'idle_instances': total_idle,
            'busy_instances': total_busy,
            'utilization_percent': (total_busy / max(total_instances, 1)) * 100
        }
        
        return stats
    
    def scale_agent_pool(self, agent_type: str, target_instances: int) -> bool:
        """Manually scale agent pool"""
        pool = self.agent_pools.get(agent_type)
        if not pool:
            return False
        
        current_count = len(pool.instances)
        
        if target_instances > current_count:
            # Scale up
            for _ in range(target_instances - current_count):
                if len(pool.instances) < pool.max_instances:
                    instance = pool.create_instance()
                    if instance:
                        instance.status = AgentStatus.IDLE
                        asyncio.create_task(self._register_instance_in_db(instance))
        
        elif target_instances < current_count:
            # Scale down - stop idle instances first
            to_remove = []
            for instance_id, instance in pool.instances.items():
                if instance.status == AgentStatus.IDLE and len(to_remove) < (current_count - target_instances):
                    to_remove.append(instance_id)
            
            for instance_id in to_remove:
                instance = pool.instances[instance_id]
                instance.status = AgentStatus.STOPPED
                del pool.instances[instance_id]
                asyncio.create_task(self._remove_instance_from_db(instance_id))
        
        return True
    
    async def _monitor_agents(self):
        """Monitor agent health and performance"""
        while True:
            try:
                for pool in self.agent_pools.values():
                    for instance in pool.instances.values():
                        # Update resource usage
                        try:
                            process = psutil.Process(instance.process_id) if instance.process_id else psutil.Process()
                            instance.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                            instance.cpu_usage_percent = process.cpu_percent()
                        except:
                            pass
                        
                        # Check for stuck tasks
                        if (instance.status == AgentStatus.BUSY and 
                            instance.last_activity < datetime.now() - timedelta(minutes=30)):
                            logger.warning(f"Agent {instance.instance_id} appears stuck")
                            instance.status = AgentStatus.ERROR
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _auto_scale_agents(self):
        """Automatic scaling based on queue size and performance"""
        while True:
            try:
                if not self.auto_scaling_enabled:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                for agent_type, pool in self.agent_pools.items():
                    # Get queue size (would need integration with task manager)
                    # For now, implement basic scaling logic
                    
                    idle_count = len([i for i in pool.instances.values() if i.status == AgentStatus.IDLE])
                    busy_count = len([i for i in pool.instances.values() if i.status == AgentStatus.BUSY])
                    total_count = len(pool.instances)
                    
                    # Scale up if all instances are busy and under max
                    if idle_count == 0 and busy_count > 0 and total_count < pool.max_instances:
                        logger.info(f"Auto-scaling up {agent_type}: {total_count} -> {total_count + 1}")
                        instance = pool.create_instance()
                        if instance:
                            instance.status = AgentStatus.IDLE
                            await self._register_instance_in_db(instance)
                    
                    # Scale down if too many idle instances
                    elif idle_count > 2 and total_count > 1:
                        # Remove one idle instance
                        for instance_id, instance in pool.instances.items():
                            if instance.status == AgentStatus.IDLE:
                                logger.info(f"Auto-scaling down {agent_type}: {total_count} -> {total_count - 1}")
                                instance.status = AgentStatus.STOPPED
                                del pool.instances[instance_id]
                                await self._remove_instance_from_db(instance_id)
                                break
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")
                await asyncio.sleep(300)
    
    async def _register_instance_in_db(self, instance: AgentInstance):
        """Register agent instance in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO agent_instances
                (instance_id, agent_type, status, current_task_id, last_activity, 
                 total_tasks, successful_tasks)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                instance.instance_id, instance.agent_type, instance.status.value,
                instance.current_task_id, instance.last_activity.isoformat(),
                instance.total_tasks, instance.successful_tasks
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to register instance in database: {e}")
    
    async def _update_instance_in_db(self, instance: AgentInstance):
        """Update agent instance in database"""
        await self._register_instance_in_db(instance)
    
    async def _remove_instance_from_db(self, instance_id: str):
        """Remove agent instance from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM agent_instances WHERE instance_id = ?', (instance_id,))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to remove instance from database: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'agent_pools': {},
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
        }
        
        for agent_type, pool in self.agent_pools.items():
            pool_stats = pool.get_instance_stats()
            health_data['agent_pools'][agent_type] = pool_stats
        
        return health_data