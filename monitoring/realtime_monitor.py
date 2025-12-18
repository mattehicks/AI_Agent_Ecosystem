#!/usr/bin/env python3
"""
Real-time GPU and System Monitoring Service
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import deque
import json

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from gpu_platform.gpu_manager import GPUManager
from config.settings import get_config
from status_tracker import status_tracker

logger = logging.getLogger(__name__)

class MetricsHistory:
    """Stores historical metrics data"""
    
    def __init__(self, max_points: int = 100):
        self.max_points = max_points
        self.gpu_history: Dict[int, deque] = {}
        self.system_history: deque = deque(maxlen=max_points)
    
    def add_gpu_metrics(self, gpu_id: int, metrics: Dict[str, Any]):
        """Add GPU metrics to history"""
        if gpu_id not in self.gpu_history:
            self.gpu_history[gpu_id] = deque(maxlen=self.max_points)
        
        metrics['timestamp'] = datetime.now().isoformat()
        self.gpu_history[gpu_id].append(metrics)
    
    def add_system_metrics(self, metrics: Dict[str, Any]):
        """Add system metrics to history"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.system_history.append(metrics)
    
    def get_gpu_history(self, gpu_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get GPU history"""
        if gpu_id not in self.gpu_history:
            return []
        
        history = list(self.gpu_history[gpu_id])
        if limit:
            history = history[-limit:]
        return history
    
    def get_system_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get system history"""
        history = list(self.system_history)
        if limit:
            history = history[-limit:]
        return history

class RealtimeMonitor:
    """Real-time monitoring service for GPU and system metrics"""
    
    def __init__(self):
        self.gpu_manager = GPUManager()
        self.metrics_history = MetricsHistory()
        self.subscribers: List[Callable] = []
        self.monitoring_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Configuration
        self.config = get_config()
        self.gpu_update_interval = self.config.monitoring.gpu_update_interval
        self.system_update_interval = self.config.monitoring.system_metrics_interval
        
        # Subscribe to config changes
        from config.settings import config_manager
        config_manager.subscribe_to_changes(self._on_config_change)
    
    def _on_config_change(self, new_config):
        """Handle configuration changes"""
        self.config = new_config
        old_gpu_interval = self.gpu_update_interval
        old_system_interval = self.system_update_interval
        
        self.gpu_update_interval = new_config.monitoring.gpu_update_interval
        self.system_update_interval = new_config.monitoring.system_metrics_interval
        
        # Restart monitoring if intervals changed
        if (old_gpu_interval != self.gpu_update_interval or 
            old_system_interval != self.system_update_interval):
            if self.is_running:
                logger.info("Restarting monitoring with new intervals")
                asyncio.create_task(self._restart_monitoring())
    
    async def _restart_monitoring(self):
        """Restart monitoring with new settings"""
        await self.stop()
        await asyncio.sleep(0.5)  # Brief pause
        await self.start()
    
    def subscribe(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Subscribe to real-time updates"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast update to all subscribers"""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(update_type, data)
                else:
                    callback(update_type, data)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")
    
    async def start(self):
        """Start real-time monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("Starting real-time monitoring service")
        
        # Start GPU monitoring if enabled
        if self.config.monitoring.enable_realtime_monitoring:
            gpu_task = asyncio.create_task(self._gpu_monitoring_loop())
            system_task = asyncio.create_task(self._system_monitoring_loop())
            
            self.monitoring_tasks = [gpu_task, system_task]
        
        # Log monitoring start
        status_tracker.log_info(
            "Monitoring Started", 
            f"Real-time monitoring active (GPU: {self.gpu_update_interval}s, System: {self.system_update_interval}s)"
        )
    
    async def stop(self):
        """Stop real-time monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping real-time monitoring service")
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        self.monitoring_tasks.clear()
        
        # Log monitoring stop
        status_tracker.log_info("Monitoring Stopped", "Real-time monitoring disabled")
    
    async def _gpu_monitoring_loop(self):
        """Main GPU monitoring loop"""
        logger.info(f"GPU monitoring started (interval: {self.gpu_update_interval}s)")
        
        while self.is_running:
            try:
                # Get current GPU metrics
                gpu_infos = self.gpu_manager.get_all_gpu_info()
                
                gpu_metrics = []
                for gpu in gpu_infos:
                    metrics = {
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_used': gpu.memory_used,
                        'memory_total': gpu.memory_total,
                        'memory_free': gpu.memory_free,
                        'utilization': gpu.utilization,
                        'temperature': gpu.temperature,
                        'power_usage': gpu.power_usage,
                        'power_limit': gpu.power_limit,
                        'current_model': gpu.current_model
                    }
                    
                    # Add to history
                    self.metrics_history.add_gpu_metrics(gpu.id, metrics.copy())
                    gpu_metrics.append(metrics)
                
                # Broadcast update
                await self._broadcast_update('gpu_metrics', {
                    'gpus': gpu_metrics,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Log GPU status update to status tracker
                status_tracker.log_gpu_status_update({
                    'gpu_count': len(gpu_metrics),
                    'avg_temperature': sum(g['temperature'] for g in gpu_metrics) / len(gpu_metrics) if gpu_metrics else 0,
                    'total_memory_used': sum(g['memory_used'] for g in gpu_metrics),
                    'models_loaded': sum(1 for g in gpu_metrics if g['current_model'])
                })
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
            
            await asyncio.sleep(self.gpu_update_interval)
    
    async def _system_monitoring_loop(self):
        """Main system monitoring loop"""
        logger.info(f"System monitoring started (interval: {self.system_update_interval}s)")
        
        while self.is_running:
            try:
                # Get system metrics
                import psutil
                
                system_metrics = {
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent if Path('/').exists() else 0,
                    'network_io': dict(psutil.net_io_counters()._asdict()) if hasattr(psutil.net_io_counters(), '_asdict') else {},
                    'process_count': len(psutil.pids()),
                    'uptime': time.time() - psutil.boot_time()
                }
                
                # Add to history
                self.metrics_history.add_system_metrics(system_metrics.copy())
                
                # Broadcast update
                await self._broadcast_update('system_metrics', {
                    'metrics': system_metrics,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
            
            await asyncio.sleep(self.system_update_interval)
    
    def get_current_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics"""
        gpu_infos = self.gpu_manager.get_all_gpu_info()
        return {
            'gpus': [
                {
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used': gpu.memory_used,
                    'memory_total': gpu.memory_total,
                    'utilization': gpu.utilization,
                    'temperature': gpu.temperature,
                    'power_usage': gpu.power_usage,
                    'current_model': gpu.current_model
                }
                for gpu in gpu_infos
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_gpu_history(self, gpu_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get GPU metrics history"""
        return self.metrics_history.get_gpu_history(gpu_id, limit)
    
    def get_system_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get system metrics history"""
        return self.metrics_history.get_system_history(limit)

# Global monitor instance
realtime_monitor = RealtimeMonitor()