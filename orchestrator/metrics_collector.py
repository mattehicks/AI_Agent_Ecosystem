#!/usr/bin/env python3
"""
Metrics Collector for AI Agent Ecosystem
Collects, stores, and provides system performance metrics
"""

import asyncio
import json
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import sqlite3
from pathlib import Path
import time

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    timestamp: datetime
    metric_name: str
    metric_value: Any
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.collection_interval = 60  # seconds
        self.retention_days = 7
        self.metrics_buffer = []
        self.collection_task = None
        self.cleanup_task = None
        
        # Performance counters
        self.start_time = time.time()
        self.task_counters = {
            'created': 0,
            'completed': 0,
            'failed': 0,
            'cancelled': 0
        }
        
    async def start(self):
        """Start metrics collection"""
        logger.info("Starting Metrics Collector")
        
        # Start collection task
        self.collection_task = asyncio.create_task(self._collect_metrics_loop())
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_old_metrics())
        
    async def stop(self):
        """Stop metrics collection"""
        logger.info("Stopping Metrics Collector")
        
        if self.collection_task:
            self.collection_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
            
        # Flush remaining metrics
        await self._flush_metrics_buffer()
    
    def record_metric(self, name: str, value: Any, tags: Dict[str, str] = None):
        """Record a custom metric"""
        metric = MetricPoint(
            timestamp=datetime.now(),
            metric_name=name,
            metric_value=value,
            tags=tags or {}
        )
        self.metrics_buffer.append(metric)
        
        # Flush buffer if it gets too large
        if len(self.metrics_buffer) > 100:
            asyncio.create_task(self._flush_metrics_buffer())
    
    def increment_counter(self, counter_name: str, amount: int = 1):
        """Increment a task counter"""
        if counter_name in self.task_counters:
            self.task_counters[counter_name] += amount
    
    async def get_metrics(self, metric_name: str = None, 
                         start_time: datetime = None, 
                         end_time: datetime = None,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve metrics from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM metrics WHERE 1=1"
            params = []
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            if start_time:
                query += " AND updated_at >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND updated_at <= ?"
                params.append(end_time.isoformat())
            
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            metrics = []
            for row in rows:
                try:
                    metric_value = json.loads(row[1])
                except:
                    metric_value = row[1]
                
                metrics.append({
                    'metric_name': row[0],
                    'metric_value': metric_value,
                    'updated_at': row[2]
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            return []
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        current_time = datetime.now()
        uptime = time.time() - self.start_time
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network stats
        try:
            network = psutil.net_io_counters()
            network_stats = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        except:
            network_stats = {}
        
        # Process info
        try:
            process = psutil.Process()
            process_stats = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        except:
            process_stats = {}
        
        metrics = {
            'timestamp': current_time.isoformat(),
            'uptime_seconds': int(uptime),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'memory_used_mb': memory.used / 1024 / 1024,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'disk_used_gb': disk.used / 1024 / 1024 / 1024
            },
            'network': network_stats,
            'process': process_stats,
            'task_stats': self.task_counters.copy(),
            'database_available': True
        }
        
        return metrics
    
    async def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        try:
            # Get task completion rates
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Task statistics
            cursor.execute('''
                SELECT status, COUNT(*) 
                FROM tasks 
                WHERE created_at >= ? 
                GROUP BY status
            ''', (start_time.isoformat(),))
            
            task_stats = dict(cursor.fetchall())
            
            # Average task completion time
            cursor.execute('''
                SELECT AVG(
                    CASE 
                        WHEN completed_at IS NOT NULL AND started_at IS NOT NULL 
                        THEN (julianday(completed_at) - julianday(started_at)) * 86400
                        ELSE NULL 
                    END
                ) as avg_completion_time
                FROM tasks 
                WHERE created_at >= ? AND status = 'completed'
            ''', (start_time.isoformat(),))
            
            avg_completion_time = cursor.fetchone()[0] or 0
            
            # Agent performance
            cursor.execute('''
                SELECT agent_type, 
                       COUNT(*) as total_tasks,
                       SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_tasks
                FROM tasks 
                WHERE created_at >= ?
                GROUP BY agent_type
            ''', (start_time.isoformat(),))
            
            agent_performance = {}
            for row in cursor.fetchall():
                agent_type, total, completed = row
                agent_performance[agent_type] = {
                    'total_tasks': total,
                    'completed_tasks': completed,
                    'success_rate': completed / max(total, 1)
                }
            
            conn.close()
            
            # System resource averages (from metrics table)
            resource_metrics = await self.get_metrics('system_resources', start_time, end_time)
            
            avg_cpu = 0
            avg_memory = 0
            if resource_metrics:
                cpu_values = [m['metric_value'].get('cpu_percent', 0) for m in resource_metrics if isinstance(m['metric_value'], dict)]
                memory_values = [m['metric_value'].get('memory_percent', 0) for m in resource_metrics if isinstance(m['metric_value'], dict)]
                
                avg_cpu = sum(cpu_values) / max(len(cpu_values), 1)
                avg_memory = sum(memory_values) / max(len(memory_values), 1)
            
            return {
                'period_hours': hours,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'task_statistics': task_stats,
                'average_completion_time_seconds': avg_completion_time,
                'agent_performance': agent_performance,
                'system_averages': {
                    'cpu_percent': avg_cpu,
                    'memory_percent': avg_memory
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {
                'period_hours': hours,
                'error': str(e)
            }
    
    async def _collect_metrics_loop(self):
        """Main metrics collection loop"""
        while True:
            try:
                # Collect system metrics
                system_metrics = await self.get_system_metrics()
                
                # Store key metrics in database
                await self._store_metric('system_resources', {
                    'cpu_percent': system_metrics['system']['cpu_percent'],
                    'memory_percent': system_metrics['system']['memory_percent'],
                    'disk_percent': system_metrics['system']['disk_percent']
                })
                
                await self._store_metric('task_counters', self.task_counters.copy())
                await self._store_metric('uptime_seconds', system_metrics['uptime_seconds'])
                
                # Flush metrics buffer
                await self._flush_metrics_buffer()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _store_metric(self, name: str, value: Any):
        """Store a metric in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO metrics (metric_name, metric_value, updated_at)
                VALUES (?, ?, ?)
            ''', (name, json.dumps(value), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store metric {name}: {e}")
    
    async def _flush_metrics_buffer(self):
        """Flush buffered metrics to database"""
        if not self.metrics_buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for metric in self.metrics_buffer:
                cursor.execute('''
                    INSERT INTO metrics (metric_name, metric_value, updated_at)
                    VALUES (?, ?, ?)
                ''', (
                    metric.metric_name,
                    json.dumps(metric.metric_value),
                    metric.timestamp.isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics buffer: {e}")
    
    async def _cleanup_old_metrics(self):
        """Cleanup old metrics data"""
        while True:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                cutoff_date = datetime.now() - timedelta(days=self.retention_days)
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM metrics 
                    WHERE updated_at < ? 
                    AND metric_name NOT IN ('system_config', 'agent_config')
                ''', (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old metric records")
                
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")

class AlertManager:
    """Manages performance alerts and thresholds"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'task_failure_rate': 0.2,
            'avg_response_time': 300  # seconds
        }
        self.active_alerts = {}
        
    async def check_alerts(self):
        """Check for alert conditions"""
        try:
            current_metrics = await self.metrics_collector.get_system_metrics()
            
            # CPU alert
            cpu_percent = current_metrics['system']['cpu_percent']
            if cpu_percent > self.alert_thresholds['cpu_percent']:
                await self._trigger_alert('high_cpu', f"CPU usage at {cpu_percent}%")
            else:
                await self._clear_alert('high_cpu')
            
            # Memory alert
            memory_percent = current_metrics['system']['memory_percent']
            if memory_percent > self.alert_thresholds['memory_percent']:
                await self._trigger_alert('high_memory', f"Memory usage at {memory_percent}%")
            else:
                await self._clear_alert('high_memory')
            
            # Disk alert
            disk_percent = current_metrics['system']['disk_percent']
            if disk_percent > self.alert_thresholds['disk_percent']:
                await self._trigger_alert('high_disk', f"Disk usage at {disk_percent}%")
            else:
                await self._clear_alert('high_disk')
            
            # Task failure rate
            task_stats = current_metrics['task_stats']
            total_tasks = sum(task_stats.values())
            if total_tasks > 10:  # Only check if we have enough data
                failure_rate = task_stats.get('failed', 0) / total_tasks
                if failure_rate > self.alert_thresholds['task_failure_rate']:
                    await self._trigger_alert('high_failure_rate', f"Task failure rate at {failure_rate:.1%}")
                else:
                    await self._clear_alert('high_failure_rate')
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert"""
        if alert_type not in self.active_alerts:
            self.active_alerts[alert_type] = {
                'triggered_at': datetime.now(),
                'message': message
            }
            logger.warning(f"ALERT: {alert_type} - {message}")
            
            # Store alert in metrics
            self.metrics_collector.record_metric(
                f"alert_{alert_type}",
                {'message': message, 'triggered_at': datetime.now().isoformat()}
            )
    
    async def _clear_alert(self, alert_type: str):
        """Clear an active alert"""
        if alert_type in self.active_alerts:
            duration = datetime.now() - self.active_alerts[alert_type]['triggered_at']
            logger.info(f"CLEARED: {alert_type} after {duration}")
            del self.active_alerts[alert_type]
    
    def get_active_alerts(self) -> Dict[str, Any]:
        """Get currently active alerts"""
        return self.active_alerts.copy()