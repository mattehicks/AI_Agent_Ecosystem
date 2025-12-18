#!/usr/bin/env python3
"""
Real-time Status and Event Tracking System
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

class EventType(Enum):
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    MODEL_LOAD_START = "model_load_start"
    MODEL_LOAD_PROGRESS = "model_load_progress"
    MODEL_LOAD_COMPLETE = "model_load_complete"
    MODEL_LOAD_ERROR = "model_load_error"
    MODEL_UNLOAD = "model_unload"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_PROCESS_START = "document_process_start"
    DOCUMENT_PROCESS_PROGRESS = "document_process_progress"
    DOCUMENT_PROCESS_COMPLETE = "document_process_complete"
    DOCUMENT_PROCESS_ERROR = "document_process_error"
    GPU_STATUS_UPDATE = "gpu_status_update"
    AGENT_START = "agent_start"
    AGENT_STOP = "agent_stop"
    TASK_QUEUE_UPDATE = "task_queue_update"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class StatusLevel(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"
    PROCESSING = "processing"

@dataclass
class StatusEvent:
    id: str
    event_type: EventType
    level: StatusLevel
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    duration: Optional[float] = None  # For tracking how long operations take
    progress: Optional[int] = None    # 0-100 for progress tracking

class StatusTracker:
    """Real-time status and event tracking system"""
    
    def __init__(self):
        self.events: List[StatusEvent] = []
        self.active_operations: Dict[str, StatusEvent] = {}
        self.subscribers: List[callable] = []
        self.max_events = 1000  # Keep last 1000 events
        
    def subscribe(self, callback: callable):
        """Subscribe to status updates"""
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback: callable):
        """Unsubscribe from status updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def emit_event(self, event: StatusEvent):
        """Emit a status event to all subscribers"""
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        
        # Update active operations
        if event.event_type.value.endswith('_start'):
            self.active_operations[event.id] = event
        elif event.event_type.value.endswith(('_complete', '_error')):
            if event.id in self.active_operations:
                start_event = self.active_operations[event.id]
                event.duration = (event.timestamp - start_event.timestamp).total_seconds()
                del self.active_operations[event.id]
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in status callback: {e}")
    
    def log_system_start(self):
        """Log system startup"""
        event = StatusEvent(
            id="system_startup",
            event_type=EventType.SYSTEM_START,
            level=StatusLevel.SUCCESS,
            title="System Started",
            message="AI Agent Ecosystem is online",
            data={},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_model_load_start(self, model_name: str, gpu_ids: List[int]):
        """Log model loading start"""
        event = StatusEvent(
            id=f"model_load_{model_name}",
            event_type=EventType.MODEL_LOAD_START,
            level=StatusLevel.PROCESSING,
            title="Loading Model",
            message=f"Loading {model_name} on GPU(s) {gpu_ids}",
            data={"model_name": model_name, "gpu_ids": gpu_ids},
            timestamp=datetime.now(),
            progress=0
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_model_load_progress(self, model_name: str, progress: int, message: str = ""):
        """Log model loading progress"""
        event = StatusEvent(
            id=f"model_load_{model_name}",
            event_type=EventType.MODEL_LOAD_PROGRESS,
            level=StatusLevel.PROCESSING,
            title="Loading Model",
            message=message or f"Loading {model_name}... {progress}%",
            data={"model_name": model_name},
            timestamp=datetime.now(),
            progress=progress
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_model_load_complete(self, model_name: str, gpu_ids: List[int], memory_used: int):
        """Log model loading completion"""
        event = StatusEvent(
            id=f"model_load_{model_name}",
            event_type=EventType.MODEL_LOAD_COMPLETE,
            level=StatusLevel.SUCCESS,
            title="Model Loaded",
            message=f"{model_name} loaded successfully on GPU(s) {gpu_ids}",
            data={"model_name": model_name, "gpu_ids": gpu_ids, "memory_used_mb": memory_used},
            timestamp=datetime.now(),
            progress=100
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_model_load_error(self, model_name: str, error: str):
        """Log model loading error"""
        event = StatusEvent(
            id=f"model_load_{model_name}",
            event_type=EventType.MODEL_LOAD_ERROR,
            level=StatusLevel.ERROR,
            title="Model Load Failed",
            message=f"Failed to load {model_name}: {error}",
            data={"model_name": model_name, "error": error},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_document_upload(self, filename: str, size: int):
        """Log document upload"""
        event = StatusEvent(
            id=f"doc_upload_{filename}_{datetime.now().timestamp()}",
            event_type=EventType.DOCUMENT_UPLOAD,
            level=StatusLevel.INFO,
            title="Document Uploaded",
            message=f"Uploaded {filename} ({self.format_file_size(size)})",
            data={"filename": filename, "size": size},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_document_process_start(self, doc_id: str, filename: str):
        """Log document processing start"""
        event = StatusEvent(
            id=f"doc_process_{doc_id}",
            event_type=EventType.DOCUMENT_PROCESS_START,
            level=StatusLevel.PROCESSING,
            title="Processing Document",
            message=f"Analyzing {filename}",
            data={"doc_id": doc_id, "filename": filename},
            timestamp=datetime.now(),
            progress=0
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_document_process_complete(self, doc_id: str, filename: str):
        """Log document processing completion"""
        event = StatusEvent(
            id=f"doc_process_{doc_id}",
            event_type=EventType.DOCUMENT_PROCESS_COMPLETE,
            level=StatusLevel.SUCCESS,
            title="Document Processed",
            message=f"Analysis complete for {filename}",
            data={"doc_id": doc_id, "filename": filename},
            timestamp=datetime.now(),
            progress=100
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_gpu_status_update(self, gpu_data: Dict[str, Any]):
        """Log GPU status update"""
        event = StatusEvent(
            id="gpu_status",
            event_type=EventType.GPU_STATUS_UPDATE,
            level=StatusLevel.INFO,
            title="GPU Status",
            message=f"GPU metrics updated",
            data=gpu_data,
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_error(self, title: str, message: str, data: Dict[str, Any] = None):
        """Log an error event"""
        event = StatusEvent(
            id=f"error_{datetime.now().timestamp()}",
            event_type=EventType.ERROR,
            level=StatusLevel.ERROR,
            title=title,
            message=message,
            data=data or {},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_warning(self, title: str, message: str, data: Dict[str, Any] = None):
        """Log a warning event"""
        event = StatusEvent(
            id=f"warning_{datetime.now().timestamp()}",
            event_type=EventType.WARNING,
            level=StatusLevel.WARNING,
            title=title,
            message=message,
            data=data or {},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_info(self, title: str, message: str, data: Dict[str, Any] = None):
        """Log an info event"""
        event = StatusEvent(
            id=f"info_{datetime.now().timestamp()}",
            event_type=EventType.INFO,
            level=StatusLevel.INFO,
            title=title,
            message=message,
            data=data or {},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent events for API"""
        recent = self.events[-limit:] if limit else self.events
        return [self.event_to_dict(event) for event in reversed(recent)]
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get currently active operations"""
        return [self.event_to_dict(event) for event in self.active_operations.values()]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        recent_errors = [e for e in self.events[-20:] if e.level == StatusLevel.ERROR]
        active_ops = len(self.active_operations)
        
        if recent_errors:
            status = "error"
            message = f"Recent errors detected. {active_ops} operations active."
        elif active_ops > 0:
            status = "processing"
            message = f"{active_ops} operations in progress"
        else:
            status = "idle"
            message = "System ready"
        
        return {
            "status": status,
            "message": message,
            "active_operations": active_ops,
            "recent_events": len(self.events),
            "last_update": datetime.now().isoformat()
        }
    
    def event_to_dict(self, event: StatusEvent) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization"""
        data = asdict(event)
        data['event_type'] = event.event_type.value
        data['level'] = event.level.value
        data['timestamp'] = event.timestamp.isoformat()
        return data
    
    def log_research_start(self, workflow_type: str, backend_config: str, document_count: int):
        """Log research workflow start"""
        event = StatusEvent(
            id=f"research_{workflow_type}_{datetime.now().timestamp()}",
            event_type=EventType.SYSTEM,
            level=StatusLevel.INFO,
            title="Research Workflow Started",
            message=f"Started {workflow_type} workflow with {backend_config} config on {document_count} documents",
            data={"workflow_type": workflow_type, "backend_config": backend_config, "document_count": document_count},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_research_complete(self, workflow_type: str, output_files: int):
        """Log research workflow completion"""
        event = StatusEvent(
            id=f"research_complete_{workflow_type}_{datetime.now().timestamp()}",
            event_type=EventType.SYSTEM,
            level=StatusLevel.SUCCESS,
            title="Research Workflow Complete",
            message=f"Completed {workflow_type} workflow, generated {output_files} output files",
            data={"workflow_type": workflow_type, "output_files": output_files},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))
    
    def log_gpu_update(self, gpu_count: int, avg_utilization: float):
        """Log GPU status update"""
        event = StatusEvent(
            id=f"gpu_status_{datetime.now().timestamp()}",
            event_type=EventType.GPU_STATUS_UPDATE,
            level=StatusLevel.INFO,
            title="GPU Status",
            message=f"GPU metrics updated",
            data={"gpu_count": gpu_count, "avg_utilization": avg_utilization},
            timestamp=datetime.now()
        )
        asyncio.create_task(self.emit_event(event))

    @staticmethod
    def format_file_size(bytes_size: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"

# Global status tracker instance
status_tracker = StatusTracker()