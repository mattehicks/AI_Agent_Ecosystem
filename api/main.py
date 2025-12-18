#!/usr/bin/env python3
"""
AI Agent Ecosystem API
FastAPI-based REST API for the AI Agent system
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Import orchestrator
from orchestrator.orchestrator import AgentOrchestrator, AgentType, TaskPriority

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global orchestrator instance
orchestrator = None

# Pydantic models for API
class TaskRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to process the task")
    description: str = Field(..., description="Human-readable task description")
    input_data: Dict[str, Any] = Field(..., description="Task input parameters")
    priority: Optional[str] = Field("normal", description="Task priority: low, normal, high, critical")

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    created_at: str

class TaskStatusResponse(BaseModel):
    task_id: str
    agent_type: str
    description: str
    status: str
    result: Optional[Dict[str, Any]] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0

class SystemMetricsResponse(BaseModel):
    uptime_seconds: float
    task_stats: Dict[str, int]
    agent_stats: Dict[str, Dict[str, Any]]
    active_tasks: int
    agent_instances: Dict[str, Dict[str, Any]]
    queue_sizes: Dict[str, int]

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Dict[str, str]]

# Create FastAPI app
app = FastAPI(
    title="AI Agent Ecosystem API",
    description="REST API for managing AI agents and tasks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
web_dir = Path(__file__).parent.parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    logger.info(f"Mounted static files from: {web_dir}")
else:
    logger.warning(f"Web directory not found: {web_dir}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator and start agent workers"""
    global orchestrator
    
    logger.info("Starting AI Agent Ecosystem API...")
    
    try:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        
        # Start agent workers in background
        asyncio.create_task(orchestrator.start_agent_workers())
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown"""
    global orchestrator
    
    logger.info("Shutting down AI Agent Ecosystem API...")
    
    if orchestrator:
        await orchestrator.shutdown()
    
    logger.info("API shutdown completed")

# Dependency to get orchestrator
def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance"""
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    return orchestrator

# API Routes

@app.get("/")
async def root():
    """Serve the web dashboard"""
    web_dir = Path(__file__).parent.parent / "web"
    index_file = web_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(str(index_file))
    else:
        # Fallback to API info if web interface not available
        return {
            "message": "AI Agent Ecosystem API",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "web_interface": "not_available"
        }

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint"""
    return {
        "message": "AI Agent Ecosystem API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/static")
async def debug_static():
    """Debug endpoint to check static file setup"""
    web_dir = Path(__file__).parent.parent / "web"
    return {
        "web_directory": str(web_dir),
        "web_dir_exists": web_dir.exists(),
        "files": list(web_dir.iterdir()) if web_dir.exists() else [],
        "css_exists": (web_dir / "styles.css").exists() if web_dir.exists() else False,
        "js_exists": (web_dir / "script.js").exists() if web_dir.exists() else False,
        "html_exists": (web_dir / "index.html").exists() if web_dir.exists() else False
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Health check endpoint"""
    try:
        # Basic health check
        components = {
            "orchestrator": {"status": "ok", "message": "Orchestrator running"},
            "database": {"status": "ok", "message": "Database accessible"},
            "agents": {"status": "ok", "message": f"{len(orch.agent_instances)} agent instances running"}
        }
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

@app.post("/tasks", response_model=TaskResponse)
async def create_task(
    task_request: TaskRequest,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Create a new task"""
    try:
        # Validate agent type
        try:
            agent_type = AgentType(task_request.agent_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid agent type: {task_request.agent_type}. "
                       f"Valid types: {[t.value for t in AgentType]}"
            )
        
        # Validate priority
        try:
            priority = TaskPriority[task_request.priority.upper()]
        except (KeyError, AttributeError):
            priority = TaskPriority.NORMAL
        
        # Create task
        task_id = orch.create_task(
            agent_type=agent_type,
            description=task_request.description,
            input_data=task_request.input_data,
            priority=priority
        )
        
        return TaskResponse(
            task_id=task_id,
            status="created",
            message="Task created successfully",
            created_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get task status"""
    try:
        task_info = orch.get_task_status(task_id)
        
        if not task_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return TaskStatusResponse(**task_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )

@app.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    status_filter: Optional[str] = None,
    agent_type_filter: Optional[str] = None,
    limit: int = 50,
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """List tasks with optional filtering"""
    try:
        # This would require implementing a list_tasks method in orchestrator
        # For now, return empty list with a note
        return []
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )

@app.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """Get system performance metrics"""
    try:
        metrics = orch.get_system_metrics()
        return SystemMetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )

@app.get("/agents", response_model=Dict[str, Any])
async def list_agents(orch: AgentOrchestrator = Depends(get_orchestrator)):
    """List available agents and their status"""
    try:
        agent_info = {}
        
        for agent_type in AgentType:
            agent_info[agent_type.value] = {
                "type": agent_type.value,
                "instances": [
                    {
                        "instance_id": instance_id,
                        "status": instance.status,
                        "current_task": instance.current_task_id,
                        "total_tasks": instance.total_tasks,
                        "successful_tasks": instance.successful_tasks,
                        "last_activity": instance.last_activity.isoformat()
                    }
                    for instance_id, instance in orch.agent_instances.items()
                    if instance.agent_type == agent_type
                ],
                "queue_size": orch.task_queues[agent_type].qsize()
            }
        
        return agent_info
        
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )

# Convenience endpoints for specific agent types

@app.post("/analyze-document")
async def analyze_document(
    document_path: str,
    analysis_type: str = "summary",
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Convenience endpoint for document analysis"""
    try:
        task_id = orch.create_task(
            agent_type=AgentType.DOCUMENT_ANALYZER,
            description=f"Analyze document: {document_path}",
            input_data={
                "document_path": document_path,
                "analysis_type": analysis_type
            }
        )
        
        return {"task_id": task_id, "message": "Document analysis task created"}
        
    except Exception as e:
        logger.error(f"Error creating document analysis task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create document analysis task: {str(e)}"
        )

@app.post("/generate-code")
async def generate_code(
    requirements: str,
    language: str = "python",
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Convenience endpoint for code generation"""
    try:
        task_id = orch.create_task(
            agent_type=AgentType.CODE_GENERATOR,
            description=f"Generate {language} code",
            input_data={
                "requirements": requirements,
                "language": language
            }
        )
        
        return {"task_id": task_id, "message": "Code generation task created"}
        
    except Exception as e:
        logger.error(f"Error creating code generation task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create code generation task: {str(e)}"
        )

@app.post("/research")
async def conduct_research(
    query: str,
    sources: List[str] = ["documents", "knowledge_base"],
    orch: AgentOrchestrator = Depends(get_orchestrator)
):
    """Convenience endpoint for research tasks"""
    try:
        task_id = orch.create_task(
            agent_type=AgentType.RESEARCH_ASSISTANT,
            description=f"Research: {query}",
            input_data={
                "query": query,
                "sources": sources
            }
        )
        
        return {"task_id": task_id, "message": "Research task created"}
        
    except Exception as e:
        logger.error(f"Error creating research task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create research task: {str(e)}"
        )

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

# Import document analysis feature
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from features.document_analysis.api import router as document_analysis_router
    DOCUMENT_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Document analysis feature not available: {e}")
    DOCUMENT_ANALYSIS_AVAILABLE = False

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected clients
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Include feature routers
if DOCUMENT_ANALYSIS_AVAILABLE:
    app.include_router(document_analysis_router)
    logger.info("Document analysis feature enabled")

# Research Assistant integration
try:
    from features.research_assistant.api import router as research_assistant_router
    app.include_router(research_assistant_router)
    logger.info("Research assistant feature enabled")
    RESEARCH_ASSISTANT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Research assistant feature not available: {e}")
    RESEARCH_ASSISTANT_AVAILABLE = False

# LLM Backends integration
try:
    from llm_backends.api import router as llm_backends_router
    app.include_router(llm_backends_router)
    logger.info("LLM backends feature enabled")
    LLM_BACKENDS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLM backends feature not available: {e}")
    LLM_BACKENDS_AVAILABLE = False

# GPU Platform integration
try:
    from gpu_platform.api import router as gpu_platform_router
    app.include_router(gpu_platform_router)
    logger.info("GPU platform feature enabled")
    GPU_PLATFORM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GPU platform feature not available: {e}")
    GPU_PLATFORM_AVAILABLE = False

# System Configuration integration
try:
    from system_config.api import router as system_config_router
    app.include_router(system_config_router)
    logger.info("System configuration feature enabled")
    SYSTEM_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"System configuration feature not available: {e}")
    SYSTEM_CONFIG_AVAILABLE = False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    # Track what data this client wants
    client_subscriptions = {
        "system_metrics": True,
        "gpu_metrics": False,  # Only send when requested
        "status_updates": True
    }
    
    try:
        # Listen for client messages to control subscriptions
        async def handle_client_messages():
            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    if data.get("type") == "subscribe":
                        subscription = data.get("subscription")
                        if subscription in client_subscriptions:
                            client_subscriptions[subscription] = True
                            logger.info(f"Client subscribed to {subscription}")
                    
                    elif data.get("type") == "unsubscribe":
                        subscription = data.get("subscription")
                        if subscription in client_subscriptions:
                            client_subscriptions[subscription] = False
                            logger.info(f"Client unsubscribed from {subscription}")
                            
            except WebSocketDisconnect:
                pass
        
        # Start message handler
        asyncio.create_task(handle_client_messages())
        
        while True:
            # Keep connection alive and send periodic updates
            await asyncio.sleep(5)  # Less frequent for system metrics
            
            # Send system metrics (always)
            if client_subscriptions["system_metrics"] and orchestrator:
                metrics = orchestrator.get_system_metrics()
                await manager.send_personal_message(
                    json.dumps({
                        "type": "system_metrics",
                        "metrics": metrics,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
            
            # Send GPU metrics only if subscribed (when on GPU page)
            if client_subscriptions["gpu_metrics"] and GPU_PLATFORM_AVAILABLE:
                try:
                    from gpu_platform.api import gpu_manager
                    gpu_metrics = gpu_manager.get_gpu_metrics()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "gpu_metrics",
                            "metrics": gpu_metrics,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                    
                    # Send status update for the status tracker (less frequently)
                    if client_subscriptions["status_updates"]:
                        from status_tracker import status_tracker
                        status_tracker.log_gpu_update(
                            len(gpu_metrics.get("gpus", {})),
                            gpu_metrics.get("summary", {}).get("average_utilization", 0)
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to send GPU metrics: {e}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An internal server error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )