#!/usr/bin/env python3
"""
GPU Platform API
FastAPI endpoints for GPU monitoring and management
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from .gpu_manager import GPUManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gpu-platform", tags=["GPU Platform"])

# Global GPU manager instance
gpu_manager = GPUManager()

# Pydantic models for API
class GPUStatusResponse(BaseModel):
    gpus: Dict[str, Any]
    summary: Dict[str, Any]
    timestamp: str
    history: Optional[Dict[str, Any]] = None

class ModelLoadRequest(BaseModel):
    model_name: str
    gpu_index: Optional[int] = None
    memory_mb: Optional[int] = None

class ModelUnloadRequest(BaseModel):
    model_name: str
    gpu_index: Optional[int] = None

@router.get("/health")
async def health_check():
    """Health check for GPU platform service"""
    return {
        "status": "healthy",
        "service": "gpu-platform",
        "timestamp": gpu_manager.get_gpu_info()["timestamp"] if gpu_manager.gpus else None,
        "gpu_count": len(gpu_manager.gpus)
    }

@router.get("/status", response_model=GPUStatusResponse)
async def get_gpu_status():
    """Get current GPU status and metrics"""
    try:
        metrics = gpu_manager.get_gpu_metrics()
        return GPUStatusResponse(**metrics)
    except Exception as e:
        logger.error(f"Failed to get GPU status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU status: {str(e)}")

@router.get("/info")
async def get_gpu_info(gpu_index: Optional[int] = None):
    """Get detailed GPU information"""
    try:
        return gpu_manager.get_gpu_info(gpu_index)
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU info: {str(e)}")

@router.get("/metrics")
async def get_gpu_metrics():
    """Get real-time GPU metrics with historical data"""
    try:
        return gpu_manager.get_gpu_metrics()
    except Exception as e:
        logger.error(f"Failed to get GPU metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get GPU metrics: {str(e)}")

@router.get("/best")
async def get_best_gpu(memory_required: int = 0):
    """Get the best available GPU for a task"""
    try:
        best_gpu = gpu_manager.get_best_gpu(memory_required)
        if best_gpu is not None:
            return {
                "gpu_index": best_gpu,
                "gpu_info": gpu_manager.get_gpu_info(best_gpu)
            }
        else:
            return {
                "gpu_index": None,
                "message": "No suitable GPU available",
                "memory_required": memory_required
            }
    except Exception as e:
        logger.error(f"Failed to find best GPU: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find best GPU: {str(e)}")

@router.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model onto a GPU"""
    try:
        # If no GPU specified, find the best one
        gpu_index = request.gpu_index
        if gpu_index is None:
            gpu_index = gpu_manager.get_best_gpu(request.memory_mb or 1000)
            if gpu_index is None:
                raise HTTPException(status_code=400, detail="No suitable GPU available")
        
        # Reserve the GPU
        success = gpu_manager.reserve_gpu(
            gpu_index, 
            request.model_name, 
            request.memory_mb or 1000
        )
        
        if success:
            return {
                "success": True,
                "message": f"Model {request.model_name} loaded on GPU {gpu_index}",
                "gpu_index": gpu_index,
                "model_name": request.model_name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to reserve GPU")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/models/unload")
async def unload_model(request: ModelUnloadRequest):
    """Unload a model from a GPU"""
    try:
        # Find GPU with the model if not specified
        gpu_index = request.gpu_index
        if gpu_index is None:
            for idx, gpu in gpu_manager.gpus.items():
                if gpu.current_model == request.model_name:
                    gpu_index = idx
                    break
        
        if gpu_index is None:
            raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found on any GPU")
        
        # Release the GPU
        success = gpu_manager.release_gpu(gpu_index, request.model_name)
        
        if success:
            return {
                "success": True,
                "message": f"Model {request.model_name} unloaded from GPU {gpu_index}",
                "gpu_index": gpu_index,
                "model_name": request.model_name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to release GPU")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@router.post("/monitoring/start")
async def start_monitoring():
    """Start GPU monitoring"""
    try:
        gpu_manager.start_monitoring()
        return {
            "success": True,
            "message": "GPU monitoring started",
            "update_interval": gpu_manager.update_interval
        }
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop GPU monitoring"""
    try:
        gpu_manager.stop_monitoring()
        return {
            "success": True,
            "message": "GPU monitoring stopped"
        }
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring status"""
    return {
        "monitoring_active": gpu_manager.monitoring_active,
        "update_interval": gpu_manager.update_interval,
        "gpu_count": len(gpu_manager.gpus),
        "last_update": gpu_manager.last_gpu_data.get("timestamp") if gpu_manager.last_gpu_data else None
    }