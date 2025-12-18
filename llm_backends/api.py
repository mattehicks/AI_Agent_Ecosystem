#!/usr/bin/env python3
"""
LLM Backend Management API
RESTful endpoints for managing modular LLM backends
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from . import InferenceRequest, InferenceResponse, BackendType, BackendStatus
from .backend_registry import backend_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-backends", tags=["LLM Backends"])

# Pydantic models for API
class BackendCreateRequest(BaseModel):
    instance_name: str
    backend_type: str
    config: Dict[str, Any]

class BackendListResponse(BaseModel):
    backends: Dict[str, Dict[str, Any]]
    active_backend: Optional[str]
    total_backends: int

class ModelListResponse(BaseModel):
    models_by_backend: Dict[str, List[str]]
    total_models: int

class GenerationRequest(BaseModel):
    prompt: str
    model: str
    backend: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

@router.get("/health")
async def health_check():
    """Health check for LLM backend system"""
    backend_count = len(backend_registry.manager.backends)
    active_backend = backend_registry.manager.active_backend
    
    return {
        "status": "healthy",
        "service": "llm-backends",
        "registered_backends": backend_count,
        "active_backend": active_backend
    }

@router.get("/discover")
async def discover_backends():
    """Auto-discover available backends on the system"""
    try:
        discovered = await backend_registry.auto_discover_backends()
        return {
            "discovered_backends": discovered,
            "total_discovered": len(discovered)
        }
    except Exception as e:
        logger.error(f"Backend discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-types")
async def get_available_backend_types():
    """Get all available backend types"""
    return {
        "backend_types": [bt.value for bt in BackendType],
        "registered_classes": list(backend_registry.backend_classes.keys()),
        "default_configs": backend_registry.get_default_configs()
    }

@router.post("/create")
async def create_backend_instance(request: BackendCreateRequest):
    """Create a new backend instance"""
    try:
        backend_type = BackendType(request.backend_type)
        
        success = await backend_registry.create_backend_instance(
            request.instance_name,
            backend_type,
            request.config
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Backend instance '{request.instance_name}' created successfully",
                "instance_name": request.instance_name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create backend instance")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid backend type: {request.backend_type}")
    except Exception as e:
        logger.error(f"Backend creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{instance_name}")
async def remove_backend_instance(instance_name: str):
    """Remove a backend instance"""
    try:
        success = await backend_registry.remove_backend_instance(instance_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Backend instance '{instance_name}' removed successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Backend instance not found")
            
    except Exception as e:
        logger.error(f"Backend removal failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=BackendListResponse)
async def list_backends():
    """List all registered backend instances"""
    try:
        backends_info = backend_registry.manager.list_backends()
        
        # Convert to API format
        backends_dict = {}
        for name, info in backends_info.items():
            backends_dict[name] = {
                "backend_type": info.backend_type.value,
                "name": info.name,
                "description": info.description,
                "endpoint_url": info.endpoint_url,
                "status": info.status.value,
                "supported_models": info.supported_models,
                "capabilities": info.capabilities,
                "gpu_requirements": info.gpu_requirements
            }
        
        return BackendListResponse(
            backends=backends_dict,
            active_backend=backend_registry.manager.active_backend,
            total_backends=len(backends_dict)
        )
        
    except Exception as e:
        logger.error(f"Failed to list backends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set-active/{instance_name}")
async def set_active_backend(instance_name: str):
    """Set the active backend for inference"""
    try:
        success = await backend_registry.manager.set_active_backend(instance_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Active backend set to: {instance_name}",
                "active_backend": instance_name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set active backend")
            
    except Exception as e:
        logger.error(f"Failed to set active backend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=ModelListResponse)
async def list_all_models():
    """List models from all backends"""
    try:
        models_by_backend = await backend_registry.manager.list_all_models()
        total_models = sum(len(models) for models in models_by_backend.values())
        
        return ModelListResponse(
            models_by_backend=models_by_backend,
            total_models=total_models
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/load")
async def load_model(model_name: str, backend_name: Optional[str] = None, gpu_ids: Optional[List[int]] = None):
    """Load a model on a specific backend"""
    try:
        backend_name = backend_name or backend_registry.manager.active_backend
        
        if not backend_name or backend_name not in backend_registry.manager.backends:
            raise HTTPException(status_code=400, detail="No active backend available")
        
        backend = backend_registry.manager.backends[backend_name]
        success = await backend.load_model(model_name, gpu_ids)
        
        if success:
            return {
                "status": "success",
                "message": f"Model '{model_name}' loaded on backend '{backend_name}'",
                "model": model_name,
                "backend": backend_name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to load model")
            
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/unload")
async def unload_model(model_name: str, backend_name: Optional[str] = None):
    """Unload a model from a specific backend"""
    try:
        backend_name = backend_name or backend_registry.manager.active_backend
        
        if not backend_name or backend_name not in backend_registry.manager.backends:
            raise HTTPException(status_code=400, detail="No active backend available")
        
        backend = backend_registry.manager.backends[backend_name]
        success = await backend.unload_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Model '{model_name}' unloaded from backend '{backend_name}'",
                "model": model_name,
                "backend": backend_name
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to unload model")
            
    except Exception as e:
        logger.error(f"Model unloading failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=InferenceResponse)
async def generate_text(request: GenerationRequest):
    """Generate text using the modular backend system"""
    try:
        # Convert to internal request format
        inference_request = InferenceRequest(
            prompt=request.prompt,
            model=request.model,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop_sequences,
            stream=request.stream
        )
        
        # Generate using backend manager
        response = await backend_registry.manager.generate(inference_request, request.backend)
        
        return response
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_system_status():
    """Get overall backend system status"""
    try:
        backends_info = backend_registry.manager.list_backends()
        
        # Calculate system status
        total_backends = len(backends_info)
        online_backends = sum(1 for info in backends_info.values() 
                            if info.status == BackendStatus.ONLINE)
        
        # Get model counts
        models_by_backend = await backend_registry.manager.list_all_models()
        total_models = sum(len(models) for models in models_by_backend.values())
        
        return {
            "system_status": "healthy" if online_backends > 0 else "offline",
            "total_backends": total_backends,
            "online_backends": online_backends,
            "active_backend": backend_registry.manager.active_backend,
            "total_models_available": total_models,
            "backends": backends_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))