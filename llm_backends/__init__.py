#!/usr/bin/env python3
"""
Modular LLM Backend System
Pluggable inference engine architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class BackendType(Enum):
    OLLAMA = "ollama"
    LOCALAI = "localai" 
    VLLM = "vllm"
    OPENAI = "openai"
    CUSTOM = "custom"

class BackendStatus(Enum):
    OFFLINE = "offline"
    STARTING = "starting"
    ONLINE = "online"
    ERROR = "error"

@dataclass
class BackendInfo:
    """Information about a backend instance"""
    backend_type: BackendType
    name: str
    description: str
    endpoint_url: str
    status: BackendStatus
    supported_models: List[str]
    capabilities: List[str]
    gpu_requirements: Dict[str, Any]
    config: Dict[str, Any]

@dataclass
class InferenceRequest:
    """Standardized inference request"""
    prompt: str
    model: str
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

@dataclass 
class InferenceResponse:
    """Standardized inference response"""
    text: str
    model: str
    backend: str
    tokens_used: int
    processing_time: float
    metadata: Dict[str, Any]

class LLMBackend(ABC):
    """Abstract base class for all LLM backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend_info: Optional[BackendInfo] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy"""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models"""
        pass
    
    @abstractmethod
    async def load_model(self, model_name: str, gpu_ids: List[int] = None) -> bool:
        """Load a model"""
        pass
    
    @abstractmethod
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model"""
        pass
    
    @abstractmethod
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text using the backend"""
        pass
    
    @abstractmethod
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a loaded model"""
        pass
    
    async def shutdown(self):
        """Shutdown the backend gracefully"""
        pass

class LLMBackendManager:
    """Manages multiple LLM backends"""
    
    def __init__(self):
        self.backends: Dict[str, LLMBackend] = {}
        self.active_backend: Optional[str] = None
        self.default_backend: str = "ollama"
    
    def register_backend(self, name: str, backend: LLMBackend):
        """Register a new backend"""
        self.backends[name] = backend
        logger.info(f"Registered backend: {name}")
    
    async def initialize_backend(self, name: str) -> bool:
        """Initialize a specific backend"""
        if name not in self.backends:
            logger.error(f"Backend not found: {name}")
            return False
        
        try:
            success = await self.backends[name].initialize()
            if success:
                logger.info(f"Backend initialized successfully: {name}")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize backend {name}: {e}")
            return False
    
    async def set_active_backend(self, name: str) -> bool:
        """Set the active backend for inference"""
        if name not in self.backends:
            return False
        
        # Health check before switching
        if await self.backends[name].health_check():
            self.active_backend = name
            logger.info(f"Switched to backend: {name}")
            return True
        
        return False
    
    async def generate(self, request: InferenceRequest, backend_name: Optional[str] = None) -> InferenceResponse:
        """Generate text using specified or active backend"""
        backend_name = backend_name or self.active_backend or self.default_backend
        
        if backend_name not in self.backends:
            raise ValueError(f"Backend not available: {backend_name}")
        
        backend = self.backends[backend_name]
        
        # Add backend info to response
        response = await backend.generate(request)
        response.backend = backend_name
        
        return response
    
    async def list_all_models(self) -> Dict[str, List[str]]:
        """List models from all backends"""
        all_models = {}
        
        for name, backend in self.backends.items():
            try:
                if await backend.health_check():
                    models = await backend.list_models()
                    all_models[name] = models
            except Exception as e:
                logger.warning(f"Failed to list models from {name}: {e}")
                all_models[name] = []
        
        return all_models
    
    def get_backend_info(self, name: str) -> Optional[BackendInfo]:
        """Get information about a backend"""
        if name in self.backends:
            return self.backends[name].backend_info
        return None
    
    def list_backends(self) -> Dict[str, BackendInfo]:
        """List all registered backends with their info"""
        return {name: backend.backend_info for name, backend in self.backends.items() 
                if backend.backend_info is not None}

# Global backend manager instance
backend_manager = LLMBackendManager()