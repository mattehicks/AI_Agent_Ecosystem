#!/usr/bin/env python3
"""
Backend Registry and Management
Central registry for all LLM backend implementations
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Type
from pathlib import Path

from . import LLMBackend, LLMBackendManager, BackendType, BackendInfo, BackendStatus
from .ollama_backend import OllamaBackend

logger = logging.getLogger(__name__)

class BackendRegistry:
    """Registry for all available backend implementations"""
    
    def __init__(self):
        self.backend_classes: Dict[BackendType, Type[LLMBackend]] = {}
        self.backend_configs: Dict[str, Dict[str, Any]] = {}
        self.manager = LLMBackendManager()
        
        # Register built-in backends
        self._register_builtin_backends()
    
    def _register_builtin_backends(self):
        """Register built-in backend implementations"""
        self.backend_classes[BackendType.OLLAMA] = OllamaBackend
        
        # Placeholder for future backends
        # self.backend_classes[BackendType.LOCALAI] = LocalAIBackend
        # self.backend_classes[BackendType.VLLM] = VLLMBackend
        # self.backend_classes[BackendType.OPENAI] = OpenAIBackend
    
    def register_backend_class(self, backend_type: BackendType, backend_class: Type[LLMBackend]):
        """Register a new backend class"""
        self.backend_classes[backend_type] = backend_class
        logger.info(f"Registered backend class: {backend_type.value}")
    
    async def create_backend_instance(self, instance_name: str, backend_type: BackendType, config: Dict[str, Any]) -> bool:
        """Create and initialize a backend instance"""
        if backend_type not in self.backend_classes:
            logger.error(f"Backend type not registered: {backend_type.value}")
            return False
        
        try:
            # Create backend instance
            backend_class = self.backend_classes[backend_type]
            backend_instance = backend_class(config)
            
            # Initialize backend
            if await backend_instance.initialize():
                # Register with manager
                self.manager.register_backend(instance_name, backend_instance)
                
                # Store config for persistence
                self.backend_configs[instance_name] = {
                    "backend_type": backend_type.value,
                    "config": config
                }
                
                logger.info(f"Created backend instance: {instance_name}")
                return True
            else:
                logger.error(f"Failed to initialize backend instance: {instance_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating backend instance {instance_name}: {e}")
            return False
    
    async def remove_backend_instance(self, instance_name: str) -> bool:
        """Remove a backend instance"""
        try:
            if instance_name in self.manager.backends:
                # Shutdown backend
                await self.manager.backends[instance_name].shutdown()
                
                # Remove from manager
                del self.manager.backends[instance_name]
                
                # Remove config
                if instance_name in self.backend_configs:
                    del self.backend_configs[instance_name]
                
                logger.info(f"Removed backend instance: {instance_name}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing backend instance {instance_name}: {e}")
            return False
    
    async def auto_discover_backends(self) -> Dict[str, Dict[str, Any]]:
        """Auto-discover available backends on the system"""
        discovered = {}
        
        # Check for Ollama
        ollama_configs = await self._discover_ollama()
        if ollama_configs:
            discovered["ollama"] = ollama_configs
        
        # Check for LocalAI
        localai_configs = await self._discover_localai()
        if localai_configs:
            discovered["localai"] = localai_configs
        
        # Check for vLLM
        vllm_configs = await self._discover_vllm()
        if vllm_configs:
            discovered["vllm"] = vllm_configs
        
        return discovered
    
    async def _discover_ollama(self) -> Optional[Dict[str, Any]]:
        """Discover Ollama installations"""
        try:
            # Check common Ollama ports
            for port in [11434, 11435, 11436]:
                url = f"http://localhost:{port}"
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"{url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                data = await response.json()
                                return {
                                    "base_url": url,
                                    "models_available": len(data.get("models", [])),
                                    "discovered": True
                                }
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Ollama discovery failed: {e}")
            return None
    
    async def _discover_localai(self) -> Optional[Dict[str, Any]]:
        """Discover LocalAI installations"""
        try:
            # Check common LocalAI ports
            for port in [8080, 8081, 8082]:
                url = f"http://localhost:{port}"
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"{url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                data = await response.json()
                                return {
                                    "base_url": url,
                                    "models_available": len(data.get("data", [])),
                                    "discovered": True
                                }
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"LocalAI discovery failed: {e}")
            return None
    
    async def _discover_vllm(self) -> Optional[Dict[str, Any]]:
        """Discover vLLM installations"""
        try:
            # Check common vLLM ports
            for port in [8000, 8001, 8002]:
                url = f"http://localhost:{port}"
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(f"{url}/v1/models", timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                data = await response.json()
                                return {
                                    "base_url": url,
                                    "models_available": len(data.get("data", [])),
                                    "discovered": True
                                }
                    except:
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"vLLM discovery failed: {e}")
            return None
    
    def get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default configurations for different deployment scenarios"""
        return {
            "ollama_local": {
                "backend_type": "ollama",
                "base_url": "http://localhost:11434",
                "timeout": 120,
                "description": "Local Ollama instance"
            },
            "ollama_remote": {
                "backend_type": "ollama", 
                "base_url": "http://192.168.2.151:11434",
                "timeout": 120,
                "description": "Remote Ollama instance"
            }
        }

# Global backend registry instance
backend_registry = BackendRegistry()