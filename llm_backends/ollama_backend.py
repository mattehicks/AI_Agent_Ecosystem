#!/usr/bin/env python3
"""
Ollama Backend Implementation
Modular backend for Ollama inference engine
"""

import aiohttp
import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from . import LLMBackend, BackendInfo, BackendType, BackendStatus, InferenceRequest, InferenceResponse

logger = logging.getLogger(__name__)

class OllamaBackend(LLMBackend):
    """Ollama backend implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 120)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize backend info
        self.backend_info = BackendInfo(
            backend_type=BackendType.OLLAMA,
            name="Ollama",
            description="Local LLM inference with automatic model management",
            endpoint_url=self.base_url,
            status=BackendStatus.OFFLINE,
            supported_models=[],
            capabilities=["text_generation", "chat", "embeddings"],
            gpu_requirements={"min_vram_gb": 4, "supports_multi_gpu": True},
            config=config
        )
    
    async def initialize(self) -> bool:
        """Initialize Ollama backend"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Test connection
            if await self.health_check():
                self.backend_info.status = BackendStatus.ONLINE
                
                # Get available models
                self.backend_info.supported_models = await self.list_models()
                
                logger.info(f"Ollama backend initialized at {self.base_url}")
                return True
            else:
                self.backend_info.status = BackendStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Ollama backend: {e}")
            self.backend_info.status = BackendStatus.ERROR
            return False
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy"""
        try:
            if not self.session:
                return False
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
                
        except Exception as e:
            logger.debug(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models from Ollama"""
        try:
            if not self.session:
                return []
            
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model["name"] for model in data.get("models", [])]
                return []
                
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def load_model(self, model_name: str, gpu_ids: List[int] = None) -> bool:
        """Load a model in Ollama"""
        try:
            if not self.session:
                return False
            
            # Ollama automatically loads models on first use
            # We can trigger loading by making a small inference request
            test_request = InferenceRequest(
                prompt="Hello",
                model=model_name,
                max_tokens=1
            )
            
            await self.generate(test_request)
            logger.info(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from Ollama"""
        try:
            if not self.session:
                return False
            
            # Ollama doesn't have explicit unload, but we can delete the model
            async with self.session.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name}
            ) as response:
                success = response.status == 200
                if success:
                    logger.info(f"Model unloaded: {model_name}")
                return success
                
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text using Ollama"""
        start_time = time.time()
        
        try:
            if not self.session:
                raise RuntimeError("Ollama backend not initialized")
            
            # Prepare request payload
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            payload = {
                "model": request.model,
                "messages": messages,
                "stream": request.stream,
                "options": {}
            }
            
            # Add optional parameters
            if request.max_tokens:
                payload["options"]["num_predict"] = request.max_tokens
            if request.temperature is not None:
                payload["options"]["temperature"] = request.temperature
            if request.top_p is not None:
                payload["options"]["top_p"] = request.top_p
            if request.top_k is not None:
                payload["options"]["top_k"] = request.top_k
            if request.stop_sequences:
                payload["options"]["stop"] = request.stop_sequences
            
            # Make request to Ollama
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {response.status} - {error_text}")
                
                result = await response.json()
                
                # Extract response text
                response_text = result.get("message", {}).get("content", "")
                
                # Calculate metrics
                processing_time = time.time() - start_time
                tokens_used = len(response_text.split())  # Rough estimate
                
                return InferenceResponse(
                    text=response_text,
                    model=request.model,
                    backend="ollama",
                    tokens_used=tokens_used,
                    processing_time=processing_time,
                    metadata={
                        "ollama_response": result,
                        "request_params": payload["options"]
                    }
                )
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        try:
            if not self.session:
                return {}
            
            async with self.session.post(
                f"{self.base_url}/api/show",
                json={"name": model_name}
            ) as response:
                if response.status == 200:
                    return await response.json()
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {}
    
    async def pull_model(self, model_name: str) -> bool:
        """Download/pull a model"""
        try:
            if not self.session:
                return False
            
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name}
            ) as response:
                success = response.status == 200
                if success:
                    logger.info(f"Model pulled successfully: {model_name}")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to pull model {model_name}: {error_text}")
                return success
                
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the backend"""
        if self.session:
            await self.session.close()
            self.session = None
        
        self.backend_info.status = BackendStatus.OFFLINE
        logger.info("Ollama backend shutdown")