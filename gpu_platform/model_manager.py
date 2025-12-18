#!/usr/bin/env python3
"""
Model management for GPU platform
"""

import logging
import asyncio
import subprocess
import json
import time
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from status_tracker import status_tracker

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"

@dataclass
class ModelInfo:
    name: str
    path: str
    size_gb: float
    status: ModelStatus
    gpu_ids: List[int]
    memory_usage_mb: int = 0
    inference_count: int = 0
    avg_inference_time: float = 0.0

class ModelManager:
    """Manages LLM models on GPU platform"""
    
    def __init__(self):
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_processes: Dict[str, subprocess.Popen] = {}
        self.cursor_integration_process: Optional[subprocess.Popen] = None
        self.default_model: Optional[str] = None
        
        # Base model directory
        self.models_base_path = "/mnt/llm/LLM-Models"
        
        # Known model paths - will be auto-discovered
        self.model_paths = {}
        
        self._scan_available_models()
    
    def _scan_available_models(self):
        """Scan for available models in /mnt/llm/LLM-Models"""
        self.available_models = []
        
        try:
            models_dir = Path(self.models_base_path)
            if not models_dir.exists():
                logger.warning(f"Models directory not found: {self.models_base_path}")
                return
            
            # Scan for model directories
            for item in models_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    try:
                        # Calculate directory size
                        size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        size_gb = size_bytes / (1024**3)
                        
                        # Skip very small directories (likely not models)
                        if size_gb < 0.1:
                            continue
                        
                        model_info = ModelInfo(
                            name=item.name,
                            path=str(item),
                            size_gb=size_gb,
                            status=ModelStatus.AVAILABLE,
                            gpu_ids=[]
                        )
                        self.available_models.append(model_info)
                        self.model_paths[item.name] = str(item)
                        
                    except Exception as e:
                        logger.warning(f"Error scanning model {item.name}: {e}")
                        continue
            
            # Also scan for .gguf files in the root
            for gguf_file in models_dir.glob("*.gguf"):
                try:
                    size_gb = gguf_file.stat().st_size / (1024**3)
                    model_name = gguf_file.stem
                    
                    model_info = ModelInfo(
                        name=model_name,
                        path=str(gguf_file),
                        size_gb=size_gb,
                        status=ModelStatus.AVAILABLE,
                        gpu_ids=[]
                    )
                    self.available_models.append(model_info)
                    self.model_paths[model_name] = str(gguf_file)
                    
                except Exception as e:
                    logger.warning(f"Error scanning GGUF file {gguf_file.name}: {e}")
                    continue
            
            logger.info(f"Found {len(self.available_models)} available models in {self.models_base_path}")
            for model in self.available_models:
                logger.info(f"  - {model.name}: {model.size_gb:.1f}GB")
                
        except Exception as e:
            logger.error(f"Failed to scan models directory: {e}")
            self.available_models = []
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models"""
        return self.available_models
    
    def get_loaded_models(self) -> List[ModelInfo]:
        """Get list of loaded models"""
        return list(self.loaded_models.values())
    
    def get_model_status(self, model_name: str) -> Optional[ModelStatus]:
        """Get status of a specific model"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name].status
        
        # Check if it's available
        for model in self.available_models:
            if model.name == model_name:
                return ModelStatus.AVAILABLE
        
        return None
    
    def get_model_gpu_ids(self, model_name: str) -> List[int]:
        """Get GPU IDs where model is loaded"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name].gpu_ids
        return []
    
    def get_model_memory_usage(self, model_name: str) -> int:
        """Get memory usage of a model in MB"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name].memory_usage_mb
        return 0
    
    async def load_model(self, model_name: str, gpu_ids: Optional[List[int]] = None, 
                        max_memory_per_gpu: Optional[str] = None, 
                        quantization: Optional[str] = None):
        """Load a model onto specified GPUs"""
        try:
            # Find model info
            model_info = None
            for model in self.available_models:
                if model.name == model_name:
                    model_info = model
                    break
            
            if not model_info:
                status_tracker.log_model_load_error(model_name, "Model not found")
                raise ValueError(f"Model {model_name} not found")
            
            # Set default GPU IDs if not specified
            if gpu_ids is None:
                gpu_ids = [0, 1]  # Use both GPUs by default
            
            # Log start of model loading
            status_tracker.log_model_load_start(model_name, gpu_ids)
            
            # Update model status
            model_info.status = ModelStatus.LOADING
            model_info.gpu_ids = gpu_ids
            self.loaded_models[model_name] = model_info
            
            logger.info(f"Loading model {model_name} on GPUs {gpu_ids}")
            
            # Simulate model loading with progress updates
            for progress in [25, 50, 75]:
                await asyncio.sleep(0.5)
                status_tracker.log_model_load_progress(model_name, progress, f"Loading {model_name}... {progress}%")
            
            await asyncio.sleep(0.5)  # Final loading step
            
            # Mark as loaded
            model_info.status = ModelStatus.LOADED
            model_info.memory_usage_mb = int(model_info.size_gb * 1024 * 0.8)  # Estimate 80% of model size
            
            # Set as default if it's the first loaded model
            if self.default_model is None:
                self.default_model = model_name
            
            # Log completion
            status_tracker.log_model_load_complete(model_name, gpu_ids, model_info.memory_usage_mb)
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            status_tracker.log_model_load_error(model_name, str(e))
            if model_name in self.loaded_models:
                self.loaded_models[model_name].status = ModelStatus.ERROR
            raise
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from GPU"""
        try:
            if model_name not in self.loaded_models:
                return False
            
            model_info = self.loaded_models[model_name]
            model_info.status = ModelStatus.UNLOADING
            
            # Stop any running processes for this model
            if model_name in self.model_processes:
                process = self.model_processes[model_name]
                process.terminate()
                process.wait(timeout=10)
                del self.model_processes[model_name]
            
            # Remove from loaded models
            del self.loaded_models[model_name]
            
            # Update default model if needed
            if self.default_model == model_name:
                self.default_model = next(iter(self.loaded_models.keys()), None)
            
            logger.info(f"Model {model_name} unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    async def run_inference(self, model_name: str, prompt: str, 
                           max_tokens: int = 100, temperature: float = 0.7, 
                           top_p: float = 0.9) -> str:
        """Run inference on a loaded model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model_info = self.loaded_models[model_name]
        if model_info.status != ModelStatus.LOADED:
            raise ValueError(f"Model {model_name} not ready for inference")
        
        start_time = time.time()
        
        try:
            # Simulate inference (in real implementation, use the actual model)
            await asyncio.sleep(0.5)  # Simulate inference time
            
            # Mock response based on model type
            if "code" in model_name.lower():
                response = f"# Generated by {model_name}\ndef example_function():\n    return 'Hello from AI!'"
            else:
                response = f"This is a response generated by {model_name} for the prompt: {prompt[:50]}..."
            
            # Update statistics
            inference_time = time.time() - start_time
            model_info.inference_count += 1
            model_info.avg_inference_time = (
                (model_info.avg_inference_time * (model_info.inference_count - 1) + inference_time) 
                / model_info.inference_count
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_name}: {e}")
            raise
    
    def get_default_model(self) -> Optional[str]:
        """Get the default model for Cursor integration"""
        return self.default_model
    
    def is_cursor_integration_active(self) -> bool:
        """Check if Cursor integration service is running"""
        return (self.cursor_integration_process is not None and 
                self.cursor_integration_process.poll() is None)
    
    async def start_cursor_integration(self):
        """Start Cursor IDE integration service"""
        try:
            if self.is_cursor_integration_active():
                logger.info("Cursor integration already active")
                return
            
            if not self.loaded_models:
                raise ValueError("No models loaded for Cursor integration")
            
            # Start a simple OpenAI-compatible API server
            # This would typically use vLLM or similar for production
            cmd = [
                "python", "-m", "http.server", "8001"
            ]
            
            self.cursor_integration_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("Cursor integration service started on port 8001")
            
        except Exception as e:
            logger.error(f"Failed to start Cursor integration: {e}")
            raise
    
    def stop_cursor_integration(self) -> bool:
        """Stop Cursor IDE integration service"""
        try:
            if self.cursor_integration_process:
                self.cursor_integration_process.terminate()
                self.cursor_integration_process.wait(timeout=10)
                self.cursor_integration_process = None
                logger.info("Cursor integration service stopped")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to stop Cursor integration: {e}")
            return False