#!/usr/bin/env python3
"""
Model Discovery System for AI Agent Ecosystem
Automatically discovers and manages available AI models across different backends
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    id: str
    name: str
    backend: str
    model_type: str  # 'chat', 'completion', 'embedding', 'multimodal'
    size_gb: float = 0.0
    context_length: int = 2048
    available: bool = False
    loaded: bool = False
    capabilities: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelDiscovery:
    """Discovers and tracks AI models across different backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.discovered_models = {}
        self.backend_configs = config.get('backends', {})
        self.discovery_interval = config.get('discovery_interval', 300)  # 5 minutes
        self.discovery_task = None
        
    async def start(self):
        """Start model discovery service"""
        logger.info("Starting Model Discovery Service")
        
        # Initial discovery
        await self.discover_all_models()
        
        # Start periodic discovery
        self.discovery_task = asyncio.create_task(self._periodic_discovery())
    
    async def stop(self):
        """Stop model discovery service"""
        logger.info("Stopping Model Discovery Service")
        
        if self.discovery_task:
            self.discovery_task.cancel()
    
    async def discover_all_models(self) -> Dict[str, Any]:
        """Discover models from all configured backends"""
        results = {
            'discovered_count': 0,
            'backends_checked': 0,
            'errors': []
        }
        
        # Discover from each backend
        discovery_tasks = []
        
        if 'ollama' in self.backend_configs:
            discovery_tasks.append(self._discover_ollama_models())
        
        if 'lm_studio' in self.backend_configs:
            discovery_tasks.append(self._discover_lmstudio_models())
        
        if 'gpt4all' in self.backend_configs:
            discovery_tasks.append(self._discover_gpt4all_models())
        
        if 'private_gpt' in self.backend_configs:
            discovery_tasks.append(self._discover_privategpt_models())
        
        if 'openai' in self.backend_configs:
            discovery_tasks.append(self._discover_openai_models())
        
        # Run discoveries concurrently
        discovery_results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        for i, result in enumerate(discovery_results):
            results['backends_checked'] += 1
            
            if isinstance(result, Exception):
                results['errors'].append(str(result))
            elif isinstance(result, dict) and result.get('success'):
                results['discovered_count'] += result.get('model_count', 0)
            else:
                results['errors'].append(f"Backend {i} discovery failed")
        
        logger.info(f"Model discovery completed: {results['discovered_count']} models from {results['backends_checked']} backends")
        return results
    
    async def _discover_ollama_models(self) -> Dict[str, Any]:
        """Discover Ollama models"""
        try:
            config = self.backend_configs['ollama']
            base_url = config.get('base_url', 'http://localhost:11434')
            
            async with aiohttp.ClientSession() as session:
                # List local models
                async with session.get(f"{base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('models', [])
                        
                        for model_data in models:
                            model_id = f"ollama:{model_data['name']}"
                            
                            model_info = ModelInfo(
                                id=model_id,
                                name=model_data['name'],
                                backend='ollama',
                                model_type='chat',
                                size_gb=model_data.get('size', 0) / (1024**3),
                                available=True,
                                capabilities=['text_generation', 'chat'],
                                metadata={
                                    'modified_at': model_data.get('modified_at'),
                                    'digest': model_data.get('digest')
                                }
                            )
                            
                            self.discovered_models[model_id] = model_info
                        
                        return {
                            'success': True,
                            'backend': 'ollama',
                            'model_count': len(models)
                        }
                    else:
                        return {
                            'success': False,
                            'backend': 'ollama',
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"Ollama model discovery failed: {e}")
            return {
                'success': False,
                'backend': 'ollama',
                'error': str(e)
            }
    
    async def _discover_lmstudio_models(self) -> Dict[str, Any]:
        """Discover LM Studio models"""
        try:
            config = self.backend_configs['lm_studio']
            base_url = config.get('base_url', 'http://localhost:1234')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/v1/models") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('data', [])
                        
                        for model_data in models:
                            model_id = f"lmstudio:{model_data['id']}"
                            
                            model_info = ModelInfo(
                                id=model_id,
                                name=model_data['id'],
                                backend='lm_studio',
                                model_type='chat',
                                context_length=model_data.get('context_length', 2048),
                                available=True,
                                loaded=model_data.get('loaded', False),
                                capabilities=['text_generation', 'chat', 'completion'],
                                metadata=model_data
                            )
                            
                            self.discovered_models[model_id] = model_info
                        
                        return {
                            'success': True,
                            'backend': 'lm_studio',
                            'model_count': len(models)
                        }
                    else:
                        return {
                            'success': False,
                            'backend': 'lm_studio',
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"LM Studio model discovery failed: {e}")
            return {
                'success': False,
                'backend': 'lm_studio',
                'error': str(e)
            }
    
    async def _discover_gpt4all_models(self) -> Dict[str, Any]:
        """Discover GPT4All models"""
        try:
            config = self.backend_configs['gpt4all']
            models_dir = Path(config.get('models_dir', '~/.cache/gpt4all')).expanduser()
            
            model_count = 0
            
            if models_dir.exists():
                for model_file in models_dir.glob('*.bin'):
                    model_id = f"gpt4all:{model_file.name}"
                    
                    model_info = ModelInfo(
                        id=model_id,
                        name=model_file.stem,
                        backend='gpt4all',
                        model_type='chat',
                        size_gb=model_file.stat().st_size / (1024**3),
                        available=True,
                        capabilities=['text_generation', 'chat'],
                        metadata={
                            'file_path': str(model_file),
                            'modified_time': datetime.fromtimestamp(model_file.stat().st_mtime)
                        }
                    )
                    
                    self.discovered_models[model_id] = model_info
                    model_count += 1
            
            return {
                'success': True,
                'backend': 'gpt4all',
                'model_count': model_count
            }
            
        except Exception as e:
            logger.error(f"GPT4All model discovery failed: {e}")
            return {
                'success': False,
                'backend': 'gpt4all',
                'error': str(e)
            }
    
    async def _discover_privategpt_models(self) -> Dict[str, Any]:
        """Discover PrivateGPT models"""
        try:
            config = self.backend_configs['private_gpt']
            base_url = config.get('base_url', 'http://localhost:8001')
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # PrivateGPT typically uses one model at a time
                        model_id = "privategpt:default"
                        
                        model_info = ModelInfo(
                            id=model_id,
                            name="PrivateGPT Model",
                            backend='private_gpt',
                            model_type='chat',
                            available=True,
                            capabilities=['text_generation', 'chat', 'document_analysis', 'embeddings'],
                            metadata={
                                'version': data.get('version'),
                                'models_loaded': data.get('models_loaded', 1)
                            }
                        )
                        
                        self.discovered_models[model_id] = model_info
                        
                        return {
                            'success': True,
                            'backend': 'private_gpt',
                            'model_count': 1
                        }
                    else:
                        return {
                            'success': False,
                            'backend': 'private_gpt',
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"PrivateGPT model discovery failed: {e}")
            return {
                'success': False,
                'backend': 'private_gpt',
                'error': str(e)
            }
    
    async def _discover_openai_models(self) -> Dict[str, Any]:
        """Discover OpenAI models (if API key provided)"""
        try:
            config = self.backend_configs['openai']
            api_key = config.get('api_key')
            
            if not api_key:
                return {
                    'success': False,
                    'backend': 'openai',
                    'error': 'No API key provided'
                }
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get('https://api.openai.com/v1/models') as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get('data', [])
                        
                        model_count = 0
                        for model_data in models:
                            # Only include chat/completion models
                            if any(cap in model_data['id'] for cap in ['gpt-', 'text-', 'davinci', 'curie']):
                                model_id = f"openai:{model_data['id']}"
                                
                                model_type = 'chat' if 'gpt-' in model_data['id'] else 'completion'
                                capabilities = ['text_generation']
                                if model_type == 'chat':
                                    capabilities.append('chat')
                                
                                model_info = ModelInfo(
                                    id=model_id,
                                    name=model_data['id'],
                                    backend='openai',
                                    model_type=model_type,
                                    available=True,
                                    capabilities=capabilities,
                                    metadata={
                                        'created': model_data.get('created'),
                                        'owned_by': model_data.get('owned_by')
                                    }
                                )
                                
                                self.discovered_models[model_id] = model_info
                                model_count += 1
                        
                        return {
                            'success': True,
                            'backend': 'openai',
                            'model_count': model_count
                        }
                    else:
                        return {
                            'success': False,
                            'backend': 'openai',
                            'error': f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            logger.error(f"OpenAI model discovery failed: {e}")
            return {
                'success': False,
                'backend': 'openai',
                'error': str(e)
            }
    
    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Get models that support a specific capability"""
        return [
            model for model in self.discovered_models.values()
            if capability in model.capabilities and model.available
        ]
    
    def get_models_by_backend(self, backend: str) -> List[ModelInfo]:
        """Get models from a specific backend"""
        return [
            model for model in self.discovered_models.values()
            if model.backend == backend and model.available
        ]
    
    def get_best_model_for_task(self, task_type: str, 
                               backend_preference: List[str] = None) -> Optional[ModelInfo]:
        """Get the best model for a specific task type"""
        
        # Define task to capability mapping
        task_capabilities = {
            'chat': ['chat', 'text_generation'],
            'completion': ['text_generation', 'completion'],
            'document_analysis': ['document_analysis', 'chat'],
            'code_generation': ['text_generation', 'chat'],
            'summarization': ['text_generation', 'chat'],
            'embedding': ['embeddings']
        }
        
        required_capabilities = task_capabilities.get(task_type, ['text_generation'])
        
        # Filter models by capabilities
        suitable_models = []
        for model in self.discovered_models.values():
            if (model.available and 
                any(cap in model.capabilities for cap in required_capabilities)):
                suitable_models.append(model)
        
        if not suitable_models:
            return None
        
        # Apply backend preference
        if backend_preference:
            for preferred_backend in backend_preference:
                for model in suitable_models:
                    if model.backend == preferred_backend:
                        return model
        
        # Fallback to best performing model
        return max(suitable_models, key=lambda m: (m.performance_score, m.usage_count))
    
    def update_model_performance(self, model_id: str, 
                               response_time: float, 
                               success: bool):
        """Update model performance metrics"""
        if model_id in self.discovered_models:
            model = self.discovered_models[model_id]
            model.last_used = datetime.now()
            model.usage_count += 1
            
            # Simple performance scoring (lower response time = higher score)
            if success:
                new_score = 100 / max(response_time, 0.1)  # Avoid division by zero
                model.performance_score = (model.performance_score + new_score) / 2
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        stats = {
            'total_models': len(self.discovered_models),
            'available_models': len([m for m in self.discovered_models.values() if m.available]),
            'backends': {},
            'capabilities': {},
            'model_types': {}
        }
        
        for model in self.discovered_models.values():
            # Backend stats
            if model.backend not in stats['backends']:
                stats['backends'][model.backend] = {'total': 0, 'available': 0}
            stats['backends'][model.backend]['total'] += 1
            if model.available:
                stats['backends'][model.backend]['available'] += 1
            
            # Capability stats
            for capability in model.capabilities:
                if capability not in stats['capabilities']:
                    stats['capabilities'][capability] = 0
                if model.available:
                    stats['capabilities'][capability] += 1
            
            # Model type stats
            if model.model_type not in stats['model_types']:
                stats['model_types'][model.model_type] = 0
            if model.available:
                stats['model_types'][model.model_type] += 1
        
        return stats
    
    async def _periodic_discovery(self):
        """Periodic model discovery task"""
        while True:
            try:
                await asyncio.sleep(self.discovery_interval)
                logger.info("Running periodic model discovery")
                await self.discover_all_models()
                
            except Exception as e:
                logger.error(f"Error in periodic discovery: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    def export_model_registry(self) -> Dict[str, Any]:
        """Export complete model registry"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.discovered_models),
            'models': {
                model_id: {
                    'id': model.id,
                    'name': model.name,
                    'backend': model.backend,
                    'model_type': model.model_type,
                    'size_gb': model.size_gb,
                    'context_length': model.context_length,
                    'available': model.available,
                    'loaded': model.loaded,
                    'capabilities': model.capabilities,
                    'performance_score': model.performance_score,
                    'usage_count': model.usage_count,
                    'last_used': model.last_used.isoformat() if model.last_used else None,
                    'metadata': model.metadata
                }
                for model_id, model in self.discovered_models.items()
            }
        }