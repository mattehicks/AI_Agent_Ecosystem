#!/usr/bin/env python3
"""
LM Studio Backend Integration for AI Agent Ecosystem
Connects to LM Studio local API for LLM inference
"""

import asyncio
import json
import logging
import aiohttp
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    id: str
    name: str
    size_gb: float
    context_length: int
    loaded: bool = False
    loading_progress: float = 0.0

class LMStudioBackend:
    """LM Studio API client for local LLM inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', 'http://localhost:1234')
        self.api_key = config.get('api_key')  # Usually not needed for local
        self.timeout = config.get('timeout', 120)
        self.max_retries = config.get('max_retries', 3)
        self.session = None
        self.current_model = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if LM Studio service is available"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    return {
                        'status': 'healthy',
                        'service': 'lm_studio',
                        'models_available': len(models_data.get('data', [])),
                        'current_model': self.current_model
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'service': 'lm_studio',
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            logger.error(f"LM Studio health check failed: {e}")
            return {
                'status': 'unavailable',
                'service': 'lm_studio',
                'error': str(e)
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    
                    for model_data in data.get('data', []):
                        model = ModelInfo(
                            id=model_data.get('id', ''),
                            name=model_data.get('id', ''),
                            size_gb=model_data.get('size', 0) / (1024**3),  # Convert to GB
                            context_length=model_data.get('context_length', 2048),
                            loaded=model_data.get('loaded', False)
                        )
                        models.append(model)
                    
                    return {
                        'success': True,
                        'models': models,
                        'total_count': len(models)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model in LM Studio"""
        try:
            payload = {'model': model_id}
            
            async with self.session.post(
                f"{self.base_url}/v1/models/load",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    self.current_model = model_id
                    return {
                        'success': True,
                        'model_id': model_id,
                        'message': result.get('message', 'Model loaded successfully')
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def unload_model(self) -> Dict[str, Any]:
        """Unload current model"""
        try:
            async with self.session.post(f"{self.base_url}/v1/models/unload") as response:
                if response.status == 200:
                    self.current_model = None
                    return {
                        'success': True,
                        'message': 'Model unloaded successfully'
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                            model: str = None,
                            temperature: float = 0.7,
                            max_tokens: int = 1000,
                            stream: bool = False) -> Dict[str, Any]:
        """Chat completion with LM Studio"""
        try:
            payload = {
                'model': model or self.current_model or 'default',
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'stream': stream
            }
            
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'response': result.get('choices', [{}])[0].get('message', {}).get('content', ''),
                        'model': result.get('model', model),
                        'usage': result.get('usage', {}),
                        'processing_time': processing_time
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def text_completion(self, prompt: str,
                            model: str = None,
                            temperature: float = 0.7,
                            max_tokens: int = 1000) -> Dict[str, Any]:
        """Text completion with LM Studio"""
        try:
            payload = {
                'model': model or self.current_model or 'default',
                'prompt': prompt,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
            
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'response': result.get('choices', [{}])[0].get('text', ''),
                        'model': result.get('model', model),
                        'usage': result.get('usage', {}),
                        'processing_time': processing_time
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Text completion failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def stream_chat_completion(self, messages: List[Dict[str, str]], 
                                   model: str = None,
                                   temperature: float = 0.7,
                                   max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """Streaming chat completion"""
        try:
            payload = {
                'model': model or self.current_model or 'default',
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'stream': True
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    logger.error(f"Streaming failed: HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
    
    async def get_model_info(self, model_id: str = None) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            target_model = model_id or self.current_model
            if not target_model:
                return {
                    'success': False,
                    'error': 'No model specified and no current model loaded'
                }
            
            async with self.session.get(f"{self.base_url}/v1/models/{target_model}") as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'model_info': result
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class LMStudioIntegration:
    """High-level integration wrapper for LM Studio"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = LMStudioBackend(config)
        self.preferred_models = config.get('preferred_models', [])
        self.auto_load_model = config.get('auto_load_model', True)
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the LM Studio integration"""
        try:
            async with self.backend:
                # Check service health
                health = await self.backend.health_check()
                if health['status'] != 'healthy':
                    return {
                        'success': False,
                        'error': f"LM Studio not healthy: {health.get('error', 'Unknown error')}"
                    }
                
                # Auto-load preferred model if enabled
                if self.auto_load_model and self.preferred_models:
                    models_result = await self.backend.list_models()
                    if models_result['success']:
                        available_models = [m.id for m in models_result['models']]
                        
                        # Find first preferred model that's available
                        for preferred in self.preferred_models:
                            if preferred in available_models:
                                load_result = await self.backend.load_model(preferred)
                                if load_result['success']:
                                    logger.info(f"Auto-loaded model: {preferred}")
                                    break
                
                return {
                    'success': True,
                    'service_health': health,
                    'current_model': self.backend.current_model
                }
                
        except Exception as e:
            logger.error(f"LM Studio initialization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_text(self, prompt: str, 
                          model: str = None,
                          max_length: int = 1000,
                          temperature: float = 0.7,
                          context: str = None) -> Dict[str, Any]:
        """Generate text with optional context"""
        try:
            # Prepare full prompt with context
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {context}\n\nPrompt: {prompt}"
            
            async with self.backend:
                result = await self.backend.text_completion(
                    prompt=full_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_length
                )
                
                if result['success']:
                    return {
                        'success': True,
                        'generated_text': result['response'],
                        'model_used': result['model'],
                        'processing_time': result['processing_time'],
                        'token_usage': result.get('usage', {})
                    }
                else:
                    return result
                    
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def chat_conversation(self, messages: List[Dict[str, str]], 
                              model: str = None,
                              system_prompt: str = None) -> Dict[str, Any]:
        """Conduct a chat conversation"""
        try:
            # Add system prompt if provided
            full_messages = []
            if system_prompt:
                full_messages.append({"role": "system", "content": system_prompt})
            full_messages.extend(messages)
            
            async with self.backend:
                result = await self.backend.chat_completion(
                    messages=full_messages,
                    model=model
                )
                
                if result['success']:
                    return {
                        'success': True,
                        'response': result['response'],
                        'model_used': result['model'],
                        'processing_time': result['processing_time'],
                        'token_usage': result.get('usage', {})
                    }
                else:
                    return result
                    
        except Exception as e:
            logger.error(f"Chat conversation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def code_generation(self, description: str, 
                            language: str = "python",
                            model: str = None) -> Dict[str, Any]:
        """Generate code based on description"""
        try:
            system_prompt = f"""You are an expert {language} programmer. Generate clean, well-commented, and functional code based on the user's description. 
            
Follow these guidelines:
- Write production-ready code
- Include proper error handling
- Add meaningful comments
- Follow {language} best practices
- Provide complete, runnable code"""
            
            messages = [
                {"role": "user", "content": f"Generate {language} code for: {description}"}
            ]
            
            result = await self.chat_conversation(
                messages=messages,
                model=model,
                system_prompt=system_prompt
            )
            
            if result['success']:
                return {
                    'success': True,
                    'generated_code': result['response'],
                    'language': language,
                    'model_used': result['model_used'],
                    'processing_time': result['processing_time']
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def summarize_text(self, text: str, 
                           summary_type: str = "concise",
                           model: str = None) -> Dict[str, Any]:
        """Summarize text with different summary types"""
        try:
            if summary_type == "concise":
                prompt = f"Provide a concise summary of the following text in 2-3 sentences:\n\n{text}"
            elif summary_type == "detailed":
                prompt = f"Provide a detailed summary of the following text, including key points and important details:\n\n{text}"
            elif summary_type == "bullet_points":
                prompt = f"Summarize the following text as bullet points, highlighting the main ideas:\n\n{text}"
            else:
                prompt = f"Summarize the following text:\n\n{text}"
            
            result = await self.generate_text(
                prompt=prompt,
                model=model,
                max_length=500
            )
            
            if result['success']:
                return {
                    'success': True,
                    'summary': result['generated_text'],
                    'summary_type': summary_type,
                    'model_used': result['model_used'],
                    'processing_time': result['processing_time']
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Text summarization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            async with self.backend:
                health = await self.backend.health_check()
                models = await self.backend.list_models()
                
                return {
                    'service_health': health,
                    'available_models': models.get('total_count', 0) if models['success'] else 0,
                    'current_model': self.backend.current_model,
                    'preferred_models': self.preferred_models,
                    'backend_url': self.backend.base_url
                }
                
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {
                'service_health': {'status': 'error', 'error': str(e)},
                'available_models': 0,
                'current_model': None,
                'preferred_models': self.preferred_models,
                'backend_url': self.backend.base_url
            }