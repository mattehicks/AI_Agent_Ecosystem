#!/usr/bin/env python3
"""
Ollama Backend Integration for AI Agent Ecosystem
Connects to Ollama API for local LLM inference
"""

import asyncio
import json
import logging
import aiohttp
from typing import Dict, List, Optional, Any, AsyncGenerator
import time

logger = logging.getLogger(__name__)

class OllamaBackend:
    """Ollama API client for local LLM inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', 'http://localhost:11434')
        self.timeout = config.get('timeout', 120)
        self.max_retries = config.get('max_retries', 3)
        self.session = None
        self.available_models = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'Content-Type': 'application/json'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if Ollama service is available"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    return {
                        'status': 'healthy',
                        'service': 'ollama',
                        'models_available': len(models),
                        'base_url': self.base_url
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'service': 'ollama',
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                'status': 'unavailable',
                'service': 'ollama',
                'error': str(e)
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('models', [])
                    
                    self.available_models = []
                    for model in models:
                        self.available_models.append({
                            'name': model['name'],
                            'size': model.get('size', 0),
                            'modified_at': model.get('modified_at'),
                            'digest': model.get('digest', ''),
                            'details': model.get('details', {})
                        })
                    
                    return {
                        'success': True,
                        'models': self.available_models,
                        'total_count': len(self.available_models)
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
    
    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull/download a model"""
        try:
            payload = {'name': model_name}
            
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                
                if response.status == 200:
                    return {
                        'success': True,
                        'model_name': model_name,
                        'message': 'Model pull started'
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                            model: str = "llama3.1:8b",
                            temperature: float = 0.7,
                            max_tokens: int = 1000,
                            stream: bool = False) -> Dict[str, Any]:
        """Chat completion with Ollama"""
        try:
            # Convert messages to Ollama format
            if len(messages) == 1 and messages[0].get('role') == 'user':
                # Simple prompt
                prompt = messages[0]['content']
            else:
                # Multi-turn conversation
                prompt_parts = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'system':
                        prompt_parts.append(f"System: {content}")
                    elif role == 'user':
                        prompt_parts.append(f"Human: {content}")
                    elif role == 'assistant':
                        prompt_parts.append(f"Assistant: {content}")
                
                prompt_parts.append("Assistant:")
                prompt = "\n".join(prompt_parts)
            
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': stream,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            }
            
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status == 200:
                    if stream:
                        # Handle streaming response
                        full_response = ""
                        async for line in response.content:
                            try:
                                chunk = json.loads(line.decode())
                                if 'response' in chunk:
                                    full_response += chunk['response']
                                if chunk.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                        
                        processing_time = time.time() - start_time
                        return {
                            'success': True,
                            'response': full_response,
                            'model': model,
                            'processing_time': processing_time
                        }
                    else:
                        # Non-streaming response
                        result = await response.json()
                        processing_time = time.time() - start_time
                        
                        return {
                            'success': True,
                            'response': result.get('response', ''),
                            'model': model,
                            'processing_time': processing_time,
                            'context': result.get('context', []),
                            'total_duration': result.get('total_duration', 0),
                            'load_duration': result.get('load_duration', 0),
                            'prompt_eval_count': result.get('prompt_eval_count', 0),
                            'eval_count': result.get('eval_count', 0)
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
    
    async def generate_text(self, prompt: str,
                          model: str = "llama3.1:8b",
                          temperature: float = 0.7,
                          max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate text using Ollama"""
        try:
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            }
            
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    return {
                        'success': True,
                        'generated_text': result.get('response', ''),
                        'model': model,
                        'processing_time': processing_time,
                        'stats': {
                            'total_duration': result.get('total_duration', 0),
                            'load_duration': result.get('load_duration', 0),
                            'prompt_eval_count': result.get('prompt_eval_count', 0),
                            'eval_count': result.get('eval_count', 0)
                        }
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def stream_generate(self, prompt: str,
                            model: str = "llama3.1:8b",
                            temperature: float = 0.7,
                            max_tokens: int = 1000) -> AsyncGenerator[str, None]:
        """Stream text generation"""
        try:
            payload = {
                'model': model,
                'prompt': prompt,
                'stream': True,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        try:
                            chunk = json.loads(line.decode())
                            if 'response' in chunk:
                                yield chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                else:
                    error_text = await response.text()
                    logger.error(f"Streaming failed: HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
    
    async def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model"""
        try:
            payload = {'name': model_name}
            
            async with self.session.delete(
                f"{self.base_url}/api/delete",
                json=payload
            ) as response:
                
                if response.status == 200:
                    return {
                        'success': True,
                        'model_name': model_name,
                        'message': 'Model deleted successfully'
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            payload = {'name': model_name}
            
            async with self.session.post(
                f"{self.base_url}/api/show",
                json=payload
            ) as response:
                
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

class OllamaIntegration:
    """High-level integration wrapper for Ollama"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = OllamaBackend(config)
        self.default_model = config.get('default_model', 'llama3.1:8b')
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Ollama integration"""
        try:
            async with self.backend:
                # Check service health
                health = await self.backend.health_check()
                if health['status'] != 'healthy':
                    return {
                        'success': False,
                        'error': f"Ollama not healthy: {health.get('error', 'Unknown error')}"
                    }
                
                # List available models
                models_result = await self.backend.list_models()
                if models_result['success']:
                    logger.info(f"Ollama initialized with {len(models_result['models'])} models")
                
                return {
                    'success': True,
                    'service_health': health,
                    'available_models': models_result.get('total_count', 0)
                }
                
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_text_generation(self, prompt: str, 
                                    model: str = None,
                                    options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process text generation request"""
        try:
            model_name = model or self.default_model
            opts = options or {}
            
            temperature = opts.get('temperature', 0.7)
            max_tokens = opts.get('max_tokens', 1000)
            
            async with self.backend:
                result = await self.backend.generate_text(
                    prompt=prompt,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if result['success']:
                    return {
                        'success': True,
                        'generated_text': result['generated_text'],
                        'model_used': result['model'],
                        'processing_time': result['processing_time'],
                        'stats': result.get('stats', {})
                    }
                else:
                    return result
                    
        except Exception as e:
            logger.error(f"Text generation processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_document_analysis(self, content: str, 
                                      analysis_type: str = "summary",
                                      model: str = None) -> Dict[str, Any]:
        """Process document analysis request"""
        try:
            model_name = model or self.default_model
            
            # Create analysis prompt based on type
            if analysis_type == "summary":
                prompt = f"Please provide a comprehensive summary of the following document:\n\n{content}\n\nSummary:"
            elif analysis_type == "key_points":
                prompt = f"Extract the key points from the following document:\n\n{content}\n\nKey Points:"
            elif analysis_type == "questions":
                prompt = f"Generate 5 important questions that this document answers:\n\n{content}\n\nQuestions:"
            elif analysis_type == "analysis":
                prompt = f"Provide a detailed analysis of the following document:\n\n{content}\n\nAnalysis:"
            else:
                prompt = f"Analyze the following document:\n\n{content}\n\nAnalysis:"
            
            async with self.backend:
                result = await self.backend.generate_text(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.3,  # Lower temperature for analysis
                    max_tokens=1500
                )
                
                if result['success']:
                    return {
                        'success': True,
                        'analysis': result['generated_text'],
                        'analysis_type': analysis_type,
                        'model_used': result['model'],
                        'processing_time': result['processing_time']
                    }
                else:
                    return result
                    
        except Exception as e:
            logger.error(f"Document analysis processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_code_generation(self, requirements: str,
                                    language: str = "python",
                                    model: str = None) -> Dict[str, Any]:
        """Process code generation request"""
        try:
            model_name = model or self.default_model
            
            prompt = f"""Generate {language} code for the following requirements:

Requirements: {requirements}

Please provide clean, well-commented, and functional {language} code that meets these requirements. Include error handling where appropriate.

{language} code:"""
            
            async with self.backend:
                result = await self.backend.generate_text(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.2,  # Lower temperature for code
                    max_tokens=2000
                )
                
                if result['success']:
                    return {
                        'success': True,
                        'generated_code': result['generated_text'],
                        'language': language,
                        'model_used': result['model'],
                        'processing_time': result['processing_time']
                    }
                else:
                    return result
                    
        except Exception as e:
            logger.error(f"Code generation processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_research_query(self, query: str,
                                   context: str = None,
                                   model: str = None) -> Dict[str, Any]:
        """Process research query"""
        try:
            model_name = model or self.default_model
            
            if context:
                prompt = f"""Context: {context}

Research Question: {query}

Please provide a comprehensive answer to this research question based on the provided context. Include relevant details, analysis, and insights.

Answer:"""
            else:
                prompt = f"""Research Question: {query}

Please provide a comprehensive answer to this research question. Include relevant analysis, insights, and any important considerations.

Answer:"""
            
            async with self.backend:
                result = await self.backend.generate_text(
                    prompt=prompt,
                    model=model_name,
                    temperature=0.5,
                    max_tokens=1500
                )
                
                if result['success']:
                    return {
                        'success': True,
                        'answer': result['generated_text'],
                        'query': query,
                        'used_context': context is not None,
                        'model_used': result['model'],
                        'processing_time': result['processing_time']
                    }
                else:
                    return result
                    
        except Exception as e:
            logger.error(f"Research query processing failed: {e}")
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
                    'default_model': self.default_model,
                    'backend_url': self.backend.base_url
                }
                
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {
                'service_health': {'status': 'error', 'error': str(e)},
                'available_models': 0,
                'default_model': self.default_model,
                'backend_url': self.backend.base_url
            }