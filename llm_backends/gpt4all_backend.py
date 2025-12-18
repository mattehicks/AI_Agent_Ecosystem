#!/usr/bin/env python3
"""
GPT4All Backend Integration for AI Agent Ecosystem
CLI interface for GPT4All local models
"""

import asyncio
import json
import logging
import subprocess
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)

class GPT4AllBackend:
    """GPT4All CLI interface for local model inference"""
    
    def __init__(self, config: Dict[str, Any]):
        self.gpt4all_path = config.get('gpt4all_path', 'gpt4all')
        self.models_dir = Path(config.get('models_dir', '~/.cache/gpt4all')).expanduser()
        self.default_model = config.get('default_model', 'orca-mini-3b.ggmlv3.q4_0.bin')
        self.timeout = config.get('timeout', 120)
        
    async def health_check(self) -> Dict[str, Any]:
        """Check if GPT4All CLI is available"""
        try:
            process = await asyncio.create_subprocess_exec(
                self.gpt4all_path, '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            
            if process.returncode == 0:
                version = stdout.decode().strip()
                return {
                    'status': 'healthy',
                    'service': 'gpt4all',
                    'version': version,
                    'models_dir': str(self.models_dir)
                }
            else:
                return {
                    'status': 'unhealthy',
                    'service': 'gpt4all',
                    'error': stderr.decode().strip()
                }
        except Exception as e:
            logger.error(f"GPT4All health check failed: {e}")
            return {
                'status': 'unavailable',
                'service': 'gpt4all',
                'error': str(e)
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """List available GPT4All models"""
        try:
            models = []
            
            # Check local models directory
            if self.models_dir.exists():
                for model_file in self.models_dir.glob('*.bin'):
                    models.append({
                        'id': model_file.name,
                        'name': model_file.stem,
                        'path': str(model_file),
                        'size_mb': model_file.stat().st_size / (1024 * 1024),
                        'available': True
                    })
            
            # Add common downloadable models
            common_models = [
                'orca-mini-3b.ggmlv3.q4_0.bin',
                'orca-mini-7b.ggmlv3.q4_0.bin',
                'orca-mini-13b.ggmlv3.q4_0.bin',
                'wizardlm-13b-v1.1.ggmlv3.q4_0.bin',
                'nous-hermes-13b.ggmlv3.q4_0.bin'
            ]
            
            for model_name in common_models:
                if not any(m['id'] == model_name for m in models):
                    models.append({
                        'id': model_name,
                        'name': model_name.replace('.ggmlv3.q4_0.bin', ''),
                        'path': str(self.models_dir / model_name),
                        'size_mb': 0,  # Unknown until downloaded
                        'available': False
                    })
            
            return {
                'success': True,
                'models': models,
                'total_count': len(models)
            }
            
        except Exception as e:
            logger.error(f"Failed to list GPT4All models: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def download_model(self, model_name: str) -> Dict[str, Any]:
        """Download a GPT4All model"""
        try:
            # Ensure models directory exists
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            process = await asyncio.create_subprocess_exec(
                self.gpt4all_path, '--download', model_name,
                '--model-dir', str(self.models_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)  # 10 min timeout
            
            if process.returncode == 0:
                return {
                    'success': True,
                    'model_name': model_name,
                    'message': 'Model downloaded successfully'
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode().strip()
                }
                
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'Model download timed out'
            }
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_text(self, prompt: str,
                          model: str = None,
                          max_tokens: int = 1000,
                          temperature: float = 0.7,
                          top_p: float = 0.9) -> Dict[str, Any]:
        """Generate text using GPT4All"""
        try:
            model_name = model or self.default_model
            model_path = self.models_dir / model_name
            
            if not model_path.exists():
                return {
                    'success': False,
                    'error': f'Model {model_name} not found. Please download it first.'
                }
            
            # Create temporary file for prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt)
                prompt_file = f.name
            
            try:
                # Run GPT4All with parameters
                cmd = [
                    self.gpt4all_path,
                    '--model', str(model_path),
                    '--prompt-file', prompt_file,
                    '--max-tokens', str(max_tokens),
                    '--temp', str(temperature),
                    '--top-p', str(top_p),
                    '--no-interactive'
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
                
                if process.returncode == 0:
                    response = stdout.decode().strip()
                    return {
                        'success': True,
                        'response': response,
                        'model': model_name,
                        'parameters': {
                            'max_tokens': max_tokens,
                            'temperature': temperature,
                            'top_p': top_p
                        }
                    }
                else:
                    return {
                        'success': False,
                        'error': stderr.decode().strip()
                    }
                    
            finally:
                # Clean up temporary file
                os.unlink(prompt_file)
                
        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': 'Text generation timed out'
            }
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def chat_completion(self, messages: List[Dict[str, str]],
                            model: str = None,
                            max_tokens: int = 1000,
                            temperature: float = 0.7) -> Dict[str, Any]:
        """Chat completion using GPT4All"""
        try:
            # Convert messages to a single prompt
            prompt_parts = []
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"Human: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt_parts.append("Assistant:")
            full_prompt = "\n".join(prompt_parts)
            
            result = await self.generate_text(
                prompt=full_prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if result['success']:
                # Clean up the response (remove "Assistant:" prefix if present)
                response = result['response']
                if response.startswith('Assistant:'):
                    response = response[10:].strip()
                
                return {
                    'success': True,
                    'response': response,
                    'model': result['model'],
                    'parameters': result['parameters']
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class GPT4AllIntegration:
    """High-level integration wrapper for GPT4All"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = GPT4AllBackend(config)
        self.preferred_models = config.get('preferred_models', ['orca-mini-3b.ggmlv3.q4_0.bin'])
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize GPT4All integration"""
        try:
            # Check service health
            health = await self.backend.health_check()
            if health['status'] != 'healthy':
                return {
                    'success': False,
                    'error': f"GPT4All not available: {health.get('error', 'Unknown error')}"
                }
            
            # Check if preferred models are available
            models_result = await self.backend.list_models()
            if models_result['success']:
                available_models = [m for m in models_result['models'] if m['available']]
                
                if not available_models:
                    # Try to download first preferred model
                    if self.preferred_models:
                        logger.info(f"No models available, downloading {self.preferred_models[0]}")
                        download_result = await self.backend.download_model(self.preferred_models[0])
                        if not download_result['success']:
                            logger.warning(f"Failed to download model: {download_result['error']}")
            
            return {
                'success': True,
                'service_health': health,
                'models_available': len([m for m in models_result.get('models', []) if m['available']])
            }
            
        except Exception as e:
            logger.error(f"GPT4All initialization failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def generate_response(self, prompt: str,
                              model: str = None,
                              max_length: int = 1000,
                              creativity: str = "balanced") -> Dict[str, Any]:
        """Generate response with creativity levels"""
        try:
            # Map creativity to temperature
            temperature_map = {
                'conservative': 0.3,
                'balanced': 0.7,
                'creative': 1.0
            }
            temperature = temperature_map.get(creativity, 0.7)
            
            result = await self.backend.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=max_length,
                temperature=temperature
            )
            
            if result['success']:
                return {
                    'success': True,
                    'generated_text': result['response'],
                    'model_used': result['model'],
                    'creativity_level': creativity,
                    'parameters': result['parameters']
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def analyze_text(self, text: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze text with different analysis types"""
        try:
            if analysis_type == "summary":
                prompt = f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
            elif analysis_type == "sentiment":
                prompt = f"Analyze the sentiment of the following text (positive, negative, or neutral):\n\n{text}\n\nSentiment:"
            elif analysis_type == "key_points":
                prompt = f"Extract the key points from the following text:\n\n{text}\n\nKey points:"
            elif analysis_type == "questions":
                prompt = f"Generate 3-5 questions that this text answers:\n\n{text}\n\nQuestions:"
            else:
                prompt = f"Analyze the following text:\n\n{text}\n\nAnalysis:"
            
            result = await self.generate_response(
                prompt=prompt,
                max_length=500,
                creativity="conservative"
            )
            
            if result['success']:
                return {
                    'success': True,
                    'analysis': result['generated_text'],
                    'analysis_type': analysis_type,
                    'model_used': result['model_used']
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def code_assistance(self, description: str, language: str = "python") -> Dict[str, Any]:
        """Generate code based on description"""
        try:
            prompt = f"""Generate {language} code for the following task:
{description}

Please provide clean, well-commented code that follows best practices.

{language} code:"""
            
            result = await self.generate_response(
                prompt=prompt,
                max_length=1500,
                creativity="balanced"
            )
            
            if result['success']:
                return {
                    'success': True,
                    'generated_code': result['generated_text'],
                    'language': language,
                    'model_used': result['model_used']
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Code assistance failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def question_answering(self, question: str, context: str = None) -> Dict[str, Any]:
        """Answer questions with optional context"""
        try:
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\n\nAnswer:"
            
            result = await self.generate_response(
                prompt=prompt,
                max_length=800,
                creativity="balanced"
            )
            
            if result['success']:
                return {
                    'success': True,
                    'answer': result['generated_text'],
                    'question': question,
                    'used_context': context is not None,
                    'model_used': result['model_used']
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            health = await self.backend.health_check()
            models = await self.backend.list_models()
            
            available_models = []
            if models['success']:
                available_models = [m for m in models['models'] if m['available']]
            
            return {
                'service_health': health,
                'available_models': len(available_models),
                'total_models': models.get('total_count', 0) if models['success'] else 0,
                'models_directory': str(self.backend.models_dir),
                'preferred_models': self.preferred_models
            }
            
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {
                'service_health': {'status': 'error', 'error': str(e)},
                'available_models': 0,
                'total_models': 0,
                'models_directory': str(self.backend.models_dir),
                'preferred_models': self.preferred_models
            }