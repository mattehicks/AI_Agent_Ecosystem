#!/usr/bin/env python3
"""
Base Agent Class
Foundation for all AI agents in the ecosystem
"""

import asyncio
import json
import logging
import os
import sys
import subprocess
import requests
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

class BaseAgent(ABC):
    """Base class for all AI agents"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        # Detect environment and set paths accordingly
        if os.name == 'nt' or os.path.exists("X:/"):
            # Windows environment
            self.base_path = Path("X:/")
        else:
            # Linux environment
            self.base_path = Path("/mnt/llm/")
        
        # Paths to existing infrastructure
        self.privateGPT_path = self.base_path / "privateGPT"
        self.gpt4all_cli_path = self.base_path / "gpt4all_cli" / "app.py"
        self.models_path = self.base_path / "LLM-Models"
        self.text_vault_path = self.base_path / "TEXT-VAULT"
        self.aiprompts_path = self.base_path / "AIPROMPTS"
        
        # LM Studio configuration
        self.lm_studio_url = "http://localhost:1234"
        
        # Load agent-specific configuration
        self.config = self._load_agent_config()
        
        self.logger.info(f"{self.__class__.__name__} initialized")

    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent-specific configuration"""
        if os.name == 'nt' or os.path.exists("X:/"):
            # Windows environment
            config_path = Path("X:/AI_Agent_Ecosystem/config/agents.yaml")
        else:
            # Linux environment
            config_path = Path("/mnt/llm/AI_Agent_Ecosystem/config/agents.yaml")
        
        # Default configuration
        default_config = {
            'timeout': 120,
            'max_retries': 3,
            'memory_limit': '2GB',
            'models': {
                'primary': 'dolphin-2.9.4-llama3.1-8b-Q4_K_M.gguf',
                'fallback': 'mistral-7b-instruct-v0.1.Q4_0.gguf'
            }
        }
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    agent_name = self.__class__.__name__.lower().replace('agent', '')
                    if agent_name in loaded_config.get('agents', {}):
                        default_config.update(loaded_config['agents'][agent_name])
            except Exception as e:
                self.logger.warning(f"Could not load agent config: {e}. Using defaults.")
        
        return default_config

    @abstractmethod
    async def process_task(self, task) -> Dict[str, Any]:
        """Process a task - to be implemented by subclasses"""
        pass

    async def call_gpt4all_cli(self, prompt: str, model: str = None) -> str:
        """Call GPT4All CLI for text generation"""
        if model is None:
            model = self.config['models']['primary']
        
        try:
            self.logger.debug(f"Calling GPT4All with model: {model}")
            
            # For now, implement a simple subprocess call
            # In production, you might want to use a more sophisticated approach
            cmd = [
                sys.executable,
                str(self.gpt4all_cli_path),
                "repl",
                "--model", model
            ]
            
            # Create a temporary script to handle the interaction
            temp_script = self._create_gpt4all_script(prompt)
            
            try:
                # Execute the temporary script
                result = subprocess.run([sys.executable, temp_script], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=self.config['timeout'])
                
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    self.logger.error(f"GPT4All CLI error: {result.stderr}")
                    return f"Error: {result.stderr}"
                    
            finally:
                # Clean up temporary script
                if os.path.exists(temp_script):
                    os.remove(temp_script)
                    
        except subprocess.TimeoutExpired:
            self.logger.error("GPT4All CLI call timed out")
            return "Error: Request timed out"
        except Exception as e:
            self.logger.error(f"Error calling GPT4All CLI: {e}")
            return f"Error: {e}"

    def _create_gpt4all_script(self, prompt: str) -> str:
        """Create a temporary script for GPT4All interaction"""
        script_content = f'''
import sys
sys.path.append("{self.gpt4all_cli_path.parent}")
from gpt4all import GPT4All

try:
    model = GPT4All("{self.config['models']['primary']}")
    response = model.generate("{prompt}", max_tokens=500)
    print(response)
except Exception as e:
    print(f"Error: {{e}}")
'''
        
        if os.name == 'nt' or os.path.exists("X:/"):
            # Windows environment
            temp_path = Path("X:/AI_Agent_Ecosystem/temp") / f"gpt4all_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py"
        else:
            # Linux environment
            temp_path = Path("/mnt/llm/AI_Agent_Ecosystem/temp") / f"gpt4all_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py"
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, 'w') as f:
            f.write(script_content)
        
        return str(temp_path)

    async def call_lm_studio(self, prompt: str, model: str = None, **kwargs) -> str:
        """Call LM Studio API for text generation"""
        if model is None:
            model = self.config['models']['primary']
        
        try:
            self.logger.debug(f"Calling LM Studio with model: {model}")
            
            # Prepare request payload
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get('temperature', 0.7),
                "max_tokens": kwargs.get('max_tokens', 500),
                "stream": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json=payload,
                timeout=self.config['timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                self.logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return f"Error: LM Studio API returned {response.status_code}"
                
        except requests.exceptions.Timeout:
            self.logger.error("LM Studio API call timed out")
            return "Error: Request timed out"
        except requests.exceptions.ConnectionError:
            self.logger.error("Could not connect to LM Studio API")
            return "Error: Could not connect to LM Studio"
        except Exception as e:
            self.logger.error(f"Error calling LM Studio API: {e}")
            return f"Error: {e}"

    async def query_privateGPT(self, query: str) -> str:
        """Query privateGPT for document-based answers"""
        try:
            self.logger.debug(f"Querying privateGPT: {query[:100]}...")
            
            # Create a temporary script to query privateGPT
            script_content = f'''
import sys
sys.path.append("{self.privateGPT_path}")
import os
os.chdir("{self.privateGPT_path}")

try:
    from privateGPT import main
    # This is a simplified approach - in practice you'd need to 
    # properly interface with privateGPT's query mechanism
    result = "privateGPT response to: {query[:100]}..."
    print(result)
except Exception as e:
    print(f"Error: {{e}}")
'''
            
            if os.name == 'nt' or os.path.exists("X:/"):
                # Windows environment
                temp_path = Path("X:/AI_Agent_Ecosystem/temp") / f"privateGPT_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py"
            else:
                # Linux environment
                temp_path = Path("/mnt/llm/AI_Agent_Ecosystem/temp") / f"privateGPT_temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.py"
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, 'w') as f:
                f.write(script_content)
            
            try:
                result = subprocess.run([sys.executable, str(temp_path)], 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=self.config['timeout'])
                
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    self.logger.error(f"privateGPT error: {result.stderr}")
                    return f"Error querying privateGPT: {result.stderr}"
                    
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            self.logger.error(f"Error querying privateGPT: {e}")
            return f"Error: {e}"

    def read_document(self, path: str) -> str:
        """Read document content with encoding detection"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    self.logger.debug(f"Successfully read {path} with encoding {encoding}")
                    return content
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try binary mode
            with open(path, 'rb') as f:
                content = f.read()
                # Try to decode as utf-8 with error handling
                return content.decode('utf-8', errors='replace')
                
        except Exception as e:
            self.logger.error(f"Error reading document {path}: {e}")
            raise

    def list_documents(self, directory: str, extensions: List[str] = None) -> List[str]:
        """List documents in a directory with optional extension filtering"""
        if extensions is None:
            extensions = ['.txt', '.rtf', '.md', '.docx', '.pdf']
        
        try:
            dir_path = Path(directory)
            documents = []
            
            for ext in extensions:
                documents.extend(dir_path.rglob(f"*{ext}"))
            
            return [str(doc) for doc in documents]
            
        except Exception as e:
            self.logger.error(f"Error listing documents in {directory}: {e}")
            return []

    def extract_text_from_rtf(self, rtf_path: str) -> str:
        """Extract plain text from RTF files"""
        try:
            # Simple RTF text extraction (basic implementation)
            with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Remove RTF control codes (basic approach)
            import re
            # Remove RTF header
            content = re.sub(r'\\rtf\d+.*?\\deff\d+', '', content)
            # Remove control words
            content = re.sub(r'\\[a-z]+\d*', '', content)
            # Remove control symbols
            content = re.sub(r'\\[^a-z]', '', content)
            # Remove braces
            content = re.sub(r'[{}]', '', content)
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error extracting text from RTF {rtf_path}: {e}")
            return ""

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get full path to a model file"""
        try:
            # Search for model in various subdirectories
            search_paths = [
                self.models_path / model_name,
                self.models_path / "dolphin-2.9.4-llama3.1-8b-gguf" / model_name,
                self.models_path / "Meta-Llama-3.1-8B-Instruct-abliterated-GGUF" / model_name,
            ]
            
            for path in search_paths:
                if path.exists():
                    return str(path)
            
            # Search recursively
            for model_file in self.models_path.rglob(model_name):
                return str(model_file)
            
            self.logger.warning(f"Model {model_name} not found")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding model {model_name}: {e}")
            return None

    def format_response(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format agent response with metadata"""
        response = {
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'agent_type': self.__class__.__name__,
            'success': True
        }
        
        if metadata:
            response.update(metadata)
        
        return response

    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Handle and format errors"""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg)
        
        return {
            'content': f"Error: {error_msg}",
            'timestamp': datetime.now().isoformat(),
            'agent_type': self.__class__.__name__,
            'success': False,
            'error': error_msg
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for the agent"""
        checks = {
            'gpt4all_cli': self._check_gpt4all_cli(),
            'lm_studio': await self._check_lm_studio(),
            'privateGPT': self._check_privateGPT(),
            'models': self._check_models(),
            'documents': self._check_documents()
        }
        
        all_healthy = all(check['status'] == 'ok' for check in checks.values())
        
        return {
            'agent_type': self.__class__.__name__,
            'overall_status': 'healthy' if all_healthy else 'unhealthy',
            'checks': checks,
            'timestamp': datetime.now().isoformat()
        }

    def _check_gpt4all_cli(self) -> Dict[str, str]:
        """Check if GPT4All CLI is available"""
        try:
            if self.gpt4all_cli_path.exists():
                return {'status': 'ok', 'message': 'GPT4All CLI available'}
            else:
                return {'status': 'error', 'message': 'GPT4All CLI not found'}
        except Exception as e:
            return {'status': 'error', 'message': f'GPT4All CLI check failed: {e}'}

    async def _check_lm_studio(self) -> Dict[str, str]:
        """Check if LM Studio API is available"""
        try:
            response = requests.get(f"{self.lm_studio_url}/health", timeout=5)
            if response.status_code == 200:
                return {'status': 'ok', 'message': 'LM Studio API available'}
            else:
                return {'status': 'warning', 'message': f'LM Studio API returned {response.status_code}'}
        except requests.exceptions.ConnectionError:
            return {'status': 'warning', 'message': 'LM Studio API not available'}
        except Exception as e:
            return {'status': 'error', 'message': f'LM Studio check failed: {e}'}

    def _check_privateGPT(self) -> Dict[str, str]:
        """Check if privateGPT is available"""
        try:
            if (self.privateGPT_path / "privateGPT.py").exists():
                return {'status': 'ok', 'message': 'privateGPT available'}
            else:
                return {'status': 'error', 'message': 'privateGPT not found'}
        except Exception as e:
            return {'status': 'error', 'message': f'privateGPT check failed: {e}'}

    def _check_models(self) -> Dict[str, str]:
        """Check if required models are available"""
        try:
            primary_model = self.config['models']['primary']
            fallback_model = self.config['models']['fallback']
            
            primary_path = self.get_model_path(primary_model)
            fallback_path = self.get_model_path(fallback_model)
            
            if primary_path:
                return {'status': 'ok', 'message': f'Primary model {primary_model} available'}
            elif fallback_path:
                return {'status': 'warning', 'message': f'Only fallback model {fallback_model} available'}
            else:
                return {'status': 'error', 'message': 'No required models found'}
        except Exception as e:
            return {'status': 'error', 'message': f'Model check failed: {e}'}

    def _check_documents(self) -> Dict[str, str]:
        """Check if document sources are available"""
        try:
            text_vault_exists = self.text_vault_path.exists()
            privateGPT_docs_exist = (self.privateGPT_path / "source_documents").exists()
            
            if text_vault_exists and privateGPT_docs_exist:
                return {'status': 'ok', 'message': 'All document sources available'}
            elif text_vault_exists or privateGPT_docs_exist:
                return {'status': 'warning', 'message': 'Some document sources available'}
            else:
                return {'status': 'error', 'message': 'No document sources found'}
        except Exception as e:
            return {'status': 'error', 'message': f'Document check failed: {e}'}