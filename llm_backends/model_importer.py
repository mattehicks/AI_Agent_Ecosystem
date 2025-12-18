#!/usr/bin/env python3
"""
Model Importer for AI Agent Ecosystem
Imports existing models from /mnt/llm/LLM-Models into Ollama
"""

import os
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
import aiohttp
import subprocess

logger = logging.getLogger(__name__)

class ModelImporter:
    """Import existing models into Ollama"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models_path = Path(config.get('models_path', '/mnt/llm/LLM-Models'))
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
        self.ollama_models_path = Path.home() / '.ollama' / 'models'
        
        # Supported model formats
        self.supported_formats = {
            '.gguf': 'GGUF',
            '.bin': 'PyTorch',
            '.safetensors': 'SafeTensors'
        }
        
    async def discover_models(self) -> Dict[str, Any]:
        """Discover available models in the LLM-Models directory"""
        try:
            discovered_models = []
            
            if not self.models_path.exists():
                return {
                    'success': False,
                    'error': f'Models directory not found: {self.models_path}'
                }
            
            for item in self.models_path.iterdir():
                if item.is_dir():
                    model_info = await self._analyze_model_directory(item)
                    if model_info:
                        discovered_models.append(model_info)
                elif item.is_file() and item.suffix.lower() in self.supported_formats:
                    model_info = await self._analyze_model_file(item)
                    if model_info:
                        discovered_models.append(model_info)
            
            return {
                'success': True,
                'models': discovered_models,
                'total_count': len(discovered_models)
            }
            
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _analyze_model_directory(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze a model directory to extract information"""
        try:
            model_files = []
            config_files = []
            
            # Look for model files
            for file_path in model_dir.rglob('*'):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    if suffix in self.supported_formats:
                        model_files.append({
                            'path': str(file_path),
                            'name': file_path.name,
                            'size_bytes': file_path.stat().st_size,
                            'format': self.supported_formats[suffix]
                        })
                    elif file_path.name.lower() in ['config.json', 'tokenizer_config.json', 'README.md']:
                        config_files.append(str(file_path))
            
            if not model_files:
                return None
            
            # Try to determine model type and size
            model_name = model_dir.name
            model_type = self._determine_model_type(model_name)
            estimated_size = self._estimate_model_size(model_name, model_files)
            
            return {
                'name': model_name,
                'path': str(model_dir),
                'type': model_type,
                'estimated_size': estimated_size,
                'model_files': model_files,
                'config_files': config_files,
                'file_count': len(model_files),
                'importable': self._is_importable(model_files)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze model directory {model_dir}: {e}")
            return None
    
    async def _analyze_model_file(self, model_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single model file"""
        try:
            suffix = model_file.suffix.lower()
            if suffix not in self.supported_formats:
                return None
            
            model_name = model_file.stem
            model_type = self._determine_model_type(model_name)
            
            return {
                'name': model_name,
                'path': str(model_file.parent),
                'type': model_type,
                'estimated_size': self._estimate_size_from_file(model_file),
                'model_files': [{
                    'path': str(model_file),
                    'name': model_file.name,
                    'size_bytes': model_file.stat().st_size,
                    'format': self.supported_formats[suffix]
                }],
                'config_files': [],
                'file_count': 1,
                'importable': suffix == '.gguf'  # Only GGUF files are easily importable to Ollama
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze model file {model_file}: {e}")
            return None
    
    def _determine_model_type(self, name: str) -> str:
        """Determine model type from name"""
        name_lower = name.lower()
        
        if 'llama' in name_lower:
            return 'Llama'
        elif 'code' in name_lower:
            return 'Code Generation'
        elif 'wizard' in name_lower:
            return 'Wizard'
        elif 'vicuna' in name_lower:
            return 'Vicuna'
        elif 'dolphin' in name_lower:
            return 'Dolphin'
        elif 'bert' in name_lower:
            return 'BERT'
        elif 'gpt' in name_lower:
            return 'GPT'
        else:
            return 'General'
    
    def _estimate_model_size(self, name: str, model_files: List[Dict]) -> str:
        """Estimate model parameter size from name and files"""
        name_lower = name.lower()
        
        # Extract size from name
        if '70b' in name_lower or '72b' in name_lower:
            return '70B'
        elif '34b' in name_lower:
            return '34B'
        elif '30b' in name_lower:
            return '30B'
        elif '16b' in name_lower:
            return '16B'
        elif '13b' in name_lower:
            return '13B'
        elif '8b' in name_lower:
            return '8B'
        elif '7b' in name_lower:
            return '7B'
        elif '3b' in name_lower:
            return '3B'
        elif '1b' in name_lower:
            return '1B'
        
        # Estimate from file sizes (rough approximation)
        total_size = sum(f['size_bytes'] for f in model_files)
        size_gb = total_size / (1024**3)
        
        if size_gb > 50:
            return '70B+'
        elif size_gb > 25:
            return '34B'
        elif size_gb > 15:
            return '16B'
        elif size_gb > 10:
            return '13B'
        elif size_gb > 6:
            return '8B'
        elif size_gb > 4:
            return '7B'
        else:
            return 'Small'
    
    def _estimate_size_from_file(self, file_path: Path) -> str:
        """Estimate model size from single file"""
        try:
            size_bytes = file_path.stat().st_size
            size_gb = size_bytes / (1024**3)
            
            if size_gb > 50:
                return '70B+'
            elif size_gb > 25:
                return '34B'
            elif size_gb > 15:
                return '16B'
            elif size_gb > 10:
                return '13B'
            elif size_gb > 6:
                return '8B'
            elif size_gb > 4:
                return '7B'
            else:
                return 'Small'
        except:
            return 'Unknown'
    
    def _is_importable(self, model_files: List[Dict]) -> bool:
        """Check if model can be imported to Ollama"""
        # Ollama works best with GGUF files
        gguf_files = [f for f in model_files if f['format'] == 'GGUF']
        return len(gguf_files) > 0
    
    async def import_gguf_model(self, model_info: Dict[str, Any], 
                              custom_name: str = None) -> Dict[str, Any]:
        """Import a GGUF model into Ollama"""
        try:
            # Find the best GGUF file (prefer Q4_K_M or Q5_K_M for balance)
            gguf_files = [f for f in model_info['model_files'] if f['format'] == 'GGUF']
            
            if not gguf_files:
                return {
                    'success': False,
                    'error': 'No GGUF files found in model'
                }
            
            # Select best quantization
            best_file = self._select_best_gguf(gguf_files)
            model_path = Path(best_file['path'])
            
            # Create Ollama modelfile
            ollama_name = custom_name or self._generate_ollama_name(model_info['name'])
            modelfile_content = self._create_modelfile(model_path, model_info)
            
            # Create temporary modelfile
            temp_modelfile = Path('/tmp') / f'{ollama_name}.modelfile'
            with open(temp_modelfile, 'w') as f:
                f.write(modelfile_content)
            
            # Import to Ollama using subprocess
            result = await self._run_ollama_create(ollama_name, temp_modelfile)
            
            # Cleanup
            temp_modelfile.unlink(missing_ok=True)
            
            if result['success']:
                return {
                    'success': True,
                    'ollama_name': ollama_name,
                    'original_name': model_info['name'],
                    'file_used': best_file['name'],
                    'message': f'Model imported as {ollama_name}'
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"GGUF import failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _select_best_gguf(self, gguf_files: List[Dict]) -> Dict[str, Any]:
        """Select the best GGUF file based on quantization quality"""
        # Priority order: Q5_K_M > Q4_K_M > Q6_K > Q8_0 > others
        priority_order = ['Q5_K_M', 'Q4_K_M', 'Q6_K', 'Q8_0', 'Q4_0', 'Q3_K_M', 'Q2_K']
        
        for priority in priority_order:
            for file_info in gguf_files:
                if priority in file_info['name']:
                    return file_info
        
        # If no priority match, return the first one
        return gguf_files[0]
    
    def _generate_ollama_name(self, original_name: str) -> str:
        """Generate a clean Ollama-compatible name"""
        # Convert to lowercase and replace problematic characters
        clean_name = original_name.lower()
        clean_name = clean_name.replace('_', '-')
        clean_name = clean_name.replace(' ', '-')
        
        # Remove common prefixes/suffixes
        clean_name = clean_name.replace('meta-', '')
        clean_name = clean_name.replace('-instruct', '')
        clean_name = clean_name.replace('-abliterated', '-uncensored')
        clean_name = clean_name.replace('-gguf', '')
        
        return clean_name
    
    def _create_modelfile(self, model_path: Path, model_info: Dict[str, Any]) -> str:
        """Create Ollama modelfile content"""
        modelfile = f"""FROM {model_path}

# Model: {model_info['name']}
# Type: {model_info['type']}
# Size: {model_info['estimated_size']}

TEMPLATE \"\"\"{{{{ if .System }}}}{{{{ .System }}}}

{{{{ end }}}}{{{{ if .Prompt }}}}### Human: {{{{ .Prompt }}}}

### Assistant: {{{{ end }}}}{{{{ .Response }}}}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

SYSTEM \"\"\"You are a helpful AI assistant. You provide accurate, helpful, and detailed responses while being concise and clear.\"\"\"
"""
        return modelfile
    
    async def _run_ollama_create(self, model_name: str, modelfile_path: Path) -> Dict[str, Any]:
        """Run ollama create command"""
        try:
            cmd = ['ollama', 'create', model_name, '-f', str(modelfile_path)]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    'success': True,
                    'message': stdout.decode().strip()
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode().strip()
                }
                
        except Exception as e:
            logger.error(f"Ollama create command failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def batch_import_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Import multiple models in batch"""
        try:
            # Discover all models first
            discovery_result = await self.discover_models()
            if not discovery_result['success']:
                return discovery_result
            
            available_models = {m['name']: m for m in discovery_result['models']}
            results = []
            
            for model_name in model_names:
                if model_name not in available_models:
                    results.append({
                        'model_name': model_name,
                        'success': False,
                        'error': 'Model not found'
                    })
                    continue
                
                model_info = available_models[model_name]
                if not model_info['importable']:
                    results.append({
                        'model_name': model_name,
                        'success': False,
                        'error': 'Model not importable (no GGUF files)'
                    })
                    continue
                
                # Import the model
                import_result = await self.import_gguf_model(model_info)
                results.append({
                    'model_name': model_name,
                    **import_result
                })
            
            successful_imports = len([r for r in results if r.get('success', False)])
            
            return {
                'success': True,
                'total_requested': len(model_names),
                'successful_imports': successful_imports,
                'failed_imports': len(model_names) - successful_imports,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Batch import failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_import_recommendations(self) -> Dict[str, Any]:
        """Get recommended models for import"""
        try:
            discovery_result = await self.discover_models()
            if not discovery_result['success']:
                return discovery_result
            
            models = discovery_result['models']
            importable_models = [m for m in models if m['importable']]
            
            # Categorize recommendations
            recommendations = {
                'general_purpose': [],
                'code_generation': [],
                'specialized': [],
                'large_models': [],
                'efficient_models': []
            }
            
            for model in importable_models:
                model_type = model['type'].lower()
                size = model['estimated_size']
                
                # Categorize by type
                if 'code' in model_type:
                    recommendations['code_generation'].append(model)
                elif model_type in ['bert', 'specialized']:
                    recommendations['specialized'].append(model)
                elif 'llama' in model_type or 'general' in model_type:
                    recommendations['general_purpose'].append(model)
                
                # Categorize by size
                if size in ['70B+', '34B', '30B']:
                    recommendations['large_models'].append(model)
                elif size in ['8B', '7B', '3B']:
                    recommendations['efficient_models'].append(model)
            
            # Remove duplicates and limit recommendations
            for category in recommendations:
                recommendations[category] = recommendations[category][:5]  # Limit to 5 per category
            
            return {
                'success': True,
                'total_importable': len(importable_models),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return {
                'success': False,
                'error': str(e)
            }