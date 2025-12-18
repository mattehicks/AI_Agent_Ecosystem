#!/usr/bin/env python3
"""
PrivateGPT Backend Integration for AI Agent Ecosystem
Connects to PrivateGPT API for document analysis and Q&A
"""

import asyncio
import json
import logging
import aiohttp
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

class PrivateGPTBackend:
    """PrivateGPT API client for document processing and Q&A"""
    
    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get('base_url', 'http://localhost:8001')
        self.api_key = config.get('api_key')
        self.timeout = config.get('timeout', 300)
        self.max_retries = config.get('max_retries', 3)
        self.session = None
        
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
        """Check if PrivateGPT service is available"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': 'healthy',
                        'service': 'privateGPT',
                        'version': data.get('version', 'unknown'),
                        'models_loaded': data.get('models_loaded', 0)
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'service': 'privateGPT',
                        'error': f"HTTP {response.status}"
                    }
        except Exception as e:
            logger.error(f"PrivateGPT health check failed: {e}")
            return {
                'status': 'unavailable',
                'service': 'privateGPT',
                'error': str(e)
            }
    
    async def ingest_document(self, file_path: Path, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ingest a document into PrivateGPT"""
        try:
            # Prepare file upload
            with open(file_path, 'rb') as file:
                data = aiohttp.FormData()
                data.add_field('file', file, filename=file_path.name)
                
                if metadata:
                    data.add_field('metadata', json.dumps(metadata))
                
                async with self.session.post(
                    f"{self.base_url}/v1/ingest/file",
                    data=data
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return {
                            'success': True,
                            'document_id': result.get('document_id'),
                            'chunks_processed': result.get('chunks_processed', 0),
                            'processing_time': result.get('processing_time', 0)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}: {error_text}"
                        }
                        
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def ingest_text(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ingest raw text into PrivateGPT"""
        try:
            payload = {
                'text': text,
                'metadata': metadata or {}
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/ingest/text",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'document_id': result.get('document_id'),
                        'chunks_processed': result.get('chunks_processed', 0)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Text ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def query_documents(self, query: str, 
                            context_filter: Dict[str, Any] = None,
                            include_sources: bool = True,
                            max_chunks: int = 4) -> Dict[str, Any]:
        """Query ingested documents"""
        try:
            payload = {
                'query': query,
                'use_context': True,
                'context_filter': context_filter or {},
                'include_sources': include_sources,
                'max_chunks': max_chunks
            }
            
            async with self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'answer': result.get('choices', [{}])[0].get('message', {}).get('content', ''),
                        'sources': result.get('sources', []),
                        'context_used': result.get('context_used', []),
                        'processing_time': result.get('processing_time', 0)
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Document query failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                            use_context: bool = False,
                            system_prompt: str = None) -> Dict[str, Any]:
        """Chat completion with optional context"""
        try:
            payload = {
                'messages': messages,
                'use_context': use_context,
                'stream': False
            }
            
            if system_prompt:
                payload['system_prompt'] = system_prompt
            
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'response': result.get('choices', [{}])[0].get('message', {}).get('content', ''),
                        'usage': result.get('usage', {}),
                        'model': result.get('model', 'unknown')
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
    
    async def list_documents(self) -> Dict[str, Any]:
        """List all ingested documents"""
        try:
            async with self.session.get(f"{self.base_url}/v1/ingest/list") as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'documents': result.get('data', []),
                        'total_count': len(result.get('data', []))
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from PrivateGPT"""
        try:
            async with self.session.delete(f"{self.base_url}/v1/ingest/{document_id}") as response:
                if response.status == 200:
                    return {
                        'success': True,
                        'message': 'Document deleted successfully'
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_embeddings(self, text: str) -> Dict[str, Any]:
        """Get embeddings for text"""
        try:
            payload = {'input': text}
            
            async with self.session.post(
                f"{self.base_url}/v1/embeddings",
                json=payload
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        'success': True,
                        'embeddings': result.get('data', [{}])[0].get('embedding', []),
                        'model': result.get('model', 'unknown')
                    }
                else:
                    error_text = await response.text()
                    return {
                        'success': False,
                        'error': f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            return {
                'success': False,
                'error': str(e)
            }

class PrivateGPTIntegration:
    """High-level integration wrapper for PrivateGPT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = PrivateGPTBackend(config)
        self.document_cache = {}
        
    async def analyze_document(self, file_path: Path, 
                             analysis_type: str = "summary",
                             custom_prompt: str = None) -> Dict[str, Any]:
        """Analyze a document with PrivateGPT"""
        try:
            # Generate cache key
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
            cache_key = f"{file_hash}_{analysis_type}"
            
            # Check cache
            if cache_key in self.document_cache:
                return self.document_cache[cache_key]
            
            async with self.backend:
                # Ingest document
                ingest_result = await self.backend.ingest_document(file_path)
                if not ingest_result['success']:
                    return ingest_result
                
                # Prepare analysis prompt
                if analysis_type == "summary":
                    prompt = f"Please provide a comprehensive summary of the document '{file_path.name}'. Include key points, main topics, and important conclusions."
                elif analysis_type == "key_points":
                    prompt = f"Extract the key points and main ideas from the document '{file_path.name}'. Format as a bullet list."
                elif analysis_type == "questions":
                    prompt = f"Generate 5-10 important questions that this document '{file_path.name}' answers or addresses."
                elif analysis_type == "custom" and custom_prompt:
                    prompt = custom_prompt
                else:
                    prompt = f"Analyze the document '{file_path.name}' and provide insights about its content."
                
                # Query the document
                query_result = await self.backend.query_documents(
                    query=prompt,
                    include_sources=True,
                    max_chunks=6
                )
                
                if query_result['success']:
                    result = {
                        'success': True,
                        'analysis_type': analysis_type,
                        'document_path': str(file_path),
                        'analysis': query_result['answer'],
                        'sources': query_result['sources'],
                        'chunks_processed': ingest_result['chunks_processed']
                    }
                    
                    # Cache result
                    self.document_cache[cache_key] = result
                    return result
                else:
                    return query_result
                    
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def research_query(self, query: str, 
                           document_filter: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform research query across ingested documents"""
        try:
            async with self.backend:
                # Get comprehensive answer
                result = await self.backend.query_documents(
                    query=query,
                    context_filter=document_filter,
                    include_sources=True,
                    max_chunks=8
                )
                
                if result['success']:
                    # Enhance with follow-up questions
                    follow_up_query = f"Based on the previous answer about '{query}', what are 3 important follow-up questions someone might ask?"
                    
                    follow_up_result = await self.backend.chat_completion([
                        {"role": "user", "content": follow_up_query}
                    ])
                    
                    result['follow_up_questions'] = follow_up_result.get('response', '') if follow_up_result['success'] else ''
                
                return result
                
        except Exception as e:
            logger.error(f"Research query failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def batch_ingest_directory(self, directory_path: Path, 
                                   file_extensions: List[str] = None) -> Dict[str, Any]:
        """Ingest all documents in a directory"""
        if file_extensions is None:
            file_extensions = ['.txt', '.pdf', '.docx', '.md', '.json']
        
        results = {
            'success': True,
            'ingested_files': [],
            'failed_files': [],
            'total_chunks': 0
        }
        
        try:
            async with self.backend:
                for file_path in directory_path.rglob('*'):
                    if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                        logger.info(f"Ingesting: {file_path}")
                        
                        ingest_result = await self.backend.ingest_document(
                            file_path,
                            metadata={'source_directory': str(directory_path)}
                        )
                        
                        if ingest_result['success']:
                            results['ingested_files'].append({
                                'file_path': str(file_path),
                                'chunks_processed': ingest_result['chunks_processed']
                            })
                            results['total_chunks'] += ingest_result['chunks_processed']
                        else:
                            results['failed_files'].append({
                                'file_path': str(file_path),
                                'error': ingest_result['error']
                            })
                
                if results['failed_files']:
                    results['success'] = False
                
                return results
                
        except Exception as e:
            logger.error(f"Batch ingestion failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        try:
            async with self.backend:
                health = await self.backend.health_check()
                documents = await self.backend.list_documents()
                
                return {
                    'service_health': health,
                    'document_count': documents.get('total_count', 0) if documents['success'] else 0,
                    'cache_size': len(self.document_cache),
                    'backend_url': self.backend.base_url
                }
                
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {
                'service_health': {'status': 'error', 'error': str(e)},
                'document_count': 0,
                'cache_size': len(self.document_cache),
                'backend_url': self.backend.base_url
            }