#!/usr/bin/env python3
"""
RAG Agent - Integrates VantaBlack RAG system with AI Agent Ecosystem
Provides semantic search over legal/technical documents
"""

import asyncio
import logging
import subprocess
from typing import Dict, List, Optional, Any
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    """Agent for RAG (Retrieval-Augmented Generation) document queries"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ssh_host = config.get('ssh_host', 'lightspeed@vantablack')
        self.ssh_key = config.get('ssh_key', 'C:\\Users\\matte\\.ssh\\id_ed25519')
        self.query_script = config.get('query_script', '/mnt/llm/rag_query.py')
        self.ingest_script = config.get('ingest_script', '/mnt/llm/rag_ingest.py')
        
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG operation"""
        task_type = task.get('type', 'query')
        
        if task_type == 'query':
            return await self._query_documents(task)
        elif task_type == 'ingest':
            return await self._ingest_documents(task)
        elif task_type == 'status':
            return await self._get_status()
        else:
            return {
                'success': False,
                'error': f'Unknown task type: {task_type}'
            }
    
    async def _query_documents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Query documents using RAG"""
        question = task.get('question', '')
        n_results = task.get('n_results', 5)
        model = task.get('model', 'qwen2.5:32b')
        
        if not question:
            return {'success': False, 'error': 'No question provided'}
        
        try:
            # Execute RAG query on VantaBlack via SSH
            cmd = [
                'ssh', '-i', self.ssh_key, self.ssh_host,
                f'python3 {self.query_script} "{question}" {n_results} {model}'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    'success': True,
                    'answer': stdout.decode('utf-8'),
                    'question': question,
                    'model': model
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode('utf-8')
                }
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _ingest_documents(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest PDF documents into RAG database"""
        directory = task.get('directory', '')
        
        if not directory:
            return {'success': False, 'error': 'No directory provided'}
        
        try:
            cmd = [
                'ssh', '-i', self.ssh_key, self.ssh_host,
                f'python3 {self.ingest_script} "{directory}"'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    'success': True,
                    'message': stdout.decode('utf-8'),
                    'directory': directory
                }
            else:
                return {
                    'success': False,
                    'error': stderr.decode('utf-8')
                }
        except Exception as e:
            logger.error(f"RAG ingestion failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _get_status(self) -> Dict[str, Any]:
        """Get RAG database status"""
        try:
            status_cmd = """python3 -c "
import chromadb
client = chromadb.PersistentClient(path='/mnt/llm/chromadb')
try:
    collection = client.get_collection('legal_docs')
    print(collection.count())
except:
    print('0')
" """
            
            cmd = ['ssh', '-i', self.ssh_key, self.ssh_host, status_cmd]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                doc_count = int(stdout.decode('utf-8').strip())
                return {
                    'success': True,
                    'documents': doc_count,
                    'collection': 'legal_docs'
                }
            else:
                return {'success': False, 'error': 'Database not initialized'}
        except Exception as e:
            logger.error(f"RAG status check failed: {e}")
            return {'success': False, 'error': str(e)}
