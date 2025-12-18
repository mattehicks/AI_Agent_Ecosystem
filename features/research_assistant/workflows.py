#!/usr/bin/env python3
"""
Research Assistant Workflows
Predefined workflow configurations and execution logic
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from .models import WorkflowType, BackendConfiguration, ResearchRequest, ResearchResult
from llm_backends import InferenceRequest, backend_manager

class WorkflowManager:
    """Manages research workflows and configurations"""
    
    def __init__(self):
        self.workflows = self._initialize_workflows()
        self.backend_configs = self._initialize_backend_configs()
        
    def _initialize_workflows(self) -> Dict[str, WorkflowType]:
        """Initialize predefined workflow types"""
        return {
            "document_summary": WorkflowType(
                id="document_summary",
                name="Document Summary",
                description="Summarize individual documents or collections",
                icon="fas fa-file-text",
                category="document_processing",
                required_models=["text_generation"],
                parameters={
                    "summary_length": {"type": "select", "options": ["brief", "detailed", "comprehensive"], "default": "detailed"},
                    "include_key_points": {"type": "boolean", "default": True},
                    "output_format": {"type": "select", "options": ["markdown", "html", "pdf"], "default": "markdown"}
                }
            ),
            
            "comparative_analysis": WorkflowType(
                id="comparative_analysis",
                name="Comparative Analysis",
                description="Compare and analyze multiple documents or datasets",
                icon="fas fa-balance-scale",
                category="technical_analysis",
                required_models=["text_generation", "analysis"],
                parameters={
                    "comparison_aspects": {"type": "text", "placeholder": "Enter aspects to compare (e.g., cost, features, performance)"},
                    "analysis_depth": {"type": "select", "options": ["surface", "detailed", "comprehensive"], "default": "detailed"},
                    "include_recommendations": {"type": "boolean", "default": True}
                }
            ),
            
            "content_generation": WorkflowType(
                id="content_generation",
                name="Content Generation",
                description="Generate new content based on source documents",
                icon="fas fa-pen-fancy",
                category="text_generation",
                required_models=["text_generation"],
                parameters={
                    "content_type": {"type": "select", "options": ["article", "report", "presentation", "email", "proposal"], "default": "article"},
                    "tone": {"type": "select", "options": ["professional", "casual", "academic", "persuasive"], "default": "professional"},
                    "length": {"type": "select", "options": ["short", "medium", "long"], "default": "medium"},
                    "target_audience": {"type": "text", "placeholder": "Describe target audience"}
                }
            ),
            
            "technical_synopsis": WorkflowType(
                id="technical_synopsis",
                name="Technical Synopsis",
                description="Create technical summaries of complex data or research",
                icon="fas fa-chart-line",
                category="technical_analysis",
                required_models=["analysis", "text_generation"],
                parameters={
                    "focus_areas": {"type": "text", "placeholder": "Key technical areas to focus on"},
                    "include_charts": {"type": "boolean", "default": False},
                    "technical_level": {"type": "select", "options": ["beginner", "intermediate", "expert"], "default": "intermediate"}
                }
            ),
            
            "research_compilation": WorkflowType(
                id="research_compilation",
                name="Research Compilation",
                description="Compile and synthesize research from multiple sources",
                icon="fas fa-books",
                category="document_processing",
                required_models=["text_generation", "analysis"],
                parameters={
                    "research_questions": {"type": "text", "placeholder": "Enter research questions to address"},
                    "citation_style": {"type": "select", "options": ["APA", "MLA", "Chicago", "IEEE"], "default": "APA"},
                    "include_bibliography": {"type": "boolean", "default": True}
                }
            ),
            
            "data_extraction": WorkflowType(
                id="data_extraction",
                name="Data Extraction",
                description="Extract structured data from unstructured documents",
                icon="fas fa-filter",
                category="technical_analysis",
                required_models=["analysis"],
                parameters={
                    "extraction_fields": {"type": "text", "placeholder": "Fields to extract (e.g., names, dates, amounts)"},
                    "output_format": {"type": "select", "options": ["json", "csv", "excel"], "default": "json"},
                    "confidence_threshold": {"type": "range", "min": 0.1, "max": 1.0, "default": 0.8}
                }
            )
        }
    
    def _initialize_backend_configs(self) -> Dict[str, BackendConfiguration]:
        """Initialize backend model configurations"""
        return {
            "lightweight": BackendConfiguration(
                id="lightweight",
                name="Lightweight Processing",
                description="Fast processing with smaller models (4-8GB VRAM)",
                models=["llama-3.1-8b", "mistral-7b"],
                parameters={
                    "max_tokens": {"type": "range", "min": 100, "max": 2048, "default": 1024},
                    "temperature": {"type": "range", "min": 0.1, "max": 2.0, "default": 0.7},
                    "top_p": {"type": "range", "min": 0.1, "max": 1.0, "default": 0.9}
                },
                workflow_types=["document_summary", "content_generation", "data_extraction"],
                gpu_requirements={"vram_gb": 8, "gpus": 1}
            ),
            
            "balanced": BackendConfiguration(
                id="balanced",
                name="Balanced Performance",
                description="Good balance of speed and quality (8-16GB VRAM)",
                models=["llama-3.1-13b", "codellama-13b", "mistral-nemo-12b"],
                parameters={
                    "max_tokens": {"type": "range", "min": 100, "max": 4096, "default": 2048},
                    "temperature": {"type": "range", "min": 0.1, "max": 2.0, "default": 0.7},
                    "top_p": {"type": "range", "min": 0.1, "max": 1.0, "default": 0.9},
                    "repeat_penalty": {"type": "range", "min": 1.0, "max": 1.3, "default": 1.1}
                },
                workflow_types=["document_summary", "comparative_analysis", "content_generation", "technical_synopsis"],
                gpu_requirements={"vram_gb": 16, "gpus": 1}
            ),
            
            "high_performance": BackendConfiguration(
                id="high_performance",
                name="High Performance",
                description="Maximum quality with large models (16-24GB VRAM)",
                models=["llama-3.1-70b", "codellama-34b", "mixtral-8x7b"],
                parameters={
                    "max_tokens": {"type": "range", "min": 100, "max": 8192, "default": 4096},
                    "temperature": {"type": "range", "min": 0.1, "max": 2.0, "default": 0.7},
                    "top_p": {"type": "range", "min": 0.1, "max": 1.0, "default": 0.9},
                    "top_k": {"type": "range", "min": 1, "max": 100, "default": 40},
                    "repeat_penalty": {"type": "range", "min": 1.0, "max": 1.3, "default": 1.1}
                },
                workflow_types=["comparative_analysis", "technical_synopsis", "research_compilation", "content_generation"],
                gpu_requirements={"vram_gb": 24, "gpus": 2}
            ),
            
            "specialized": BackendConfiguration(
                id="specialized",
                name="Specialized Models",
                description="Task-specific models for optimal results",
                models=["code-generation", "document-analysis", "research-assistant"],
                parameters={
                    "max_tokens": {"type": "range", "min": 100, "max": 4096, "default": 2048},
                    "temperature": {"type": "range", "min": 0.1, "max": 1.5, "default": 0.5},
                    "top_p": {"type": "range", "min": 0.1, "max": 1.0, "default": 0.95},
                    "system_prompt": {"type": "text", "placeholder": "Custom system prompt"}
                },
                workflow_types=["data_extraction", "research_compilation", "technical_synopsis"],
                gpu_requirements={"vram_gb": 12, "gpus": 1}
            )
        }
    
    def get_workflows(self) -> Dict[str, WorkflowType]:
        """Get all available workflows"""
        return self.workflows
    
    def get_backend_configs(self) -> Dict[str, BackendConfiguration]:
        """Get all backend configurations"""
        return self.backend_configs
    
    def get_workflows_by_category(self, category: str) -> Dict[str, WorkflowType]:
        """Get workflows filtered by category"""
        return {k: v for k, v in self.workflows.items() if v.category == category}
    
    def get_compatible_configs(self, workflow_id: str) -> Dict[str, BackendConfiguration]:
        """Get backend configs compatible with a workflow"""
        if workflow_id not in self.workflows:
            return {}
        
        return {k: v for k, v in self.backend_configs.items() 
                if workflow_id in v.workflow_types}
    
    async def execute_workflow(self, request: ResearchRequest) -> ResearchResult:
        """Execute a research workflow"""
        session_id = str(uuid.uuid4())
        
        # Validate request
        if request.workflow_type not in self.workflows:
            raise ValueError(f"Unknown workflow type: {request.workflow_type}")
        
        if request.backend_config not in self.backend_configs:
            raise ValueError(f"Unknown backend config: {request.backend_config}")
        
        # Create result object
        result = ResearchResult(
            id=session_id,
            workflow_type=request.workflow_type,
            status="processing",
            input_summary=await self._analyze_inputs(request),
            created_at=datetime.now()
        )
        
        try:
            # Execute workflow based on type
            workflow = self.workflows[request.workflow_type]
            
            if workflow.id == "document_summary":
                result.results = await self._execute_document_summary(request)
            elif workflow.id == "comparative_analysis":
                result.results = await self._execute_comparative_analysis(request)
            elif workflow.id == "content_generation":
                result.results = await self._execute_content_generation(request)
            elif workflow.id == "technical_synopsis":
                result.results = await self._execute_technical_synopsis(request)
            elif workflow.id == "research_compilation":
                result.results = await self._execute_research_compilation(request)
            elif workflow.id == "data_extraction":
                result.results = await self._execute_data_extraction(request)
            else:
                raise ValueError(f"Workflow execution not implemented: {workflow.id}")
            
            # Save results if requested
            if request.save_results:
                result.output_files = await self._save_results(session_id, result.results, request.output_format)
            
            result.status = "completed"
            result.completed_at = datetime.now()
            
        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.completed_at = datetime.now()
        
        return result
    
    async def _analyze_inputs(self, request: ResearchRequest) -> Dict[str, Any]:
        """Analyze input documents and folders"""
        summary = {
            "document_count": len(request.input_documents),
            "folder_count": len(request.input_folders),
            "total_size": 0,
            "file_types": {},
            "documents": []
        }
        
        # Analyze documents
        for doc_path in request.input_documents:
            if os.path.exists(doc_path):
                stat = os.stat(doc_path)
                file_type = Path(doc_path).suffix.lower()
                
                summary["total_size"] += stat.st_size
                summary["file_types"][file_type] = summary["file_types"].get(file_type, 0) + 1
                summary["documents"].append({
                    "path": doc_path,
                    "name": Path(doc_path).name,
                    "size": stat.st_size,
                    "type": file_type,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
        
        # Analyze folders
        for folder_path in request.input_folders:
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        stat = os.stat(file_path)
                        file_type = Path(file).suffix.lower()
                        
                        summary["total_size"] += stat.st_size
                        summary["file_types"][file_type] = summary["file_types"].get(file_type, 0) + 1
                        summary["documents"].append({
                            "path": file_path,
                            "name": file,
                            "size": stat.st_size,
                            "type": file_type,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
        
        return summary
    
    async def _execute_document_summary(self, request: ResearchRequest) -> Dict[str, Any]:
        """Execute document summary workflow using real LLM backend"""
        try:
            # Read document contents
            document_contents = []
            for doc_path in request.document_paths:
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        content = f.read()[:5000]  # Limit to first 5000 chars
                        document_contents.append(f"Document: {Path(doc_path).name}\n{content}")
                except Exception as e:
                    logger.warning(f"Failed to read document {doc_path}: {e}")
            
            if not document_contents:
                return {"error": "No documents could be read", "summary": "Failed to process documents"}
            
            # Prepare prompt for LLM
            combined_content = "\n\n---\n\n".join(document_contents)
            prompt = f"""Please analyze the following documents and provide a comprehensive summary:

{combined_content}

Please provide:
1. Executive Summary
2. Key Findings (as bullet points)
3. Main Themes
4. Recommendations
5. Conclusion

Format your response as a professional report."""
            
            # Create inference request
            inference_request = InferenceRequest(
                prompt=prompt,
                model=request.model_name or "llama3.1:8b",
                system_prompt="You are a professional research analyst. Provide clear, structured summaries of documents.",
                max_tokens=1000,
                temperature=0.3
            )
            
            # Generate using backend manager
            response = await backend_manager.generate(inference_request)
            
            # Extract key points from response (simple extraction)
            lines = response.text.split('\n')
            key_points = [line.strip('- ') for line in lines if line.strip().startswith('-')]
            
            return {
                "summary": response.text,
                "key_points": key_points[:5] if key_points else ["Summary generated successfully"],
                "word_count": len(response.text.split()),
                "processing_time": f"{response.processing_time:.2f} seconds",
                "backend_used": response.backend,
                "model_used": response.model,
                "tokens_used": response.tokens_used
            }
            
        except Exception as e:
            logger.error(f"Document summary workflow failed: {e}")
            return {
                "error": str(e),
                "summary": f"Workflow failed: {str(e)}",
                "key_points": ["Error occurred during processing"],
                "word_count": 0,
                "processing_time": "0 seconds"
            }
    
    async def _execute_comparative_analysis(self, request: ResearchRequest) -> Dict[str, Any]:
        """Execute comparative analysis workflow"""
        return {
            "comparison": "Comparative analysis would be generated here",
            "similarities": ["Similarity 1", "Similarity 2"],
            "differences": ["Difference 1", "Difference 2"],
            "recommendations": ["Recommendation 1", "Recommendation 2"]
        }
    
    async def _execute_content_generation(self, request: ResearchRequest) -> Dict[str, Any]:
        """Execute content generation workflow"""
        return {
            "generated_content": "Generated content would appear here",
            "content_type": request.parameters.get("content_type", "article"),
            "word_count": 500,
            "tone": request.parameters.get("tone", "professional")
        }
    
    async def _execute_technical_synopsis(self, request: ResearchRequest) -> Dict[str, Any]:
        """Execute technical synopsis workflow"""
        return {
            "synopsis": "Technical synopsis would be generated here",
            "technical_highlights": ["Highlight 1", "Highlight 2"],
            "complexity_score": 0.7,
            "recommended_actions": ["Action 1", "Action 2"]
        }
    
    async def _execute_research_compilation(self, request: ResearchRequest) -> Dict[str, Any]:
        """Execute research compilation workflow"""
        return {
            "compiled_research": "Research compilation would be generated here",
            "sources_analyzed": 5,
            "key_findings": ["Finding 1", "Finding 2"],
            "bibliography": ["Source 1", "Source 2"]
        }
    
    async def _execute_data_extraction(self, request: ResearchRequest) -> Dict[str, Any]:
        """Execute data extraction workflow"""
        return {
            "extracted_data": {"field1": "value1", "field2": "value2"},
            "extraction_confidence": 0.85,
            "fields_found": 2,
            "total_records": 10
        }
    
    async def _save_results(self, session_id: str, results: Dict[str, Any], output_format: str) -> List[str]:
        """Save workflow results to files"""
        output_dir = Path(f"/mnt/llm/AI_Agent_Ecosystem/outputs/research/{session_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        # Save main results
        if output_format == "json":
            output_file = output_dir / "results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            output_files.append(str(output_file))
        
        elif output_format == "markdown":
            output_file = output_dir / "results.md"
            with open(output_file, 'w') as f:
                f.write(f"# Research Results\n\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                for key, value in results.items():
                    f.write(f"## {key.replace('_', ' ').title()}\n\n")
                    f.write(f"{value}\n\n")
            output_files.append(str(output_file))
        
        return output_files

# Global workflow manager instance
workflow_manager = WorkflowManager()