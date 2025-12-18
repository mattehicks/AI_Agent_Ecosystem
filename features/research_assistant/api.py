#!/usr/bin/env python3
"""
Research Assistant API
FastAPI endpoints for research workflows and document processing
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse

from .models import (
    WorkflowType, BackendConfiguration, ResearchRequest, 
    ResearchResult, DocumentInfo
)
from .workflows import workflow_manager
from status_tracker import status_tracker

router = APIRouter(prefix="/research-assistant", tags=["research-assistant"])

@router.get("/health")
async def health_check():
    """Health check for research assistant service"""
    return {
        "status": "healthy",
        "service": "research-assistant",
        "workflows_available": len(workflow_manager.get_workflows()),
        "backend_configs_available": len(workflow_manager.get_backend_configs())
    }

@router.get("/workflows", response_model=Dict[str, WorkflowType])
async def get_workflows():
    """Get all available research workflows"""
    return workflow_manager.get_workflows()

@router.get("/workflows/category/{category}")
async def get_workflows_by_category(category: str):
    """Get workflows filtered by category"""
    workflows = workflow_manager.get_workflows_by_category(category)
    if not workflows:
        raise HTTPException(status_code=404, detail=f"No workflows found for category: {category}")
    return workflows

@router.get("/backend-configs", response_model=Dict[str, BackendConfiguration])
async def get_backend_configs():
    """Get all available backend configurations"""
    return workflow_manager.get_backend_configs()

@router.get("/backend-configs/workflow/{workflow_id}")
async def get_compatible_configs(workflow_id: str):
    """Get backend configs compatible with a specific workflow"""
    configs = workflow_manager.get_compatible_configs(workflow_id)
    if not configs:
        raise HTTPException(status_code=404, detail=f"No compatible configs found for workflow: {workflow_id}")
    return configs

@router.post("/execute", response_model=ResearchResult)
async def execute_workflow(request: ResearchRequest):
    """Execute a research workflow"""
    try:
        # Log workflow start
        status_tracker.log_research_start(
            workflow_type=request.workflow_type,
            backend_config=request.backend_config,
            document_count=len(request.input_documents) + len(request.input_folders)
        )
        
        # Execute workflow
        result = await workflow_manager.execute_workflow(request)
        
        # Log completion or error
        if result.status == "completed":
            status_tracker.log_research_complete(
                workflow_type=request.workflow_type,
                output_files=len(result.output_files)
            )
        else:
            status_tracker.log_error(
                "research_workflow_failed",
                f"Workflow {request.workflow_type} failed: {result.error_message}"
            )
        
        return result
        
    except Exception as e:
        status_tracker.log_error("research_execution_error", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}")
async def get_session_result(session_id: str):
    """Get results for a specific research session"""
    # This would typically query a database
    # For now, return a placeholder
    return {
        "session_id": session_id,
        "status": "completed",
        "message": "Session result retrieval not yet implemented"
    }

@router.get("/browse-documents")
async def browse_documents(path: str = Query("/mnt/llm", description="Path to browse")):
    """Browse available documents and folders"""
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Path not found")
        
        items = []
        
        # Get directory contents
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            stat = os.stat(item_path)
            
            if os.path.isdir(item_path):
                items.append({
                    "name": item,
                    "path": item_path,
                    "type": "folder",
                    "size": 0,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "item_count": len(os.listdir(item_path)) if os.access(item_path, os.R_OK) else 0
                })
            else:
                file_ext = Path(item).suffix.lower()
                # Filter for document types
                if file_ext in ['.txt', '.md', '.pdf', '.docx', '.doc', '.rtf', '.csv', '.json', '.xml']:
                    items.append({
                        "name": item,
                        "path": item_path,
                        "type": "file",
                        "size": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "file_type": file_ext
                    })
        
        # Sort: folders first, then files
        items.sort(key=lambda x: (x["type"] != "folder", x["name"].lower()))
        
        return {
            "current_path": path,
            "parent_path": str(Path(path).parent) if path != "/" else None,
            "items": items,
            "total_items": len(items)
        }
        
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/document-preview")
async def preview_document(path: str = Query(..., description="Document path to preview")):
    """Get a preview of a document's content"""
    try:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not os.path.isfile(path):
            raise HTTPException(status_code=400, detail="Path is not a file")
        
        file_ext = Path(path).suffix.lower()
        preview_text = ""
        
        # Read file based on type
        if file_ext in ['.txt', '.md']:
            with open(path, 'r', encoding='utf-8') as f:
                preview_text = f.read(2000)  # First 2000 characters
        elif file_ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                preview_text = json.dumps(data, indent=2)[:2000]
        else:
            preview_text = f"Preview not available for {file_ext} files"
        
        stat = os.stat(path)
        
        return {
            "path": path,
            "name": Path(path).name,
            "size": stat.st_size,
            "type": file_ext,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "preview": preview_text,
            "is_truncated": len(preview_text) >= 2000
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/outputs/{session_id}")
async def list_session_outputs(session_id: str):
    """List output files for a research session"""
    output_dir = Path(f"/mnt/llm/AI_Agent_Ecosystem/outputs/research/{session_id}")
    
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Session outputs not found")
    
    files = []
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "name": file_path.name,
                "path": str(file_path),
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "download_url": f"/research-assistant/download/{session_id}/{file_path.name}"
            })
    
    return {
        "session_id": session_id,
        "output_files": files,
        "total_files": len(files)
    }

@router.get("/download/{session_id}/{filename}")
async def download_output_file(session_id: str, filename: str):
    """Download a specific output file"""
    file_path = Path(f"/mnt/llm/AI_Agent_Ecosystem/outputs/research/{session_id}/{filename}")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

@router.get("/categories")
async def get_workflow_categories():
    """Get available workflow categories with counts"""
    workflows = workflow_manager.get_workflows()
    categories = {}
    
    for workflow in workflows.values():
        category = workflow.category
        if category not in categories:
            categories[category] = {
                "name": category.replace("_", " ").title(),
                "count": 0,
                "workflows": []
            }
        categories[category]["count"] += 1
        categories[category]["workflows"].append({
            "id": workflow.id,
            "name": workflow.name,
            "icon": workflow.icon
        })
    
    return categories

@router.post("/validate-config")
async def validate_workflow_config(request: ResearchRequest):
    """Validate a workflow configuration before execution"""
    try:
        # Check if workflow exists
        workflows = workflow_manager.get_workflows()
        if request.workflow_type not in workflows:
            return {"valid": False, "error": f"Unknown workflow type: {request.workflow_type}"}
        
        # Check if backend config exists
        configs = workflow_manager.get_backend_configs()
        if request.backend_config not in configs:
            return {"valid": False, "error": f"Unknown backend config: {request.backend_config}"}
        
        # Check compatibility
        compatible_configs = workflow_manager.get_compatible_configs(request.workflow_type)
        if request.backend_config not in compatible_configs:
            return {"valid": False, "error": f"Backend config '{request.backend_config}' is not compatible with workflow '{request.workflow_type}'"}
        
        # Validate input files exist
        missing_files = []
        for doc_path in request.input_documents:
            if not os.path.exists(doc_path):
                missing_files.append(doc_path)
        
        for folder_path in request.input_folders:
            if not os.path.exists(folder_path):
                missing_files.append(folder_path)
        
        if missing_files:
            return {"valid": False, "error": f"Missing files/folders: {missing_files}"}
        
        return {
            "valid": True,
            "workflow": workflows[request.workflow_type],
            "backend_config": configs[request.backend_config],
            "estimated_processing_time": "2-5 minutes"  # Placeholder
        }
        
    except Exception as e:
        return {"valid": False, "error": str(e)}