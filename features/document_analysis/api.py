#!/usr/bin/env python3
"""
Document Analysis API Endpoints
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text

from .config import config
from .models import (
    DocumentUpload, DocumentInfo, AnalysisRequest, AnalysisResult,
    TaskStatus, AnalysisType, DocumentAnalysis, TaskUpdate
)
from .file_handler import file_handler
from .document_parser import document_parser
from .task_queue import task_queue, TaskPriority
from .database import get_db, DocumentRecord, db_manager

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/document-analysis", tags=["Document Analysis"])

@router.post("/upload", response_model=List[DocumentInfo])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    analysis_types: List[AnalysisType] = [AnalysisType.SUMMARY],
    db: Session = Depends(get_db)
):
    """Upload documents for analysis"""
    
    try:
        # Upload files
        uploaded_docs = await file_handler.upload_multiple_files(files)
        
        # Store document records in database
        for doc_info in uploaded_docs:
            doc_record = DocumentRecord(
                id=doc_info.id,
                filename=doc_info.filename,
                original_filename=doc_info.filename,
                content_type=doc_info.content_type,
                file_size=doc_info.size,
                file_path=str(file_handler.get_file_path(doc_info.id)),
                status=TaskStatus.PENDING.value
            )
            db.add(doc_record)
        
        db.commit()
        
        # Queue processing tasks
        for doc_info in uploaded_docs:
            background_tasks.add_task(
                process_document_async,
                doc_info.id,
                analysis_types
            )
        
        logger.info(f"Uploaded {len(uploaded_docs)} documents for analysis")
        return uploaded_docs
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    status: TaskStatus = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """List uploaded documents"""
    
    try:
        query = db.query(DocumentRecord)
        
        if status:
            query = query.filter(DocumentRecord.status == status.value)
        
        documents = query.offset(offset).limit(limit).all()
        
        doc_infos = []
        for doc in documents:
            doc_info = DocumentInfo(
                id=doc.id,
                filename=doc.original_filename,
                content_type=doc.content_type,
                size=doc.file_size,
                upload_time=doc.created_at,
                status=TaskStatus(doc.status),
                progress=doc.progress,
                error_message=doc.error_message
            )
            doc_infos.append(doc_info)
        
        return doc_infos
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document_info(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get document information"""
    
    try:
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        
        if not doc_record:
            raise HTTPException(status_code=404, detail="Document not found")
        
        doc_info = DocumentInfo(
            id=doc_record.id,
            filename=doc_record.original_filename,
            content_type=doc_record.content_type,
            size=doc_record.file_size,
            upload_time=doc_record.created_at,
            status=TaskStatus(doc_record.status),
            progress=doc_record.progress,
            error_message=doc_record.error_message
        )
        
        return doc_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document info: {str(e)}")

@router.get("/documents/{document_id}/analysis", response_model=DocumentAnalysis)
async def get_document_analysis(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Get document analysis results"""
    
    try:
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        
        if not doc_record:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if doc_record.status != TaskStatus.COMPLETED.value:
            raise HTTPException(status_code=400, detail="Document analysis not completed")
        
        if not doc_record.analysis_results:
            raise HTTPException(status_code=404, detail="Analysis results not found")
        
        # Parse analysis results
        results = doc_record.analysis_results
        
        analysis = DocumentAnalysis(
            document_id=document_id,
            filename=doc_record.original_filename,
            summary=results.get('summary'),
            key_points=results.get('key_points', []),
            entities=results.get('entities', []),
            sentiment=results.get('sentiment'),
            questions=results.get('questions', []),
            actions=results.get('actions', []),
            metadata=results.get('metadata', {}),
            created_at=doc_record.created_at,
            processing_time=doc_record.processing_time
        )
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

@router.post("/documents/{document_id}/reanalyze")
async def reanalyze_document(
    document_id: str,
    analysis_types: List[AnalysisType],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Reanalyze document with different analysis types"""
    
    try:
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        
        if not doc_record:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Reset document status
        doc_record.status = TaskStatus.PENDING.value
        doc_record.progress = 0.0
        doc_record.error_message = None
        doc_record.analysis_results = None
        db.commit()
        
        # Queue new analysis task
        background_tasks.add_task(
            process_document_async,
            document_id,
            analysis_types
        )
        
        return {"message": "Document queued for reanalysis", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reanalyze document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reanalyze document: {str(e)}")

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """Delete document and its analysis"""
    
    try:
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        
        if not doc_record:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file
        file_handler.delete_file(document_id)
        
        # Delete database record
        db.delete(doc_record)
        db.commit()
        
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.get("/queue/stats")
async def get_queue_stats():
    """Get task queue statistics"""
    
    try:
        stats = await task_queue.get_queue_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")

@router.post("/tasks/{task_id}/retry")
async def retry_task(task_id: str):
    """Retry a failed task"""
    
    try:
        success = await task_queue.retry_failed_task(task_id)
        
        if success:
            return {"message": "Task queued for retry", "task_id": task_id}
        else:
            raise HTTPException(status_code=400, detail="Task cannot be retried")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retry task: {str(e)}")

# Background task functions

async def process_document_async(document_id: str, analysis_types: List[AnalysisType]):
    """Process document asynchronously"""
    
    try:
        logger.info(f"Starting document processing: {document_id}")
        
        # Update status to processing
        with file_handler.db_manager.get_session() as db:
            doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
            if doc_record:
                doc_record.status = TaskStatus.PROCESSING.value
                doc_record.processing_started_at = datetime.now()
                db.commit()
        
        # Parse document
        file_path = file_handler.get_file_path(document_id)
        parsed_data = document_parser.parse_document(file_path)
        
        # Update progress
        await update_document_progress(document_id, 30.0, "Document parsed")
        
        # Store extracted text
        with file_handler.db_manager.get_session() as db:
            doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
            if doc_record:
                doc_record.text_content = parsed_data['text']
                doc_record.metadata = parsed_data['metadata']
                db.commit()
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            document_id=document_id,
            text_content=parsed_data['text'],
            analysis_types=analysis_types,
            metadata=parsed_data['metadata']
        )
        
        # Queue AI analysis task
        task_id = await task_queue.enqueue_task(
            task_type="ai_analysis",
            document_id=document_id,
            parameters={
                "analysis_request": analysis_request.dict(),
                "chunks": parsed_data['chunks']
            },
            priority=TaskPriority.NORMAL
        )
        
        logger.info(f"Queued AI analysis task {task_id} for document {document_id}")
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}")
        
        # Update status to failed
        with file_handler.db_manager.get_session() as db:
            doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
            if doc_record:
                doc_record.status = TaskStatus.FAILED.value
                doc_record.error_message = str(e)
                db.commit()

async def update_document_progress(document_id: str, progress: float, message: str = None):
    """Update document processing progress"""
    
    with file_handler.db_manager.get_session() as db:
        doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
        if doc_record:
            doc_record.progress = progress
            db.commit()
    
    logger.info(f"Document {document_id} progress: {progress:.1f}% - {message or ''}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for document analysis service"""
    
    try:
        # Check file handler
        upload_dir_exists = config.upload_path.exists()
        
        # Check database
        db_healthy = True
        try:
            with db_manager.get_session() as db:
                # Test database connection with a simple query
                result = db.execute(text("SELECT 1")).fetchone()
                db_healthy = result is not None
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            db_healthy = False
        
        # Check task queue
        queue_stats = await task_queue.get_queue_stats()
        
        status = "healthy" if upload_dir_exists and db_healthy else "unhealthy"
        
        return {
            "status": status,
            "upload_directory": upload_dir_exists,
            "database": db_healthy,
            "task_queue": queue_stats.get('backend', 'unknown'),
            "pending_tasks": queue_stats.get('pending_tasks', 0)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }