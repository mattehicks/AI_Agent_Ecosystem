#!/usr/bin/env python3
"""
File Upload and Management for Document Analysis
"""

import os
import uuid
import magic
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import aiofiles
from fastapi import UploadFile, HTTPException

from .config import config
from .models import DocumentUpload, DocumentInfo, TaskStatus

logger = logging.getLogger(__name__)

class FileHandler:
    """Handles file upload, validation, and storage"""
    
    def __init__(self):
        self.upload_dir = config.upload_path
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize magic for MIME type detection
        try:
            self.mime_detector = magic.Magic(mime=True)
        except Exception as e:
            logger.warning(f"Could not initialize python-magic: {e}")
            self.mime_detector = None
    
    async def upload_file(self, file: UploadFile) -> DocumentInfo:
        """Upload and validate a single file"""
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Validate file
        await self._validate_file(file)
        
        # Generate file path
        file_extension = Path(file.filename).suffix.lower()
        safe_filename = f"{doc_id}{file_extension}"
        file_path = self.upload_dir / safe_filename
        
        # Save file
        file_size = await self._save_file(file, file_path)
        
        # Detect MIME type
        content_type = await self._detect_mime_type(file_path, file.content_type)
        
        # Create document info
        doc_info = DocumentInfo(
            id=doc_id,
            filename=file.filename,
            content_type=content_type,
            size=file_size,
            upload_time=datetime.now(),
            status=TaskStatus.PENDING,
            progress=0.0
        )
        
        logger.info(f"File uploaded successfully: {file.filename} -> {doc_id}")
        return doc_info
    
    async def upload_multiple_files(self, files: List[UploadFile]) -> List[DocumentInfo]:
        """Upload multiple files with batch validation"""
        
        # Validate batch size
        if len(files) > config.max_files_per_batch:
            raise HTTPException(
                status_code=400,
                detail=f"Too many files. Maximum {config.max_files_per_batch} files per batch."
            )
        
        # Upload files
        uploaded_docs = []
        for file in files:
            try:
                doc_info = await self.upload_file(file)
                uploaded_docs.append(doc_info)
            except Exception as e:
                logger.error(f"Failed to upload file {file.filename}: {e}")
                # Clean up any successfully uploaded files if one fails
                await self._cleanup_uploaded_files(uploaded_docs)
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to upload file {file.filename}: {str(e)}"
                )
        
        return uploaded_docs
    
    async def _validate_file(self, file: UploadFile) -> None:
        """Validate uploaded file"""
        
        # Check file size
        if hasattr(file, 'size') and file.size > config.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {config.max_file_size / (1024*1024):.1f}MB"
            )
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in config.supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. "
                       f"Supported types: {', '.join(config.supported_extensions)}"
            )
        
        # Check filename
        if not file.filename or file.filename.strip() == "":
            raise HTTPException(
                status_code=400,
                detail="Invalid filename"
            )
        
        # Basic security check for filename
        if any(char in file.filename for char in ['..', '/', '\\', '|', ':', '*', '?', '"', '<', '>']):
            raise HTTPException(
                status_code=400,
                detail="Invalid characters in filename"
            )
    
    async def _save_file(self, file: UploadFile, file_path: Path) -> int:
        """Save uploaded file to disk"""
        
        file_size = 0
        chunk_size = 8192  # 8KB chunks
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                while chunk := await file.read(chunk_size):
                    await f.write(chunk)
                    file_size += len(chunk)
                    
                    # Check size limit during upload
                    if file_size > config.max_file_size:
                        await f.close()
                        file_path.unlink(missing_ok=True)  # Delete partial file
                        raise HTTPException(
                            status_code=413,
                            detail=f"File too large. Maximum size: {config.max_file_size / (1024*1024):.1f}MB"
                        )
            
            return file_size
            
        except Exception as e:
            # Clean up on error
            file_path.unlink(missing_ok=True)
            raise e
        finally:
            # Reset file position for potential re-reading
            await file.seek(0)
    
    async def _detect_mime_type(self, file_path: Path, declared_type: str) -> str:
        """Detect actual MIME type of file"""
        
        if self.mime_detector:
            try:
                detected_type = self.mime_detector.from_file(str(file_path))
                
                # Validate against allowed types
                if detected_type in config.allowed_mime_types:
                    return detected_type
                else:
                    logger.warning(f"Detected MIME type {detected_type} not in allowed list")
                    # Fall back to declared type if it's allowed
                    if declared_type in config.allowed_mime_types:
                        return declared_type
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File type not allowed: {detected_type}"
                        )
            except Exception as e:
                logger.warning(f"MIME type detection failed: {e}")
        
        # Fall back to declared content type
        if declared_type in config.allowed_mime_types:
            return declared_type
        
        # Final fallback based on extension
        extension_mime_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.rtf': 'application/rtf',
            '.txt': 'text/plain',
            '.md': 'text/markdown'
        }
        
        file_extension = file_path.suffix.lower()
        return extension_mime_map.get(file_extension, 'application/octet-stream')
    
    async def _cleanup_uploaded_files(self, doc_infos: List[DocumentInfo]) -> None:
        """Clean up uploaded files on error"""
        for doc_info in doc_infos:
            try:
                file_path = self.get_file_path(doc_info.id)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Cleaned up file: {doc_info.id}")
            except Exception as e:
                logger.error(f"Failed to clean up file {doc_info.id}: {e}")
    
    def get_file_path(self, document_id: str) -> Path:
        """Get file path for document ID"""
        # Find file with matching document ID prefix
        for file_path in self.upload_dir.glob(f"{document_id}.*"):
            return file_path
        
        raise FileNotFoundError(f"File not found for document ID: {document_id}")
    
    def delete_file(self, document_id: str) -> bool:
        """Delete file by document ID"""
        try:
            file_path = self.get_file_path(document_id)
            file_path.unlink()
            logger.info(f"Deleted file: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {document_id}: {e}")
            return False
    
    def get_file_info(self, document_id: str) -> Dict[str, Any]:
        """Get file information"""
        try:
            file_path = self.get_file_path(document_id)
            stat = file_path.stat()
            
            return {
                'path': str(file_path),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'exists': True
            }
        except Exception as e:
            logger.error(f"Failed to get file info for {document_id}: {e}")
            return {'exists': False, 'error': str(e)}
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up old uploaded files"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        cleaned_count = 0
        
        try:
            for file_path in self.upload_dir.iterdir():
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.info(f"Cleaned up old file: {file_path.name}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned_count

# Global file handler instance
file_handler = FileHandler()