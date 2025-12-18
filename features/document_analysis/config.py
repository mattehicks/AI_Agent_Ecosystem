#!/usr/bin/env python3
"""
Document Analysis Feature Configuration
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import List, Optional

class DocumentAnalysisConfig(BaseSettings):
    """Configuration for document analysis feature"""
    
    # File Upload Settings
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Maximum file size in bytes (100MB)")
    max_files_per_batch: int = Field(default=10, description="Maximum files per upload batch")
    upload_dir: str = Field(default="temp/uploads", description="Temporary upload directory")
    
    # Supported File Types
    supported_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".doc", ".rtf", ".txt", ".md"],
        description="Supported file extensions"
    )
    
    # Processing Settings
    max_text_length: int = Field(default=1000000, description="Maximum text length to process")
    chunk_size: int = Field(default=4000, description="Text chunk size for processing")
    chunk_overlap: int = Field(default=200, description="Overlap between text chunks")
    
    # Task Queue Settings
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL for task queue")
    task_timeout: int = Field(default=300, description="Task timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum task retries")
    
    # Database Settings
    database_url: str = Field(default="sqlite:///./document_analysis.db", description="Database URL")
    
    @property
    def resolved_database_url(self) -> str:
        """Get resolved database URL with proper path"""
        if self.database_url.startswith("sqlite:///"):
            # Use absolute path for SQLite
            db_path = self.base_path / "document_analysis.db"
            return f"sqlite:///{db_path}"
        return self.database_url
    
    # Remote Processing Settings
    remote_api_url: str = Field(default="http://localhost:8000", description="Remote API URL")
    remote_timeout: int = Field(default=120, description="Remote API timeout in seconds")
    
    # Security Settings
    allowed_mime_types: List[str] = Field(
        default=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "application/rtf",
            "text/plain",
            "text/markdown"
        ],
        description="Allowed MIME types"
    )
    
    # Environment Detection
    @property
    def is_windows(self) -> bool:
        return os.name == 'nt'
    
    @property
    def base_path(self) -> Path:
        if self.is_windows or os.path.exists("X:/"):
            return Path("X:/AI_Agent_Ecosystem")
        else:
            return Path("/mnt/llm/AI_Agent_Ecosystem")
    
    @property
    def upload_path(self) -> Path:
        # Use userfiles structure as mentioned in notes
        if self.is_windows or os.path.exists("X:/"):
            base = Path("X:/AI_Agent_Ecosystem")
        else:
            base = Path("/mnt/llm")
        
        upload_path = base / "userfiles" / "default" / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)
        return upload_path
    
    class Config:
        env_prefix = "DOC_ANALYSIS_"
        case_sensitive = False

# Global configuration instance
config = DocumentAnalysisConfig()