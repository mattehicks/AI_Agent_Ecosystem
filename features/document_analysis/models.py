#!/usr/bin/env python3
"""
Document Analysis Data Models
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class TaskStatus(str, Enum):
    """Task processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AnalysisType(str, Enum):
    """Types of document analysis"""
    SUMMARY = "summary"
    KEY_POINTS = "key_points"
    ENTITIES = "entities"
    SENTIMENT = "sentiment"
    QUESTIONS = "questions"
    ACTIONS = "actions"

# Pydantic Models for API

class DocumentUpload(BaseModel):
    """Document upload request"""
    filename: str
    content_type: str
    size: int
    analysis_types: List[AnalysisType] = Field(default=[AnalysisType.SUMMARY])

class DocumentInfo(BaseModel):
    """Document information"""
    id: str
    filename: str
    content_type: str
    size: int
    upload_time: datetime
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: Optional[str] = None

class AnalysisRequest(BaseModel):
    """Analysis request for remote processing"""
    document_id: str
    text_content: str
    analysis_types: List[AnalysisType]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnalysisResult(BaseModel):
    """Analysis result from processing"""
    document_id: str
    analysis_type: AnalysisType
    result: Dict[str, Any]
    confidence: Optional[float] = None
    processing_time: Optional[float] = None

class DocumentAnalysis(BaseModel):
    """Complete document analysis"""
    document_id: str
    filename: str
    summary: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment: Optional[Dict[str, Any]] = None
    questions: List[str] = Field(default_factory=list)
    actions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    processing_time: Optional[float] = None

class TaskUpdate(BaseModel):
    """Task status update for WebSocket"""
    task_id: str
    status: TaskStatus
    progress: float
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# SQLAlchemy Models for Database

class DocumentRecord(Base):
    """Database model for document records"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    
    # Processing status
    status = Column(String, nullable=False, default=TaskStatus.PENDING.value)
    progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    
    # Extracted content
    text_content = Column(Text, nullable=True)
    document_metadata = Column(JSON, nullable=True)
    
    # Analysis results
    analysis_results = Column(JSON, nullable=True)
    processing_time = Column(Float, nullable=True)

class TaskRecord(Base):
    """Database model for task records"""
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default=TaskStatus.PENDING.value)
    progress = Column(Float, default=0.0)
    
    # Task configuration
    parameters = Column(JSON, nullable=True)
    
    # Results
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Performance metrics
    processing_time = Column(Float, nullable=True)
    retry_count = Column(Integer, default=0)