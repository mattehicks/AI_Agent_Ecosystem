#!/usr/bin/env python3
"""
Research Assistant Models
Data models for research workflows and configurations
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

# Pydantic Models for API
class WorkflowType(BaseModel):
    """Workflow type definition"""
    id: str
    name: str
    description: str
    icon: str
    category: Literal["text_generation", "technical_analysis", "document_processing"]
    required_models: List[str] = []
    parameters: Dict[str, Any] = {}

class BackendConfiguration(BaseModel):
    """Backend model configuration"""
    id: str
    name: str
    description: str
    models: List[str]
    parameters: Dict[str, Any]
    workflow_types: List[str]
    gpu_requirements: Dict[str, Any] = {}

class ResearchRequest(BaseModel):
    """Research workflow request"""
    workflow_type: str
    backend_config: str
    input_documents: List[str] = []
    input_folders: List[str] = []
    parameters: Dict[str, Any] = {}
    output_format: str = "markdown"
    save_results: bool = True

class ResearchResult(BaseModel):
    """Research workflow result"""
    id: str
    workflow_type: str
    status: Literal["pending", "processing", "completed", "failed"]
    input_summary: Dict[str, Any]
    results: Dict[str, Any] = {}
    output_files: List[str] = []
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

class DocumentInfo(BaseModel):
    """Document information"""
    path: str
    name: str
    size: int
    type: str
    modified_at: datetime
    content_preview: Optional[str] = None

# SQLAlchemy Models for Database
class ResearchSession(Base):
    """Research session record"""
    __tablename__ = "research_sessions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), unique=True, nullable=False)
    workflow_type = Column(String(100), nullable=False)
    backend_config = Column(String(100), nullable=False)
    status = Column(String(20), default="pending")
    input_summary = Column(JSON)
    parameters = Column(JSON)
    results = Column(JSON)
    output_files = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(Text)

class ResearchOutput(Base):
    """Research output file record"""
    __tablename__ = "research_outputs"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), ForeignKey("research_sessions.session_id"))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50))
    file_size = Column(Integer)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    session = relationship("ResearchSession", backref="outputs")

class WorkflowTemplate(Base):
    """Saved workflow template"""
    __tablename__ = "workflow_templates"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    workflow_type = Column(String(100), nullable=False)
    backend_config = Column(String(100), nullable=False)
    parameters = Column(JSON)
    is_public = Column(Boolean, default=False)
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)