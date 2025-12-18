#!/usr/bin/env python3
"""
System Configuration API
FastAPI endpoints for configuration management
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from .config_manager import ConfigurationManager
from .models import ConfigurationProfile, ConfigurationResponse, ValidationResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System Configuration"])

# Global configuration manager
config_manager = ConfigurationManager()

class ConfigurationRequest(BaseModel):
    profile: Dict[str, Any]

@router.get("/health")
async def health_check():
    """Health check for system configuration service"""
    return {
        "status": "healthy",
        "service": "system-configuration",
        "profiles_available": len(config_manager.get_available_profiles())
    }

@router.get("/configuration")
async def get_current_configuration():
    """Get the current system configuration"""
    try:
        current_config = config_manager.get_current_configuration()
        if current_config:
            return ConfigurationResponse(
                success=True,
                message="Current configuration retrieved successfully",
                configuration=current_config
            )
        else:
            return ConfigurationResponse(
                success=False,
                message="No current configuration found"
            )
    except Exception as e:
        logger.error(f"Failed to get current configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration: {str(e)}")

@router.post("/configuration")
async def apply_configuration(request: ConfigurationRequest):
    """Apply a new system configuration"""
    try:
        # Convert request to configuration profile
        profile_data = request.profile
        profile = ConfigurationProfile(**profile_data)
        
        # Validate configuration
        validation = config_manager.validate_configuration(profile)
        
        if not validation.isValid:
            return ConfigurationResponse(
                success=False,
                message="Configuration validation failed",
                validation=validation
            )
        
        # Apply configuration
        success = config_manager.apply_configuration(profile)
        
        if success:
            return ConfigurationResponse(
                success=True,
                message="Configuration applied successfully",
                configuration=profile,
                validation=validation
            )
        else:
            return ConfigurationResponse(
                success=False,
                message="Failed to apply configuration"
            )
            
    except Exception as e:
        logger.error(f"Failed to apply configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to apply configuration: {str(e)}")

@router.get("/profiles")
async def get_available_profiles():
    """Get all available configuration profiles"""
    try:
        profiles = config_manager.get_available_profiles()
        return {
            "success": True,
            "profiles": profiles
        }
    except Exception as e:
        logger.error(f"Failed to get profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profiles: {str(e)}")

@router.get("/profiles/{profile_name}")
async def get_configuration_profile(profile_name: str):
    """Get a specific configuration profile"""
    try:
        profile = config_manager.get_profile(profile_name)
        if profile:
            return ConfigurationResponse(
                success=True,
                message=f"Profile '{profile_name}' retrieved successfully",
                configuration=profile
            )
        else:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get profile '{profile_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get profile: {str(e)}")

@router.post("/profiles/{profile_name}")
async def save_configuration_profile(profile_name: str, request: ConfigurationRequest):
    """Save a configuration profile"""
    try:
        profile_data = request.profile
        profile_data["name"] = profile_name
        profile = ConfigurationProfile(**profile_data)
        
        success = config_manager.save_profile(profile)
        
        if success:
            return ConfigurationResponse(
                success=True,
                message=f"Profile '{profile_name}' saved successfully",
                configuration=profile
            )
        else:
            return ConfigurationResponse(
                success=False,
                message=f"Failed to save profile '{profile_name}'"
            )
            
    except Exception as e:
        logger.error(f"Failed to save profile '{profile_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save profile: {str(e)}")

@router.post("/profiles/{profile_name}/load")
async def load_configuration_profile(profile_name: str):
    """Load and apply a configuration profile"""
    try:
        profile = config_manager.load_profile(profile_name)
        success = config_manager.apply_configuration(profile)
        
        if success:
            return ConfigurationResponse(
                success=True,
                message=f"Profile '{profile_name}' loaded and applied successfully",
                configuration=profile
            )
        else:
            return ConfigurationResponse(
                success=False,
                message=f"Failed to apply profile '{profile_name}'"
            )
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load profile '{profile_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load profile: {str(e)}")

@router.post("/configuration/validate")
async def validate_configuration(request: ConfigurationRequest):
    """Validate a configuration without applying it"""
    try:
        profile_data = request.profile
        profile = ConfigurationProfile(**profile_data)
        
        validation = config_manager.validate_configuration(profile)
        
        return {
            "success": True,
            "validation": validation,
            "message": "Configuration validated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate configuration: {str(e)}")

@router.get("/configuration/export/{profile_name}")
async def export_configuration_profile(profile_name: str):
    """Export a configuration profile"""
    try:
        profile = config_manager.get_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found")
        
        return {
            "success": True,
            "profile": profile.dict(),
            "filename": f"{profile_name}_config.json"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export profile '{profile_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export profile: {str(e)}")

@router.post("/configuration/import")
async def import_configuration_profile(file: UploadFile = File(...)):
    """Import a configuration profile from file"""
    try:
        # Read file content
        content = await file.read()
        
        # Save temporarily and import
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, 'wb') as f:
            f.write(content)
        
        profile = config_manager.import_configuration(temp_path)
        
        if profile:
            # Save the imported profile
            config_manager.save_profile(profile)
            
            return ConfigurationResponse(
                success=True,
                message=f"Configuration imported successfully as '{profile.name}'",
                configuration=profile
            )
        else:
            return ConfigurationResponse(
                success=False,
                message="Failed to import configuration file"
            )
            
    except Exception as e:
        logger.error(f"Failed to import configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to import configuration: {str(e)}")

@router.post("/configuration/reset")
async def reset_configuration():
    """Reset configuration to default development mode"""
    try:
        profile = config_manager.get_profile("development")
        success = config_manager.apply_configuration(profile)
        
        if success:
            return ConfigurationResponse(
                success=True,
                message="Configuration reset to development mode defaults",
                configuration=profile
            )
        else:
            return ConfigurationResponse(
                success=False,
                message="Failed to reset configuration"
            )
            
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset configuration: {str(e)}")

@router.get("/models/available")
async def get_available_models():
    """Get list of available models for configuration"""
    try:
        # This would typically scan the models directory
        models = [
            {"id": "codellama-34b", "name": "CodeLlama 34B", "size": "34B", "type": "code"},
            {"id": "llama-70b", "name": "Llama 70B", "size": "70B", "type": "general"},
            {"id": "dolphin-8b", "name": "Dolphin 8B", "size": "8B", "type": "chat"},
            {"id": "mistral-7b", "name": "Mistral 7B", "size": "7B", "type": "general"},
            {"id": "llama-13b", "name": "Llama 13B", "size": "13B", "type": "general"}
        ]
        
        return {
            "success": True,
            "models": models
        }
        
    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available models: {str(e)}")

@router.get("/agents/types")
async def get_available_agent_types():
    """Get list of available agent types"""
    try:
        agent_types = [
            {
                "id": "code_generator",
                "name": "Code Generator",
                "description": "Generates code based on prompts and context",
                "icon": "fas fa-code"
            },
            {
                "id": "research_assistant", 
                "name": "Research Assistant",
                "description": "Analyzes documents and performs research tasks",
                "icon": "fas fa-search"
            },
            {
                "id": "document_analyzer",
                "name": "Document Analyzer", 
                "description": "Extracts and analyzes content from documents",
                "icon": "fas fa-file-alt"
            }
        ]
        
        return {
            "success": True,
            "agent_types": agent_types
        }
        
    except Exception as e:
        logger.error(f"Failed to get agent types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent types: {str(e)}")