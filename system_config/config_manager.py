#!/usr/bin/env python3
"""
Configuration Manager
Handles system configuration loading, validation, and application
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import (
    SystemConfiguration, ConfigurationProfile, WorkMode, 
    GPUAllocationStrategy, ValidationResult, ProfileInfo,
    AgentInstanceConfig, ModelAssignment
)

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Manages system configuration profiles and settings"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "config/profiles")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_config: Optional[SystemConfiguration] = None
        self.available_profiles: Dict[str, ConfigurationProfile] = {}
        
        # Initialize with default profiles
        self._initialize_default_profiles()
    
    def _initialize_default_profiles(self):
        """Initialize default configuration profiles"""
        
        # Development Mode Profile
        dev_profile = ConfigurationProfile(
            name="development",
            workMode=WorkMode.DEVELOPMENT,
            gpuStrategy=GPUAllocationStrategy.SPECIALIZED,
            primaryModel="codellama-34b",
            fallbackModel="dolphin-8b",
            agents={
                "code_generator": {
                    "instances": 2,
                    "gpu": "0",
                    "autoscale": True,
                    "priority": 1
                },
                "research_assistant": {
                    "instances": 1,
                    "gpu": "both",
                    "autoscale": False,
                    "priority": 2
                },
                "document_analyzer": {
                    "instances": 3,
                    "gpu": "1",
                    "autoscale": True,
                    "priority": 1
                }
            },
            performance={
                "reservedMemory": 1.0,
                "memoryBuffer": 0.5,
                "maxConcurrentTasks": 8,
                "taskTimeout": 300,
                "cacheSize": 4.0,
                "autoCleanup": True,
                "batchProcessing": True
            },
            localProcessing={
                "fileIndexing": True,
                "gitIntegration": True,
                "shellCommands": True,
                "autoScanning": False,
                "allowedDirectories": ["/mnt/llm/userfiles", "/projects", "/code"]
            }
        )
        
        # Research Mode Profile
        research_profile = ConfigurationProfile(
            name="research",
            workMode=WorkMode.RESEARCH,
            gpuStrategy=GPUAllocationStrategy.BALANCED,
            primaryModel="llama-70b",
            fallbackModel="dolphin-8b",
            agents={
                "code_generator": {
                    "instances": 1,
                    "gpu": "0",
                    "autoscale": False,
                    "priority": 3
                },
                "research_assistant": {
                    "instances": 3,
                    "gpu": "both",
                    "autoscale": True,
                    "priority": 1
                },
                "document_analyzer": {
                    "instances": 5,
                    "gpu": "both",
                    "autoscale": True,
                    "priority": 1
                }
            },
            performance={
                "reservedMemory": 2.0,
                "memoryBuffer": 1.0,
                "maxConcurrentTasks": 6,
                "taskTimeout": 600,
                "cacheSize": 8.0,
                "autoCleanup": True,
                "batchProcessing": True
            },
            localProcessing={
                "fileIndexing": True,
                "gitIntegration": False,
                "shellCommands": True,
                "autoScanning": True,
                "allowedDirectories": ["/mnt/llm/userfiles", "/documents", "/research"]
            }
        )
        
        # Production Mode Profile
        production_profile = ConfigurationProfile(
            name="production",
            workMode=WorkMode.PRODUCTION,
            gpuStrategy=GPUAllocationStrategy.HIGH_THROUGHPUT,
            primaryModel="llama-13b",
            fallbackModel="mistral-7b",
            agents={
                "code_generator": {
                    "instances": 4,
                    "gpu": "both",
                    "autoscale": True,
                    "priority": 2
                },
                "research_assistant": {
                    "instances": 2,
                    "gpu": "both",
                    "autoscale": True,
                    "priority": 2
                },
                "document_analyzer": {
                    "instances": 6,
                    "gpu": "both",
                    "autoscale": True,
                    "priority": 1
                }
            },
            performance={
                "reservedMemory": 0.5,
                "memoryBuffer": 0.25,
                "maxConcurrentTasks": 16,
                "taskTimeout": 120,
                "cacheSize": 2.0,
                "autoCleanup": True,
                "batchProcessing": True
            },
            localProcessing={
                "fileIndexing": False,
                "gitIntegration": False,
                "shellCommands": False,
                "autoScanning": False,
                "allowedDirectories": ["/mnt/llm/userfiles"]
            }
        )
        
        self.available_profiles = {
            "development": dev_profile,
            "research": research_profile,
            "production": production_profile
        }
    
    def get_available_profiles(self) -> Dict[str, ConfigurationProfile]:
        """Get all available configuration profiles"""
        return self.available_profiles
    
    def get_profile(self, profile_name: str) -> Optional[ConfigurationProfile]:
        """Get a specific configuration profile"""
        return self.available_profiles.get(profile_name)
    
    def load_profile(self, profile_name: str) -> ConfigurationProfile:
        """Load a configuration profile"""
        profile = self.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Configuration profile '{profile_name}' not found")
        
        return profile
    
    def save_profile(self, profile: ConfigurationProfile) -> bool:
        """Save a configuration profile"""
        try:
            # Save to available profiles
            self.available_profiles[profile.name] = profile
            
            # Save to file
            profile_file = self.config_dir / f"{profile.name}.json"
            with open(profile_file, 'w') as f:
                json.dump(profile.dict(), f, indent=2, default=str)
            
            logger.info(f"Configuration profile '{profile.name}' saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration profile: {e}")
            return False
    
    def validate_configuration(self, profile: ConfigurationProfile) -> ValidationResult:
        """Validate a configuration profile"""
        errors = []
        warnings = []
        
        try:
            # Validate GPU requirements
            total_instances = sum(
                agent_config.get("instances", 0) 
                for agent_config in profile.agents.values()
            )
            
            if total_instances > 20:
                errors.append(f"Total agent instances ({total_instances}) exceeds maximum (20)")
            
            # Validate memory requirements
            reserved_memory = profile.performance.get("reservedMemory", 0)
            memory_buffer = profile.performance.get("memoryBuffer", 0)
            total_reserved = reserved_memory + memory_buffer
            
            if total_reserved > 8:  # Assuming 24GB GPU with 8GB max reservation
                warnings.append(f"High memory reservation ({total_reserved}GB) may limit model loading")
            
            # Validate concurrent tasks
            max_tasks = profile.performance.get("maxConcurrentTasks", 8)
            if max_tasks > 32:
                warnings.append(f"High concurrent task limit ({max_tasks}) may impact performance")
            
            # Validate directories
            allowed_dirs = profile.localProcessing.get("allowedDirectories", [])
            for directory in allowed_dirs:
                if not os.path.exists(directory):
                    warnings.append(f"Directory does not exist: {directory}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return ValidationResult(
            isValid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def apply_configuration(self, profile: ConfigurationProfile) -> bool:
        """Apply a configuration profile to the system"""
        try:
            # Validate configuration first
            validation = self.validate_configuration(profile)
            if not validation.isValid:
                logger.error(f"Configuration validation failed: {validation.errors}")
                return False
            
            # Log warnings
            for warning in validation.warnings:
                logger.warning(warning)
            
            # Convert to full system configuration
            system_config = self._profile_to_system_config(profile)
            
            # Apply configuration (this would integrate with actual system components)
            self.current_config = system_config
            
            # Save as current configuration
            self._save_current_config()
            
            logger.info(f"Configuration '{profile.name}' applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply configuration: {e}")
            return False
    
    def get_current_configuration(self) -> Optional[ConfigurationProfile]:
        """Get the current system configuration"""
        if self.current_config:
            return self._system_config_to_profile(self.current_config)
        
        # Try to load from file
        try:
            current_file = self.config_dir / "current.json"
            if current_file.exists():
                with open(current_file, 'r') as f:
                    data = json.load(f)
                    return ConfigurationProfile(**data)
        except Exception as e:
            logger.error(f"Failed to load current configuration: {e}")
        
        # Return default development configuration
        return self.get_profile("development")
    
    def _profile_to_system_config(self, profile: ConfigurationProfile) -> SystemConfiguration:
        """Convert a configuration profile to full system configuration"""
        
        # Create profile info
        profile_info = ProfileInfo(
            name=profile.name,
            description=f"{profile.workMode.value.title()} mode configuration",
            mode=profile.workMode,
            last_modified=datetime.now()
        )
        
        # Create agent configurations
        agent_instances = []
        for agent_type, config in profile.agents.items():
            gpu_affinity = []
            if config.get("gpu") == "0":
                gpu_affinity = [0]
            elif config.get("gpu") == "1":
                gpu_affinity = [1]
            elif config.get("gpu") == "both":
                gpu_affinity = [0, 1]
            
            agent_instance = AgentInstanceConfig(
                agent_type=agent_type,
                instance_count=config.get("instances", 1),
                gpu_affinity=gpu_affinity,
                auto_scale=config.get("autoscale", True),
                priority=config.get("priority", 2)
            )
            agent_instances.append(agent_instance)
        
        # Create system configuration
        system_config = SystemConfiguration(profile=profile_info)
        system_config.gpu_config.allocation_strategy = profile.gpuStrategy
        system_config.agent_config.agent_deployment.agents = agent_instances
        
        # Apply performance settings
        perf = profile.performance
        system_config.performance_config.max_concurrent_tasks = perf.get("maxConcurrentTasks", 8)
        system_config.performance_config.task_timeout = perf.get("taskTimeout", 300)
        system_config.performance_config.cache_size = int(perf.get("cacheSize", 4) * 1024)  # Convert GB to MB
        system_config.performance_config.batch_processing = perf.get("batchProcessing", True)
        
        # Apply GPU memory settings
        system_config.gpu_config.memory_management.reservation_per_gpu = perf.get("reservedMemory", 2) * 1024  # Convert GB to MB
        system_config.gpu_config.memory_management.memory_buffer = perf.get("memoryBuffer", 1) * 1024  # Convert GB to MB
        system_config.gpu_config.memory_management.auto_cleanup = perf.get("autoCleanup", True)
        
        # Apply local processing settings
        local = profile.localProcessing
        system_config.local_config.file_indexing = local.get("fileIndexing", True)
        system_config.local_config.git_integration = local.get("gitIntegration", False)
        system_config.local_config.shell_integration = local.get("shellCommands", True)
        system_config.local_config.project_scanning = local.get("autoScanning", False)
        system_config.local_config.allowed_directories = local.get("allowedDirectories", [])
        
        return system_config
    
    def _system_config_to_profile(self, system_config: SystemConfiguration) -> ConfigurationProfile:
        """Convert system configuration to profile format"""
        
        # Extract agent configurations
        agents = {}
        for agent in system_config.agent_config.agent_deployment.agents:
            gpu_assignment = "both"
            if len(agent.gpu_affinity) == 1:
                gpu_assignment = str(agent.gpu_affinity[0])
            elif len(agent.gpu_affinity) == 0:
                gpu_assignment = "both"
            
            agents[agent.agent_type] = {
                "instances": agent.instance_count,
                "gpu": gpu_assignment,
                "autoscale": agent.auto_scale,
                "priority": agent.priority
            }
        
        # Extract performance settings
        performance = {
            "reservedMemory": system_config.gpu_config.memory_management.reservation_per_gpu / 1024,  # Convert MB to GB
            "memoryBuffer": system_config.gpu_config.memory_management.memory_buffer / 1024,  # Convert MB to GB
            "maxConcurrentTasks": system_config.performance_config.max_concurrent_tasks,
            "taskTimeout": system_config.performance_config.task_timeout,
            "cacheSize": system_config.performance_config.cache_size / 1024,  # Convert MB to GB
            "autoCleanup": system_config.gpu_config.memory_management.auto_cleanup,
            "batchProcessing": system_config.performance_config.batch_processing
        }
        
        # Extract local processing settings
        local_processing = {
            "fileIndexing": system_config.local_config.file_indexing,
            "gitIntegration": system_config.local_config.git_integration,
            "shellCommands": system_config.local_config.shell_integration,
            "autoScanning": system_config.local_config.project_scanning,
            "allowedDirectories": system_config.local_config.allowed_directories
        }
        
        return ConfigurationProfile(
            name=system_config.profile.name,
            workMode=system_config.profile.mode,
            gpuStrategy=system_config.gpu_config.allocation_strategy,
            primaryModel="codellama-34b",  # Default, would be extracted from model config
            fallbackModel="dolphin-8b",    # Default, would be extracted from model config
            agents=agents,
            performance=performance,
            localProcessing=local_processing
        )
    
    def _save_current_config(self):
        """Save current configuration to file"""
        try:
            if self.current_config:
                profile = self._system_config_to_profile(self.current_config)
                current_file = self.config_dir / "current.json"
                with open(current_file, 'w') as f:
                    json.dump(profile.dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save current configuration: {e}")
    
    def export_configuration(self, profile: ConfigurationProfile, file_path: str) -> bool:
        """Export configuration to file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(profile.dict(), f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, file_path: str) -> Optional[ConfigurationProfile]:
        """Import configuration from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return ConfigurationProfile(**data)
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return None