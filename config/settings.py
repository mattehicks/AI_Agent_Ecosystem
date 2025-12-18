#!/usr/bin/env python3
"""
System Configuration Settings
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

class MonitoringSettings(BaseModel):
    """GPU and system monitoring settings"""
    gpu_update_interval: float = Field(default=3.0, ge=1.0, le=30.0, description="GPU status update interval in seconds")
    system_metrics_interval: float = Field(default=5.0, ge=2.0, le=60.0, description="System metrics update interval in seconds")
    enable_realtime_monitoring: bool = Field(default=True, description="Enable real-time monitoring")
    max_history_points: int = Field(default=100, ge=10, le=1000, description="Maximum data points to keep in history")

class UISettings(BaseModel):
    """User interface settings"""
    theme: str = Field(default="dark", description="UI theme (dark/light)")
    compact_mode: bool = Field(default=True, description="Use compact UI layout")
    show_advanced_metrics: bool = Field(default=False, description="Show advanced GPU metrics")
    auto_refresh_dashboard: bool = Field(default=True, description="Auto-refresh dashboard data")

class ModelSettings(BaseModel):
    """Model management settings"""
    auto_unload_idle_models: bool = Field(default=False, description="Auto-unload models after idle time")
    idle_timeout_minutes: int = Field(default=30, ge=5, le=480, description="Minutes before auto-unloading idle models")
    default_gpu_allocation: str = Field(default="balanced", description="Default GPU allocation strategy")
    enable_model_preloading: bool = Field(default=False, description="Enable model preloading")

class SystemSettings(BaseModel):
    """System-wide settings"""
    log_level: str = Field(default="INFO", description="Logging level")
    max_concurrent_tasks: int = Field(default=10, ge=1, le=100, description="Maximum concurrent tasks")
    task_timeout_seconds: int = Field(default=300, ge=30, le=3600, description="Default task timeout")
    enable_performance_profiling: bool = Field(default=False, description="Enable performance profiling")

class EcosystemConfig(BaseSettings):
    """Main configuration class"""
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    ui: UISettings = Field(default_factory=UISettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    
    class Config:
        env_prefix = "AI_ECOSYSTEM_"
        case_sensitive = False

class ConfigManager:
    """Manages system configuration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            # Auto-detect environment
            if Path("/mnt/llm").exists():
                config_path = Path("/mnt/llm/AI_Agent_Ecosystem/config/settings.json")
            else:
                config_path = Path("X:/AI_Agent_Ecosystem/config/settings.json")
        
        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._config = self._load_config()
        self._subscribers = []
    
    def _load_config(self) -> EcosystemConfig:
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                return EcosystemConfig(**data)
            else:
                # Create default config
                config = EcosystemConfig()
                self.save_config(config)
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return EcosystemConfig()
    
    def save_config(self, config: Optional[EcosystemConfig] = None):
        """Save configuration to file"""
        if config is None:
            config = self._config
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config.dict(), f, indent=2)
            
            # Notify subscribers of config change
            for callback in self._subscribers:
                try:
                    callback(config)
                except Exception as e:
                    logger.error(f"Config subscriber error: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_config(self) -> EcosystemConfig:
        """Get current configuration"""
        return self._config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        try:
            # Convert flat dict to nested structure
            config_dict = self._config.dict()
            
            for key, value in updates.items():
                if '.' in key:
                    # Handle nested keys like 'monitoring.gpu_update_interval'
                    parts = key.split('.')
                    current = config_dict
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    config_dict[key] = value
            
            # Validate and create new config
            new_config = EcosystemConfig(**config_dict)
            self._config = new_config
            self.save_config()
            
            logger.info(f"Configuration updated: {updates}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False
    
    def subscribe_to_changes(self, callback):
        """Subscribe to configuration changes"""
        self._subscribers.append(callback)
    
    def unsubscribe_from_changes(self, callback):
        """Unsubscribe from configuration changes"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> EcosystemConfig:
    """Get current system configuration"""
    return config_manager.get_config()

def update_config(updates: Dict[str, Any]) -> bool:
    """Update system configuration"""
    return config_manager.update_config(updates)