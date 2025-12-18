#!/usr/bin/env python3
"""
System Configuration Models
Pydantic models for configuration management
"""

from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class WorkMode(str, Enum):
    RESEARCH = "research"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    CUSTOM = "custom"

class GPUAllocationStrategy(str, Enum):
    BALANCED = "balanced"
    SPECIALIZED = "specialized"
    HIGH_THROUGHPUT = "high_throughput"
    SINGLE_LARGE = "single_large"

class CoolingStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

class TaskType(str, Enum):
    CODE_GENERATION = "code_generation"
    DOCUMENT_ANALYSIS = "document_analysis"
    RESEARCH = "research"
    CHAT = "chat"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

# Configuration Models
class ProfileInfo(BaseModel):
    name: str
    description: str
    mode: WorkMode
    created_at: datetime = Field(default_factory=datetime.now)
    last_modified: datetime = Field(default_factory=datetime.now)
    is_default: bool = False

class MemoryManagement(BaseModel):
    reservation_per_gpu: float = 2048  # MB
    dynamic_allocation: bool = True
    memory_buffer: float = 1024  # MB
    auto_cleanup: bool = True
    swap_threshold: float = 85  # Percentage

class ThermalManagement(BaseModel):
    max_temperature: int = 83  # Celsius
    throttle_temperature: int = 78  # Celsius
    cooling_strategy: CoolingStrategy = CoolingStrategy.BALANCED

class PowerManagement(BaseModel):
    max_power_limit: Optional[float] = None  # Watts
    power_efficiency_mode: bool = False

class LoadBalancingConfig(BaseModel):
    algorithm: str = "least_loaded"
    health_check_interval: int = 30  # seconds
    failover_enabled: bool = True

class GPUConfiguration(BaseModel):
    allocation_strategy: GPUAllocationStrategy = GPUAllocationStrategy.SPECIALIZED
    memory_management: MemoryManagement = Field(default_factory=MemoryManagement)
    thermal_management: ThermalManagement = Field(default_factory=ThermalManagement)
    power_management: PowerManagement = Field(default_factory=PowerManagement)
    load_balancing: LoadBalancingConfig = Field(default_factory=LoadBalancingConfig)

class ModelAssignment(BaseModel):
    model_name: str
    model_path: str
    gpu_assignment: List[int] = Field(default_factory=list)
    memory_requirement: int = 4096  # MB
    context_length: int = 8192
    specialization: List[str] = Field(default_factory=list)
    priority: int = 1

class SpecializedModelConfig(BaseModel):
    task_type: TaskType
    model_name: str
    gpu_preference: int = 0
    auto_load: bool = False
    unload_timeout: int = 300  # seconds

class ModelSelection(BaseModel):
    primary_models: List[ModelAssignment] = Field(default_factory=list)
    fallback_models: List[ModelAssignment] = Field(default_factory=list)
    specialized_models: List[SpecializedModelConfig] = Field(default_factory=list)

class ContextManagement(BaseModel):
    max_context_length: int = 8192
    context_window_strategy: str = "sliding"
    context_compression: bool = False

class QuantizationConfig(BaseModel):
    enabled: bool = True
    quantization_type: str = "Q4_K_M"
    dynamic_quantization: bool = False

class ModelLoadingStrategy(BaseModel):
    preload_models: bool = True
    lazy_loading: bool = False
    model_switching_delay: int = 5  # seconds

class ModelConfiguration(BaseModel):
    model_selection: ModelSelection = Field(default_factory=ModelSelection)
    loading_strategy: ModelLoadingStrategy = Field(default_factory=ModelLoadingStrategy)
    context_management: ContextManagement = Field(default_factory=ContextManagement)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)

class AgentInstanceConfig(BaseModel):
    agent_type: str
    instance_count: int = 1
    min_instances: int = 0
    max_instances: int = 5
    gpu_affinity: List[int] = Field(default_factory=list)
    memory_limit: int = 2048  # MB
    cpu_cores: int = 2
    auto_scale: bool = True
    priority: int = 2

class ScalingPolicy(BaseModel):
    scale_up_threshold: int = 5
    scale_down_threshold: int = 30  # seconds
    scale_up_cooldown: int = 10  # seconds
    scale_down_cooldown: int = 60  # seconds
    max_scale_up_rate: int = 2

class ResourceAllocation(BaseModel):
    max_memory_per_agent: int = 4096  # MB
    max_cpu_per_agent: int = 4
    shared_resources: bool = True

class TaskRouting(BaseModel):
    routing_algorithm: str = "round_robin"
    priority_based: bool = True
    load_balancing: bool = True

class AgentDeployment(BaseModel):
    agents: List[AgentInstanceConfig] = Field(default_factory=list)
    max_total_instances: int = 20
    startup_delay: int = 2  # seconds
    health_check_interval: int = 30  # seconds

class AgentConfiguration(BaseModel):
    agent_deployment: AgentDeployment = Field(default_factory=AgentDeployment)
    resource_allocation: ResourceAllocation = Field(default_factory=ResourceAllocation)
    scaling_policy: ScalingPolicy = Field(default_factory=ScalingPolicy)
    task_routing: TaskRouting = Field(default_factory=TaskRouting)

class PerformanceConfiguration(BaseModel):
    max_concurrent_tasks: int = 8
    task_timeout: int = 300  # seconds
    cache_size: int = 4096  # MB
    prefetch_enabled: bool = True
    batch_processing: bool = True
    load_balancing: bool = False

class LocalProcessingConfiguration(BaseModel):
    file_indexing: bool = True
    text_extraction: bool = True
    metadata_parsing: bool = True
    shell_integration: bool = True
    git_integration: bool = False
    project_scanning: bool = False
    allowed_directories: List[str] = Field(default_factory=lambda: ["/mnt/llm/userfiles", "/projects", "/documents"])

class SecurityConfiguration(BaseModel):
    api_key_required: bool = False
    rate_limiting: bool = True
    ip_whitelisting: bool = False
    audit_logging: bool = True
    data_encryption: bool = False

class SystemConfiguration(BaseModel):
    profile: ProfileInfo
    gpu_config: GPUConfiguration = Field(default_factory=GPUConfiguration)
    model_configuration: ModelConfiguration = Field(default_factory=ModelConfiguration)
    agent_config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    performance_config: PerformanceConfiguration = Field(default_factory=PerformanceConfiguration)
    local_config: LocalProcessingConfiguration = Field(default_factory=LocalProcessingConfiguration)
    security_config: SecurityConfiguration = Field(default_factory=SecurityConfiguration)

class ConfigurationProfile(BaseModel):
    """Simplified configuration profile for API responses"""
    name: str
    workMode: WorkMode
    gpuStrategy: GPUAllocationStrategy
    primaryModel: str
    fallbackModel: str
    agents: Dict[str, Dict[str, Any]]
    performance: Dict[str, Any]
    localProcessing: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class ValidationResult(BaseModel):
    isValid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class ConfigurationResponse(BaseModel):
    success: bool
    message: str
    configuration: Optional[ConfigurationProfile] = None
    validation: Optional[ValidationResult] = None