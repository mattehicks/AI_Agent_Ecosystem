#!/usr/bin/env python3
"""
GPU Manager for AI Agent Ecosystem
Handles GPU detection, monitoring, and resource allocation
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available. GPU monitoring will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. GPU inference will not work.")

@dataclass
class GPUInfo:
    """GPU information and metrics"""
    index: int
    name: str
    memory_total: int  # MB
    memory_used: int   # MB
    memory_free: int   # MB
    utilization: float # Percentage
    temperature: int   # Celsius
    power_usage: float # Watts
    power_limit: float # Watts
    is_available: bool
    current_model: Optional[str] = None
    last_updated: Optional[datetime] = None

class GPUManager:
    """Manages GPU resources and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpus: Dict[int, GPUInfo] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_interval = 2.0  # seconds
        
        # Enhanced state management for real-time updates
        self.last_gpu_data = {}
        self.is_loading_gpu_data = False
        self.gpu_update_debounce = 0
        self.callbacks = []  # For WebSocket updates
        self.metrics_history = {}  # Store historical data for charts
        
        self._initialize_nvml()
        self._discover_gpus()
        
    def _initialize_nvml(self):
        """Initialize NVIDIA Management Library"""
        if not NVML_AVAILABLE:
            self.logger.warning("NVML not available - GPU monitoring disabled")
            return
            
        try:
            pynvml.nvmlInit()
            self.logger.info("NVML initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize NVML: {e}")
            # Fix: Remove the problematic global declaration
            # NVML_AVAILABLE is already declared at module level
    
    def _discover_gpus(self):
        """Discover available GPUs"""
        if not NVML_AVAILABLE:
            # Fallback to torch detection
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    self.gpus[i] = GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total=props.total_memory // (1024 * 1024),  # Convert to MB
                        memory_used=0,
                        memory_free=props.total_memory // (1024 * 1024),
                        utilization=0.0,
                        temperature=0,
                        power_usage=0.0,
                        power_limit=0.0,
                        is_available=True,
                        last_updated=datetime.now()
                    )
                self.logger.info(f"Discovered {len(self.gpus)} GPUs via PyTorch")
            return
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            self.logger.info(f"Discovered {device_count} GPU(s)")
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Handle both bytes and string return types
                name_raw = pynvml.nvmlDeviceGetName(handle)
                name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else name_raw
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except:
                    gpu_util = 0.0
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = 0
                
                # Get power info
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                except:
                    power_usage = 0.0
                    power_limit = 0.0
                
                self.gpus[i] = GPUInfo(
                    index=i,
                    name=name,
                    memory_total=mem_info.total // (1024 * 1024),  # Convert to MB
                    memory_used=mem_info.used // (1024 * 1024),
                    memory_free=mem_info.free // (1024 * 1024),
                    utilization=gpu_util,
                    temperature=temp,
                    power_usage=power_usage,
                    power_limit=power_limit,
                    is_available=True,
                    last_updated=datetime.now()
                )
                
                self.logger.info(f"GPU {i}: {name} - {mem_info.total // (1024**3)}GB VRAM")
                
        except Exception as e:
            self.logger.error(f"Error discovering GPUs: {e}")
    
    def start_monitoring(self):
        """Start GPU monitoring thread"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("GPU monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_gpu_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _update_gpu_metrics(self):
        """Update GPU metrics"""
        if not NVML_AVAILABLE:
            return
            
        try:
            for gpu_index in self.gpus:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                
                # Update memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpus[gpu_index].memory_used = mem_info.used // (1024 * 1024)
                self.gpus[gpu_index].memory_free = mem_info.free // (1024 * 1024)
                
                # Update utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpus[gpu_index].utilization = util.gpu
                except:
                    pass
                
                # Update temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    self.gpus[gpu_index].temperature = temp
                except:
                    pass
                
                # Update power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    self.gpus[gpu_index].power_usage = power_usage
                except:
                    pass
                
                self.gpus[gpu_index].last_updated = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error updating GPU metrics: {e}")
    
    def get_gpu_info(self, gpu_index: Optional[int] = None) -> Dict[str, Any]:
        """Get GPU information"""
        if gpu_index is not None:
            if gpu_index in self.gpus:
                gpu = self.gpus[gpu_index]
                return {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory": {
                        "total": gpu.memory_total,
                        "used": gpu.memory_used,
                        "free": gpu.memory_free,
                        "usage_percent": (gpu.memory_used / gpu.memory_total * 100) if gpu.memory_total > 0 else 0
                    },
                    "utilization": gpu.utilization,
                    "temperature": gpu.temperature,
                    "power": {
                        "usage": gpu.power_usage,
                        "limit": gpu.power_limit,
                        "usage_percent": (gpu.power_usage / gpu.power_limit * 100) if gpu.power_limit > 0 else 0
                    },
                    "is_available": gpu.is_available,
                    "current_model": gpu.current_model,
                    "last_updated": gpu.last_updated.isoformat() if gpu.last_updated else None
                }
            else:
                return {"error": f"GPU {gpu_index} not found"}
        
        # Return all GPUs
        return {
            "gpus": [self.get_gpu_info(i) for i in self.gpus.keys()],
            "total_gpus": len(self.gpus),
            "total_vram": sum(gpu.memory_total for gpu in self.gpus.values()),
            "available_vram": sum(gpu.memory_free for gpu in self.gpus.values()),
            "average_utilization": sum(gpu.utilization for gpu in self.gpus.values()) / len(self.gpus) if self.gpus else 0,
            "average_temperature": sum(gpu.temperature for gpu in self.gpus.values()) / len(self.gpus) if self.gpus else 0
        }
    
    def add_callback(self, callback):
        """Add callback for real-time updates"""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback):
        """Remove callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, data):
        """Notify all callbacks with new data"""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics for all GPUs with enhanced real-time data"""
        # Debounce rapid updates
        current_time = time.time()
        if current_time - self.gpu_update_debounce < 0.5:  # 500ms debounce
            return self.last_gpu_data if self.last_gpu_data else self._get_empty_metrics()
        
        if self.is_loading_gpu_data:
            return self.last_gpu_data if self.last_gpu_data else self._get_empty_metrics()
        
        self.is_loading_gpu_data = True
        self.gpu_update_debounce = current_time
        
        try:
            # Update GPU data
            self._update_gpu_metrics()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "gpus": {},
                "summary": {
                    "total_gpus": len(self.gpus),
                    "available_gpus": sum(1 for gpu in self.gpus.values() if gpu.is_available),
                    "total_memory": sum(gpu.memory_total for gpu in self.gpus.values()),
                    "used_memory": sum(gpu.memory_used for gpu in self.gpus.values()),
                    "average_utilization": sum(gpu.utilization for gpu in self.gpus.values()) / len(self.gpus) if self.gpus else 0,
                    "average_temperature": sum(gpu.temperature for gpu in self.gpus.values()) / len(self.gpus) if self.gpus else 0,
                    "total_power": sum(gpu.power_usage for gpu in self.gpus.values())
                }
            }
            
            for gpu_id, gpu in self.gpus.items():
                gpu_data = {
                    "name": gpu.name,
                    "memory_total": gpu.memory_total,
                    "memory_used": gpu.memory_used,
                    "memory_free": gpu.memory_free,
                    "memory_percent": (gpu.memory_used / gpu.memory_total * 100) if gpu.memory_total > 0 else 0,
                    "utilization": gpu.utilization,
                    "temperature": gpu.temperature,
                    "power_usage": gpu.power_usage,
                    "power_limit": gpu.power_limit,
                    "power_percent": (gpu.power_usage / gpu.power_limit * 100) if gpu.power_limit > 0 else 0,
                    "is_available": gpu.is_available,
                    "current_model": gpu.current_model,
                    "last_updated": gpu.last_updated.isoformat() if gpu.last_updated else None
                }
                metrics["gpus"][str(gpu_id)] = gpu_data
                
                # Store historical data for charts
                if str(gpu_id) not in self.metrics_history:
                    self.metrics_history[str(gpu_id)] = []
                
                self.metrics_history[str(gpu_id)].append({
                    "timestamp": current_time,
                    "utilization": gpu.utilization,
                    "memory_percent": gpu_data["memory_percent"],
                    "temperature": gpu.temperature,
                    "power_percent": gpu_data["power_percent"]
                })
                
                # Keep only last 60 data points (2 minutes at 2s intervals)
                if len(self.metrics_history[str(gpu_id)]) > 60:
                    self.metrics_history[str(gpu_id)] = self.metrics_history[str(gpu_id)][-60:]
            
            # Add historical data for charts
            metrics["history"] = self.metrics_history
            
            self.last_gpu_data = metrics
            
            # Notify callbacks
            self._notify_callbacks(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting GPU metrics: {e}")
            return self._get_empty_metrics()
        finally:
            self.is_loading_gpu_data = False
    
    def _get_empty_metrics(self):
        """Return empty metrics structure"""
        return {
            "timestamp": datetime.now().isoformat(),
            "gpus": {},
            "summary": {
                "total_gpus": 0,
                "available_gpus": 0,
                "total_memory": 0,
                "used_memory": 0,
                "average_utilization": 0,
                "average_temperature": 0,
                "total_power": 0
            },
            "history": {}
        }
    
    def get_best_gpu(self, memory_required: int = 0) -> Optional[int]:
        """Get the best GPU for a task based on available memory and utilization"""
        available_gpus = [
            (idx, gpu) for idx, gpu in self.gpus.items() 
            if gpu.is_available and gpu.memory_free >= memory_required
        ]
        
        if not available_gpus:
            return None
        
        # Sort by utilization (lower is better), then by free memory (higher is better)
        best_gpu = min(available_gpus, key=lambda x: (x[1].utilization, -x[1].memory_free))
        return best_gpu[0]
    
    def reserve_gpu(self, gpu_index: int, model_name: str, memory_mb: int) -> bool:
        """Reserve GPU for a model"""
        if gpu_index not in self.gpus:
            return False
        
        gpu = self.gpus[gpu_index]
        if not gpu.is_available or gpu.memory_free < memory_mb:
            return False
        
        gpu.current_model = model_name
        gpu.memory_used += memory_mb
        gpu.memory_free -= memory_mb
        
        self.logger.info(f"Reserved GPU {gpu_index} for model {model_name} ({memory_mb}MB)")
        return True
    
    def release_gpu(self, gpu_index: int, memory_mb: int) -> bool:
        """Release GPU resources"""
        if gpu_index not in self.gpus:
            return False
        
        gpu = self.gpus[gpu_index]
        gpu.current_model = None
        gpu.memory_used = max(0, gpu.memory_used - memory_mb)
        gpu.memory_free = min(gpu.memory_total, gpu.memory_free + memory_mb)
        
        self.logger.info(f"Released GPU {gpu_index} resources ({memory_mb}MB)")
        return True
    
    def get_gpu_health(self) -> Dict[str, Any]:
        """Get overall GPU health status"""
        if not self.gpus:
            return {"status": "no_gpus", "message": "No GPUs detected"}
        
        health_issues = []
        warning_issues = []
        
        for gpu_index, gpu in self.gpus.items():
            # Check temperature
            if gpu.temperature > 85:
                health_issues.append(f"GPU {gpu_index} temperature critical: {gpu.temperature}°C")
            elif gpu.temperature > 75:
                warning_issues.append(f"GPU {gpu_index} temperature high: {gpu.temperature}°C")
            
            # Check memory usage
            memory_usage_percent = (gpu.memory_used / gpu.memory_total * 100) if gpu.memory_total > 0 else 0
            if memory_usage_percent > 95:
                health_issues.append(f"GPU {gpu_index} memory critical: {memory_usage_percent:.1f}%")
            elif memory_usage_percent > 85:
                warning_issues.append(f"GPU {gpu_index} memory high: {memory_usage_percent:.1f}%")
            
            # Check power usage
            if gpu.power_limit > 0:
                power_usage_percent = (gpu.power_usage / gpu.power_limit * 100)
                if power_usage_percent > 95:
                    warning_issues.append(f"GPU {gpu_index} power usage high: {power_usage_percent:.1f}%")
        
        if health_issues:
            status = "critical"
        elif warning_issues:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "health_issues": health_issues,
            "warning_issues": warning_issues,
            "gpu_count": len(self.gpus),
            "monitoring_active": self.monitoring_active,
            "last_check": datetime.now().isoformat()
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_monitoring()
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass