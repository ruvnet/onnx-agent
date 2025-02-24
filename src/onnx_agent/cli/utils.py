"""Utility functions for CLI operations."""

import os
import sys
import time
import psutil
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # Add handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"

def format_time(seconds: float) -> str:
    """Format time duration to human readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds:.2f}s")
    
    return " ".join(parts)

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information and statistics.
    
    Returns:
        Dictionary containing GPU information
    """
    if not torch.cuda.is_available():
        return {}
        
    info = {
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "devices": []
    }
    
    for i in range(info["device_count"]):
        device_info = {
            "name": torch.cuda.get_device_name(i),
            "memory_allocated": format_size(torch.cuda.memory_allocated(i)),
            "memory_cached": format_size(torch.cuda.memory_cached(i)),
            "max_memory_allocated": format_size(torch.cuda.max_memory_allocated(i))
        }
        info["devices"].append(device_info)
        
    return info

def get_system_info() -> Dict[str, Any]:
    """Get system information and statistics.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "cpu_count": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=1, percpu=True),
        "memory": {
            "total": format_size(psutil.virtual_memory().total),
            "available": format_size(psutil.virtual_memory().available),
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total": format_size(psutil.disk_usage('/').total),
            "free": format_size(psutil.disk_usage('/').free),
            "percent": psutil.disk_usage('/').percent
        }
    }
    
    # Add GPU info if available
    gpu_info = get_gpu_info()
    if gpu_info:
        info["gpu"] = gpu_info
        
    return info

def measure_time(func: callable) -> Tuple[Any, float]:
    """Measure execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Tuple of (function result, execution time in seconds)
    """
    start_time = time.time()
    result = func()
    end_time = time.time()
    return result, end_time - start_time

def format_table(data: List[Dict[str, Any]], columns: List[str]) -> str:
    """Format data as ASCII table.
    
    Args:
        data: List of dictionaries containing data
        columns: List of column names to include
        
    Returns:
        Formatted table string
    """
    if not data:
        return ""
        
    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in data:
        for col in columns:
            if col in row:
                widths[col] = max(widths[col], len(str(row[col])))
                
    # Create format string
    format_str = "| " + " | ".join(f"{{:{widths[col]}}}" for col in columns) + " |"
    
    # Create separator line
    separator = "+" + "+".join("-" * (widths[col] + 2) for col in columns) + "+"
    
    # Format table
    lines = [separator]
    lines.append(format_str.format(*columns))
    lines.append(separator)
    
    for row in data:
        values = [str(row.get(col, "")) for col in columns]
        lines.append(format_str.format(*values))
        
    lines.append(separator)
    return "\n".join(lines)

def format_json(data: Any, indent: int = 2) -> str:
    """Format data as JSON string.
    
    Args:
        data: Data to format
        indent: Number of spaces for indentation
        
    Returns:
        Formatted JSON string
    """
    import json
    
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            return super().default(obj)
            
    return json.dumps(data, indent=indent, cls=CustomEncoder)

def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
    """Validate and normalize file path.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        
    Returns:
        Normalized Path object
        
    Raises:
        ValueError: If path validation fails
    """
    path = Path(path).resolve()
    
    if must_exist and not path.exists():
        raise ValueError(f"Path does not exist: {path}")
        
    return path

def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
