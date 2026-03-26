"""GPU utilities for gradient boosting model analysis."""

import subprocess
import sys
from typing import Dict, Any


def is_gpu_available() -> bool:
    """Check if GPU is available for acceleration.
    
    Returns:
        True if GPU is available, False otherwise
    """
    try:
        # Try to import cupy or check for CUDA
        import cupy as cp
        return cp.cuda.is_available()
    except ImportError:
        try:
            # Try torch
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Try tensorflow
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                # Fallback: check nvidia-smi
                try:
                    result = subprocess.run(
                        ['nvidia-smi'], 
                        capture_output=True, 
                        text=True, 
                        timeout=5
                    )
                    return result.returncode == 0
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    return False


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPU.
    
    Returns:
        Dictionary with GPU information
    """
    info = {
        "available": False,
        "device_count": 0,
        "device_name": None,
        "driver_version": None,
        "cuda_version": None
    }
    
    try:
        import cupy as cp
        if cp.cuda.is_available():
            info["available"] = True
            info["device_count"] = cp.cuda.runtime.getDeviceCount()
            # Get device properties
            if info["device_count"] > 0:
                props = cp.cuda.runtime.getDeviceProperties(0)
                info["device_name"] = props["name"].decode('utf-8')
    except ImportError:
        pass
        
    return info