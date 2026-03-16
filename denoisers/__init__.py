import os
import importlib
import inspect
from .base import BaseDenoiser

def discover_denoisers():
    """Dynamically discover all denoiser plugins in the current directory."""
    denoisers = {}
    current_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
            module_name = f".{filename[:-3]}"
            module = importlib.import_module(module_name, package=__package__)
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseDenoiser) and 
                    obj is not BaseDenoiser):
                    
                    instance = obj()
                    denoisers[instance.key] = instance
                    
    return denoisers
