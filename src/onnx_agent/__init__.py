"""ONNX Agent for model optimization and deployment."""

from .infrastructure.project import ProjectStructure
from .infrastructure.config import ConfigLoader
from .infrastructure.export import ONNXExporter
from .infrastructure.dspy_integration import DSPyIntegration

__version__ = "0.1.0"

__all__ = [
    "ProjectStructure",
    "ConfigLoader", 
    "ONNXExporter",
    "DSPyIntegration"
]
