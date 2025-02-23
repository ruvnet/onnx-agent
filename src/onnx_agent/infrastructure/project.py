import os
from pathlib import Path
from typing import Dict, Any, Optional

class ProjectStructure:
    """Manages project directory structure and paths."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root_dir = Path(os.getenv("PROJECT_ROOT", os.getcwd()))
        self.model_dir = self.root_dir / "models"
        self.config_dir = self.root_dir / "config"
        self.data_dir = self.root_dir / "data"
        self.cache_dir = self.root_dir / "cache"
        self.export_dir = self.root_dir / "exports"
        
    def create_directories(self) -> None:
        """Create all required project directories."""
        dirs = [
            self.model_dir,
            self.config_dir,
            self.data_dir,
            self.cache_dir,
            self.export_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a model file."""
        return self.model_dir / model_name
        
    def get_config_path(self, config_name: str) -> Path:
        """Get path for a configuration file."""
        return self.config_dir / config_name
        
    def get_data_path(self, data_name: str) -> Path:
        """Get path for a data file."""
        return self.data_dir / data_name
        
    def get_export_path(self, model_name: str) -> Path:
        """Get path for an exported model."""
        return self.export_dir / f"{model_name}.onnx"
        
    def validate_structure(self) -> bool:
        """Validate that all required directories exist."""
        return all(
            dir_path.exists() and dir_path.is_dir()
            for dir_path in [
                self.model_dir,
                self.config_dir,
                self.data_dir,
                self.cache_dir,
                self.export_dir
            ]
        )
