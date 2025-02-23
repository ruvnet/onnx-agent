import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from dotenv import load_dotenv

class ConfigLoader:
    """Handles loading and merging of configuration from files and environment."""
    
    def __init__(self, env_prefix: str = "ONNX_AGENT_"):
        self.env_prefix = env_prefix
        load_dotenv()
        
    def load(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with path.open("r") as f:
            config = yaml.safe_load(f)
            
        return self._process_config(config)
        
    def load_with_env(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file and override with environment variables."""
        # Start with empty or loaded config
        config = self._process_config({}) if not config_path else self.load(config_path)
        
        # Create a new dict for environment overrides
        env_overrides = {}
        
        # Process environment variables
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                config_key = key[len(self.env_prefix):].lower()
                parts = config_key.split("_")
                
                # Convert the value before storing
                converted_value = self._convert_value(value)
                
                # Handle special case for model_config
                if len(parts) >= 2 and parts[0] == "model" and parts[1] == "config":
                    if "model_config" not in env_overrides:
                        env_overrides["model_config"] = {}
                    if len(parts) > 2:
                        env_overrides["model_config"][parts[2]] = converted_value
                else:
                    # Regular key handling - store directly in env_overrides
                    env_overrides[config_key] = converted_value
        
        # Update config with environment overrides
        config.update(env_overrides)
        return config
        
    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate configuration."""
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
            
        # Add any default values
        config.setdefault("environment", "development")
        config.setdefault("log_level", "INFO")
        config.setdefault("model_config", {})
        
        return config
        
    def _convert_value(self, value: str) -> Any:
        """Convert string values to appropriate Python types."""
        # Handle boolean values
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value.lower() == "null":
            return None
            
        # Handle numeric values
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            # If not a number, return as string
            return value
