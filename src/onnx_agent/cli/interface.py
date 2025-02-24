"""Command Line Interface implementation."""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm

class CommandLineInterface:
    """Main CLI interface for ONNX Agent."""
    
    def __init__(self):
        """Initialize CLI interface."""
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging system."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def run_command(self, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a CLI command with arguments.
        
        Args:
            command: Command name to execute
            args: Dictionary of command arguments
            
        Returns:
            Dictionary containing command results
            
        Raises:
            ValueError: If command or arguments are invalid
            Exception: If command execution fails
        """
        # Validate command
        if command not in self.get_available_commands():
            raise ValueError(f"Unknown command: {command}")
            
        # Load configuration if specified
        if "config" in args:
            config = self.load_config(args["config"])
            args = {**config, **args}  # Args override config
            
        # Validate arguments
        self.validate_arguments(command, args)
        
        # Setup progress reporting
        show_progress = args.get("show_progress", True)
        if show_progress:
            self.setup_progress_bar()
            
        try:
            # Execute command
            if command == "train":
                result = self.handle_train(args)
            elif command == "export":
                result = self.handle_export(args)
            elif command == "optimize":
                result = self.handle_optimize(args)
            elif command == "infer":
                result = self.handle_infer(args)
            elif command == "evaluate":
                result = self.handle_evaluate(args)
            elif command == "compare":
                result = self.handle_compare(args)
            else:
                raise ValueError(f"Command not implemented: {command}")
                
            # Add status to result
            result["status"] = "success"
            
            # Save results if requested
            if "save_results" in args:
                self.save_results(result, args["save_results"])
                
            return result
            
        except Exception as e:
            # Add error details
            error = type(e)("Error executing command: " + str(e))
            error.error_code = getattr(e, "error_code", "UNKNOWN_ERROR")
            error.details = getattr(e, "details", {})
            raise error
            
        finally:
            # Clean up progress bar
            if show_progress:
                self.cleanup_progress_bar()
                
    def get_available_commands(self) -> List[str]:
        """Get list of available commands."""
        return [
            "train",
            "export", 
            "optimize",
            "infer",
            "evaluate",
            "compare"
        ]
        
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config: {e}")
            
    def validate_arguments(self, command: str, args: Dict[str, Any]):
        """Validate command arguments."""
        # Common validation
        if "batch_size" in args and args["batch_size"] <= 0:
            raise ValueError("batch_size must be positive")
            
        # Command-specific validation
        if command == "train":
            if "data" not in args:
                raise ValueError("data is required for training")
        elif command == "export":
            if "model" not in args:
                raise ValueError("model is required for export")
        elif command == "optimize":
            if "model" not in args:
                raise ValueError("model is required for optimization")
            if args.get("quantization") == "static" and "calibration_data" not in args:
                raise ValueError("calibration_data required for static quantization")
        elif command == "infer":
            if "model" not in args or "data" not in args:
                raise ValueError("model and data required for inference")
                
    def setup_progress_bar(self):
        """Configure progress bar."""
        self.pbar = None
        
    def update_progress(self, progress: float, desc: str = ""):
        """Update progress bar."""
        if self.pbar is None:
            self.pbar = tqdm(total=100, desc=desc)
        self.pbar.n = int(progress * 100)
        self.pbar.refresh()
        
    def cleanup_progress_bar(self):
        """Clean up progress bar."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
            
    def save_results(self, results: Dict[str, Any], path: Union[str, Path]):
        """Save results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
            
    def get_help(self, command: Optional[str] = None) -> str:
        """Get help documentation."""
        if command is None:
            return self._get_main_help()
        elif command in self.get_available_commands():
            return self._get_command_help(command)
        else:
            raise ValueError(f"Unknown command: {command}")
            
    def _get_main_help(self) -> str:
        """Get main help documentation."""
        return """
        ONNX Agent CLI

        Usage:
            onnx-agent <command> [options]

        Commands:
            train     Train a model
            export    Export model to ONNX format
            optimize  Optimize ONNX model
            infer     Run inference
            evaluate  Evaluate model performance
            compare   Compare models

        Use 'onnx-agent help <command>' for command-specific help.
        """
        
    def _get_command_help(self, command: str) -> str:
        """Get command-specific help documentation."""
        help_text = {
            "train": """
            train: Train a model

            Arguments:
                data            Training data
                val_data       Validation data (optional)
                batch_size     Batch size (default: 32)
                epochs         Number of epochs (default: 10)
                learning_rate  Learning rate (default: 0.001)
            """,
            "export": """
            export: Export model to ONNX format

            Arguments:
                model    Model to export
                output   Output path
            """,
            "optimize": """
            optimize: Optimize ONNX model

            Arguments:
                model            Model to optimize
                quantization     Quantization mode (dynamic/static)
                calibration_data Calibration data (required for static)
                provider        Execution provider
            """,
            "infer": """
            infer: Run inference

            Arguments:
                model       Model to use
                data       Input data
                batch_size Batch size (default: 32)
            """
        }
        return help_text.get(command, "Help not available")
        
    # Command handlers
    
    def handle_train(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle train command."""
        raise NotImplementedError
        
    def handle_export(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle export command."""
        raise NotImplementedError
        
    def handle_optimize(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle optimize command."""
        raise NotImplementedError
        
    def handle_infer(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle infer command."""
        raise NotImplementedError
        
    def handle_evaluate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle evaluate command."""
        raise NotImplementedError
        
    def handle_compare(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compare command."""
        raise NotImplementedError
