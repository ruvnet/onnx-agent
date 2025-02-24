"""CLI command validation tests."""

import os
import unittest
import pytest
from pathlib import Path

from onnx_agent.cli import CommandLineInterface
from tests.integration.pipeline.base_tests import BasePipelineTest

class TestCommandValidation(BasePipelineTest):
    """Test CLI command validation and error handling."""
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        self.cli = CommandLineInterface()
        
    def test_argument_parsing(self):
        """Test argument parsing and validation."""
        # Test missing required arguments
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("train", {})
        self.assertIn("data", str(ctx.exception))
        
        # Test invalid argument type
        with self.assertRaises(TypeError) as ctx:
            self.cli.run_command("train", {
                "data": self.create_synthetic_dataset(100),
                "batch_size": "invalid"  # Should be int
            })
        self.assertIn("batch_size", str(ctx.exception))
        
        # Test invalid argument value
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("train", {
                "data": self.create_synthetic_dataset(100),
                "batch_size": -1  # Should be positive
            })
        self.assertIn("batch_size", str(ctx.exception))
        
    def test_configuration_loading(self):
        """Test configuration file loading and validation."""
        config_path = self.data_dir / "test_config.yaml"
        
        # Test missing config file
        with self.assertRaises(FileNotFoundError) as ctx:
            self.cli.run_command("train", {
                "config": str(config_path)
            })
            
        # Test invalid config format
        config_path.write_text("invalid: yaml: content")
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("train", {
                "config": str(config_path)
            })
            
        # Test valid config
        config_path.write_text("""
        model_type: classifier
        batch_size: 32
        epochs: 2
        learning_rate: 1e-4
        """)
        result = self.cli.run_command("train", {
            "data": self.create_synthetic_dataset(100),
            "config": str(config_path)
        })
        self.assertIsNotNone(result)
        
        # Clean up
        config_path.unlink()
        
    def test_path_resolution(self):
        """Test path resolution and validation."""
        # Test relative paths
        result = self.cli.run_command("train", {
            "data": self.create_synthetic_dataset(100),
            "checkpoint_path": "checkpoint.pt"
        })
        self.assertTrue(Path("checkpoint.pt").exists())
        Path("checkpoint.pt").unlink()
        
        # Test absolute paths
        abs_path = self.data_dir / "model.pt"
        result = self.cli.run_command("train", {
            "data": self.create_synthetic_dataset(100),
            "checkpoint_path": str(abs_path)
        })
        self.assertTrue(abs_path.exists())
        abs_path.unlink()
        
        # Test invalid paths
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("export", {
                "model": "/invalid/path/model.pt",
                "output": "model.onnx"
            })
            
    def test_parameter_dependencies(self):
        """Test parameter dependency validation."""
        # Test mutually exclusive parameters
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("optimize", {
                "model": "model.onnx",
                "quantization": "dynamic",
                "no_quantization": True
            })
            
        # Test dependent parameters
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("optimize", {
                "model": "model.onnx",
                "quantization": "static",
                # Missing calibration_data for static quantization
            })
            
        # Test conditional requirements
        with self.assertRaises(ValueError) as ctx:
            self.cli.run_command("train", {
                "data": self.create_synthetic_dataset(100),
                "distributed": True,
                # Missing num_gpus for distributed training
            })
            
    def test_help_documentation(self):
        """Test help documentation generation."""
        # Test main help
        help_text = self.cli.get_help()
        self.assertIsInstance(help_text, str)
        self.assertIn("Usage", help_text)
        self.assertIn("Commands", help_text)
        
        # Test command-specific help
        train_help = self.cli.get_help("train")
        self.assertIn("train", train_help)
        self.assertIn("Arguments", train_help)
        
        optimize_help = self.cli.get_help("optimize")
        self.assertIn("optimize", optimize_help)
        self.assertIn("Arguments", optimize_help)
        
        # Test invalid command help
        with self.assertRaises(ValueError):
            self.cli.get_help("invalid_command")
