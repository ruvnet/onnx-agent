"""Tests for CLI main entry point."""

import os
import sys
import unittest
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from onnx_agent.cli.main import main, create_parser
from ..pipeline.base_tests import BasePipelineTest

class TestMain(BasePipelineTest):
    """Test CLI main entry point."""
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        self.test_data = self.create_synthetic_dataset(100)
        
    def test_argument_parsing(self):
        """Test command line argument parsing."""
        parser = create_parser()
        
        # Test train command
        args = parser.parse_args([
            "train",
            "data.pt",
            "--batch-size", "64",
            "--epochs", "5"
        ])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.data, "data.pt")
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.epochs, 5)
        
        # Test export command
        args = parser.parse_args([
            "export",
            "model.pt",
            "model.onnx",
            "--input-names", "input1", "input2"
        ])
        self.assertEqual(args.command, "export")
        self.assertEqual(args.model, "model.pt")
        self.assertEqual(args.output, "model.onnx")
        self.assertEqual(args.input_names, ["input1", "input2"])
        
    def test_config_loading(self):
        """Test configuration file loading."""
        config_path = self.data_dir / "test_config.yaml"
        config_path.write_text("""
        batch_size: 128
        epochs: 10
        learning_rate: 0.01
        """)
        
        with patch("sys.argv", ["prog", "--config", str(config_path), "train", "data.pt"]):
            with patch("onnx_agent.cli.interface.CommandLineInterface") as mock_cli:
                main()
                
                # Verify config was loaded and merged with args
                args = mock_cli.return_value.run_command.call_args[0][1]
                self.assertEqual(args["batch_size"], 128)
                self.assertEqual(args["epochs"], 10)
                self.assertEqual(args["learning_rate"], 0.01)
                
    def test_logging_setup(self):
        """Test logging configuration."""
        with patch("sys.argv", ["prog", "--log-level", "debug", "train", "data.pt"]):
            with patch("onnx_agent.cli.main.setup_logger") as mock_setup:
                main()
                mock_setup.assert_called_once_with("onnx_agent", "debug")
                
    def test_command_execution(self):
        """Test command execution flow."""
        with patch("sys.argv", ["prog", "train", "data.pt"]):
            with patch("onnx_agent.cli.interface.CommandLineInterface") as mock_cli:
                instance = mock_cli.return_value
                instance.run_command.return_value = {
                    "status": "success",
                    "model": "model.pt"
                }
                
                exit_code = main()
                
                # Verify command was executed
                instance.run_command.assert_called_once()
                self.assertEqual(exit_code, 0)
                
    def test_error_handling(self):
        """Test error handling."""
        with patch("sys.argv", ["prog", "train", "data.pt"]):
            with patch("onnx_agent.cli.interface.CommandLineInterface") as mock_cli:
                instance = mock_cli.return_value
                instance.run_command.side_effect = ValueError("Invalid data")
                
                exit_code = main()
                
                # Verify error was handled
                self.assertEqual(exit_code, 1)
                
    def test_quiet_mode(self):
        """Test quiet mode output suppression."""
        with patch("sys.argv", ["prog", "--quiet", "train", "data.pt"]):
            with patch("onnx_agent.cli.interface.CommandLineInterface") as mock_cli:
                with patch("logging.Logger.info") as mock_info:
                    instance = mock_cli.return_value
                    instance.run_command.return_value = {"status": "success"}
                    
                    main()
                    
                    # Verify no output was logged
                    mock_info.assert_not_called()
                    
    def test_help_display(self):
        """Test help text display."""
        parser = create_parser()
        
        # Test main help
        with patch("sys.stdout") as mock_stdout:
            with self.assertRaises(SystemExit):
                parser.parse_args(["--help"])
                
        # Test command help
        with patch("sys.stdout") as mock_stdout:
            with self.assertRaises(SystemExit):
                parser.parse_args(["train", "--help"])
                
    def test_command_validation(self):
        """Test command validation."""
        parser = create_parser()
        
        # Test invalid command
        with self.assertRaises(SystemExit):
            parser.parse_args(["invalid"])
            
        # Test missing required args
        with self.assertRaises(SystemExit):
            parser.parse_args(["train"])
            
    def test_global_options(self):
        """Test global option handling."""
        parser = create_parser()
        
        args = parser.parse_args([
            "--config", "config.yaml",
            "--log-level", "debug",
            "--quiet",
            "--color",
            "train",
            "data.pt"
        ])
        
        self.assertEqual(args.config, "config.yaml")
        self.assertEqual(args.log_level, "debug")
        self.assertTrue(args.quiet)
        self.assertTrue(args.color)
        
    def test_command_specific_options(self):
        """Test command-specific option handling."""
        parser = create_parser()
        
        # Test train command options
        args = parser.parse_args([
            "train",
            "data.pt",
            "--val-data", "val.pt",
            "--batch-size", "64",
            "--epochs", "5",
            "--learning-rate", "0.01",
            "--checkpoint-path", "checkpoints",
            "--early-stopping",
            "--patience", "3",
            "--mixed-precision",
            "--distributed",
            "--num-gpus", "2"
        ])
        
        self.assertEqual(args.val_data, "val.pt")
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.epochs, 5)
        self.assertEqual(args.learning_rate, 0.01)
        self.assertEqual(args.checkpoint_path, "checkpoints")
        self.assertTrue(args.early_stopping)
        self.assertEqual(args.patience, 3)
        self.assertTrue(args.mixed_precision)
        self.assertTrue(args.distributed)
        self.assertEqual(args.num_gpus, 2)
