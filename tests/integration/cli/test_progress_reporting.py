"""CLI progress reporting tests."""

import os
import sys
import unittest
import pytest
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

from onnx_agent.cli import CommandLineInterface
from tests.integration.pipeline.base_tests import BasePipelineTest

class TestProgressReporting(BasePipelineTest):
    """Test CLI progress reporting and output handling."""
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        self.cli = CommandLineInterface()
        self.test_data = self.create_synthetic_dataset(100)
        
    def capture_output(self, func, *args, **kwargs):
        """Capture stdout and stderr output."""
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = func(*args, **kwargs)
        return result, stdout.getvalue(), stderr.getvalue()
        
    def test_progress_bar_updates(self):
        """Test progress bar functionality."""
        # Test training progress
        _, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "train",
            {
                "data": self.test_data,
                "epochs": 2,
                "batch_size": 10,
                "show_progress": True
            }
        )
        
        # Verify progress indicators
        self.assertIn("Training", stdout)
        self.assertIn("Epoch", stdout)
        self.assertIn("Progress", stdout)
        self.assertIn("%", stdout)
        
        # Verify completion message
        self.assertIn("Training complete", stdout)
        
    def test_status_messages(self):
        """Test status message display."""
        # Test optimization status
        _, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "optimize",
            {
                "model": "model.onnx",
                "quantization": "dynamic",
                "show_status": True
            }
        )
        
        # Verify status updates
        self.assertIn("Loading model", stdout)
        self.assertIn("Optimizing", stdout)
        self.assertIn("Quantizing", stdout)
        self.assertIn("Saving", stdout)
        
    def test_error_messages(self):
        """Test error message formatting."""
        # Test invalid model path
        with self.assertRaises(Exception) as ctx:
            _, stdout, stderr = self.capture_output(
                self.cli.run_command,
                "infer",
                {"model": "nonexistent.onnx"}
            )
            
        error_msg = str(ctx.exception)
        self.assertIn("Error", error_msg)
        self.assertIn("nonexistent.onnx", error_msg)
        
    def test_warning_handling(self):
        """Test warning message handling."""
        # Test deprecated parameter warning
        _, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "train",
            {
                "data": self.test_data,
                "old_param": True  # Deprecated parameter
            }
        )
        
        self.assertIn("Warning", stderr)
        self.assertIn("deprecated", stderr)
        
    def test_log_levels(self):
        """Test different logging levels."""
        # Test debug output
        _, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "train",
            {
                "data": self.test_data,
                "log_level": "debug"
            }
        )
        
        self.assertIn("DEBUG", stderr)
        
        # Test quiet mode
        _, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "train",
            {
                "data": self.test_data,
                "quiet": True
            }
        )
        
        self.assertEqual("", stdout)
        self.assertEqual("", stderr)
        
    def test_output_formatting(self):
        """Test output formatting options."""
        # Test JSON output
        result, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "infer",
            {
                "model": "model.onnx",
                "data": self.test_data,
                "format": "json"
            }
        )
        
        self.assertIn("{", stdout)
        self.assertIn("}", stdout)
        
        # Test table output
        result, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "infer",
            {
                "model": "model.onnx",
                "data": self.test_data,
                "format": "table"
            }
        )
        
        self.assertIn("|", stdout)
        self.assertIn("-", stdout)
        
    def test_color_support(self):
        """Test colored output support."""
        # Test with color enabled
        _, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "train",
            {
                "data": self.test_data,
                "color": True
            }
        )
        
        self.assertIn("\033[", stdout)  # ANSI color codes
        
        # Test with color disabled
        _, stdout, stderr = self.capture_output(
            self.cli.run_command,
            "train",
            {
                "data": self.test_data,
                "color": False
            }
        )
        
        self.assertNotIn("\033[", stdout)
        
    def test_interactive_mode(self):
        """Test interactive mode features."""
        # Mock interactive input
        input_values = ["y", "n"]
        def mock_input(prompt):
            return input_values.pop(0)
            
        with unittest.mock.patch("builtins.input", mock_input):
            _, stdout, stderr = self.capture_output(
                self.cli.run_command,
                "train",
                {
                    "data": self.test_data,
                    "interactive": True
                }
            )
            
        self.assertIn("Continue?", stdout)
        self.assertIn("Proceed?", stdout)
