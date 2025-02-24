"""Main entry point for CLI application."""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from .interface import CommandLineInterface
from .utils import setup_logger

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="ONNX Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Enable colored output"
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("data", help="Training data")
    train_parser.add_argument("--val-data", help="Validation data")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--checkpoint-path", help="Path to save checkpoints")
    train_parser.add_argument("--early-stopping", action="store_true")
    train_parser.add_argument("--patience", type=int, default=5)
    train_parser.add_argument("--mixed-precision", action="store_true")
    train_parser.add_argument("--distributed", action="store_true")
    train_parser.add_argument("--num-gpus", type=int, default=1)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("model", help="Model to export")
    export_parser.add_argument("output", help="Output path")
    export_parser.add_argument("--dynamic-axes", help="Dynamic axes configuration")
    export_parser.add_argument("--input-names", nargs="+", default=["input"])
    export_parser.add_argument("--output-names", nargs="+", default=["output"])
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize ONNX model")
    optimize_parser.add_argument("model", help="Model to optimize")
    optimize_parser.add_argument("output", help="Output path")
    optimize_parser.add_argument(
        "--quantization",
        choices=["dynamic", "static"],
        help="Quantization mode"
    )
    optimize_parser.add_argument(
        "--calibration-data",
        help="Data for static quantization"
    )
    optimize_parser.add_argument(
        "--provider",
        default="CPUExecutionProvider",
        help="Execution provider"
    )
    
    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("model", help="Model to use")
    infer_parser.add_argument("data", help="Input data")
    infer_parser.add_argument("--batch-size", type=int, default=32)
    infer_parser.add_argument("--collect-metrics", action="store_true")
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    evaluate_parser.add_argument("model", help="Model to evaluate")
    evaluate_parser.add_argument("data", help="Evaluation data")
    evaluate_parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy"],
        help="Metrics to compute"
    )
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare models")
    compare_parser.add_argument(
        "models",
        nargs="+",
        help="Models to compare"
    )
    compare_parser.add_argument("data", help="Test data")
    compare_parser.add_argument(
        "--metrics",
        nargs="+",
        default=["accuracy"],
        help="Metrics to compare"
    )
    
    return parser

def main(args: Optional[List[str]] = None) -> int:
    """Main entry point.
    
    Args:
        args: Command line arguments (uses sys.argv if None)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args(args)
    
    # Set up logging
    logger = setup_logger("onnx_agent", args.log_level)
    
    try:
        # Create CLI interface
        cli = CommandLineInterface()
        
        # Load config if specified
        config = {}
        if args.config:
            config = cli.load_config(args.config)
            
        # Convert args to dict
        command_args = vars(args)
        command = command_args.pop("command")
        
        # Remove global options
        for key in ["config", "log_level", "quiet", "color"]:
            command_args.pop(key, None)
            
        # Add config values (command args take precedence)
        command_args = {**config, **command_args}
        
        # Execute command
        result = cli.run_command(command, command_args)
        
        # Display results
        if not args.quiet:
            if isinstance(result, dict) and "status" in result:
                logger.info(f"Command completed successfully: {result['status']}")
            else:
                logger.info("Command completed successfully")
                
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if logger.level <= logging.DEBUG:
            logger.exception(e)
        return 1

if __name__ == "__main__":
    sys.exit(main())
