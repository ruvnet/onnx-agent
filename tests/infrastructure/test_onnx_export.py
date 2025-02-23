import os
import pytest
import torch
import onnx
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

from onnx_agent.infrastructure.export import ONNXExporter

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

def test_basic_export(temp_dir, export_config):
    """Test basic model export functionality."""
    model = SimpleModel()
    exporter = ONNXExporter(export_config)
    
    # Create sample input
    sample_input = torch.randn(1, 10)
    export_path = Path(temp_dir) / "model.onnx"
    
    # Export model
    onnx_model = exporter.export(model, sample_input, export_path)
    
    # Verify export
    assert export_path.exists()
    assert isinstance(onnx_model, onnx.ModelProto)

def test_dynamic_axes(temp_dir, export_config):
    """Test export with dynamic axes configuration."""
    model = SimpleModel()
    
    # Configure dynamic batch size
    export_config["dynamic_axes"] = {"input": {0: "batch_size"}}
    exporter = ONNXExporter(export_config)
    
    # Test with different batch sizes
    for batch_size in [1, 4, 8]:
        sample_input = torch.randn(batch_size, 10)
        export_path = Path(temp_dir) / f"model_batch_{batch_size}.onnx"
        
        onnx_model = exporter.export(model, sample_input, export_path)
        assert export_path.exists()

def test_model_validation(temp_dir, export_config):
    """Test model validation during export."""
    model = SimpleModel()
    exporter = ONNXExporter(export_config)
    
    # Export valid model
    sample_input = torch.randn(1, 10)
    export_path = Path(temp_dir) / "valid_model.onnx"
    onnx_model = exporter.export(model, sample_input, export_path)
    
    # Validate should not raise error
    exporter.validate_model(onnx_model)

def test_default_input_handling(temp_dir, export_config):
    """Test handling of default input when none provided."""
    model = SimpleModel()
    export_config["input_shape"] = [1, 10]  # Match model input shape
    exporter = ONNXExporter(export_config)
    
    export_path = Path(temp_dir) / "default_input_model.onnx"
    onnx_model = exporter.export(model, path=export_path)
    
    assert export_path.exists()
    assert isinstance(onnx_model, onnx.ModelProto)

def test_model_metadata(temp_dir, export_config):
    """Test extraction of model metadata."""
    model = SimpleModel()
    exporter = ONNXExporter(export_config)
    
    sample_input = torch.randn(1, 10)
    export_path = Path(temp_dir) / "metadata_model.onnx"
    onnx_model = exporter.export(model, sample_input, export_path)
    
    metadata = exporter.get_model_metadata(onnx_model)
    
    assert metadata["opset_version"] == export_config["opset_version"]
    assert "input" in metadata["input_names"]
    assert "output" in metadata["output_names"]
    assert metadata["num_nodes"] > 0

@patch('onnx.checker.check_model')
@patch('onnx.load')
@patch('onnx.save')
@patch('onnxruntime.InferenceSession')
@patch('onnxruntime.SessionOptions')
def test_model_optimization(mock_session_options, mock_inference_session, mock_save, mock_load, mock_check_model, temp_dir, export_config):
    """Test model optimization functionality."""
    model = SimpleModel()
    exporter = ONNXExporter(export_config)
    
    # Export basic model
    sample_input = torch.randn(1, 10)
    export_path = Path(temp_dir) / "optimize_model.onnx"
    
    # Mock ONNX model
    mock_model = MagicMock(spec=onnx.ModelProto)
    mock_load.return_value = mock_model
    mock_check_model.return_value = None
    
    # Mock session options
    mock_options = MagicMock()
    mock_session_options.return_value = mock_options
    mock_options.optimized_model_filepath = str(Path(temp_dir) / "optimized_model.onnx")
    
    # Test optimization
    try:
        optimized_model = exporter.optimize_model(mock_model)
        assert isinstance(optimized_model, onnx.ModelProto)
        mock_save.assert_called_once()
        mock_load.assert_called()
    except ImportError:
        pytest.skip("onnxruntime not available")

def test_custom_input_output_names(temp_dir):
    """Test export with custom input/output names."""
    model = SimpleModel()
    custom_config = {
        "opset_version": 13,
        "input_names": ["custom_input"],
        "output_names": ["custom_output"]
    }
    exporter = ONNXExporter(custom_config)
    
    sample_input = torch.randn(1, 10)
    export_path = Path(temp_dir) / "custom_names_model.onnx"
    onnx_model = exporter.export(model, sample_input, export_path)
    
    metadata = exporter.get_model_metadata(onnx_model)
    assert "custom_input" in metadata["input_names"]
    assert "custom_output" in metadata["output_names"]

def test_export_error_handling(temp_dir, export_config):
    """Test error handling during export."""
    class BrokenModel(torch.nn.Module):
        def forward(self, x):
            # This will raise a RuntimeError during ONNX export
            raise RuntimeError("Test error")
    
    model = BrokenModel()
    exporter = ONNXExporter(export_config)
    
    sample_input = torch.randn(1, 10)
    export_path = Path(temp_dir) / "invalid_model.onnx"
    
    with pytest.raises(RuntimeError):
        exporter.export(model, sample_input, export_path)
