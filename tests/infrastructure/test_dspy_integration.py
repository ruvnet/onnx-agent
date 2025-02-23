import os
import pytest
import torch
import dspy
from pathlib import Path
from unittest.mock import Mock, patch

from onnx_agent.infrastructure.dspy_integration import DSPyIntegration

@pytest.fixture
def dspy_config():
    return {
        "model_type": "classifier",
        "model_config": {
            "input_field": "text",
            "output_field": "label"
        },
        "optimizer": {
            "max_demos": 5,
            "max_rounds": 2
        },
        "num_epochs": 2,
        "batch_size": 32
    }

def test_device_setup(dspy_config):
    """Test compute device setup."""
    integration = DSPyIntegration(dspy_config)
    assert isinstance(integration.device, torch.device)
    
    # Force CPU
    dspy_config["force_cpu"] = True
    integration = DSPyIntegration(dspy_config)
    assert integration.device == torch.device("cpu")

@patch('dspy.Predict')
def test_model_initialization(mock_predict, dspy_config):
    """Test model initialization."""
    integration = DSPyIntegration(dspy_config)
    
    # Test with valid model type
    model = integration.initialize_model()
    mock_predict.assert_called_once_with(
        "text -> label"
    )
    assert model == integration.model
    
    # Test with transformer type
    dspy_config["model_type"] = "transformer"
    model = integration.initialize_model()
    assert isinstance(model, dspy.ChainOfThought)
    
    # Test with invalid model type
    dspy_config["model_type"] = "invalid_type"
    with pytest.raises(ValueError):
        integration.initialize_model()

@patch('dspy.teleprompt.BootstrapFewShot')
def test_optimizer_setup(mock_bootstrap, dspy_config):
    """Test optimizer setup."""
    integration = DSPyIntegration(dspy_config)
    integration.model = Mock(spec=dspy.Predict)
    
    # Test with valid setup
    integration.setup_optimizer()
    mock_bootstrap.assert_called_once_with(
        max_bootstrapped_demos=5,
        max_rounds=2
    )
    
    # Test without model
    integration = DSPyIntegration(dspy_config)
    with pytest.raises(ValueError):
        integration.setup_optimizer()

def test_training_data_loading(temp_dir, dspy_config):
    """Test training data loading."""
    integration = DSPyIntegration(dspy_config)
    
    # Test with nonexistent file
    with pytest.raises(FileNotFoundError):
        integration.load_training_data("nonexistent.json")

@patch('dspy.teleprompt.BootstrapFewShot')
def test_model_training(mock_bootstrap, dspy_config):
    """Test model training functionality."""
    integration = DSPyIntegration(dspy_config)
    integration.model = Mock(spec=dspy.Predict)
    
    # Create dummy training data
    training_data = [
        {"input": "Sample text 1", "output": "positive"},
        {"input": "Sample text 2", "output": "negative"}
    ]
    
    # Mock teleprompter
    mock_teleprompter = Mock()
    mock_bootstrap.return_value = mock_teleprompter
    mock_teleprompter.compile.return_value = "compiled_model"
    mock_teleprompter.metrics = {"accuracy": 0.9}
    
    # Test training
    result = integration.train(training_data)
    assert result["compiled_model"] == "compiled_model"
    assert result["metrics"] == {"accuracy": 0.9}

def test_checkpoint_saving_loading(temp_dir, dspy_config):
    """Test model checkpoint functionality."""
    integration = DSPyIntegration(dspy_config)
    integration.model = Mock(spec=dspy.Predict)
    integration.model.compiled_prompts = ["prompt1", "prompt2"]
    
    # Save checkpoint
    checkpoint_path = Path(temp_dir) / "checkpoint.pt"
    integration.save_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()
    
    # Load checkpoint
    new_integration = DSPyIntegration(dspy_config)
    new_integration.model = Mock(spec=dspy.Predict)  # Mock model before loading
    new_integration.load_checkpoint(checkpoint_path)
    assert new_integration.model is not None
    assert new_integration.model.compiled_prompts == ["prompt1", "prompt2"]
    
    # Test loading nonexistent checkpoint
    with pytest.raises(FileNotFoundError):
        integration.load_checkpoint("nonexistent.pt")

def test_mixed_precision_not_supported(dspy_config):
    """Test mixed precision raises NotImplementedError."""
    integration = DSPyIntegration(dspy_config)
    with pytest.raises(NotImplementedError):
        integration.enable_mixed_precision()

def test_distributed_training_not_supported(dspy_config):
    """Test distributed training raises NotImplementedError."""
    integration = DSPyIntegration(dspy_config)
    with pytest.raises(NotImplementedError):
        integration.setup_distributed_training(world_size=2)

def test_training_validation(dspy_config):
    """Test training input validation."""
    integration = DSPyIntegration(dspy_config)
    
    # Test training without model
    with pytest.raises(ValueError):
        integration.train([])

def test_checkpoint_validation(dspy_config):
    """Test checkpoint validation."""
    integration = DSPyIntegration(dspy_config)
    
    # Test saving without model
    with pytest.raises(ValueError):
        integration.save_checkpoint("test.pt")
