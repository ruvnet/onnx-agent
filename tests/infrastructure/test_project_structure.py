import os
import pytest
from pathlib import Path
import yaml

from onnx_agent.infrastructure.project import ProjectStructure
from onnx_agent.infrastructure.config import ConfigLoader

@pytest.fixture(autouse=True)
def clean_env():
    """Clean up environment variables before and after each test."""
    # Store original environment
    original_env = {k: v for k, v in os.environ.items() if k.startswith("ONNX_AGENT_")}
    
    # Clean environment
    for k in list(os.environ.keys()):
        if k.startswith("ONNX_AGENT_"):
            del os.environ[k]
            
    yield
    
    # Restore original environment
    for k in list(os.environ.keys()):
        if k.startswith("ONNX_AGENT_"):
            del os.environ[k]
    os.environ.update(original_env)

def test_directory_creation(temp_dir, test_config):
    """Test creating project directories."""
    os.environ["PROJECT_ROOT"] = temp_dir
    project = ProjectStructure(test_config)
    project.create_directories()
    
    assert project.model_dir.exists()
    assert project.config_dir.exists()
    assert project.data_dir.exists()
    assert project.cache_dir.exists()
    assert project.export_dir.exists()

def test_validate_structure(temp_dir, test_config):
    """Test validating project structure."""
    os.environ["PROJECT_ROOT"] = temp_dir
    project = ProjectStructure(test_config)
    
    # Initially should not be valid
    assert not project.validate_structure()
    
    # Create directories
    project.create_directories()
    assert project.validate_structure()

def test_get_paths(temp_dir, test_config):
    """Test getting various file paths."""
    os.environ["PROJECT_ROOT"] = temp_dir
    project = ProjectStructure(test_config)
    project.create_directories()
    
    model_path = project.get_model_path("model.pt")
    assert model_path == Path(temp_dir) / "models" / "model.pt"
    
    config_path = project.get_config_path("config.yaml")
    assert config_path == Path(temp_dir) / "config" / "config.yaml"
    
    data_path = project.get_data_path("data.txt")
    assert data_path == Path(temp_dir) / "data" / "data.txt"
    
    export_path = project.get_export_path("model")
    assert export_path == Path(temp_dir) / "exports" / "model.onnx"

def test_config_loading(temp_dir):
    """Test loading configuration from file."""
    config_loader = ConfigLoader()
    
    # Create a test config file
    config_path = Path(temp_dir) / "test_config.yaml"
    test_config = {
        "environment": "test",
        "model_config": {"type": "classifier"}
    }
    
    with config_path.open("w") as f:
        yaml.dump(test_config, f)
    
    # Load and verify config
    loaded_config = config_loader.load(str(config_path))
    assert loaded_config["environment"] == "test"
    assert loaded_config["model_config"]["type"] == "classifier"
    assert loaded_config["log_level"] == "INFO"  # Default value

def test_config_environment_override(temp_dir):
    """Test environment variable override of config values."""
    # Create base config
    config_path = Path(temp_dir) / "test_config.yaml"
    base_config = {
        "environment": "development",
        "model_config": {"type": "classifier"}
    }
    
    with config_path.open("w") as f:
        yaml.dump(base_config, f)
    
    # Set environment variables
    os.environ["ONNX_AGENT_ENVIRONMENT"] = "production"
    os.environ["ONNX_AGENT_MODEL_CONFIG_TYPE"] = "regressor"
    
    config_loader = ConfigLoader()
    config = config_loader.load_with_env(str(config_path))
    
    assert config["environment"] == "production"
    assert config["model_config"]["type"] == "regressor"

def test_config_type_conversion(temp_dir):
    """Test configuration value type conversion."""
    # Create base config
    config_path = Path(temp_dir) / "test_config.yaml"
    base_config = {
        "debug": False,
        "batch_size": 16,
        "learning_rate": 0.01
    }
    
    with config_path.open("w") as f:
        yaml.dump(base_config, f)
    
    # Set environment variables
    os.environ["ONNX_AGENT_DEBUG"] = "true"
    os.environ["ONNX_AGENT_BATCH_SIZE"] = "32"
    os.environ["ONNX_AGENT_LEARNING_RATE"] = "0.001"
    
    config_loader = ConfigLoader()
    config = config_loader.load_with_env(str(config_path))
    
    # Verify type conversions
    assert isinstance(config["debug"], bool)
    assert config["debug"] is True
    assert isinstance(config["batch_size"], int)
    assert config["batch_size"] == 32
    assert isinstance(config["learning_rate"], float)
    assert config["learning_rate"] == 0.001

def test_config_file_not_found():
    """Test error handling for missing config file."""
    config_loader = ConfigLoader()
    with pytest.raises(FileNotFoundError):
        config_loader.load("nonexistent.yaml")

def test_invalid_config_format(temp_dir):
    """Test error handling for invalid config format."""
    config_path = Path(temp_dir) / "invalid_config.yaml"
    
    # Create an invalid config file (not a dictionary)
    with config_path.open("w") as f:
        f.write("- just\n- a\n- list")
    
    config_loader = ConfigLoader()
    with pytest.raises(ValueError):
        config_loader.load(str(config_path))
