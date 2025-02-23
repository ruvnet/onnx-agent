import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock

import torch
import dspy

@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)

@pytest.fixture
def test_config():
    return {
        "environment": "test",
        "log_level": "DEBUG",
        "model_config": {
            "type": "classifier",
            "architecture": "resnet18"
        }
    }

@pytest.fixture
def mock_dspy_module():
    return Mock(spec=dspy.Module)

@pytest.fixture
def mock_optimizer():
    return Mock(spec=dspy.Optimizer)

@pytest.fixture
def mock_torch_model():
    return Mock(spec=torch.nn.Module)

@pytest.fixture
def export_config():
    return {
        "opset_version": 13,
        "dynamic_axes": {"input": {0: "batch_size"}},
        "input_names": ["input"],
        "output_names": ["output"]
    }
