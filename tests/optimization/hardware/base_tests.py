import unittest
import numpy as np
import onnx
import onnxruntime as ort

class BaseQuantizationTest(unittest.TestCase):
    def setUp(self):
        self.model_path = self.get_test_model()
        self.quantizer = self.setup_quantizer()
        self.calibration_data = self.load_calibration_data()
        
    def get_test_model(self):
        return str(self.test_model_path)
        
    def setup_quantizer(self):
        raise NotImplementedError("Subclasses must implement setup_quantizer")
        
    def load_calibration_data(self):
        return np.random.randn(100, 3, 224, 224).astype(np.float32)

class BaseProviderTest(unittest.TestCase):
    def setUp(self):
        self.session_options = self.get_session_options()
        self.providers = self.get_available_providers()
        self.test_input = self.create_test_input()
        
    def get_session_options(self):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options
        
    def get_available_providers(self):
        return ort.get_available_providers()
        
    def create_test_input(self):
        return np.random.randn(1, 3, 224, 224).astype(np.float32)

class BaseOptimizationTest(unittest.TestCase):
    def setUp(self):
        self.model = self.load_test_model()
        self.optimizer = self.setup_optimizer()
        self.optimization_config = self.get_optimization_config()
        
    def load_test_model(self):
        return onnx.load(str(self.test_model_path))
        
    def setup_optimizer(self):
        raise NotImplementedError("Subclasses must implement setup_optimizer")
        
    def get_optimization_config(self):
        return {
            "level": "ORT_ENABLE_ALL",
            "optimization_passes": ["constant_folding", "node_fusion"]
        }
