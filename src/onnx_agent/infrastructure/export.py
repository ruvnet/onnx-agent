from typing import Dict, Any, Optional, Union, Tuple, List
import torch
import onnx
from pathlib import Path

class ONNXExporter:
    """Handles ONNX model export and validation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opset_version = config.get("opset_version", 13)
        self.dynamic_axes = config.get("dynamic_axes", {})
        self.input_names = config.get("input_names", ["input"])
        self.output_names = config.get("output_names", ["output"])
        
    def export(
        self,
        model: torch.nn.Module,
        sample_input: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...], List[torch.Tensor]]] = None,
        path: Optional[Union[str, Path]] = None
    ) -> onnx.ModelProto:
        """Export PyTorch model to ONNX format."""
        if sample_input is None:
            # Create dummy input based on config
            input_shape = self.config.get("input_shape", [1, 3, 224, 224])
            sample_input = torch.randn(*input_shape)
            
        if path is None:
            path = "model.onnx"
            
        path = Path(path)
        
        # Ensure model is in eval mode
        model.eval()
        
        # Export the model
        torch.onnx.export(
            model,
            sample_input,
            path,
            input_names=self.input_names,
            output_names=self.output_names,
            dynamic_axes=self.dynamic_axes,
            opset_version=self.opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )
        
        # Load and validate the exported model
        onnx_model = onnx.load(path)
        self.validate_model(onnx_model)
        
        return onnx_model
        
    def validate_model(self, model: onnx.ModelProto) -> None:
        """Validate exported ONNX model."""
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            raise ValueError(f"Invalid ONNX model: {str(e)}")
            
    def optimize_model(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Apply ONNX optimizations to the model."""
        import onnxruntime
        
        # Save model to temporary file
        temp_path = Path("temp_model.onnx")
        onnx.save(model, temp_path)
        
        # Create a session with optimization level
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(Path("optimized_model.onnx"))
        
        # Create session which will optimize the model
        _ = onnxruntime.InferenceSession(
            str(temp_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
        # Load the optimized model
        optimized_model = onnx.load(sess_options.optimized_model_filepath)
        
        # Clean up temporary files
        temp_path.unlink(missing_ok=True)
        Path(sess_options.optimized_model_filepath).unlink(missing_ok=True)
        
        return optimized_model
        
    def get_model_metadata(self, model: onnx.ModelProto) -> Dict[str, Any]:
        """Get metadata about the ONNX model."""
        return {
            "opset_version": model.opset_import[0].version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "domain": model.domain,
            "model_version": model.model_version,
            "doc_string": model.doc_string,
            "input_names": [input.name for input in model.graph.input],
            "output_names": [output.name for output in model.graph.output],
            "num_nodes": len(model.graph.node)
        }
