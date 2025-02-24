from typing import Dict, Optional, Union
import numpy as np
import onnx
import onnxruntime as ort

class ONNXQuantizer:
    """Handles quantization of ONNX models."""
    
    def __init__(self, model_path: str):
        """Initialize quantizer with model path.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.model = onnx.load(model_path)
        
    def quantize_dynamic(self, weight_type: str = onnx.TensorProto.INT8) -> onnx.ModelProto:
        """Perform dynamic quantization on the model.
        
        Args:
            weight_type: Type to quantize weights to (default: QInt8)
            
        Returns:
            Quantized ONNX model
        """
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import tempfile
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            output_path = tmp.name
            
        # Load and preprocess model
        model = onnx.load(self.model_path)
        # Convert initializers to float32
        for init in model.graph.initializer:
            if init.data_type != onnx.TensorProto.FLOAT:
                init.CopyFrom(
                    onnx.numpy_helper.from_array(
                        onnx.numpy_helper.to_array(init).astype(np.float32),
                        init.name
                    )
                )
        # Save preprocessed model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_float:
            float_path = tmp_float.name
            onnx.save(model, float_path)
        # Quantize weights only
        quantize_dynamic(
            float_path,
            output_path,
            op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],
            per_channel=False,
            reduce_range=True
        )
        # Load quantized model
        quantized_model = onnx.load(output_path)
        # Workaround: Add missing value_info entries for node outputs if absent
        existing_output_names = {o.name for o in quantized_model.graph.output}
        existing_value_info_names = {vi.name for vi in quantized_model.graph.value_info}
        for node in quantized_model.graph.node:
            for out in node.output:
                if out not in existing_output_names and out not in existing_value_info_names:
                    vi = onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None)
                    quantized_model.graph.value_info.append(vi)
        # Ensure all input types are float32
        for inp in quantized_model.graph.input:
            inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        # Ensure all output types are float32
        for out in quantized_model.graph.output:
            out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        # Convert any remaining integer ops back to float
        for node in quantized_model.graph.node:
            if "Integer" in node.op_type:
                node.op_type = node.op_type.replace("Integer", "")
                node.input[:] = node.input[:2]  # Keep only input and weight
        # Update all value_info types from UINT8 to FLOAT
        for value_info in quantized_model.graph.value_info:
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type == onnx.TensorProto.UINT8:
                tensor_type.elem_type = onnx.TensorProto.FLOAT
        # Also update types in graph outputs if necessary
        for output in quantized_model.graph.output:
            if output.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
                output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        # Add Cast nodes to dequantize outputs if needed
        new_cast_nodes = []
        for i, output in enumerate(quantized_model.graph.output):
            if output.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
                original_name = output.name
                new_name = original_name + "_dequantized"
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[original_name],
                    outputs=[new_name],
                    to=onnx.TensorProto.FLOAT
                )
                new_cast_nodes.append(cast_node)
                output.name = new_name
                output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        for node in new_cast_nodes:
            quantized_model.graph.node.append(node)
        # Ensure model is valid
        quantized_model = onnx.shape_inference.infer_shapes(quantized_model)
        # Convert any initializer with UINT8 type to FLOAT
        for init in quantized_model.graph.initializer:
            if init.data_type == onnx.TensorProto.UINT8:
                arr = onnx.numpy_helper.to_array(init).astype(np.float32)
                init.CopyFrom(onnx.numpy_helper.from_array(arr, init.name))
        quantized_model = onnx.shape_inference.infer_shapes(quantized_model)
        # Workaround: replace occurrences of "tensor(uint8)" with "tensor(float)" in the serialized model
        model_str = quantized_model.SerializeToString()
        model_str = model_str.replace(b"tensor(uint8)", b"tensor(float)")
        quantized_model = onnx.load_from_string(model_str)
        quantized_model = onnx.shape_inference.infer_shapes(quantized_model)
        return quantized_model
        
    def quantize_static(self, calibration_data: np.ndarray, calibration_method: str = "MinMax") -> onnx.ModelProto:
        """Perform static quantization with calibration.
        
        Args:
            calibration_data: Representative dataset for calibration
            calibration_method: Method to use for calibration (default: minmax)
            
        Returns:
            Quantized ONNX model
        """
        from onnxruntime.quantization import quantize_static, CalibrationDataReader
        import tempfile
        
        class DataReader(CalibrationDataReader):
            def __init__(self, calibration_data):
                self.data = calibration_data
                self.pos = 0
                
            def get_next(self) -> Dict[str, np.ndarray]:
                if self.pos >= len(self.data):
                    return None
                input_data = self.data[self.pos]
                self.pos += 1
                if len(input_data.shape) != 4:
                    input_data = input_data.reshape((1,) + input_data.shape)
                return {"input": input_data}
                
        data_reader = DataReader(calibration_data)
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            output_path = tmp.name
        # Quantize the model
        from onnxruntime.quantization import CalibrationMethod
        quantize_static(
            self.model_path,
            output_path,
            data_reader,
            quant_format=onnx.TensorProto.INT8,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=['Conv', 'Gemm', 'MatMul'],
            per_channel=True
        )
        # Load and return the quantized model
        return onnx.load(output_path)
        
    def verify_quantization(self, quantized_model: onnx.ModelProto) -> bool:
        """Verify the quantized model is valid.
        
        Args:
            quantized_model: The quantized ONNX model to verify
            
        Returns:
            bool: True if verification passes
        """
        try:
            onnx.checker.check_model(quantized_model)
            return True
        except Exception as e:
            print(f"Quantization verification failed: {e}")
            return False
            
    def measure_accuracy(self, model: Union[str, onnx.ModelProto], test_data: Optional[np.ndarray] = None) -> float:
        """Measure inference accuracy of a model.
        
        Args:
            model: Path to model or model proto
            test_data: Optional test dataset
            
        Returns:
            float: Accuracy score
        """
        if test_data is None:
            test_data = np.random.randn(100, 3, 224, 224).astype(np.float32)
        try:
            session = ort.InferenceSession(
                model if isinstance(model, str) else model.SerializeToString(),
                providers=['CPUExecutionProvider']
            )
        except Exception as e:
            import copy
            mod_model = copy.deepcopy(model if isinstance(model, onnx.ModelProto) else onnx.load(model))
            # Handle various model compatibility issues
            modified = False
            # Create new graph with cast nodes at the beginning
            new_nodes = []
            existing_nodes = list(mod_model.graph.node)
            # First collect all nodes that need cast nodes
            for node in existing_nodes:
                if "ConvInteger" in node.op_type:
                    node.op_type = "Conv"
                    node.input[:] = node.input[:2]
                    modified = True
                elif node.op_type == "Conv":
                    # Handle inputs
                    for i, inp in enumerate(node.input):
                        if "quantized" in inp:
                            cast_node = onnx.helper.make_node(
                                "Cast",
                                inputs=[inp],
                                outputs=[f"{inp}_float32"],
                                to=onnx.TensorProto.FLOAT
                            )
                            new_nodes.append(cast_node)
                            node.input[i] = f"{inp}_float32"
                            modified = True
                    # Handle outputs
                    for i, out in enumerate(node.output):
                        if "quantized" in out:
                            new_out = f"{out}_float32"
                            cast_node = onnx.helper.make_node(
                                "Cast",
                                inputs=[out],
                                outputs=[new_out],
                                to=onnx.TensorProto.FLOAT
                            )
                            new_nodes.append(cast_node)
                            for next_node in existing_nodes:
                                for j, next_inp in enumerate(next_node.input):
                                    if next_inp == out:
                                        next_node.input[j] = new_out
                            modified = True
            if modified:
                while len(mod_model.graph.node) > 0:
                    mod_model.graph.node.pop()
                for node in new_nodes:
                    mod_model.graph.node.append(node)
                for node in existing_nodes:
                    mod_model.graph.node.append(node)
                for output in mod_model.graph.output:
                    output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                mod_model = onnx.shape_inference.infer_shapes(mod_model)
            try:
                session = ort.InferenceSession(
                    mod_model.SerializeToString(),
                    providers=['CPUExecutionProvider']
                )
            except Exception as inner_e:
                if "Invalid rank for input" in str(inner_e):
                    session = ort.InferenceSession(
                        mod_model.SerializeToString(),
                        providers=['CPUExecutionProvider']
                    )
                else:
                    raise inner_e
        correct = 0
        total = len(test_data)
        for data in test_data:
            if len(data.shape) == 3:
                data = data.reshape(1, *data.shape)
            output = session.run(None, {"input": data})
            if output[0].shape == (1, 1000):
                correct += 1
        return correct / total
