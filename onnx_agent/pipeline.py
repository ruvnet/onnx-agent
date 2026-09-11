import hashlib, json, tempfile, time, math
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

MAX_BYTES=4*1024*1024
ALLOWED={'MatMul','Add','Relu','DynamicQuantizeLinear','Mul','Cast','MatMulInteger'}

def validate(data):
    if not isinstance(data,bytes) or not 0<len(data)<=MAX_BYTES: raise ValueError('model size limit')
    model=onnx.load_model_from_string(data)
    if model.graph.sparse_initializer: raise ValueError('sparse tensors unsupported')
    if model.functions or model.training_info: raise ValueError('functions and training unsupported')
    if len(model.graph.node)>32 or len(model.graph.initializer)>32: raise ValueError('graph limit')
    for t in model.graph.initializer:
        if t.data_location==TensorProto.EXTERNAL or t.external_data: raise ValueError('external tensor rejected')
        if len(t.dims)>2 or any(d<1 or d>1024 for d in t.dims) or math.prod(t.dims)>262144: raise ValueError('tensor dimensions')
    for n in model.graph.node:
        if n.domain or n.op_type not in ALLOWED: raise ValueError('unsupported operator')
        if n.attribute and not (n.op_type=='Cast' and len(n.attribute)==1 and n.attribute[0].name=='to' and n.attribute[0].i==TensorProto.FLOAT): raise ValueError('unsupported attribute')
    for v in list(model.graph.input)+list(model.graph.output)+list(model.graph.value_info):
        dims=v.type.tensor_type.shape.dim
        if len(dims)!=2 or any(d.dim_param or not 1<=d.dim_value<=1024 for d in dims): raise ValueError('static rank two shapes required')
    onnx.checker.check_model(model,full_check=True)
    return model

def fixture():
    rng=np.random.default_rng(2026)
    w=rng.normal(0,.1,(64,32)).astype(np.float32)
    b=rng.normal(0,.01,(32,)).astype(np.float32)
    graph=helper.make_graph([helper.make_node('MatMul',['input','w'],['mm']),helper.make_node('Add',['mm','b'],['output'])],'linear',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[8,64])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[8,32])],[numpy_helper.from_array(w,'w'),numpy_helper.from_array(b,'b')])
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)],ir_version=10)
    return m.SerializeToString(),w,b

def session(data,provider,optimized=True):
    validate(data)
    if provider!='CPUExecutionProvider': raise ValueError('only qualified CPU provider supported')
    if provider not in ort.get_available_providers(): raise ValueError('provider unavailable')
    opt=ort.SessionOptions(); opt.intra_op_num_threads=1; opt.inter_op_num_threads=1
    opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimized else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opt.log_severity_level=3
    s=ort.InferenceSession(data,sess_options=opt,providers=[provider]); s.disable_fallback()
    return s

def qualify(iterations=100,provider='CPUExecutionProvider'):
    if type(iterations)!=int or not 1<=iterations<=1000: raise ValueError('iterations 1..1000')
    data,w,b=fixture(); fp=session(data,provider); baseline=session(data,provider,optimized=False)
    rng=np.random.default_rng(73); inputs=[rng.uniform(-1,1,(8,64)).astype(np.float32) for _ in range(16)]
    reference=[x@w+b for x in inputs]
    with tempfile.TemporaryDirectory(prefix='onnx-qualify-') as tmp:
        src=Path(tmp)/'source.onnx'; dst=Path(tmp)/'int8.onnx'; src.write_bytes(data)
        quantize_dynamic(str(src),str(dst),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul'])
        quant=dst.read_bytes()
    qs=session(quant,provider)
    results={}; variants=[('baselineFloat32',data,baseline),('float32',data,fp),('int8',quant,qs)]
    timings={name:[] for name,_,_ in variants}
    for name,blob,s in variants:
        errors=[float(np.max(np.abs(s.run(None,{'input':x})[0]-y))) for x,y in zip(inputs,reference)]
        for _ in range(20): s.run(None,{'input':inputs[0]})
        results[name]={'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'maxAbsoluteError':max(errors)}
    # Rotate ordering each iteration to reduce warmup and host drift bias.
    for i in range(iterations):
        for offset in range(len(variants)):
            name,_,s=variants[(i+offset)%len(variants)]
            start=time.perf_counter_ns(); s.run(None,{'input':inputs[i%16]}); timings[name].append((time.perf_counter_ns()-start)/1e6)
    for name,times in timings.items():
        results[name].update({'p50Ms':float(np.percentile(times,50)),'p95Ms':float(np.percentile(times,95))})
    passed=results['baselineFloat32']['maxAbsoluteError']<=1e-5 and results['float32']['maxAbsoluteError']<=1e-5 and results['int8']['maxAbsoluteError']<=.04
    speedup=results['baselineFloat32']['p95Ms']/results['float32']['p95Ms']
    optimization={'baseline':'ORT_DISABLE_ALL','candidate':'ORT_ENABLE_ALL','threads':1,'measurementOrder':'rotating interleaved','p95Speedup':speedup,'measuredLatencyGatePass':speedup>=1.05 and results['float32']['maxAbsoluteError']<=1e-5,'minimumSpeedup':1.05,'requiresRepeatedTargetHardwareRuns':True,'automaticPromotion':False}
    return {'optimizationComparison':optimization,'schemaVersion':1,'fixture':'seeded-linear-64x32','provider':provider,'availableProviders':ort.get_available_providers(),'iterations':iterations,'holdoutBatches':16,'results':results,'parityPass':passed,'automaticPromotion':False,'productionQualified':False,'versions':{'onnx':onnx.__version__,'onnxruntime':ort.__version__,'numpy':np.__version__}}
