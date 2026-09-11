import argparse,json,sys
from .pipeline import qualify,validate,MAX_BYTES
p=argparse.ArgumentParser(description='Local ONNX CPU qualification; no remote models or training')
p.add_argument('command',choices=['status','qualify','benchmark','validate'])
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--model',help='Operator selected local file for structural validation only')
a=p.parse_args()
try:
    if a.command=='status': result={'project':'onnx-agent','commands':['qualify','benchmark','validate'],'provider':'CPUExecutionProvider','automaticPromotion':False}
    elif a.command=='validate':
        if not a.model: raise ValueError('--model required')
        with open(a.model,'rb') as f: data=f.read(MAX_BYTES+1)
        validate(data);result={'valid':True,'bytes':len(data),'executionPerformed':False}
    else: result=qualify(a.iterations)
    print(json.dumps(result,allow_nan=False));sys.exit(0 if result.get('parityPass',True) else 1)
except Exception as e:
    print(json.dumps({'error':type(e).__name__,'detail':'qualification rejected; inspect local inputs and limits'}));sys.exit(2)
