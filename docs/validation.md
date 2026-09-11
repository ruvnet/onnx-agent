# Validation record

Recorded 2026-09-11 on managed Linux x86_64, Python3.12 and Node24.19. Eleven Python tests exercise actual ONNX Runtime CPU and NumPy parity, actual quantized size reduction, external data rejection, unsupported operators/providers, bytes and iteration bounds, actual CLI and initializer shape product overflow before checker invocation. One actual official SDK MCP test exercises CPU qualification, policy resource and denied validation/custom model arguments. All pass.

pip-audit over requirements.txt and npm audit over package-lock.json report zero known vulnerabilities at execution time. Build produces a wheel containing only onnx_agent v2, excluding historical src. docs/benchmark.json contains a real1000iteration warm CPU benchmark with16holdout batches, version/hash/size provenance and explicit productionQualified:false. No GPU, physical hardware, trained semantic model, training or live deployment is validated.

Independent review found numpy product overflow could bypass initializer count bounds. Fixed with rank<=2 and Python arbitrary precision math.prod, with a regression asserting rejection before checker invocation. MCP only accepts the internal fixture, so no caller graph reaches native inference.

Qualification tolerances are fixture-specific: fp32 max absolute error<=1e-5; int8<=.04. Tiny matrix int8 was smaller and slightly slower in a reference run; do not extrapolate superiority. Remaining real-model, provider and sandbox gates are tracked in issue1. Historical tests are preserved but excluded, and no claim is made that old DSPy APIs pass current dependencies.

CI qualification caught NumPy2.5 requiring Python>=3.12. The declared package and supported matrix now match that real dependency minimum; Python3.11 is unsupported.

RuFlo3.25.6 deep security scan completed with zero findings. This heuristic result supplements, and does not replace, the manual native parser trust boundary review and dependency audits.

Current optimization review adds ORT_DISABLE_ALL versus ORT_ENABLE_ALL on the same single-thread CPU fixture, with independent parity,20warmups and1000measurements per configuration. Reports a measured latency gate requiring>=1.05x p95 speedup plus fp32 parity, never promotion; repeat on target hardware before selection. Native graph optimization is measured separately from dynamic int8 size reduction.

Cross CPU validation caught full range int8 error0.21932176 on a CI runner while both fp32 variants differed from NumPy by at most5.96e-7. The supported quantizer now uses reduce_range=True per official ORT guidance for non-VNNI x86 saturation. Local reduced-range error0.02836808 passes the unchanged0.04 bound. Latest docs/benchmark.json reports current7bit effective weight range stored in int8. Python3.13 was also independently installed and tested locally.

The saturation regression inspects weights actually emitted by the production quantization path and verifies2*255*max(abs(weight))<=32767, preventing the signed16 pair accumulator saturation mechanism.
