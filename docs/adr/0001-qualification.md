# ADR 0001: Differential qualification before optimization claims

Status: accepted for v2.

The original implementation includes experimental DSPy and PyTorch components, placeholder dataset loading and broad unpinned dependencies. Model optimization may change numerical outputs. The supported increment must establish actual runtime and reference evidence before adding general training or arbitrary model ingestion.

Decision: export a deterministic ONNX linear fixture, run official ONNX checker, perform real ONNX Runtime dynamic int8 quantization, execute CPU sessions with graph optimizations and compare against independently computed NumPy outputs on 16 held out batches. Keep fixed iteration bounds and single thread runtime configuration. Record source and quantized hashes, sizes, versions and latency quantiles. Do not call repeated inference training. Do not claim reduced latency merely because the file is smaller.

Only CPU is qualified. Provider requests outside that boundary fail, and fallback is disabled. MCP accepts no arbitrary graphs. Operator local structural validation limits bytes, static dimensions, operators, attributes and graph count; external tensors, custom domains/functions, sparse tensors and training graphs are rejected. Native ONNX parsers are not a sandbox for hostile binary models. A later general model service requires process isolation and deployment resource limits.

The historical package is excluded from the v2 wheel. Its torch.load call now explicitly requires weights_only=True to reduce legacy unsafe checkpoint loading risk, but the legacy pipeline remains unsupported. No training/GPU behavior is certified by the v2 suite.

Evidence: [ONNX Runtime quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) explains quantization accuracy and performance tradeoffs. [OODTE differential testing research](https://arxiv.org/abs/2505.01892) motivates checking optimizer correctness against original outputs. [ONNX external data](https://onnx.ai/onnx/repo-docs/ExternalData.html) documents external tensor references that this service rejects.

Acceptance: exact CPU execution passes declared numerical tolerances; report actual size and latency tradeoff; reject external data, unsupported providers and malformed input; CLI and actual MCP execution produce passing qualification reports.

Optimization extension: compare graph optimizations disabled/enabled on identical single-thread CPU inputs. A local latency recommendation requires at least5% p95 improvement and unchanged fp32 parity; repeated target hardware runs remain mandatory. Dynamic activation quantization has per-inference overhead, so file size and latency are reported independently. No graph allowlist expansion is needed.
