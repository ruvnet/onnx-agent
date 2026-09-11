# Security and trust boundaries

Supported v2: onnx_agent/, mcp/, requirements.txt, current setup.py and generated harness. Historical src/, tests/ and old proposal documents are not installed or certified by this release. The legacy checkpoint loader explicitly uses weights_only=True. Never load an untrusted PyTorch checkpoint through historical code.

The MCP server only executes an internally generated fixture. It accepts no file path, URL, provider selection, shell command, custom operator, training code or credential. Operator local validation reads at most 4 MiB and rejects external/sparse tensors, functions, training graphs, custom domains, unsupported attributes, more than 32 operators or initializers, dynamic/rank other than two input/output shapes, and dimensions larger than 1024. Initializers contain at most 262144 elements. ONNX and ONNX Runtime are native parsers; these checks are not an operating system sandbox. Do not feed attacker supplied model binaries to the direct local CLI without a disposable sandbox.

The subprocess boundary allows one process, 30 seconds, 128 KiB combined output and a scrubbed environment. Validation tests require operator opt in. MCP framing is bounded. Results contain only fixture provenance and metrics. No federation writes or automatic deployment/promotion occur.

Dependency audit: pip-audit over pinned requirements and npm audit over package lock; advisory feeds are dated by the CI execution. No zero day immunity is claimed. Report vulnerabilities via private GitHub security advisories.
