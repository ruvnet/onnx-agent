"""Setup configuration for onnx-agent package."""

from setuptools import setup, find_packages

setup(
    name="onnx-agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "onnx",
        "numpy",
        "tqdm",
        "pyyaml",
        "psutil",
        "dspy-ai",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-benchmark",
            "pytest-timeout",
            "pytest-xdist",
            "pytest-cov",
        ],
    },
    python_requires=">=3.8",
)
