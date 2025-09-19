# Copyright 2025 Xiaomi Corporation.
from setuptools import setup, find_packages

setup(
    name="slm_eval",
    version="0.1.0",
    packages=find_packages(include=["slm_eval", "slm_eval.*"]),
    install_requires=[
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "datasets==3.2.0",
        "transformers==4.49.0",
    ],
    python_requires=">=3.10",
)