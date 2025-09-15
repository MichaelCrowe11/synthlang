#!/usr/bin/env python
"""Setup script for SynthLang - The Generative AI Pipeline DSL"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from package
version = "1.0.0"

setup(
    name="synthlang",
    version=version,
    author="Michael Benjamin Crowe",
    author_email="michael@crowelogic.com",
    description="The Generative AI Pipeline DSL - Compose, evaluate, and deploy LLM pipelines with confidence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichaelCrowe11/synthlang",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0",
        "pyyaml>=6.0",
        "requests>=2.28",
        "rich>=13.0",
        "aiohttp>=3.8",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "mypy>=1.0",
            "ruff>=0.1.0",
        ],
        "ide": [
            "flask>=2.3",
            "flask-cors>=4.0",
            "websockets>=11.0",
        ],
        "monitoring": [
            "prometheus-client>=0.16",
            "grafana-api>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "synthlang=synthlang.cli:main",
            "synth=synthlang.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "synthlang": [
            "templates/*.synth",
            "static/*",
            "examples/*.synth",
        ],
    },
    zip_safe=False,
    keywords="llm ai pipeline dsl generative-ai orchestration langchain prompt-engineering",
    project_urls={
        "Documentation": "https://synthlang.ai/docs",
        "Bug Reports": "https://github.com/MichaelCrowe11/synthlang/issues",
        "Source": "https://github.com/MichaelCrowe11/synthlang",
        "Discord": "https://discord.gg/synthlang",
    },
)