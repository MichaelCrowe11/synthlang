"""
SynthLang - The Generative AI Pipeline DSL

Compose, evaluate, and deploy LLM pipelines with confidence.
"""

__version__ = "1.0.0"
__author__ = "Michael Benjamin Crowe"
__email__ = "michael@crowelogic.com"

from .core import Pipeline, Model, Prompt, Router, Guardrail, Cache
from .parser import SynthParser
from .executor import PipelineExecutor
from .evaluator import Evaluator
from .monitor import Monitor
from .optimization import CostOptimizer

__all__ = [
    "Pipeline",
    "Model",
    "Prompt",
    "Router",
    "Guardrail",
    "Cache",
    "SynthParser",
    "PipelineExecutor",
    "Evaluator",
    "Monitor",
    "CostOptimizer",
]