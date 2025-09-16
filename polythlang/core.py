"""
Core components for SynthLang pipelines
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import yaml

class ProviderType(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

class RouteStrategy(Enum):
    """Routing strategies for pipeline execution"""
    ROUND_ROBIN = "round_robin"
    LEAST_LATENCY = "least_latency"
    LOWEST_COST = "lowest_cost"
    AB_SPLIT = "ab_split"
    CONDITIONAL = "conditional"
    WEIGHTED = "weighted"

@dataclass
class Model:
    """LLM model configuration"""
    name: str
    provider: ProviderType
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation"""
        return {
            "name": self.name,
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

@dataclass
class Prompt:
    """Prompt template configuration"""
    name: str
    template: str
    variables: List[str]
    system_prompt: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None

    def format(self, **kwargs) -> str:
        """Format prompt with provided variables"""
        formatted = self.template
        for var in self.variables:
            if var in kwargs:
                formatted = formatted.replace(f"{{{{{var}}}}}", str(kwargs[var]))
        return formatted

@dataclass
class Router:
    """Request routing configuration"""
    name: str
    strategy: RouteStrategy
    routes: List[Dict[str, Any]]
    metrics: Optional[List[str]] = None
    auto_optimize: bool = False

    def select_route(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Select route based on strategy"""
        if self.strategy == RouteStrategy.ROUND_ROBIN:
            # Simple round-robin implementation
            return self.routes[0]["target"]
        elif self.strategy == RouteStrategy.CONDITIONAL and context:
            # Evaluate conditions
            for route in self.routes:
                if "condition" in route:
                    # Simplified condition evaluation
                    if self._evaluate_condition(route["condition"], context):
                        return route["target"]
        return self.routes[0]["target"]

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Simple condition evaluation"""
        # This would need proper expression evaluation in production
        return True

@dataclass
class Guardrail:
    """Safety and compliance guardrail"""
    name: str
    toxicity_threshold: Optional[float] = None
    pii_detection: bool = False
    bias_check: Optional[List[str]] = None
    profanity_filter: bool = False
    max_length: Optional[int] = None
    min_length: Optional[int] = None

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """Validate text against guardrail rules"""
        # Simplified validation
        if self.max_length and len(text) > self.max_length:
            return False, f"Text exceeds maximum length of {self.max_length}"
        if self.min_length and len(text) < self.min_length:
            return False, f"Text below minimum length of {self.min_length}"
        return True, None

@dataclass
class Cache:
    """Caching configuration"""
    name: str
    ttl: int  # Time to live in seconds
    strategy: str = "exact_match"
    max_size: Optional[int] = None
    invalidate_on: Optional[List[str]] = None

    def cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key from input"""
        return json.dumps(input_data, sort_keys=True)

@dataclass
class Pipeline:
    """Main pipeline configuration"""
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    models: List[Model] = None
    prompts: List[Prompt] = None
    routers: List[Router] = None
    guardrails: List[Guardrail] = None
    caches: List[Cache] = None
    edges: List[tuple[str, str]] = None

    def __post_init__(self):
        """Initialize empty lists if None"""
        self.models = self.models or []
        self.prompts = self.prompts or []
        self.routers = self.routers or []
        self.guardrails = self.guardrails or []
        self.caches = self.caches or []
        self.edges = self.edges or []

    def add_model(self, model: Model):
        """Add a model to the pipeline"""
        self.models.append(model)

    def add_prompt(self, prompt: Prompt):
        """Add a prompt to the pipeline"""
        self.prompts.append(prompt)

    def add_router(self, router: Router):
        """Add a router to the pipeline"""
        self.routers.append(router)

    def add_guardrail(self, guardrail: Guardrail):
        """Add a guardrail to the pipeline"""
        self.guardrails.append(guardrail)

    def add_cache(self, cache: Cache):
        """Add a cache to the pipeline"""
        self.caches.append(cache)

    def add_edge(self, source: str, target: str):
        """Add an edge to the pipeline graph"""
        self.edges.append((source, target))

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary representation"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "models": [m.to_dict() for m in self.models],
            "prompts": [{"name": p.name, "template": p.template} for p in self.prompts],
            "routers": [{"name": r.name, "strategy": r.strategy.value} for r in self.routers],
            "guardrails": [{"name": g.name, "toxicity_threshold": g.toxicity_threshold} for g in self.guardrails],
            "caches": [{"name": c.name, "ttl": c.ttl} for c in self.caches],
            "edges": self.edges,
        }

    def to_yaml(self) -> str:
        """Export pipeline as YAML"""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    def to_json(self) -> str:
        """Export pipeline as JSON"""
        return json.dumps(self.to_dict(), indent=2)