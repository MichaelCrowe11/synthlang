"""
Pipeline executor for SynthLang
"""

import time
import json
from typing import Dict, Any, Optional, List
from .core import Pipeline, Model, Prompt, Guardrail, Cache

class PipelineExecutor:
    """Execute SynthLang pipelines"""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.cache_store = {}
        self.metrics = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline with given input"""
        start_time = time.time()
        result = {"input": input_data}

        # Build execution graph from edges
        execution_order = self._build_execution_order()

        # Execute each node in order
        current_data = input_data
        for node in execution_order:
            if node == "input":
                continue
            elif node == "output":
                result["output"] = current_data
            else:
                current_data = self._execute_node(node, current_data)

        # Update metrics
        self.metrics["execution_time"] = time.time() - start_time
        result["metrics"] = self.metrics.copy()

        return result

    def _build_execution_order(self) -> List[str]:
        """Build execution order from pipeline edges"""
        order = []
        visited = set()

        def visit(node):
            if node in visited:
                return
            visited.add(node)

            # Find dependencies
            for source, target in self.pipeline.edges:
                if target == node and source not in visited:
                    visit(source)

            order.append(node)

        # Start from output nodes
        output_nodes = set()
        for source, target in self.pipeline.edges:
            if target == "output":
                output_nodes.add(source)

        for node in output_nodes:
            visit(node)

        return order

    def _execute_node(self, node_name: str, input_data: Any) -> Any:
        """Execute a single node in the pipeline"""
        # Find node by name
        node = self._find_node(node_name)

        if isinstance(node, Model):
            return self._execute_model(node, input_data)
        elif isinstance(node, Prompt):
            return self._execute_prompt(node, input_data)
        elif isinstance(node, Guardrail):
            return self._execute_guardrail(node, input_data)
        elif isinstance(node, Cache):
            return self._execute_cache(node, input_data)
        else:
            # Router or unknown node
            return input_data

    def _find_node(self, name: str):
        """Find node by name in pipeline"""
        for model in self.pipeline.models:
            if model.name == name:
                return model
        for prompt in self.pipeline.prompts:
            if prompt.name == name:
                return prompt
        for guardrail in self.pipeline.guardrails:
            if guardrail.name == name:
                return guardrail
        for cache in self.pipeline.caches:
            if cache.name == name:
                return cache
        for router in self.pipeline.routers:
            if router.name == name:
                return router
        return None

    def _execute_model(self, model: Model, input_data: Any) -> str:
        """Execute a model node"""
        # Simulate model execution
        tokens_used = len(str(input_data).split()) * 2
        self.metrics["total_tokens"] += tokens_used
        self.metrics["total_cost"] += tokens_used * 0.00001  # Simulated cost

        # Return simulated response
        return f"Response from {model.name} using {model.model}: Processing '{input_data}'"

    def _execute_prompt(self, prompt: Prompt, input_data: Any) -> str:
        """Execute a prompt node"""
        # Format prompt with input data
        if isinstance(input_data, dict):
            formatted = prompt.format(**input_data)
        else:
            formatted = prompt.template.replace("{{input}}", str(input_data))

        return formatted

    def _execute_guardrail(self, guardrail: Guardrail, input_data: Any) -> Any:
        """Execute a guardrail node"""
        text = str(input_data)
        valid, error = guardrail.validate(text)

        if not valid:
            raise ValueError(f"Guardrail {guardrail.name} failed: {error}")

        return input_data

    def _execute_cache(self, cache: Cache, input_data: Any) -> Any:
        """Execute a cache node"""
        cache_key = cache.cache_key({"input": input_data})

        # Check cache
        if cache_key in self.cache_store:
            cached_entry = self.cache_store[cache_key]
            if time.time() - cached_entry["timestamp"] < cache.ttl:
                self.metrics["cache_hits"] += 1
                return cached_entry["data"]

        # Cache miss
        self.metrics["cache_misses"] += 1

        # Store in cache
        self.cache_store[cache_key] = {
            "data": input_data,
            "timestamp": time.time()
        }

        return input_data