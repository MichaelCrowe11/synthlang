"""
Parser for SynthLang DSL files
"""

import re
from typing import Dict, List, Any, Optional
from .core import Pipeline, Model, Prompt, Router, Guardrail, Cache, ProviderType, RouteStrategy

class SynthParser:
    """Parser for .synth files"""

    def __init__(self):
        self.pipeline = None
        self.current_block = None
        self.current_indent = 0

    def parse(self, content: str) -> Pipeline:
        """Parse SynthLang DSL content"""
        lines = content.split('\n')
        self.pipeline = Pipeline(name="unnamed")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip empty lines and comments
            if not line or line.startswith('//'):
                i += 1
                continue

            # Parse pipeline declaration
            if line.startswith('pipeline'):
                match = re.match(r'pipeline\s+(\w+)\s*{', line)
                if match:
                    self.pipeline.name = match.group(1)

            # Parse model block
            elif line.startswith('model'):
                i = self._parse_model(lines, i)

            # Parse prompt block
            elif line.startswith('prompt'):
                i = self._parse_prompt(lines, i)

            # Parse router block
            elif line.startswith('router'):
                i = self._parse_router(lines, i)

            # Parse guardrail block
            elif line.startswith('guardrail'):
                i = self._parse_guardrail(lines, i)

            # Parse cache block
            elif line.startswith('cache'):
                i = self._parse_cache(lines, i)

            # Parse edges
            elif line.startswith('edges:'):
                i = self._parse_edges(lines, i)

            i += 1

        return self.pipeline

    def _parse_model(self, lines: List[str], start: int) -> int:
        """Parse a model block"""
        line = lines[start].strip()
        match = re.match(r'model\s+(\w+)\s*{', line)
        if not match:
            return start

        model_name = match.group(1)
        model_config = {"name": model_name}

        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == '}':
                break

            # Parse model properties
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"')

                if key == 'provider':
                    model_config['provider'] = ProviderType(value)
                elif key == 'model':
                    model_config['model'] = value
                elif key == 'temperature':
                    model_config['temperature'] = float(value)
                elif key == 'max_tokens':
                    model_config['max_tokens'] = int(value)

            i += 1

        # Create and add model
        model = Model(**model_config)
        self.pipeline.add_model(model)

        return i

    def _parse_prompt(self, lines: List[str], start: int) -> int:
        """Parse a prompt block"""
        line = lines[start].strip()
        match = re.match(r'prompt\s+(\w+)\s*{', line)
        if not match:
            return start

        prompt_name = match.group(1)
        template = ""
        variables = []

        i = start + 1
        in_template = False
        while i < len(lines):
            line = lines[i]

            if line.strip() == '}':
                break

            if 'template:' in line:
                in_template = True
                # Check if template starts with triple quotes
                if '"""' in line:
                    template_start = line.index('"""') + 3
                    template = line[template_start:]
                    if template.endswith('"""'):
                        template = template[:-3]
                        in_template = False
            elif in_template:
                if '"""' in line:
                    template += '\n' + line[:line.index('"""')]
                    in_template = False
                else:
                    template += '\n' + line

            i += 1

        # Extract variables from template
        variables = re.findall(r'{{(\w+)}}', template)

        prompt = Prompt(name=prompt_name, template=template, variables=variables)
        self.pipeline.add_prompt(prompt)

        return i

    def _parse_router(self, lines: List[str], start: int) -> int:
        """Parse a router block"""
        line = lines[start].strip()
        match = re.match(r'router\s+(\w+)\s*{', line)
        if not match:
            return start

        router_name = match.group(1)
        router_config = {"name": router_name, "routes": []}

        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == '}':
                break

            if 'strategy:' in line:
                strategy = line.split(':', 1)[1].strip()
                if '(' in strategy:
                    strategy = strategy[:strategy.index('(')]
                router_config['strategy'] = RouteStrategy(strategy)
            elif 'routes:' in line:
                # Parse routes array
                i = self._parse_routes(lines, i, router_config)

            i += 1

        router = Router(**router_config)
        self.pipeline.add_router(router)

        return i

    def _parse_guardrail(self, lines: List[str], start: int) -> int:
        """Parse a guardrail block"""
        line = lines[start].strip()
        match = re.match(r'guardrail\s+(\w+)\s*{', line)
        if not match:
            return start

        guardrail_name = match.group(1)
        guardrail_config = {"name": guardrail_name}

        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == '}':
                break

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'toxicity_threshold':
                    guardrail_config['toxicity_threshold'] = float(value)
                elif key == 'pii_detection':
                    guardrail_config['pii_detection'] = value.lower() == 'true'
                elif key == 'bias_check':
                    # Parse array
                    guardrail_config['bias_check'] = eval(value)

            i += 1

        guardrail = Guardrail(**guardrail_config)
        self.pipeline.add_guardrail(guardrail)

        return i

    def _parse_cache(self, lines: List[str], start: int) -> int:
        """Parse a cache block"""
        line = lines[start].strip()
        match = re.match(r'cache\s+(\w+)\s*{', line)
        if not match:
            return start

        cache_name = match.group(1)
        cache_config = {"name": cache_name}

        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == '}':
                break

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'ttl':
                    cache_config['ttl'] = int(value)
                elif key == 'strategy':
                    cache_config['strategy'] = value.strip('"')

            i += 1

        cache = Cache(**cache_config)
        self.pipeline.add_cache(cache)

        return i

    def _parse_edges(self, lines: List[str], start: int) -> int:
        """Parse edges definition"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == ']':
                break

            # Parse edge chain like: input -> greeting -> gpt -> output
            if '->' in line:
                nodes = [n.strip() for n in line.split('->')]
                for j in range(len(nodes) - 1):
                    self.pipeline.add_edge(nodes[j], nodes[j + 1])

            i += 1

        return i

    def _parse_routes(self, lines: List[str], start: int, router_config: Dict) -> int:
        """Parse routes array"""
        i = start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line == ']':
                break

            # Simple route parsing
            if '{' in line:
                route = {}
                # Extract route properties
                if 'name:' in line:
                    match = re.search(r'name:\s*"([^"]+)"', line)
                    if match:
                        route['name'] = match.group(1)
                if 'target:' in line:
                    match = re.search(r'target:\s*(\w+)', line)
                    if match:
                        route['target'] = match.group(1)

                router_config['routes'].append(route)

            i += 1

        return i