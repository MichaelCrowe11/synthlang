# SynthLang User Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Pipeline Development](#pipeline-development)
4. [Cost Optimization](#cost-optimization)
5. [Monitoring & Dashboards](#monitoring--dashboards)
6. [IDE Integration](#ide-integration)
7. [Best Practices](#best-practices)

## Getting Started

### Installation

#### VS Code Extension
1. Open VS Code
2. Navigate to Extensions (Ctrl+Shift+X)
3. Search for "SynthLang"
4. Click Install

#### CLI Installation
```bash
# NPM
npm install -g synthlang

# Cargo
cargo install synthlang

# Binary download
curl -sSL https://get.synthlang.ai | sh
```

### Your First Pipeline

Create a file named `hello.synth`:

```synth
pipeline HelloWorld {
    prompt greeting {
        template: """
        Generate a friendly greeting for {{name}}.
        """
    }

    model gpt {
        provider: "openai"
        model: "gpt-3.5-turbo"
        temperature: 0.7
    }

    edges: [
        input -> greeting -> gpt -> output
    ]
}
```

Run the pipeline:
```bash
synth run hello.synth --input '{"name": "Alice"}'
```

## Core Concepts

### Pipelines
Pipelines are the core abstraction in SynthLang. They define:
- Data flow between components
- Model configurations
- Safety guardrails
- Caching strategies
- Evaluation metrics

### Components

#### Models
Define LLM providers and configurations:
```synth
model assistant {
    provider: "anthropic"
    model: "claude-3-sonnet"
    temperature: 0.3
    max_tokens: 2000
}
```

#### Prompts
Template-based prompt engineering:
```synth
prompt classifier {
    template: """
    Classify the following text:
    {{text}}

    Categories: {{categories}}
    """
}
```

#### Routers
Intelligent request routing:
```synth
router load_balancer {
    strategy: round_robin
    routes: [
        {name: "primary", target: gpt4},
        {name: "fallback", target: gpt35}
    ]
}
```

#### Guardrails
Safety and compliance checks:
```synth
guardrail safety {
    toxicity_threshold: 0.1
    pii_detection: true
    bias_check: ["gender", "race"]
}
```

## Pipeline Development

### Multi-Stage Pipelines

```synth
pipeline DocumentProcessor {
    // Stage 1: Document extraction
    model extractor {
        provider: "openai"
        model: "gpt-4-vision"
    }

    // Stage 2: Summary generation
    prompt summarizer {
        template: """
        Summarize: {{extracted_text}}
        Max length: 500 words
        """
    }

    // Stage 3: Quality check
    guardrail quality {
        min_length: 100
        max_length: 500
        readability_score: 8.0
    }

    edges: [
        input -> extractor -> summarizer -> quality -> output
    ]
}
```

### Conditional Routing

```synth
router intent_router {
    strategy: conditional
    routes: [
        {
            condition: "intent == 'technical'",
            target: technical_expert
        },
        {
            condition: "intent == 'billing'",
            target: billing_expert
        },
        {
            condition: "true",
            target: general_assistant
        }
    ]
}
```

### A/B Testing

```synth
router experiment {
    strategy: ab_split(0.5)
    metrics: ["response_quality", "latency", "cost"]
    auto_optimize: true
}
```

## Cost Optimization

### Budget Management

```synth
budget monthly_limit {
    amount: 1000.00
    period: monthly
    alert_thresholds: [50, 75, 90]
    auto_shutdown: true
}
```

### Caching Strategies

```synth
cache semantic_cache {
    ttl: 3600
    strategy: semantic_similarity(0.95)
    max_size: 1000
}
```

### Model Selection

```synth
optimizer cost_optimizer {
    strategy: "lowest_cost"
    constraints: {
        min_quality: 0.8,
        max_latency: 2000
    }
}
```

## Monitoring & Dashboards

### Real-time Monitoring

Access the dashboard at `http://localhost:8080` after starting the server:

```bash
synth dashboard --port 8080
```

### Metrics Available
- Request volume and latency
- Cost tracking per pipeline/model
- Cache hit rates
- Error rates and alerts
- A/B test performance

### Custom Dashboards

```synth
dashboard custom {
    panels: [
        {
            type: "line_chart",
            metric: "latency_p95",
            timerange: "1h"
        },
        {
            type: "cost_breakdown",
            groupby: "model"
        }
    ]
}
```

## IDE Integration

### Web IDE
Access the web-based IDE at `http://localhost:8080/ide`

Features:
- Syntax highlighting
- Auto-completion
- Live validation
- Pipeline visualization
- Real-time metrics

### VS Code Extension

Features:
- IntelliSense support
- Pipeline debugging
- Live metrics overlay
- Git integration
- Terminal integration

Shortcuts:
- `Ctrl+Shift+P`: Command palette
- `F5`: Run pipeline
- `Ctrl+S`: Save and validate
- `Shift+F10`: View metrics

## Best Practices

### 1. Start Simple
Begin with basic pipelines and gradually add complexity:
```synth
// Start with this
pipeline Simple {
    model gpt { ... }
    edges: [input -> gpt -> output]
}

// Then add features
pipeline Advanced {
    model gpt { ... }
    guardrail safety { ... }
    cache responses { ... }
    edges: [input -> safety -> gpt -> cache -> output]
}
```

### 2. Use Guardrails
Always include safety checks for production:
```synth
guardrail production_safety {
    toxicity_threshold: 0.05
    pii_detection: true
    profanity_filter: true
    bias_check: ["all"]
}
```

### 3. Implement Caching
Reduce costs with intelligent caching:
```synth
cache smart_cache {
    strategy: semantic_similarity(0.9)
    ttl: 7200
    invalidate_on: ["model_update", "prompt_change"]
}
```

### 4. Monitor Performance
Set up alerts for critical metrics:
```synth
alert high_latency {
    metric: "latency_p95"
    threshold: 3000
    action: "notify"
}
```

### 5. Version Control
Use semantic versioning for pipelines:
```synth
pipeline CustomerSupport {
    version: "2.1.0"
    changelog: """
    - Added multilingual support
    - Improved response quality
    - Fixed timeout issues
    """
}
```

### 6. Test Thoroughly
Create evaluation datasets:
```synth
eval test_suite {
    dataset: "customer_queries_v1"
    metrics: {
        accuracy: 0.95,
        toxicity: 0.01,
        latency_p95: 2000
    }
}
```

### 7. Document Pipelines
Add clear documentation:
```synth
pipeline DocumentedPipeline {
    description: """
    Customer support pipeline with multilingual capabilities.
    Handles technical, billing, and general queries.
    """

    // Component documentation
    model assistant {
        // Uses GPT-4 for complex queries
        provider: "openai"
        model: "gpt-4"
    }
}
```

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check model selection
   - Verify caching is enabled
   - Review pipeline complexity

2. **Budget Exceeded**
   - Enable cost optimization
   - Implement stricter caching
   - Use cheaper models for simple tasks

3. **Low Quality Responses**
   - Adjust temperature settings
   - Improve prompt templates
   - Switch to more capable models

4. **Safety Violations**
   - Lower toxicity thresholds
   - Enable all safety checks
   - Review input validation

### Getting Help

- Documentation: https://docs.synthlang.ai
- Discord: https://discord.gg/synthlang
- GitHub Issues: https://github.com/synth-lang/synth/issues
- Email: support@synthlang.ai

## Next Steps

1. Explore the [examples directory](../examples/)
2. Join our [Discord community](https://discord.gg/synthlang)
3. Read the [API documentation](https://docs.synthlang.ai/api)
4. Try the [interactive tutorials](https://synthlang.ai/tutorials)