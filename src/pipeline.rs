/*!
 * SynthLang - LLM Pipeline Composition DSL
 * Core pipeline primitives for model synthesis and orchestration
 */

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use crate::monitoring::{MonitoringSystem, TraceStatus, LogLevel};

/// Pipeline node representing a model or transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineNode {
    pub id: String,
    pub kind: NodeKind,
    pub config: NodeConfig,
    pub metrics: Arc<RwLock<NodeMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    /// LLM model endpoint
    Model {
        provider: String,
        model: String,
        temperature: f32,
        max_tokens: usize,
    },
    
    /// Prompt template
    Prompt {
        template: String,
        variables: Vec<String>,
    },
    
    /// Router for A/B testing or load balancing
    Router {
        strategy: RoutingStrategy,
        routes: Vec<Route>,
    },
    
    /// Transform node for data processing
    Transform {
        operation: TransformOp,
    },
    
    /// Evaluation node
    Evaluator {
        metrics: Vec<EvalMetric>,
    },
    
    /// Cache node
    Cache {
        ttl_seconds: u64,
        max_size: usize,
    },
    
    /// Guardrail for safety
    Guardrail {
        checks: Vec<SafetyCheck>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub retry_policy: RetryPolicy,
    pub timeout_ms: u64,
    pub rate_limit: Option<RateLimit>,
    pub cost_tracking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff_ms: u64,
    pub exponential: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_second: f64,
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingStrategy {
    RoundRobin,
    WeightedRandom,
    LatencyBased,
    CostOptimized,
    ABTest { split: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub id: String,
    pub weight: f64,
    pub condition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformOp {
    JsonExtract { path: String },
    Regex { pattern: String, replace: String },
    Template { template: String },
    Function { name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalMetric {
    Accuracy,
    Perplexity,
    BLEU,
    ROUGE,
    Toxicity,
    Bias,
    Coherence,
    Relevance,
    Custom { name: String, function: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyCheck {
    PII,
    Toxicity { threshold: f64 },
    Bias { categories: Vec<String> },
    ContentFilter { blocked_terms: Vec<String> },
    RateLimiting,
}

#[derive(Debug, Default)]
pub struct NodeMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_latency_ms: u64,
    pub total_tokens: u64,
    pub total_cost: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Pipeline graph for execution
pub struct Pipeline {
    pub id: String,
    pub name: String,
    pub nodes: HashMap<String, PipelineNode>,
    pub edges: Vec<Edge>,
    pub global_config: PipelineConfig,
    pub execution_engine: Arc<ExecutionEngine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub condition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub max_parallel: usize,
    pub cache_enabled: bool,
    pub tracing_enabled: bool,
    pub cost_limit: Option<f64>,
    pub timeout_ms: u64,
}

/// Execution engine for running pipelines
pub struct ExecutionEngine {
    cache: Arc<RwLock<ResponseCache>>,
    router: Arc<Router>,
    evaluator: Arc<Evaluator>,
    guardrails: Arc<GuardrailEngine>,
    monitoring: Arc<MonitoringSystem>,
}

/// Response cache for model outputs
pub struct ResponseCache {
    entries: HashMap<String, CacheEntry>,
    lru: VecDeque<String>,
    max_size: usize,
}

#[derive(Clone)]
struct CacheEntry {
    response: String,
    timestamp: std::time::Instant,
    ttl: std::time::Duration,
}

/// Router for load balancing and A/B testing
pub struct Router {
    strategies: HashMap<String, Box<dyn RoutingStrategy + Send + Sync>>,
}

/// Evaluator for model outputs
pub struct Evaluator {
    metrics: HashMap<String, Box<dyn EvalMetric + Send + Sync>>,
}

/// Guardrail engine for safety checks
pub struct GuardrailEngine {
    checks: Vec<Box<dyn SafetyCheck + Send + Sync>>,
}

/// Pipeline builder DSL
pub struct PipelineBuilder {
    nodes: Vec<PipelineNode>,
    edges: Vec<Edge>,
    config: PipelineConfig,
}

impl PipelineBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            config: PipelineConfig::default(),
        }
    }

    /// Add a model node
    pub fn model(mut self, id: &str, provider: &str, model: &str) -> Self {
        self.nodes.push(PipelineNode {
            id: id.to_string(),
            kind: NodeKind::Model {
                provider: provider.to_string(),
                model: model.to_string(),
                temperature: 0.7,
                max_tokens: 1000,
            },
            config: NodeConfig::default(),
            metrics: Arc::new(RwLock::new(NodeMetrics::default())),
        });
        self
    }

    /// Add a prompt template
    pub fn prompt(mut self, id: &str, template: &str) -> Self {
        self.nodes.push(PipelineNode {
            id: id.to_string(),
            kind: NodeKind::Prompt {
                template: template.to_string(),
                variables: extract_variables(template),
            },
            config: NodeConfig::default(),
            metrics: Arc::new(RwLock::new(NodeMetrics::default())),
        });
        self
    }

    /// Add a router for A/B testing
    pub fn router(mut self, id: &str, strategy: RoutingStrategy) -> Self {
        self.nodes.push(PipelineNode {
            id: id.to_string(),
            kind: NodeKind::Router {
                strategy,
                routes: Vec::new(),
            },
            config: NodeConfig::default(),
            metrics: Arc::new(RwLock::new(NodeMetrics::default())),
        });
        self
    }

    /// Add an edge between nodes
    pub fn edge(mut self, from: &str, to: &str) -> Self {
        self.edges.push(Edge {
            from: from.to_string(),
            to: to.to_string(),
            condition: None,
        });
        self
    }

    /// Configure caching
    pub fn with_cache(mut self, ttl_seconds: u64) -> Self {
        self.config.cache_enabled = true;
        self
    }

    /// Configure tracing
    pub fn with_tracing(mut self) -> Self {
        self.config.tracing_enabled = true;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> Pipeline {
        let mut nodes = HashMap::new();
        for node in self.nodes {
            nodes.insert(node.id.clone(), node);
        }

        Pipeline {
            id: uuid::Uuid::new_v4().to_string(),
            name: "pipeline".to_string(),
            nodes,
            edges: self.edges,
            global_config: self.config,
            execution_engine: Arc::new(ExecutionEngine::new(
                Arc::new(crate::monitoring::MonitoringSystem::new())
            )),
        }
    }
}

impl ExecutionEngine {
    pub fn new(monitoring: Arc<MonitoringSystem>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(ResponseCache::new(1000))),
            router: Arc::new(Router::new()),
            evaluator: Arc::new(Evaluator::new()),
            guardrails: Arc::new(GuardrailEngine::new()),
            monitoring,
        }
    }

    /// Execute a pipeline with given input
    pub async fn execute(
        &self,
        pipeline: &Pipeline,
        input: PipelineInput,
    ) -> Result<PipelineOutput, PipelineError> {
        // Start pipeline-level tracing
        let pipeline_span_id = self.monitoring.start_span(
            "pipeline_execution", 
            &pipeline.id, 
            "root", 
            None
        );

        // Record pipeline execution start
        let mut labels = std::collections::HashMap::new();
        labels.insert("pipeline_id".to_string(), pipeline.id.clone());
        labels.insert("pipeline_name".to_string(), pipeline.name.clone());
        self.monitoring.record_metric("pipeline_executions_started", 1.0, labels.clone());

        let mut context = ExecutionContext {
            variables: input.variables,
            trace: Vec::new(),
            cost: 0.0,
            start_time: std::time::Instant::now(),
        };

        let result = async {
            // Topological sort for execution order
            let execution_order = self.topological_sort(&pipeline.nodes, &pipeline.edges)?;

            // Execute nodes in order
            for node_id in execution_order {
                let node = pipeline.nodes.get(&node_id)
                    .ok_or(PipelineError::NodeNotFound(node_id.clone()))?;

                // Start node-level tracing
                let node_span_id = self.monitoring.start_span(
                    "node_execution",
                    &pipeline.id,
                    &node_id,
                    Some(pipeline_span_id.clone())
                );

                // Log node execution start
                let mut fields = std::collections::HashMap::new();
                fields.insert("node_type".to_string(), format!("{:?}", node.kind));
                self.monitoring.add_span_log(
                    &node_span_id,
                    LogLevel::Info,
                    "Starting node execution",
                    fields
                );

                // Check cache if enabled
                if pipeline.global_config.cache_enabled {
                    if let Some(cached) = self.check_cache(&node_id, &context).await {
                        self.monitoring.add_span_log(
                            &node_span_id,
                            LogLevel::Info,
                            "Cache hit",
                            std::collections::HashMap::new()
                        );
                        
                        // Record cache hit metric
                        let mut cache_labels = labels.clone();
                        cache_labels.insert("node_id".to_string(), node_id.clone());
                        cache_labels.insert("cache_result".to_string(), "hit".to_string());
                        self.monitoring.record_metric("cache_requests", 1.0, cache_labels);

                        context.trace.push(TraceEntry {
                            node_id: node_id.clone(),
                            cached: true,
                            latency_ms: 0,
                            tokens: 0,
                            cost: 0.0,
                        });

                        self.monitoring.finish_span(&node_span_id, TraceStatus::Success);
                        continue;
                    } else {
                        // Record cache miss
                        let mut cache_labels = labels.clone();
                        cache_labels.insert("node_id".to_string(), node_id.clone());
                        cache_labels.insert("cache_result".to_string(), "miss".to_string());
                        self.monitoring.record_metric("cache_requests", 1.0, cache_labels);
                    }
                }

                // Execute node with monitoring
                let node_start = std::time::Instant::now();
                let result = self.execute_node_with_monitoring(node, &mut context, &node_span_id).await;
                let node_duration = node_start.elapsed();

                match result {
                    Ok(result) => {
                        // Record successful node execution
                        let mut node_labels = labels.clone();
                        node_labels.insert("node_id".to_string(), node_id.clone());
                        node_labels.insert("status".to_string(), "success".to_string());
                        self.monitoring.record_metric("node_executions", 1.0, node_labels.clone());
                        self.monitoring.record_metric("node_duration_ms", node_duration.as_millis() as f64, node_labels);

                        // Run guardrails with monitoring
                        if let Err(e) = self.guardrails.check(&result).await {
                            self.monitoring.add_span_log(
                                &node_span_id,
                                LogLevel::Error,
                                "Guardrail violation",
                                std::collections::HashMap::new()
                            );
                            self.monitoring.finish_span(&node_span_id, TraceStatus::Error(e.to_string()));
                            return Err(PipelineError::GuardrailViolation(e.to_string()));
                        }

                        self.monitoring.finish_span(&node_span_id, TraceStatus::Success);
                    }
                    Err(e) => {
                        // Record failed node execution
                        let mut node_labels = labels.clone();
                        node_labels.insert("node_id".to_string(), node_id.clone());
                        node_labels.insert("status".to_string(), "error".to_string());
                        self.monitoring.record_metric("node_executions", 1.0, node_labels);

                        self.monitoring.add_span_log(
                            &node_span_id,
                            LogLevel::Error,
                            "Node execution failed",
                            std::collections::HashMap::new()
                        );
                        self.monitoring.finish_span(&node_span_id, TraceStatus::Error(format!("{:?}", e)));
                        return Err(e);
                    }
                }

                // Cache result if enabled
                if pipeline.global_config.cache_enabled {
                    self.cache_result(&node_id, &result).await;
                }
            }

            Ok(PipelineOutput {
                result: context.variables.get("output").cloned().unwrap_or_default(),
                trace: context.trace,
                total_cost: context.cost,
                total_latency_ms: context.start_time.elapsed().as_millis() as u64,
            })
        }.await;

        // Finish pipeline span with appropriate status
        match &result {
            Ok(_) => {
                // Record successful pipeline execution
                labels.insert("status".to_string(), "success".to_string());
                self.monitoring.record_metric("pipeline_executions_completed", 1.0, labels.clone());
                self.monitoring.record_metric("pipeline_cost_usd", context.cost, labels.clone());
                self.monitoring.record_metric("pipeline_total_tokens", 
                    context.trace.iter().map(|t| t.tokens as f64).sum(), labels);
                self.monitoring.finish_span(&pipeline_span_id, TraceStatus::Success);
            }
            Err(e) => {
                // Record failed pipeline execution
                labels.insert("status".to_string(), "error".to_string());
                self.monitoring.record_metric("pipeline_executions_completed", 1.0, labels);
                self.monitoring.finish_span(&pipeline_span_id, TraceStatus::Error(format!("{:?}", e)));
            }
        }

        result
    }

    async fn execute_node_with_monitoring(
        &self,
        node: &PipelineNode,
        context: &mut ExecutionContext,
        span_id: &str,
    ) -> Result<NodeResult, PipelineError> {
        let start = std::time::Instant::now();

        let result = match &node.kind {
            NodeKind::Model { provider, model, temperature, max_tokens } => {
                self.monitoring.add_span_log(
                    span_id,
                    LogLevel::Info,
                    "Executing model",
                    [("provider".to_string(), provider.clone()), 
                     ("model".to_string(), model.clone())].into_iter().collect()
                );
                self.execute_model(provider, model, *temperature, *max_tokens, context).await?
            }
            NodeKind::Prompt { template, variables } => {
                self.monitoring.add_span_log(
                    span_id,
                    LogLevel::Info,
                    "Processing prompt template",
                    [("variables_count".to_string(), variables.len().to_string())].into_iter().collect()
                );
                self.execute_prompt(template, variables, context)?
            }
            NodeKind::Router { strategy, routes } => {
                self.monitoring.add_span_log(
                    span_id,
                    LogLevel::Info,
                    "Routing request",
                    [("routes_count".to_string(), routes.len().to_string())].into_iter().collect()
                );
                self.execute_router(strategy, routes, context).await?
            }
            _ => {
                self.monitoring.add_span_log(
                    span_id,
                    LogLevel::Warn,
                    "Unknown node type",
                    std::collections::HashMap::new()
                );
                NodeResult::default()
            }
        };

        let latency_ms = start.elapsed().as_millis() as u64;
        
        // Log execution completion
        let mut completion_fields = std::collections::HashMap::new();
        completion_fields.insert("latency_ms".to_string(), latency_ms.to_string());
        completion_fields.insert("tokens".to_string(), result.tokens.to_string());
        completion_fields.insert("cost".to_string(), result.cost.to_string());
        self.monitoring.add_span_log(
            span_id,
            LogLevel::Info,
            "Node execution completed",
            completion_fields
        );
        
        context.trace.push(TraceEntry {
            node_id: node.id.clone(),
            cached: false,
            latency_ms,
            tokens: result.tokens,
            cost: result.cost,
        });

        // Update node metrics
        self.update_metrics(node, &result).await;

        Ok(result)
    }

    async fn execute_node(
        &self,
        node: &PipelineNode,
        context: &mut ExecutionContext,
    ) -> Result<NodeResult, PipelineError> {
        let start = std::time::Instant::now();

        let result = match &node.kind {
            NodeKind::Model { provider, model, temperature, max_tokens } => {
                self.execute_model(provider, model, *temperature, *max_tokens, context).await?
            }
            NodeKind::Prompt { template, variables } => {
                self.execute_prompt(template, variables, context)?
            }
            NodeKind::Router { strategy, routes } => {
                self.execute_router(strategy, routes, context).await?
            }
            _ => NodeResult::default(),
        };

        let latency_ms = start.elapsed().as_millis() as u64;
        
        context.trace.push(TraceEntry {
            node_id: node.id.clone(),
            cached: false,
            latency_ms,
            tokens: result.tokens,
            cost: result.cost,
        });

        Ok(result)
    }

    async fn execute_model(
        &self,
        provider: &str,
        model: &str,
        temperature: f32,
        max_tokens: usize,
        context: &mut ExecutionContext,
    ) -> Result<NodeResult, PipelineError> {
        // Model execution logic would go here
        Ok(NodeResult {
            output: "model response".to_string(),
            tokens: 100,
            cost: 0.002,
        })
    }

    fn execute_prompt(
        &self,
        template: &str,
        variables: &[String],
        context: &mut ExecutionContext,
    ) -> Result<NodeResult, PipelineError> {
        let mut result = template.to_string();
        for var in variables {
            if let Some(value) = context.variables.get(var) {
                result = result.replace(&format!("{{{{{}}}}}", var), value);
            }
        }
        Ok(NodeResult {
            output: result,
            tokens: 0,
            cost: 0.0,
        })
    }

    async fn execute_router(
        &self,
        strategy: &RoutingStrategy,
        routes: &[Route],
        context: &mut ExecutionContext,
    ) -> Result<NodeResult, PipelineError> {
        // Routing logic would go here
        Ok(NodeResult::default())
    }

    fn topological_sort(
        &self,
        nodes: &HashMap<String, PipelineNode>,
        edges: &[Edge],
    ) -> Result<Vec<String>, PipelineError> {
        // Topological sort implementation
        let mut result = Vec::new();
        let mut visited = HashMap::new();
        
        for node_id in nodes.keys() {
            if !visited.contains_key(node_id) {
                self.dfs(node_id, nodes, edges, &mut visited, &mut result)?;
            }
        }
        
        result.reverse();
        Ok(result)
    }

    fn dfs(
        &self,
        node_id: &str,
        nodes: &HashMap<String, PipelineNode>,
        edges: &[Edge],
        visited: &mut HashMap<String, bool>,
        result: &mut Vec<String>,
    ) -> Result<(), PipelineError> {
        visited.insert(node_id.to_string(), true);
        
        for edge in edges {
            if edge.from == node_id && !visited.contains_key(&edge.to) {
                self.dfs(&edge.to, nodes, edges, visited, result)?;
            }
        }
        
        result.push(node_id.to_string());
        Ok(())
    }

    async fn check_cache(
        &self,
        node_id: &str,
        context: &ExecutionContext,
    ) -> Option<String> {
        let cache = self.cache.read().await;
        cache.get(node_id)
    }

    async fn cache_result(&self, node_id: &str, result: &NodeResult) {
        let mut cache = self.cache.write().await;
        cache.put(node_id.to_string(), result.output.clone());
    }

    async fn update_metrics(&self, node: &PipelineNode, result: &NodeResult) {
        let mut metrics = node.metrics.write().await;
        metrics.total_requests += 1;
        metrics.successful_requests += 1;
        metrics.total_tokens += result.tokens as u64;
        metrics.total_cost += result.cost;
    }
}

impl ResponseCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            lru: VecDeque::new(),
            max_size,
        }
    }

    fn get(&self, key: &str) -> Option<String> {
        self.entries.get(key).and_then(|entry| {
            if entry.timestamp.elapsed() < entry.ttl {
                Some(entry.response.clone())
            } else {
                None
            }
        })
    }

    fn put(&mut self, key: String, value: String) {
        if self.entries.len() >= self.max_size {
            if let Some(oldest) = self.lru.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        
        self.entries.insert(key.clone(), CacheEntry {
            response: value,
            timestamp: std::time::Instant::now(),
            ttl: std::time::Duration::from_secs(300),
        });
        self.lru.push_back(key);
    }
}

impl Router {
    fn new() -> Self {
        Self {
            strategies: HashMap::new(),
        }
    }
}

impl Evaluator {
    fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
}

impl GuardrailEngine {
    fn new() -> Self {
        Self {
            checks: Vec::new(),
        }
    }

    async fn check(&self, result: &NodeResult) -> Result<(), GuardrailError> {
        // Run safety checks
        Ok(())
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            retry_policy: RetryPolicy {
                max_attempts: 3,
                backoff_ms: 1000,
                exponential: true,
            },
            timeout_ms: 30000,
            rate_limit: None,
            cost_tracking: true,
        }
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_parallel: 10,
            cache_enabled: true,
            tracing_enabled: true,
            cost_limit: None,
            timeout_ms: 60000,
        }
    }
}

/// Input to pipeline execution
#[derive(Debug, Clone)]
pub struct PipelineInput {
    pub variables: HashMap<String, String>,
}

/// Output from pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineOutput {
    pub result: String,
    pub trace: Vec<TraceEntry>,
    pub total_cost: f64,
    pub total_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub node_id: String,
    pub cached: bool,
    pub latency_ms: u64,
    pub tokens: usize,
    pub cost: f64,
}

#[derive(Debug, Default)]
struct NodeResult {
    output: String,
    tokens: usize,
    cost: f64,
}

struct ExecutionContext {
    variables: HashMap<String, String>,
    trace: Vec<TraceEntry>,
    cost: f64,
    start_time: std::time::Instant,
}

#[derive(Debug)]
pub enum PipelineError {
    NodeNotFound(String),
    ExecutionError(String),
    GuardrailViolation(String),
    Timeout,
    CostLimitExceeded,
}

#[derive(Debug)]
pub enum GuardrailError {
    PIIDetected,
    ToxicityThresholdExceeded,
    BiasDetected,
    BlockedContent,
}

fn extract_variables(template: &str) -> Vec<String> {
    // Extract {{variable}} patterns
    let mut vars = Vec::new();
    let re = regex::Regex::new(r"\{\{(\w+)\}\}").unwrap();
    for cap in re.captures_iter(template) {
        vars.push(cap[1].to_string());
    }
    vars
}

#[async_trait]
pub trait RoutingStrategy {
    async fn route(&self, routes: &[Route]) -> String;
}

#[async_trait]
pub trait EvalMetric {
    async fn evaluate(&self, output: &str, expected: &str) -> f64;
}

#[async_trait]
pub trait SafetyCheck {
    async fn check(&self, content: &str) -> Result<(), GuardrailError>;
}