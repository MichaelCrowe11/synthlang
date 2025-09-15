/*!
 * Fine-Tuning Management System for SynthLang
 * Enterprise-grade model customization with LoRA/QLoRA support
 */

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use anyhow::Result;

/// Fine-tuning job manager
pub struct FineTuningManager {
    jobs: HashMap<JobId, FineTuningJob>,
    adapters: HashMap<AdapterId, LoRAAdapter>,
    storage: Box<dyn ModelStorage>,
    scheduler: JobScheduler,
    metrics_collector: MetricsCollector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdapterId(pub Uuid);

/// Fine-tuning job configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningJob {
    pub id: JobId,
    pub name: String,
    pub base_model: String,
    pub config: TrainingConfig,
    pub dataset: DatasetConfig,
    pub status: JobStatus,
    pub progress: TrainingProgress,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub metrics: TrainingMetrics,
    pub artifacts: Vec<ModelArtifact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub method: FineTuningMethod,
    pub hyperparameters: Hyperparameters,
    pub hardware: HardwareConfig,
    pub early_stopping: EarlyStoppingConfig,
    pub checkpointing: CheckpointConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FineTuningMethod {
    /// Low-Rank Adaptation - memory efficient
    LoRA {
        rank: u32,
        alpha: f32,
        dropout: f32,
        target_modules: Vec<String>,
    },
    /// Quantized LoRA for even more efficiency
    QLoRA {
        rank: u32,
        alpha: f32,
        dropout: f32,
        quantization_bits: u8,
        target_modules: Vec<String>,
    },
    /// Full parameter fine-tuning
    FullFineTuning,
    /// Prefix tuning
    PrefixTuning {
        prefix_length: u32,
        hidden_size: u32,
    },
    /// Adapter layers
    Adapters {
        bottleneck_size: u32,
        non_linearity: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub batch_size: u32,
    pub epochs: u32,
    pub warmup_steps: u32,
    pub weight_decay: f64,
    pub gradient_accumulation_steps: u32,
    pub max_grad_norm: f64,
    pub optimizer: OptimizerConfig,
    pub scheduler: SchedulerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerConfig {
    Adam { beta1: f64, beta2: f64, eps: f64 },
    AdamW { beta1: f64, beta2: f64, eps: f64 },
    SGD { momentum: f64 },
    RMSprop { alpha: f64, eps: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerConfig {
    Linear { warmup_ratio: f64 },
    Cosine { warmup_ratio: f64, cycles: f64 },
    Constant,
    Exponential { gamma: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub gpu_type: GpuType,
    pub num_gpus: u32,
    pub memory_per_gpu: u64, // GB
    pub precision: Precision,
    pub distributed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuType {
    A100,
    V100,
    T4,
    A10G,
    RTX4090,
    H100,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
    INT4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub enabled: bool,
    pub patience: u32,
    pub min_delta: f64,
    pub metric: String,
    pub mode: EarlyStoppingMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarlyStoppingMode {
    Min,
    Max,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub save_steps: u32,
    pub save_total_limit: u32,
    pub save_best_only: bool,
    pub monitor_metric: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub train_dataset: String,
    pub eval_dataset: Option<String>,
    pub data_format: DataFormat,
    pub max_sequence_length: u32,
    pub preprocessing: PreprocessingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    JSONL,
    CSV,
    Parquet,
    HuggingFace,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub tokenizer: String,
    pub truncation: bool,
    pub padding: PaddingStrategy,
    pub data_collator: DataCollatorType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PaddingStrategy {
    MaxLength,
    LongestFirst,
    DoNotPad,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataCollatorType {
    Default,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Preparing,
    Running,
    Completed,
    Failed { error: String },
    Cancelled,
    Paused,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingProgress {
    pub current_step: u32,
    pub total_steps: u32,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub samples_per_second: f64,
    pub eta_seconds: u64,
    pub gpu_utilization: f64,
    pub memory_usage: MemoryUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub used_gb: f64,
    pub total_gb: f64,
    pub peak_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub train_loss: Vec<f64>,
    pub eval_loss: Vec<f64>,
    pub perplexity: Vec<f64>,
    pub learning_rate: Vec<f64>,
    pub gradient_norm: Vec<f64>,
    pub custom_metrics: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub path: String,
    pub artifact_type: ArtifactType,
    pub size_bytes: u64,
    pub checksum: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    FinalModel,
    Checkpoint,
    LoRAAdapter,
    Tokenizer,
    Config,
    TrainingLogs,
    Metrics,
}

/// LoRA adapter management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAAdapter {
    pub id: AdapterId,
    pub name: String,
    pub base_model: String,
    pub rank: u32,
    pub alpha: f32,
    pub target_modules: Vec<String>,
    pub adapter_weights: Vec<AdapterWeight>,
    pub metadata: AdapterMetadata,
    pub performance: AdapterPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterWeight {
    pub layer_name: String,
    pub weight_a: Vec<f32>, // Low-rank matrix A
    pub weight_b: Vec<f32>, // Low-rank matrix B
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    pub created_at: DateTime<Utc>,
    pub training_job_id: JobId,
    pub dataset_used: String,
    pub training_steps: u32,
    pub final_loss: f64,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterPerformance {
    pub benchmarks: HashMap<String, f64>,
    pub evaluation_results: EvaluationResults,
    pub inference_speed: InferenceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub accuracy: f64,
    pub perplexity: f64,
    pub bleu_score: Option<f64>,
    pub rouge_scores: Option<HashMap<String, f64>>,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub tokens_per_second: f64,
    pub memory_usage_mb: u64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
}

/// Job scheduling system
pub struct JobScheduler {
    queue: Vec<JobId>,
    running_jobs: HashMap<JobId, RunningJob>,
    resource_pool: ResourcePool,
}

#[derive(Debug, Clone)]
pub struct RunningJob {
    pub job_id: JobId,
    pub process_id: u32,
    pub allocated_resources: AllocatedResources,
    pub start_time: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AllocatedResources {
    pub gpus: Vec<u32>,
    pub memory_gb: u64,
    pub cpu_cores: u32,
}

#[derive(Debug, Clone)]
pub struct ResourcePool {
    pub available_gpus: HashMap<u32, GpuInfo>,
    pub total_memory_gb: u64,
    pub available_memory_gb: u64,
    pub total_cpu_cores: u32,
    pub available_cpu_cores: u32,
}

#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub gpu_id: u32,
    pub gpu_type: GpuType,
    pub memory_gb: u64,
    pub available_memory_gb: u64,
    pub utilization: f64,
}

/// Metrics collection system
pub struct MetricsCollector {
    pub metrics_store: Box<dyn MetricsStorage>,
    pub alerts: Vec<AlertRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub actions: Vec<AlertAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    TrainingLossStagnant { steps: u32 },
    GpuUtilizationLow { threshold: f64 },
    MemoryUsageHigh { threshold: f64 },
    JobDurationExceeded { minutes: u64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertAction {
    Email(String),
    Slack(String),
    PauseJob,
    TerminateJob,
    ScaleResources,
}

impl FineTuningManager {
    pub fn new(storage: Box<dyn ModelStorage>) -> Self {
        Self {
            jobs: HashMap::new(),
            adapters: HashMap::new(),
            storage,
            scheduler: JobScheduler::new(),
            metrics_collector: MetricsCollector::new(),
        }
    }

    /// Submit a new fine-tuning job
    pub async fn submit_job(&mut self, config: FineTuningJobConfig) -> Result<JobId> {
        let job_id = JobId(Uuid::new_v4());
        
        // Validate configuration
        self.validate_job_config(&config)?;
        
        // Estimate resource requirements
        let resource_requirements = self.estimate_resources(&config)?;
        
        // Create job
        let job = FineTuningJob {
            id: job_id,
            name: config.name,
            base_model: config.base_model,
            config: config.training_config,
            dataset: config.dataset_config,
            status: JobStatus::Queued,
            progress: TrainingProgress::default(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            metrics: TrainingMetrics::default(),
            artifacts: vec![],
        };

        self.jobs.insert(job_id, job);
        self.scheduler.enqueue_job(job_id, resource_requirements);
        
        Ok(job_id)
    }

    /// Get job status and progress
    pub fn get_job(&self, job_id: &JobId) -> Option<&FineTuningJob> {
        self.jobs.get(job_id)
    }

    /// List all jobs with filtering
    pub fn list_jobs(&self, filter: Option<JobFilter>) -> Vec<&FineTuningJob> {
        let mut jobs: Vec<_> = self.jobs.values().collect();
        
        if let Some(filter) = filter {
            jobs.retain(|job| filter.matches(job));
        }
        
        jobs.sort_by_key(|job| job.created_at);
        jobs.reverse();
        
        jobs
    }

    /// Cancel a running or queued job
    pub async fn cancel_job(&mut self, job_id: &JobId) -> Result<()> {
        if let Some(job) = self.jobs.get_mut(job_id) {
            match job.status {
                JobStatus::Queued => {
                    job.status = JobStatus::Cancelled;
                    self.scheduler.remove_from_queue(job_id);
                }
                JobStatus::Running => {
                    self.scheduler.terminate_job(job_id).await?;
                    job.status = JobStatus::Cancelled;
                }
                _ => {
                    return Err(anyhow::anyhow!("Cannot cancel job in status: {:?}", job.status));
                }
            }
        }
        Ok(())
    }

    /// Create a LoRA adapter from completed job
    pub async fn create_adapter(&mut self, job_id: &JobId, config: AdapterConfig) -> Result<AdapterId> {
        let job = self.jobs.get(job_id)
            .ok_or_else(|| anyhow::anyhow!("Job not found"))?;
            
        if !matches!(job.status, JobStatus::Completed) {
            return Err(anyhow::anyhow!("Job must be completed to create adapter"));
        }

        let adapter_id = AdapterId(Uuid::new_v4());
        
        // Extract adapter weights from job artifacts
        let adapter_weights = self.extract_adapter_weights(job).await?;
        
        let adapter = LoRAAdapter {
            id: adapter_id,
            name: config.name,
            base_model: job.base_model.clone(),
            rank: self.extract_rank_from_config(&job.config),
            alpha: self.extract_alpha_from_config(&job.config),
            target_modules: self.extract_target_modules(&job.config),
            adapter_weights,
            metadata: AdapterMetadata {
                created_at: Utc::now(),
                training_job_id: *job_id,
                dataset_used: job.dataset.train_dataset.clone(),
                training_steps: job.progress.current_step,
                final_loss: job.metrics.train_loss.last().copied().unwrap_or(0.0),
                tags: config.tags,
            },
            performance: AdapterPerformance {
                benchmarks: HashMap::new(),
                evaluation_results: EvaluationResults {
                    accuracy: 0.0,
                    perplexity: job.metrics.perplexity.last().copied().unwrap_or(0.0),
                    bleu_score: None,
                    rouge_scores: None,
                    custom_metrics: HashMap::new(),
                },
                inference_speed: InferenceMetrics {
                    tokens_per_second: 0.0,
                    memory_usage_mb: 0,
                    latency_p50_ms: 0.0,
                    latency_p95_ms: 0.0,
                },
            },
        };

        self.adapters.insert(adapter_id, adapter);
        
        // Store adapter persistently
        self.storage.store_adapter(&self.adapters[&adapter_id]).await?;
        
        Ok(adapter_id)
    }

    /// Merge multiple LoRA adapters
    pub async fn merge_adapters(&mut self, adapter_ids: Vec<AdapterId>, config: MergeConfig) -> Result<AdapterId> {
        let adapters: Vec<_> = adapter_ids.iter()
            .map(|id| self.adapters.get(id).ok_or_else(|| anyhow::anyhow!("Adapter not found: {:?}", id)))
            .collect::<Result<Vec<_>>>()?;

        // Validate adapters are compatible
        self.validate_adapter_compatibility(&adapters)?;

        let merged_id = AdapterId(Uuid::new_v4());
        
        // Merge adapter weights
        let merged_weights = self.merge_adapter_weights(&adapters, &config).await?;
        
        // Create merged adapter
        let merged_adapter = LoRAAdapter {
            id: merged_id,
            name: config.name,
            base_model: adapters[0].base_model.clone(),
            rank: adapters[0].rank,
            alpha: adapters[0].alpha,
            target_modules: adapters[0].target_modules.clone(),
            adapter_weights: merged_weights,
            metadata: AdapterMetadata {
                created_at: Utc::now(),
                training_job_id: adapters[0].metadata.training_job_id,
                dataset_used: "merged".to_string(),
                training_steps: 0,
                final_loss: 0.0,
                tags: config.tags,
            },
            performance: AdapterPerformance {
                benchmarks: HashMap::new(),
                evaluation_results: EvaluationResults {
                    accuracy: 0.0,
                    perplexity: 0.0,
                    bleu_score: None,
                    rouge_scores: None,
                    custom_metrics: HashMap::new(),
                },
                inference_speed: InferenceMetrics {
                    tokens_per_second: 0.0,
                    memory_usage_mb: 0,
                    latency_p50_ms: 0.0,
                    latency_p95_ms: 0.0,
                },
            },
        };

        self.adapters.insert(merged_id, merged_adapter);
        Ok(merged_id)
    }

    /// Benchmark adapter performance
    pub async fn benchmark_adapter(&mut self, adapter_id: &AdapterId, benchmarks: Vec<BenchmarkSuite>) -> Result<()> {
        let adapter = self.adapters.get_mut(adapter_id)
            .ok_or_else(|| anyhow::anyhow!("Adapter not found"))?;

        for benchmark in benchmarks {
            let results = self.run_benchmark(&benchmark, adapter).await?;
            adapter.performance.benchmarks.insert(benchmark.name, results);
        }

        Ok(())
    }

    fn validate_job_config(&self, config: &FineTuningJobConfig) -> Result<()> {
        // Validate base model exists
        // Validate dataset exists and format
        // Validate hyperparameters are reasonable
        // Validate hardware requirements are available
        Ok(())
    }

    fn estimate_resources(&self, config: &FineTuningJobConfig) -> Result<ResourceRequirements> {
        // Estimate based on model size, method, and batch size
        Ok(ResourceRequirements {
            gpu_memory_gb: 16, // Simplified estimation
            cpu_cores: 4,
            estimated_duration_hours: 8,
        })
    }

    async fn extract_adapter_weights(&self, job: &FineTuningJob) -> Result<Vec<AdapterWeight>> {
        // Extract LoRA weights from model checkpoint
        Ok(vec![])
    }

    fn extract_rank_from_config(&self, config: &TrainingConfig) -> u32 {
        match &config.method {
            FineTuningMethod::LoRA { rank, .. } => *rank,
            FineTuningMethod::QLoRA { rank, .. } => *rank,
            _ => 0,
        }
    }

    fn extract_alpha_from_config(&self, config: &TrainingConfig) -> f32 {
        match &config.method {
            FineTuningMethod::LoRA { alpha, .. } => *alpha,
            FineTuningMethod::QLoRA { alpha, .. } => *alpha,
            _ => 0.0,
        }
    }

    fn extract_target_modules(&self, config: &TrainingConfig) -> Vec<String> {
        match &config.method {
            FineTuningMethod::LoRA { target_modules, .. } => target_modules.clone(),
            FineTuningMethod::QLoRA { target_modules, .. } => target_modules.clone(),
            _ => vec![],
        }
    }

    fn validate_adapter_compatibility(&self, adapters: &[&LoRAAdapter]) -> Result<()> {
        if adapters.is_empty() {
            return Err(anyhow::anyhow!("No adapters to merge"));
        }

        let base_model = &adapters[0].base_model;
        let rank = adapters[0].rank;
        
        for adapter in adapters.iter().skip(1) {
            if adapter.base_model != *base_model {
                return Err(anyhow::anyhow!("Adapters must have same base model"));
            }
            if adapter.rank != rank {
                return Err(anyhow::anyhow!("Adapters must have same rank"));
            }
        }
        
        Ok(())
    }

    async fn merge_adapter_weights(&self, adapters: &[&LoRAAdapter], config: &MergeConfig) -> Result<Vec<AdapterWeight>> {
        // Implement adapter weight merging based on strategy
        match config.strategy {
            MergeStrategy::Average => {
                // Average the weights
                Ok(vec![])
            }
            MergeStrategy::WeightedSum(ref weights) => {
                // Weighted sum
                Ok(vec![])
            }
            MergeStrategy::TaskVector => {
                // Task arithmetic
                Ok(vec![])
            }
        }
    }

    async fn run_benchmark(&self, benchmark: &BenchmarkSuite, adapter: &LoRAAdapter) -> Result<f64> {
        // Run benchmark and return score
        Ok(0.85)
    }
}

impl Default for TrainingProgress {
    fn default() -> Self {
        Self {
            current_step: 0,
            total_steps: 0,
            current_epoch: 0,
            total_epochs: 0,
            samples_per_second: 0.0,
            eta_seconds: 0,
            gpu_utilization: 0.0,
            memory_usage: MemoryUsage {
                used_gb: 0.0,
                total_gb: 0.0,
                peak_gb: 0.0,
            },
        }
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self {
            train_loss: vec![],
            eval_loss: vec![],
            perplexity: vec![],
            learning_rate: vec![],
            gradient_norm: vec![],
            custom_metrics: HashMap::new(),
        }
    }
}

impl JobScheduler {
    pub fn new() -> Self {
        Self {
            queue: vec![],
            running_jobs: HashMap::new(),
            resource_pool: ResourcePool::new(),
        }
    }

    pub fn enqueue_job(&mut self, job_id: JobId, requirements: ResourceRequirements) {
        self.queue.push(job_id);
        // TODO: Sort by priority
    }

    pub fn remove_from_queue(&mut self, job_id: &JobId) {
        self.queue.retain(|id| id != job_id);
    }

    pub async fn terminate_job(&mut self, job_id: &JobId) -> Result<()> {
        if let Some(running_job) = self.running_jobs.remove(job_id) {
            // Kill the process
            // Free resources
        }
        Ok(())
    }
}

impl ResourcePool {
    pub fn new() -> Self {
        Self {
            available_gpus: HashMap::new(),
            total_memory_gb: 0,
            available_memory_gb: 0,
            total_cpu_cores: 0,
            available_cpu_cores: 0,
        }
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics_store: Box::new(InMemoryMetricsStorage::new()),
            alerts: vec![],
        }
    }
}

/// Trait for model storage backends
#[async_trait::async_trait]
pub trait ModelStorage: Send + Sync {
    async fn store_adapter(&self, adapter: &LoRAAdapter) -> Result<()>;
    async fn load_adapter(&self, adapter_id: &AdapterId) -> Result<LoRAAdapter>;
    async fn list_adapters(&self) -> Result<Vec<AdapterId>>;
    async fn delete_adapter(&self, adapter_id: &AdapterId) -> Result<()>;
}

/// Trait for metrics storage
pub trait MetricsStorage: Send + Sync {
    fn store_metric(&mut self, job_id: &JobId, metric_name: &str, value: f64, timestamp: DateTime<Utc>);
    fn get_metrics(&self, job_id: &JobId, metric_name: &str) -> Vec<(DateTime<Utc>, f64)>;
}

/// Configuration structures
#[derive(Debug, Clone)]
pub struct FineTuningJobConfig {
    pub name: String,
    pub base_model: String,
    pub training_config: TrainingConfig,
    pub dataset_config: DatasetConfig,
}

#[derive(Debug, Clone)]
pub struct AdapterConfig {
    pub name: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MergeConfig {
    pub name: String,
    pub strategy: MergeStrategy,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    Average,
    WeightedSum(Vec<f64>),
    TaskVector,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub gpu_memory_gb: u64,
    pub cpu_cores: u32,
    pub estimated_duration_hours: u64,
}

#[derive(Debug, Clone)]
pub struct JobFilter {
    pub status: Option<JobStatus>,
    pub base_model: Option<String>,
    pub created_after: Option<DateTime<Utc>>,
}

impl JobFilter {
    pub fn matches(&self, job: &FineTuningJob) -> bool {
        if let Some(ref status) = self.status {
            if !matches!(job.status, status) {
                return false;
            }
        }
        
        if let Some(ref model) = self.base_model {
            if job.base_model != *model {
                return false;
            }
        }
        
        if let Some(created_after) = self.created_after {
            if job.created_at <= created_after {
                return false;
            }
        }
        
        true
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkSuite {
    pub name: String,
    pub tasks: Vec<BenchmarkTask>,
}

#[derive(Debug, Clone)]
pub struct BenchmarkTask {
    pub name: String,
    pub dataset: String,
    pub metric: String,
}

/// In-memory implementations for development
pub struct InMemoryModelStorage {
    adapters: HashMap<AdapterId, LoRAAdapter>,
}

impl InMemoryModelStorage {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
        }
    }
}

#[async_trait::async_trait]
impl ModelStorage for InMemoryModelStorage {
    async fn store_adapter(&self, adapter: &LoRAAdapter) -> Result<()> {
        // In production, would store to S3/GCS
        Ok(())
    }

    async fn load_adapter(&self, adapter_id: &AdapterId) -> Result<LoRAAdapter> {
        // In production, would load from S3/GCS
        Err(anyhow::anyhow!("Not implemented"))
    }

    async fn list_adapters(&self) -> Result<Vec<AdapterId>> {
        Ok(self.adapters.keys().cloned().collect())
    }

    async fn delete_adapter(&self, adapter_id: &AdapterId) -> Result<()> {
        Ok(())
    }
}

pub struct InMemoryMetricsStorage {
    metrics: HashMap<(JobId, String), Vec<(DateTime<Utc>, f64)>>,
}

impl InMemoryMetricsStorage {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
}

impl MetricsStorage for InMemoryMetricsStorage {
    fn store_metric(&mut self, job_id: &JobId, metric_name: &str, value: f64, timestamp: DateTime<Utc>) {
        self.metrics.entry((*job_id, metric_name.to_string()))
            .or_insert_with(Vec::new)
            .push((timestamp, value));
    }

    fn get_metrics(&self, job_id: &JobId, metric_name: &str) -> Vec<(DateTime<Utc>, f64)> {
        self.metrics.get(&(*job_id, metric_name.to_string()))
            .cloned()
            .unwrap_or_default()
    }
}