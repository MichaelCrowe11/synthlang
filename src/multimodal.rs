use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::path::PathBuf;
use bytes::Bytes;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalPipeline {
    pub id: String,
    pub name: String,
    pub description: String,
    pub input_modalities: Vec<Modality>,
    pub output_modalities: Vec<Modality>,
    pub processors: Vec<MultiModalProcessor>,
    pub fusion_strategy: FusionStrategy,
    pub quality_settings: QualitySettings,
    pub created_at: u64,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum Modality {
    Text {
        format: TextFormat,
        language: Option<String>,
        encoding: String,
    },
    Image {
        format: ImageFormat,
        max_resolution: (u32, u32),
        color_space: ColorSpace,
    },
    Audio {
        format: AudioFormat,
        sample_rate: u32,
        channels: u8,
        bitrate: Option<u32>,
    },
    Video {
        format: VideoFormat,
        resolution: (u32, u32),
        fps: f32,
        codec: String,
    },
    Document {
        format: DocumentFormat,
        extract_images: bool,
        extract_tables: bool,
    },
    ThreeD {
        format: ThreeDFormat,
        vertex_count: Option<u32>,
        texture_support: bool,
    },
    Sensor {
        sensor_type: SensorType,
        sampling_rate: f64,
        data_format: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum TextFormat {
    PlainText,
    Markdown,
    HTML,
    JSON,
    XML,
    Code(String), // Programming language
    LaTeX,
    RTF,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum ImageFormat {
    JPEG,
    PNG,
    WEBP,
    TIFF,
    BMP,
    SVG,
    HEIC,
    RAW(String), // Camera RAW format
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum ColorSpace {
    RGB,
    RGBA,
    CMYK,
    HSV,
    LAB,
    Grayscale,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum AudioFormat {
    WAV,
    MP3,
    FLAC,
    AAC,
    OGG,
    M4A,
    OPUS,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum VideoFormat {
    MP4,
    AVI,
    MOV,
    WEBM,
    MKV,
    FLV,
    H264,
    H265,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum DocumentFormat {
    PDF,
    DOCX,
    PPTX,
    XLSX,
    ODT,
    RTF,
    TXT,
    CSV,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum ThreeDFormat {
    OBJ,
    STL,
    PLY,
    GLTF,
    FBX,
    DAE,
    X3D,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub enum SensorType {
    IMU,          // Inertial Measurement Unit
    GPS,          // Global Positioning System
    Temperature,
    Humidity,
    Pressure,
    LiDAR,
    Radar,
    Accelerometer,
    Gyroscope,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalProcessor {
    pub id: String,
    pub name: String,
    pub processor_type: ProcessorType,
    pub input_modality: Modality,
    pub output_modality: Modality,
    pub model_config: ModelConfig,
    pub preprocessing: Vec<PreprocessingStep>,
    pub postprocessing: Vec<PostprocessingStep>,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessorType {
    // Vision processing
    ObjectDetection,
    ImageClassification,
    ImageSegmentation,
    FaceRecognition,
    OpticalCharacterRecognition,
    ImageGeneration,
    ImageToText,
    ImageToImage,
    
    // Audio processing
    SpeechToText,
    TextToSpeech,
    AudioClassification,
    MusicGeneration,
    AudioEnhancement,
    SpeakerIdentification,
    
    // Video processing
    VideoClassification,
    ActionRecognition,
    VideoSummarization,
    MotionDetection,
    VideoToText,
    
    // Document processing
    DocumentParsing,
    TableExtraction,
    LayoutAnalysis,
    DocumentClassification,
    
    // Multi-modal fusion
    VisionLanguageModel,
    AudioVisualFusion,
    TextImageFusion,
    CrossModalRetrieval,
    
    // 3D processing
    PointCloudProcessing,
    MeshAnalysis,
    CADAnalysis,
    
    // Sensor processing
    TimeSeriesAnalysis,
    AnomalyDetection,
    PatternRecognition,
    
    // Custom processors
    Custom {
        name: String,
        description: String,
        endpoint: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: String,
    pub model_name: String,
    pub version: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub compute_requirements: ComputeRequirements,
    pub cost_per_unit: f64,
    pub rate_limits: RateLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeRequirements {
    pub gpu_memory_gb: Option<f32>,
    pub cpu_cores: Option<u32>,
    pub ram_gb: f32,
    pub storage_gb: Option<f32>,
    pub specialized_hardware: Option<String>, // TPU, NPU, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_minute: u32,
    pub requests_per_day: Option<u32>,
    pub concurrent_requests: u32,
    pub max_file_size_mb: u32,
    pub max_processing_time_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    EarlyFusion,   // Combine features at input level
    LateFusion,    // Combine predictions at output level
    Attention,     // Cross-modal attention mechanism
    Transformer,   // Multi-modal transformer
    Ensemble,      // Ensemble of single-modal models
    Custom {
        strategy_name: String,
        fusion_points: Vec<String>,
        weights: HashMap<String, f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitySettings {
    pub target_accuracy: f64,
    pub max_latency_ms: u32,
    pub quality_vs_speed: f64, // 0.0 = speed priority, 1.0 = quality priority
    pub confidence_threshold: f64,
    pub fallback_enabled: bool,
    pub adaptive_quality: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStep {
    pub step_type: PreprocessingType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub order: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingType {
    // Image preprocessing
    Resize { width: u32, height: u32 },
    Normalize { mean: Vec<f32>, std: Vec<f32> },
    Crop { x: u32, y: u32, width: u32, height: u32 },
    Rotate { angle: f32 },
    Flip { direction: FlipDirection },
    ColorCorrection,
    NoiseReduction,
    
    // Audio preprocessing
    Resample { target_rate: u32 },
    Normalize,
    Trim { start_ms: u32, end_ms: u32 },
    AmplitudeNormalization,
    SpectrogramConversion,
    
    // Text preprocessing
    Tokenization { tokenizer: String },
    Cleaning,
    LanguageDetection,
    Translation { target_language: String },
    
    // Video preprocessing
    FrameExtraction { fps: f32 },
    VideoResize { width: u32, height: u32 },
    TemporalSegmentation,
    
    // Document preprocessing
    TextExtraction,
    LayoutDetection,
    TableDetection,
    ImageExtraction,
    
    // Generic preprocessing
    Custom { name: String, config: HashMap<String, serde_json::Value> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlipDirection {
    Horizontal,
    Vertical,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessingStep {
    pub step_type: PostprocessingType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub order: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PostprocessingType {
    // Result filtering and enhancement
    ConfidenceFiltering { min_confidence: f64 },
    NonMaximumSuppression { iou_threshold: f64 },
    ResultRanking { metric: String },
    Deduplication,
    
    // Output formatting
    FormatConversion { target_format: String },
    Annotation { style: String },
    Overlay { position: String },
    
    // Quality enhancement
    Upscaling { factor: f32 },
    Enhancement,
    Smoothing,
    
    // Custom postprocessing
    Custom { name: String, config: HashMap<String, serde_json::Value> },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    pub avg_processing_time_ms: f64,
    pub throughput_per_second: f64,
    pub accuracy: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization: f64,
    pub cost_per_request: f64,
    pub error_rate: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalInput {
    pub id: String,
    pub modalities: HashMap<String, ModalityData>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub timestamp: u64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModalityData {
    Text {
        content: String,
        encoding: String,
        language: Option<String>,
    },
    Image {
        data: Bytes,
        format: ImageFormat,
        width: u32,
        height: u32,
        metadata: ImageMetadata,
    },
    Audio {
        data: Bytes,
        format: AudioFormat,
        duration_ms: u32,
        sample_rate: u32,
        channels: u8,
        metadata: AudioMetadata,
    },
    Video {
        data: Bytes,
        format: VideoFormat,
        duration_ms: u32,
        width: u32,
        height: u32,
        fps: f32,
        metadata: VideoMetadata,
    },
    Document {
        data: Bytes,
        format: DocumentFormat,
        pages: u32,
        metadata: DocumentMetadata,
    },
    ThreeD {
        data: Bytes,
        format: ThreeDFormat,
        vertex_count: u32,
        metadata: ThreeDMetadata,
    },
    Sensor {
        data: Vec<SensorReading>,
        sensor_type: SensorType,
        sampling_rate: f64,
        metadata: SensorMetadata,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    pub camera_model: Option<String>,
    pub exposure_time: Option<f64>,
    pub iso: Option<u32>,
    pub focal_length: Option<f32>,
    pub gps_coordinates: Option<(f64, f64)>,
    pub creation_date: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMetadata {
    pub bitrate: Option<u32>,
    pub codec: Option<String>,
    pub artist: Option<String>,
    pub title: Option<String>,
    pub genre: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    pub codec: String,
    pub bitrate: Option<u32>,
    pub aspect_ratio: Option<String>,
    pub color_space: Option<String>,
    pub creation_date: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub author: Option<String>,
    pub title: Option<String>,
    pub creation_date: Option<u64>,
    pub modification_date: Option<u64>,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreeDMetadata {
    pub scale_units: Option<String>,
    pub material_count: Option<u32>,
    pub texture_count: Option<u32>,
    pub creation_software: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorMetadata {
    pub device_id: String,
    pub location: Option<(f64, f64, f64)>, // lat, lon, alt
    pub accuracy: Option<f64>,
    pub calibration_date: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub timestamp: u64,
    pub values: Vec<f64>,
    pub quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalOutput {
    pub id: String,
    pub results: HashMap<String, ProcessingResult>,
    pub fusion_result: Option<FusionResult>,
    pub confidence_scores: HashMap<String, f64>,
    pub processing_time_ms: u64,
    pub cost: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingResult {
    Classification {
        classes: Vec<ClassificationResult>,
        confidence: f64,
    },
    Detection {
        objects: Vec<DetectionResult>,
        confidence: f64,
    },
    Segmentation {
        segments: Vec<SegmentationResult>,
        mask: Option<Bytes>,
    },
    Generation {
        generated_content: ModalityData,
        quality_score: f64,
    },
    Transcription {
        text: String,
        confidence: f64,
        word_timestamps: Option<Vec<WordTimestamp>>,
    },
    Analysis {
        insights: Vec<AnalysisInsight>,
        summary: String,
        confidence: f64,
    },
    Custom {
        result_type: String,
        data: serde_json::Value,
        confidence: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub class: String,
    pub confidence: f64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub class: String,
    pub confidence: f64,
    pub bounding_box: BoundingBox,
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub normalized: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationResult {
    pub class: String,
    pub confidence: f64,
    pub polygon: Vec<(f32, f32)>,
    pub area: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_ms: u32,
    pub end_ms: u32,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisInsight {
    pub insight_type: String,
    pub description: String,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionResult {
    pub fusion_strategy: FusionStrategy,
    pub combined_confidence: f64,
    pub modality_contributions: HashMap<String, f64>,
    pub final_prediction: serde_json::Value,
    pub explanation: String,
}

pub struct MultiModalEngine {
    processors: Arc<RwLock<HashMap<String, MultiModalProcessor>>>,
    pipelines: Arc<RwLock<HashMap<String, MultiModalPipeline>>>,
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    provider_registry: Arc<RwLock<ProviderRegistry>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedResult {
    pub input_hash: String,
    pub output: MultiModalOutput,
    pub created_at: u64,
    pub ttl_seconds: u64,
    pub access_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceMonitor {
    pub processing_times: VecDeque<u64>,
    pub throughput_history: VecDeque<f64>,
    pub error_counts: HashMap<String, u32>,
    pub cost_tracking: VecDeque<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProviderRegistry {
    pub providers: HashMap<String, ProviderInfo>,
    pub model_mappings: HashMap<String, Vec<String>>,
    pub capability_matrix: HashMap<String, Vec<ProcessorType>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProviderInfo {
    pub name: String,
    pub endpoint: String,
    pub api_key: Option<String>,
    pub supported_modalities: Vec<Modality>,
    pub rate_limits: RateLimits,
    pub cost_model: CostModel,
    pub availability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CostModel {
    pub base_cost: f64,
    pub per_token_cost: Option<f64>,
    pub per_second_cost: Option<f64>,
    pub per_pixel_cost: Option<f64>,
    pub per_request_cost: Option<f64>,
}

impl MultiModalEngine {
    pub fn new() -> Self {
        Self {
            processors: Arc::new(RwLock::new(HashMap::new())),
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            cache: Arc::new(RwLock::new(HashMap::new())),
            performance_monitor: Arc::new(RwLock::new(PerformanceMonitor {
                processing_times: VecDeque::new(),
                throughput_history: VecDeque::new(),
                error_counts: HashMap::new(),
                cost_tracking: VecDeque::new(),
            })),
            provider_registry: Arc::new(RwLock::new(ProviderRegistry {
                providers: HashMap::new(),
                model_mappings: HashMap::new(),
                capability_matrix: HashMap::new(),
            })),
        }
    }

    // Pipeline management
    pub async fn create_pipeline(&self, pipeline: MultiModalPipeline) -> Result<String, MultiModalError> {
        let pipeline_id = pipeline.id.clone();
        
        // Validate pipeline configuration
        self.validate_pipeline(&pipeline).await?;
        
        // Register pipeline
        let mut pipelines = self.pipelines.write().await;
        pipelines.insert(pipeline_id.clone(), pipeline);
        
        Ok(pipeline_id)
    }

    pub async fn get_pipeline(&self, pipeline_id: &str) -> Option<MultiModalPipeline> {
        let pipelines = self.pipelines.read().await;
        pipelines.get(pipeline_id).cloned()
    }

    pub async fn execute_pipeline(
        &self,
        pipeline_id: &str,
        input: MultiModalInput,
    ) -> Result<MultiModalOutput, MultiModalError> {
        let pipeline = self.get_pipeline(pipeline_id).await
            .ok_or(MultiModalError::PipelineNotFound(pipeline_id.to_string()))?;

        let start_time = std::time::Instant::now();
        
        // Check cache
        let input_hash = self.compute_input_hash(&input)?;
        if let Some(cached) = self.check_cache(&input_hash).await {
            return Ok(cached.output);
        }

        // Validate input modalities
        self.validate_input(&pipeline, &input)?;

        // Execute processors
        let mut results = HashMap::new();
        let mut total_cost = 0.0;
        let mut confidence_scores = HashMap::new();

        for processor in &pipeline.processors {
            let processor_result = self.execute_processor(processor, &input).await?;
            
            match &processor_result {
                ProcessingResult::Classification { confidence, .. } => {
                    confidence_scores.insert(processor.id.clone(), *confidence);
                },
                ProcessingResult::Detection { confidence, .. } => {
                    confidence_scores.insert(processor.id.clone(), *confidence);
                },
                ProcessingResult::Transcription { confidence, .. } => {
                    confidence_scores.insert(processor.id.clone(), *confidence);
                },
                ProcessingResult::Analysis { confidence, .. } => {
                    confidence_scores.insert(processor.id.clone(), *confidence);
                },
                ProcessingResult::Custom { confidence, .. } => {
                    confidence_scores.insert(processor.id.clone(), *confidence);
                },
                _ => {}
            }
            
            results.insert(processor.id.clone(), processor_result);
            total_cost += processor.model_config.cost_per_unit;
        }

        // Apply fusion strategy
        let fusion_result = self.apply_fusion_strategy(
            &pipeline.fusion_strategy,
            &results,
            &confidence_scores,
        ).await?;

        let processing_time = start_time.elapsed().as_millis() as u64;

        let output = MultiModalOutput {
            id: Uuid::new_v4().to_string(),
            results,
            fusion_result: Some(fusion_result),
            confidence_scores,
            processing_time_ms: processing_time,
            cost: total_cost,
            metadata: HashMap::new(),
        };

        // Cache result
        self.cache_result(input_hash, &output).await;

        // Update performance metrics
        self.update_performance_metrics(processing_time, total_cost).await;

        Ok(output)
    }

    async fn validate_pipeline(&self, pipeline: &MultiModalPipeline) -> Result<(), MultiModalError> {
        // Check if required processors are available
        let processors = self.processors.read().await;
        
        for processor_config in &pipeline.processors {
            if !processors.contains_key(&processor_config.id) {
                return Err(MultiModalError::ProcessorNotFound(processor_config.id.clone()));
            }
        }

        // Validate modality compatibility
        for i in 0..pipeline.processors.len() {
            if i > 0 {
                let prev_output = &pipeline.processors[i-1].output_modality;
                let curr_input = &pipeline.processors[i].input_modality;
                
                if !self.are_modalities_compatible(prev_output, curr_input) {
                    return Err(MultiModalError::ModalityMismatch {
                        expected: format!("{:?}", curr_input),
                        received: format!("{:?}", prev_output),
                    });
                }
            }
        }

        Ok(())
    }

    fn validate_input(&self, pipeline: &MultiModalPipeline, input: &MultiModalInput) -> Result<(), MultiModalError> {
        // Check if all required input modalities are present
        for required_modality in &pipeline.input_modalities {
            let found = input.modalities.values().any(|data| {
                self.matches_modality_type(data, required_modality)
            });
            
            if !found {
                return Err(MultiModalError::MissingInputModality(format!("{:?}", required_modality)));
            }
        }

        Ok(())
    }

    async fn execute_processor(
        &self,
        processor: &MultiModalProcessor,
        input: &MultiModalInput,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Find matching input modality data
        let input_data = input.modalities.values()
            .find(|data| self.matches_modality_type(data, &processor.input_modality))
            .ok_or(MultiModalError::NoMatchingInputModality)?;

        // Apply preprocessing
        let preprocessed_data = self.apply_preprocessing(input_data, &processor.preprocessing).await?;

        // Execute the actual processing based on processor type
        let raw_result = match &processor.processor_type {
            ProcessorType::ObjectDetection => {
                self.execute_object_detection(&processor.model_config, &preprocessed_data).await?
            },
            ProcessorType::ImageClassification => {
                self.execute_image_classification(&processor.model_config, &preprocessed_data).await?
            },
            ProcessorType::SpeechToText => {
                self.execute_speech_to_text(&processor.model_config, &preprocessed_data).await?
            },
            ProcessorType::TextToSpeech => {
                self.execute_text_to_speech(&processor.model_config, &preprocessed_data).await?
            },
            ProcessorType::VisionLanguageModel => {
                self.execute_vision_language_model(&processor.model_config, &preprocessed_data, input).await?
            },
            ProcessorType::DocumentParsing => {
                self.execute_document_parsing(&processor.model_config, &preprocessed_data).await?
            },
            ProcessorType::Custom { name, endpoint, .. } => {
                self.execute_custom_processor(name, endpoint, &processor.model_config, &preprocessed_data).await?
            },
            _ => {
                return Err(MultiModalError::UnsupportedProcessorType(format!("{:?}", processor.processor_type)));
            }
        };

        // Apply postprocessing
        let final_result = self.apply_postprocessing(raw_result, &processor.postprocessing).await?;

        Ok(final_result)
    }

    async fn apply_fusion_strategy(
        &self,
        strategy: &FusionStrategy,
        results: &HashMap<String, ProcessingResult>,
        confidence_scores: &HashMap<String, f64>,
    ) -> Result<FusionResult, MultiModalError> {
        let combined_confidence = confidence_scores.values().sum::<f64>() / confidence_scores.len() as f64;
        
        let modality_contributions = confidence_scores.clone();

        let final_prediction = match strategy {
            FusionStrategy::EarlyFusion => {
                // Combine at feature level - simplified for demo
                serde_json::json!({
                    "fusion_type": "early_fusion",
                    "combined_features": "feature_vector_placeholder"
                })
            },
            FusionStrategy::LateFusion => {
                // Combine predictions with weighted average
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;
                
                for (_, confidence) in confidence_scores {
                    weighted_sum += confidence;
                    total_weight += 1.0;
                }
                
                serde_json::json!({
                    "fusion_type": "late_fusion",
                    "weighted_average": if total_weight > 0.0 { weighted_sum / total_weight } else { 0.0 },
                    "predictions": results.keys().collect::<Vec<_>>()
                })
            },
            FusionStrategy::Attention => {
                // Cross-modal attention mechanism
                serde_json::json!({
                    "fusion_type": "attention",
                    "attention_weights": modality_contributions,
                    "combined_representation": "attention_output_placeholder"
                })
            },
            FusionStrategy::Transformer => {
                // Multi-modal transformer fusion
                serde_json::json!({
                    "fusion_type": "transformer",
                    "transformer_output": "multimodal_transformer_result"
                })
            },
            FusionStrategy::Ensemble => {
                // Ensemble of models
                serde_json::json!({
                    "fusion_type": "ensemble",
                    "ensemble_prediction": "ensemble_result_placeholder",
                    "model_votes": results.len()
                })
            },
            FusionStrategy::Custom { strategy_name, fusion_points, weights } => {
                serde_json::json!({
                    "fusion_type": "custom",
                    "strategy_name": strategy_name,
                    "fusion_points": fusion_points,
                    "weights": weights
                })
            },
        };

        let explanation = format!(
            "Applied {:?} fusion strategy combining {} modalities with average confidence {:.2}",
            strategy,
            results.len(),
            combined_confidence
        );

        Ok(FusionResult {
            fusion_strategy: strategy.clone(),
            combined_confidence,
            modality_contributions,
            final_prediction,
            explanation,
        })
    }

    // Processor implementations (simplified for demo)
    async fn execute_object_detection(
        &self,
        config: &ModelConfig,
        data: &ModalityData,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simulate object detection
        let objects = vec![
            DetectionResult {
                class: "person".to_string(),
                confidence: 0.92,
                bounding_box: BoundingBox {
                    x: 0.1, y: 0.2, width: 0.3, height: 0.6, normalized: true
                },
                attributes: HashMap::new(),
            },
            DetectionResult {
                class: "car".to_string(),
                confidence: 0.87,
                bounding_box: BoundingBox {
                    x: 0.6, y: 0.4, width: 0.3, height: 0.4, normalized: true
                },
                attributes: HashMap::new(),
            },
        ];

        Ok(ProcessingResult::Detection {
            objects,
            confidence: 0.90,
        })
    }

    async fn execute_image_classification(
        &self,
        config: &ModelConfig,
        data: &ModalityData,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simulate image classification
        let classes = vec![
            ClassificationResult {
                class: "outdoor_scene".to_string(),
                confidence: 0.85,
                metadata: HashMap::new(),
            },
            ClassificationResult {
                class: "urban_environment".to_string(),
                confidence: 0.78,
                metadata: HashMap::new(),
            },
        ];

        Ok(ProcessingResult::Classification {
            classes,
            confidence: 0.85,
        })
    }

    async fn execute_speech_to_text(
        &self,
        config: &ModelConfig,
        data: &ModalityData,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simulate speech-to-text
        Ok(ProcessingResult::Transcription {
            text: "Hello, this is a sample transcription of the audio input.".to_string(),
            confidence: 0.93,
            word_timestamps: Some(vec![
                WordTimestamp { word: "Hello".to_string(), start_ms: 0, end_ms: 500, confidence: 0.95 },
                WordTimestamp { word: "this".to_string(), start_ms: 600, end_ms: 800, confidence: 0.92 },
            ]),
        })
    }

    async fn execute_text_to_speech(
        &self,
        config: &ModelConfig,
        data: &ModalityData,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simulate text-to-speech
        let audio_data = Bytes::from("fake_audio_data_placeholder");
        
        Ok(ProcessingResult::Generation {
            generated_content: ModalityData::Audio {
                data: audio_data,
                format: AudioFormat::WAV,
                duration_ms: 5000,
                sample_rate: 44100,
                channels: 2,
                metadata: AudioMetadata {
                    bitrate: Some(320),
                    codec: Some("PCM".to_string()),
                    artist: None,
                    title: None,
                    genre: None,
                },
            },
            quality_score: 0.88,
        })
    }

    async fn execute_vision_language_model(
        &self,
        config: &ModelConfig,
        data: &ModalityData,
        input: &MultiModalInput,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simulate vision-language model processing
        let insights = vec![
            AnalysisInsight {
                insight_type: "scene_description".to_string(),
                description: "The image shows a busy city street with pedestrians and vehicles".to_string(),
                confidence: 0.89,
                supporting_evidence: vec!["Multiple people detected".to_string(), "Urban environment classified".to_string()],
            },
            AnalysisInsight {
                insight_type: "activity_analysis".to_string(),
                description: "People are walking and commuting during what appears to be daytime".to_string(),
                confidence: 0.82,
                supporting_evidence: vec!["Movement patterns".to_string(), "Lighting conditions".to_string()],
            },
        ];

        Ok(ProcessingResult::Analysis {
            insights,
            summary: "Urban street scene with active pedestrian and vehicle traffic".to_string(),
            confidence: 0.85,
        })
    }

    async fn execute_document_parsing(
        &self,
        config: &ModelConfig,
        data: &ModalityData,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simulate document parsing
        let insights = vec![
            AnalysisInsight {
                insight_type: "document_structure".to_string(),
                description: "Document contains header, body text, and tables".to_string(),
                confidence: 0.94,
                supporting_evidence: vec!["Layout analysis".to_string(), "Text formatting".to_string()],
            },
        ];

        Ok(ProcessingResult::Analysis {
            insights,
            summary: "Structured document with multiple content types extracted".to_string(),
            confidence: 0.94,
        })
    }

    async fn execute_custom_processor(
        &self,
        name: &str,
        endpoint: &str,
        config: &ModelConfig,
        data: &ModalityData,
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simulate custom processor call
        Ok(ProcessingResult::Custom {
            result_type: name.to_string(),
            data: serde_json::json!({
                "processed": true,
                "endpoint": endpoint,
                "model": config.model_name
            }),
            confidence: 0.75,
        })
    }

    // Utility methods
    fn are_modalities_compatible(&self, output: &Modality, input: &Modality) -> bool {
        // Simplified compatibility check
        std::mem::discriminant(output) == std::mem::discriminant(input)
    }

    fn matches_modality_type(&self, data: &ModalityData, modality: &Modality) -> bool {
        match (data, modality) {
            (ModalityData::Text { .. }, Modality::Text { .. }) => true,
            (ModalityData::Image { .. }, Modality::Image { .. }) => true,
            (ModalityData::Audio { .. }, Modality::Audio { .. }) => true,
            (ModalityData::Video { .. }, Modality::Video { .. }) => true,
            (ModalityData::Document { .. }, Modality::Document { .. }) => true,
            (ModalityData::ThreeD { .. }, Modality::ThreeD { .. }) => true,
            (ModalityData::Sensor { .. }, Modality::Sensor { .. }) => true,
            _ => false,
        }
    }

    fn compute_input_hash(&self, input: &MultiModalInput) -> Result<String, MultiModalError> {
        // Simplified hash computation
        Ok(format!("hash_{}", input.id))
    }

    async fn check_cache(&self, input_hash: &str) -> Option<CachedResult> {
        let cache = self.cache.read().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if let Some(cached) = cache.get(input_hash) {
            if cached.created_at + cached.ttl_seconds > now {
                return Some(cached.clone());
            }
        }
        None
    }

    async fn cache_result(&self, input_hash: String, output: &MultiModalOutput) {
        let mut cache = self.cache.write().await;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let cached_result = CachedResult {
            input_hash: input_hash.clone(),
            output: output.clone(),
            created_at: now,
            ttl_seconds: 3600, // 1 hour default TTL
            access_count: 1,
        };

        cache.insert(input_hash, cached_result);

        // Keep cache size reasonable
        if cache.len() > 10000 {
            let oldest_keys: Vec<_> = cache.iter()
                .sorted_by_key(|(_, v)| v.created_at)
                .take(5000)
                .map(|(k, _)| k.clone())
                .collect();
            
            for key in oldest_keys {
                cache.remove(&key);
            }
        }
    }

    async fn update_performance_metrics(&self, processing_time: u64, cost: f64) {
        let mut monitor = self.performance_monitor.write().await;
        
        monitor.processing_times.push_back(processing_time);
        monitor.cost_tracking.push_back(cost);
        
        // Keep only recent metrics (last 1000 entries)
        if monitor.processing_times.len() > 1000 {
            monitor.processing_times.pop_front();
        }
        if monitor.cost_tracking.len() > 1000 {
            monitor.cost_tracking.pop_front();
        }
    }

    async fn apply_preprocessing(
        &self,
        data: &ModalityData,
        steps: &[PreprocessingStep],
    ) -> Result<ModalityData, MultiModalError> {
        // Simplified preprocessing - in practice would apply actual transformations
        Ok(data.clone())
    }

    async fn apply_postprocessing(
        &self,
        result: ProcessingResult,
        steps: &[PostprocessingStep],
    ) -> Result<ProcessingResult, MultiModalError> {
        // Simplified postprocessing - in practice would apply actual transformations
        Ok(result)
    }

    // Public API methods
    pub async fn register_processor(&self, processor: MultiModalProcessor) -> String {
        let processor_id = processor.id.clone();
        let mut processors = self.processors.write().await;
        processors.insert(processor_id.clone(), processor);
        processor_id
    }

    pub async fn list_processors(&self) -> Vec<MultiModalProcessor> {
        let processors = self.processors.read().await;
        processors.values().cloned().collect()
    }

    pub async fn get_performance_stats(&self) -> PerformanceStats {
        let monitor = self.performance_monitor.read().await;
        
        let avg_processing_time = if !monitor.processing_times.is_empty() {
            monitor.processing_times.iter().sum::<u64>() as f64 / monitor.processing_times.len() as f64
        } else {
            0.0
        };

        let avg_cost = if !monitor.cost_tracking.is_empty() {
            monitor.cost_tracking.iter().sum::<f64>() / monitor.cost_tracking.len() as f64
        } else {
            0.0
        };

        PerformanceStats {
            avg_processing_time_ms: avg_processing_time,
            total_processed: monitor.processing_times.len(),
            avg_cost_per_request: avg_cost,
            error_rate: monitor.error_counts.values().sum::<u32>() as f64 / 
                       monitor.processing_times.len().max(1) as f64,
            cache_hit_rate: 0.0, // Would calculate from cache statistics
        }
    }
}

use itertools::Itertools;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub avg_processing_time_ms: f64,
    pub total_processed: usize,
    pub avg_cost_per_request: f64,
    pub error_rate: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug)]
pub enum MultiModalError {
    PipelineNotFound(String),
    ProcessorNotFound(String),
    ModalityMismatch { expected: String, received: String },
    MissingInputModality(String),
    NoMatchingInputModality,
    UnsupportedProcessorType(String),
    ProcessingFailed(String),
    InvalidInput(String),
    ConfigurationError(String),
}

impl std::fmt::Display for MultiModalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MultiModalError::PipelineNotFound(id) => write!(f, "Pipeline not found: {}", id),
            MultiModalError::ProcessorNotFound(id) => write!(f, "Processor not found: {}", id),
            MultiModalError::ModalityMismatch { expected, received } => {
                write!(f, "Modality mismatch - expected: {}, received: {}", expected, received)
            },
            MultiModalError::MissingInputModality(modality) => {
                write!(f, "Missing required input modality: {}", modality)
            },
            MultiModalError::NoMatchingInputModality => write!(f, "No matching input modality found"),
            MultiModalError::UnsupportedProcessorType(ptype) => {
                write!(f, "Unsupported processor type: {}", ptype)
            },
            MultiModalError::ProcessingFailed(reason) => write!(f, "Processing failed: {}", reason),
            MultiModalError::InvalidInput(reason) => write!(f, "Invalid input: {}", reason),
            MultiModalError::ConfigurationError(reason) => write!(f, "Configuration error: {}", reason),
        }
    }
}

impl std::error::Error for MultiModalError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multimodal_pipeline_creation() {
        let engine = MultiModalEngine::new();
        
        let pipeline = MultiModalPipeline {
            id: "test_pipeline".to_string(),
            name: "Test Vision-Language Pipeline".to_string(),
            description: "Pipeline for processing images and generating text descriptions".to_string(),
            input_modalities: vec![
                Modality::Image {
                    format: ImageFormat::JPEG,
                    max_resolution: (1920, 1080),
                    color_space: ColorSpace::RGB,
                },
            ],
            output_modalities: vec![
                Modality::Text {
                    format: TextFormat::PlainText,
                    language: Some("en".to_string()),
                    encoding: "utf-8".to_string(),
                },
            ],
            processors: vec![],
            fusion_strategy: FusionStrategy::LateFusion,
            quality_settings: QualitySettings {
                target_accuracy: 0.9,
                max_latency_ms: 5000,
                quality_vs_speed: 0.7,
                confidence_threshold: 0.8,
                fallback_enabled: true,
                adaptive_quality: false,
            },
            created_at: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            version: "1.0.0".to_string(),
        };

        let pipeline_id = engine.create_pipeline(pipeline).await.unwrap();
        assert_eq!(pipeline_id, "test_pipeline");

        let retrieved = engine.get_pipeline(&pipeline_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Test Vision-Language Pipeline");
    }

    #[tokio::test]
    async fn test_processor_registration() {
        let engine = MultiModalEngine::new();
        
        let processor = MultiModalProcessor {
            id: "image_classifier".to_string(),
            name: "Image Classification Processor".to_string(),
            processor_type: ProcessorType::ImageClassification,
            input_modality: Modality::Image {
                format: ImageFormat::JPEG,
                max_resolution: (1024, 1024),
                color_space: ColorSpace::RGB,
            },
            output_modality: Modality::Text {
                format: TextFormat::JSON,
                language: None,
                encoding: "utf-8".to_string(),
            },
            model_config: ModelConfig {
                provider: "test_provider".to_string(),
                model_name: "test_classifier".to_string(),
                version: "1.0".to_string(),
                parameters: HashMap::new(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: Some(4.0),
                    cpu_cores: Some(4),
                    ram_gb: 8.0,
                    storage_gb: None,
                    specialized_hardware: None,
                },
                cost_per_unit: 0.01,
                rate_limits: RateLimits {
                    requests_per_minute: 60,
                    requests_per_day: None,
                    concurrent_requests: 5,
                    max_file_size_mb: 10,
                    max_processing_time_seconds: 30,
                },
            },
            preprocessing: vec![],
            postprocessing: vec![],
            performance_metrics: PerformanceMetrics::default(),
        };

        let processor_id = engine.register_processor(processor).await;
        assert_eq!(processor_id, "image_classifier");

        let processors = engine.list_processors().await;
        assert_eq!(processors.len(), 1);
        assert_eq!(processors[0].name, "Image Classification Processor");
    }
}