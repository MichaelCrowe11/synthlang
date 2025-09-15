use std::collections::HashMap;
use crate::multimodal::*;
use uuid::Uuid;

pub struct MultiModalPipelineBuilder {
    pipeline: MultiModalPipeline,
    processors: Vec<MultiModalProcessor>,
}

impl MultiModalPipelineBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            pipeline: MultiModalPipeline {
                id: Uuid::new_v4().to_string(),
                name: name.to_string(),
                description: String::new(),
                input_modalities: Vec::new(),
                output_modalities: Vec::new(),
                processors: Vec::new(),
                fusion_strategy: FusionStrategy::LateFusion,
                quality_settings: QualitySettings::default(),
                created_at: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                version: "1.0.0".to_string(),
            },
            processors: Vec::new(),
        }
    }

    pub fn description(mut self, desc: &str) -> Self {
        self.pipeline.description = desc.to_string();
        self
    }

    pub fn version(mut self, version: &str) -> Self {
        self.pipeline.version = version.to_string();
        self
    }

    // Input/Output modality configuration
    pub fn input_text(mut self) -> Self {
        self.pipeline.input_modalities.push(Modality::Text {
            format: TextFormat::PlainText,
            language: Some("en".to_string()),
            encoding: "utf-8".to_string(),
        });
        self
    }

    pub fn input_image(mut self, format: ImageFormat, max_resolution: (u32, u32)) -> Self {
        self.pipeline.input_modalities.push(Modality::Image {
            format,
            max_resolution,
            color_space: ColorSpace::RGB,
        });
        self
    }

    pub fn input_audio(mut self, format: AudioFormat, sample_rate: u32, channels: u8) -> Self {
        self.pipeline.input_modalities.push(Modality::Audio {
            format,
            sample_rate,
            channels,
            bitrate: None,
        });
        self
    }

    pub fn input_video(mut self, format: VideoFormat, resolution: (u32, u32), fps: f32) -> Self {
        self.pipeline.input_modalities.push(Modality::Video {
            format,
            resolution,
            fps,
            codec: "h264".to_string(),
        });
        self
    }

    pub fn input_document(mut self, format: DocumentFormat) -> Self {
        self.pipeline.input_modalities.push(Modality::Document {
            format,
            extract_images: true,
            extract_tables: true,
        });
        self
    }

    pub fn output_text(mut self) -> Self {
        self.pipeline.output_modalities.push(Modality::Text {
            format: TextFormat::JSON,
            language: Some("en".to_string()),
            encoding: "utf-8".to_string(),
        });
        self
    }

    pub fn output_image(mut self, format: ImageFormat, max_resolution: (u32, u32)) -> Self {
        self.pipeline.output_modalities.push(Modality::Image {
            format,
            max_resolution,
            color_space: ColorSpace::RGB,
        });
        self
    }

    pub fn output_audio(mut self, format: AudioFormat, sample_rate: u32, channels: u8) -> Self {
        self.pipeline.output_modalities.push(Modality::Audio {
            format,
            sample_rate,
            channels,
            bitrate: Some(320),
        });
        self
    }

    // Processor builders
    pub fn add_image_classifier(mut self, model_name: &str, provider: &str) -> Self {
        let processor = MultiModalProcessor {
            id: format!("image_classifier_{}", Uuid::new_v4()),
            name: "Image Classification".to_string(),
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
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                version: "latest".to_string(),
                parameters: HashMap::new(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: Some(4.0),
                    cpu_cores: Some(4),
                    ram_gb: 8.0,
                    storage_gb: None,
                    specialized_hardware: None,
                },
                cost_per_unit: 0.001,
                rate_limits: RateLimits {
                    requests_per_minute: 100,
                    requests_per_day: None,
                    concurrent_requests: 10,
                    max_file_size_mb: 10,
                    max_processing_time_seconds: 30,
                },
            },
            preprocessing: vec![
                PreprocessingStep {
                    step_type: PreprocessingType::Resize { width: 224, height: 224 },
                    parameters: HashMap::new(),
                    order: 1,
                },
                PreprocessingStep {
                    step_type: PreprocessingType::Normalize {
                        mean: vec![0.485, 0.456, 0.406],
                        std: vec![0.229, 0.224, 0.225],
                    },
                    parameters: HashMap::new(),
                    order: 2,
                },
            ],
            postprocessing: vec![
                PostprocessingStep {
                    step_type: PostprocessingType::ConfidenceFiltering { min_confidence: 0.5 },
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    pub fn add_object_detector(mut self, model_name: &str, provider: &str) -> Self {
        let processor = MultiModalProcessor {
            id: format!("object_detector_{}", Uuid::new_v4()),
            name: "Object Detection".to_string(),
            processor_type: ProcessorType::ObjectDetection,
            input_modality: Modality::Image {
                format: ImageFormat::JPEG,
                max_resolution: (1920, 1080),
                color_space: ColorSpace::RGB,
            },
            output_modality: Modality::Text {
                format: TextFormat::JSON,
                language: None,
                encoding: "utf-8".to_string(),
            },
            model_config: ModelConfig {
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                version: "latest".to_string(),
                parameters: [
                    ("confidence_threshold".to_string(), serde_json::json!(0.5)),
                    ("nms_threshold".to_string(), serde_json::json!(0.4)),
                    ("max_detections".to_string(), serde_json::json!(100)),
                ].into_iter().collect(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: Some(6.0),
                    cpu_cores: Some(8),
                    ram_gb: 16.0,
                    storage_gb: None,
                    specialized_hardware: None,
                },
                cost_per_unit: 0.002,
                rate_limits: RateLimits {
                    requests_per_minute: 60,
                    requests_per_day: None,
                    concurrent_requests: 5,
                    max_file_size_mb: 20,
                    max_processing_time_seconds: 60,
                },
            },
            preprocessing: vec![
                PreprocessingStep {
                    step_type: PreprocessingType::Resize { width: 640, height: 640 },
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            postprocessing: vec![
                PostprocessingStep {
                    step_type: PostprocessingType::NonMaximumSuppression { iou_threshold: 0.4 },
                    parameters: HashMap::new(),
                    order: 1,
                },
                PostprocessingStep {
                    step_type: PostprocessingType::ConfidenceFiltering { min_confidence: 0.5 },
                    parameters: HashMap::new(),
                    order: 2,
                },
            ],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    pub fn add_speech_to_text(mut self, model_name: &str, provider: &str) -> Self {
        let processor = MultiModalProcessor {
            id: format!("speech_to_text_{}", Uuid::new_v4()),
            name: "Speech Recognition".to_string(),
            processor_type: ProcessorType::SpeechToText,
            input_modality: Modality::Audio {
                format: AudioFormat::WAV,
                sample_rate: 16000,
                channels: 1,
                bitrate: None,
            },
            output_modality: Modality::Text {
                format: TextFormat::PlainText,
                language: Some("en".to_string()),
                encoding: "utf-8".to_string(),
            },
            model_config: ModelConfig {
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                version: "latest".to_string(),
                parameters: [
                    ("language".to_string(), serde_json::json!("en-US")),
                    ("enable_word_timestamps".to_string(), serde_json::json!(true)),
                    ("enable_automatic_punctuation".to_string(), serde_json::json!(true)),
                ].into_iter().collect(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: Some(2.0),
                    cpu_cores: Some(4),
                    ram_gb: 4.0,
                    storage_gb: None,
                    specialized_hardware: None,
                },
                cost_per_unit: 0.0001, // Per second of audio
                rate_limits: RateLimits {
                    requests_per_minute: 120,
                    requests_per_day: None,
                    concurrent_requests: 10,
                    max_file_size_mb: 100,
                    max_processing_time_seconds: 300,
                },
            },
            preprocessing: vec![
                PreprocessingStep {
                    step_type: PreprocessingType::Resample { target_rate: 16000 },
                    parameters: HashMap::new(),
                    order: 1,
                },
                PreprocessingStep {
                    step_type: PreprocessingType::AmplitudeNormalization,
                    parameters: HashMap::new(),
                    order: 2,
                },
            ],
            postprocessing: vec![
                PostprocessingStep {
                    step_type: PostprocessingType::ConfidenceFiltering { min_confidence: 0.8 },
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    pub fn add_text_to_speech(mut self, model_name: &str, provider: &str, voice: &str) -> Self {
        let processor = MultiModalProcessor {
            id: format!("text_to_speech_{}", Uuid::new_v4()),
            name: "Text to Speech".to_string(),
            processor_type: ProcessorType::TextToSpeech,
            input_modality: Modality::Text {
                format: TextFormat::PlainText,
                language: Some("en".to_string()),
                encoding: "utf-8".to_string(),
            },
            output_modality: Modality::Audio {
                format: AudioFormat::MP3,
                sample_rate: 44100,
                channels: 2,
                bitrate: Some(320),
            },
            model_config: ModelConfig {
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                version: "latest".to_string(),
                parameters: [
                    ("voice".to_string(), serde_json::json!(voice)),
                    ("speed".to_string(), serde_json::json!(1.0)),
                    ("pitch".to_string(), serde_json::json!(0.0)),
                    ("format".to_string(), serde_json::json!("mp3")),
                ].into_iter().collect(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: Some(1.0),
                    cpu_cores: Some(2),
                    ram_gb: 2.0,
                    storage_gb: None,
                    specialized_hardware: None,
                },
                cost_per_unit: 0.000016, // Per character
                rate_limits: RateLimits {
                    requests_per_minute: 100,
                    requests_per_day: None,
                    concurrent_requests: 20,
                    max_file_size_mb: 1,
                    max_processing_time_seconds: 60,
                },
            },
            preprocessing: vec![
                PreprocessingStep {
                    step_type: PreprocessingType::Cleaning,
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            postprocessing: vec![],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    pub fn add_vision_language_model(mut self, model_name: &str, provider: &str) -> Self {
        let processor = MultiModalProcessor {
            id: format!("vision_language_model_{}", Uuid::new_v4()),
            name: "Vision-Language Model".to_string(),
            processor_type: ProcessorType::VisionLanguageModel,
            input_modality: Modality::Image {
                format: ImageFormat::JPEG,
                max_resolution: (1024, 1024),
                color_space: ColorSpace::RGB,
            },
            output_modality: Modality::Text {
                format: TextFormat::JSON,
                language: Some("en".to_string()),
                encoding: "utf-8".to_string(),
            },
            model_config: ModelConfig {
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                version: "latest".to_string(),
                parameters: [
                    ("max_tokens".to_string(), serde_json::json!(500)),
                    ("temperature".to_string(), serde_json::json!(0.7)),
                    ("detail_level".to_string(), serde_json::json!("high")),
                ].into_iter().collect(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: Some(16.0),
                    cpu_cores: Some(8),
                    ram_gb: 32.0,
                    storage_gb: None,
                    specialized_hardware: Some("A100".to_string()),
                },
                cost_per_unit: 0.01, // Per image
                rate_limits: RateLimits {
                    requests_per_minute: 30,
                    requests_per_day: Some(1000),
                    concurrent_requests: 3,
                    max_file_size_mb: 20,
                    max_processing_time_seconds: 120,
                },
            },
            preprocessing: vec![
                PreprocessingStep {
                    step_type: PreprocessingType::Resize { width: 512, height: 512 },
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            postprocessing: vec![
                PostprocessingStep {
                    step_type: PostprocessingType::ConfidenceFiltering { min_confidence: 0.6 },
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    pub fn add_document_parser(mut self, model_name: &str, provider: &str) -> Self {
        let processor = MultiModalProcessor {
            id: format!("document_parser_{}", Uuid::new_v4()),
            name: "Document Parser".to_string(),
            processor_type: ProcessorType::DocumentParsing,
            input_modality: Modality::Document {
                format: DocumentFormat::PDF,
                extract_images: true,
                extract_tables: true,
            },
            output_modality: Modality::Text {
                format: TextFormat::JSON,
                language: None,
                encoding: "utf-8".to_string(),
            },
            model_config: ModelConfig {
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                version: "latest".to_string(),
                parameters: [
                    ("extract_images".to_string(), serde_json::json!(true)),
                    ("extract_tables".to_string(), serde_json::json!(true)),
                    ("extract_metadata".to_string(), serde_json::json!(true)),
                    ("ocr_engine".to_string(), serde_json::json!("tesseract")),
                ].into_iter().collect(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: None,
                    cpu_cores: Some(4),
                    ram_gb: 8.0,
                    storage_gb: Some(10.0),
                    specialized_hardware: None,
                },
                cost_per_unit: 0.005, // Per page
                rate_limits: RateLimits {
                    requests_per_minute: 50,
                    requests_per_day: None,
                    concurrent_requests: 5,
                    max_file_size_mb: 50,
                    max_processing_time_seconds: 300,
                },
            },
            preprocessing: vec![
                PreprocessingStep {
                    step_type: PreprocessingType::TextExtraction,
                    parameters: HashMap::new(),
                    order: 1,
                },
                PreprocessingStep {
                    step_type: PreprocessingType::LayoutDetection,
                    parameters: HashMap::new(),
                    order: 2,
                },
            ],
            postprocessing: vec![
                PostprocessingStep {
                    step_type: PostprocessingType::Deduplication,
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    pub fn add_video_analyzer(mut self, model_name: &str, provider: &str) -> Self {
        let processor = MultiModalProcessor {
            id: format!("video_analyzer_{}", Uuid::new_v4()),
            name: "Video Analysis".to_string(),
            processor_type: ProcessorType::VideoClassification,
            input_modality: Modality::Video {
                format: VideoFormat::MP4,
                resolution: (1920, 1080),
                fps: 30.0,
                codec: "h264".to_string(),
            },
            output_modality: Modality::Text {
                format: TextFormat::JSON,
                language: Some("en".to_string()),
                encoding: "utf-8".to_string(),
            },
            model_config: ModelConfig {
                provider: provider.to_string(),
                model_name: model_name.to_string(),
                version: "latest".to_string(),
                parameters: [
                    ("frame_sample_rate".to_string(), serde_json::json!(1.0)), // Sample 1 frame per second
                    ("max_duration_seconds".to_string(), serde_json::json!(300)),
                    ("analyze_audio".to_string(), serde_json::json!(true)),
                ].into_iter().collect(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: Some(12.0),
                    cpu_cores: Some(8),
                    ram_gb: 24.0,
                    storage_gb: Some(100.0),
                    specialized_hardware: None,
                },
                cost_per_unit: 0.1, // Per minute of video
                rate_limits: RateLimits {
                    requests_per_minute: 10,
                    requests_per_day: Some(100),
                    concurrent_requests: 2,
                    max_file_size_mb: 500,
                    max_processing_time_seconds: 600,
                },
            },
            preprocessing: vec![
                PreprocessingStep {
                    step_type: PreprocessingType::FrameExtraction { fps: 1.0 },
                    parameters: HashMap::new(),
                    order: 1,
                },
                PreprocessingStep {
                    step_type: PreprocessingType::VideoResize { width: 640, height: 360 },
                    parameters: HashMap::new(),
                    order: 2,
                },
            ],
            postprocessing: vec![
                PostprocessingStep {
                    step_type: PostprocessingType::ConfidenceFiltering { min_confidence: 0.7 },
                    parameters: HashMap::new(),
                    order: 1,
                },
            ],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    pub fn add_custom_processor(
        mut self,
        name: &str,
        endpoint: &str,
        input_modality: Modality,
        output_modality: Modality,
    ) -> Self {
        let processor = MultiModalProcessor {
            id: format!("custom_processor_{}", Uuid::new_v4()),
            name: name.to_string(),
            processor_type: ProcessorType::Custom {
                name: name.to_string(),
                description: format!("Custom processor: {}", name),
                endpoint: endpoint.to_string(),
            },
            input_modality,
            output_modality,
            model_config: ModelConfig {
                provider: "custom".to_string(),
                model_name: name.to_string(),
                version: "1.0".to_string(),
                parameters: HashMap::new(),
                compute_requirements: ComputeRequirements {
                    gpu_memory_gb: None,
                    cpu_cores: Some(2),
                    ram_gb: 4.0,
                    storage_gb: None,
                    specialized_hardware: None,
                },
                cost_per_unit: 0.01,
                rate_limits: RateLimits {
                    requests_per_minute: 60,
                    requests_per_day: None,
                    concurrent_requests: 10,
                    max_file_size_mb: 50,
                    max_processing_time_seconds: 120,
                },
            },
            preprocessing: vec![],
            postprocessing: vec![],
            performance_metrics: PerformanceMetrics::default(),
        };

        self.processors.push(processor);
        self
    }

    // Fusion strategy configuration
    pub fn with_early_fusion(mut self) -> Self {
        self.pipeline.fusion_strategy = FusionStrategy::EarlyFusion;
        self
    }

    pub fn with_late_fusion(mut self) -> Self {
        self.pipeline.fusion_strategy = FusionStrategy::LateFusion;
        self
    }

    pub fn with_attention_fusion(mut self) -> Self {
        self.pipeline.fusion_strategy = FusionStrategy::Attention;
        self
    }

    pub fn with_transformer_fusion(mut self) -> Self {
        self.pipeline.fusion_strategy = FusionStrategy::Transformer;
        self
    }

    pub fn with_ensemble_fusion(mut self) -> Self {
        self.pipeline.fusion_strategy = FusionStrategy::Ensemble;
        self
    }

    pub fn with_custom_fusion(mut self, strategy_name: &str, fusion_points: Vec<String>, weights: HashMap<String, f64>) -> Self {
        self.pipeline.fusion_strategy = FusionStrategy::Custom {
            strategy_name: strategy_name.to_string(),
            fusion_points,
            weights,
        };
        self
    }

    // Quality settings
    pub fn with_quality_settings(mut self, settings: QualitySettings) -> Self {
        self.pipeline.quality_settings = settings;
        self
    }

    pub fn target_accuracy(mut self, accuracy: f64) -> Self {
        self.pipeline.quality_settings.target_accuracy = accuracy;
        self
    }

    pub fn max_latency_ms(mut self, latency_ms: u32) -> Self {
        self.pipeline.quality_settings.max_latency_ms = latency_ms;
        self
    }

    pub fn quality_vs_speed(mut self, ratio: f64) -> Self {
        self.pipeline.quality_settings.quality_vs_speed = ratio.clamp(0.0, 1.0);
        self
    }

    pub fn confidence_threshold(mut self, threshold: f64) -> Self {
        self.pipeline.quality_settings.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn enable_fallback(mut self) -> Self {
        self.pipeline.quality_settings.fallback_enabled = true;
        self
    }

    pub fn enable_adaptive_quality(mut self) -> Self {
        self.pipeline.quality_settings.adaptive_quality = true;
        self
    }

    // Build the pipeline
    pub fn build(mut self) -> MultiModalPipeline {
        self.pipeline.processors = self.processors;
        self.pipeline
    }
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self {
            target_accuracy: 0.85,
            max_latency_ms: 5000,
            quality_vs_speed: 0.7,
            confidence_threshold: 0.8,
            fallback_enabled: true,
            adaptive_quality: false,
        }
    }
}

// Predefined pipeline templates
pub struct PipelineTemplates;

impl PipelineTemplates {
    /// Creates a vision-language pipeline for image understanding and description
    pub fn vision_language_pipeline() -> MultiModalPipelineBuilder {
        MultiModalPipelineBuilder::new("Vision-Language Analysis")
            .description("Analyze images and generate detailed textual descriptions")
            .input_image(ImageFormat::JPEG, (1920, 1080))
            .output_text()
            .add_object_detector("yolov8", "ultralytics")
            .add_image_classifier("resnet50", "torchvision")
            .add_vision_language_model("gpt-4-vision", "openai")
            .with_transformer_fusion()
            .target_accuracy(0.9)
            .max_latency_ms(8000)
    }

    /// Creates a speech processing pipeline for transcription and analysis
    pub fn speech_processing_pipeline() -> MultiModalPipelineBuilder {
        MultiModalPipelineBuilder::new("Speech Processing")
            .description("Transcribe speech and perform audio analysis")
            .input_audio(AudioFormat::WAV, 16000, 1)
            .output_text()
            .add_speech_to_text("whisper-large", "openai")
            .with_late_fusion()
            .target_accuracy(0.95)
            .max_latency_ms(3000)
    }

    /// Creates a document processing pipeline for parsing and analysis
    pub fn document_processing_pipeline() -> MultiModalPipelineBuilder {
        MultiModalPipelineBuilder::new("Document Processing")
            .description("Parse documents and extract structured information")
            .input_document(DocumentFormat::PDF)
            .output_text()
            .add_document_parser("layout-lm", "microsoft")
            .with_late_fusion()
            .target_accuracy(0.92)
            .max_latency_ms(10000)
    }

    /// Creates a video analysis pipeline for content understanding
    pub fn video_analysis_pipeline() -> MultiModalPipelineBuilder {
        MultiModalPipelineBuilder::new("Video Analysis")
            .description("Analyze video content for scenes, actions, and objects")
            .input_video(VideoFormat::MP4, (1920, 1080), 30.0)
            .output_text()
            .add_video_analyzer("video-swin-transformer", "microsoft")
            .with_attention_fusion()
            .target_accuracy(0.85)
            .max_latency_ms(30000)
    }

    /// Creates a multi-modal content creation pipeline
    pub fn content_creation_pipeline() -> MultiModalPipelineBuilder {
        MultiModalPipelineBuilder::new("Content Creation")
            .description("Create multi-modal content from text descriptions")
            .input_text()
            .output_image(ImageFormat::JPEG, (1024, 1024))
            .output_audio(AudioFormat::MP3, 44100, 2)
            .add_text_to_speech("neural-voice", "azure", "en-US-JennyNeural")
            .with_ensemble_fusion()
            .target_accuracy(0.8)
            .max_latency_ms(15000)
    }

    /// Creates a comprehensive multi-modal analysis pipeline
    pub fn comprehensive_analysis_pipeline() -> MultiModalPipelineBuilder {
        MultiModalPipelineBuilder::new("Comprehensive Multi-Modal Analysis")
            .description("Comprehensive analysis of images, audio, and text")
            .input_image(ImageFormat::JPEG, (1920, 1080))
            .input_audio(AudioFormat::WAV, 44100, 2)
            .input_text()
            .output_text()
            .add_image_classifier("efficientnet-b7", "tensorflow")
            .add_object_detector("yolov8-large", "ultralytics")
            .add_speech_to_text("whisper-large", "openai")
            .add_vision_language_model("flamingo", "deepmind")
            .with_transformer_fusion()
            .target_accuracy(0.92)
            .max_latency_ms(12000)
            .enable_adaptive_quality()
            .enable_fallback()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let pipeline = MultiModalPipelineBuilder::new("Test Pipeline")
            .description("A test pipeline for unit testing")
            .input_image(ImageFormat::JPEG, (1024, 1024))
            .output_text()
            .add_image_classifier("resnet50", "torchvision")
            .add_object_detector("yolov8", "ultralytics")
            .with_late_fusion()
            .target_accuracy(0.9)
            .max_latency_ms(5000)
            .build();

        assert_eq!(pipeline.name, "Test Pipeline");
        assert_eq!(pipeline.description, "A test pipeline for unit testing");
        assert_eq!(pipeline.processors.len(), 2);
        assert_eq!(pipeline.input_modalities.len(), 1);
        assert_eq!(pipeline.output_modalities.len(), 1);
        assert!(matches!(pipeline.fusion_strategy, FusionStrategy::LateFusion));
        assert_eq!(pipeline.quality_settings.target_accuracy, 0.9);
        assert_eq!(pipeline.quality_settings.max_latency_ms, 5000);
    }

    #[test]
    fn test_vision_language_template() {
        let pipeline = PipelineTemplates::vision_language_pipeline().build();
        
        assert_eq!(pipeline.name, "Vision-Language Analysis");
        assert!(pipeline.processors.len() >= 3);
        assert!(!pipeline.input_modalities.is_empty());
        assert!(!pipeline.output_modalities.is_empty());
        assert!(matches!(pipeline.fusion_strategy, FusionStrategy::Transformer));
    }

    #[test]
    fn test_speech_processing_template() {
        let pipeline = PipelineTemplates::speech_processing_pipeline().build();
        
        assert_eq!(pipeline.name, "Speech Processing");
        assert!(!pipeline.processors.is_empty());
        assert!(pipeline.input_modalities.iter().any(|m| matches!(m, Modality::Audio { .. })));
        assert!(pipeline.quality_settings.target_accuracy >= 0.9);
    }

    #[test]
    fn test_comprehensive_analysis_template() {
        let pipeline = PipelineTemplates::comprehensive_analysis_pipeline().build();
        
        assert_eq!(pipeline.name, "Comprehensive Multi-Modal Analysis");
        assert!(pipeline.processors.len() >= 4);
        assert!(pipeline.input_modalities.len() >= 3);
        assert!(pipeline.quality_settings.adaptive_quality);
        assert!(pipeline.quality_settings.fallback_enabled);
    }
}