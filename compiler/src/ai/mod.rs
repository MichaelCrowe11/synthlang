/*!
 * SYNTH AI Integration Module
 * Provides real AI API integrations for SYNTH programs
 */

pub mod openai;
pub mod anthropic;
pub mod embeddings;

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct AIConfig {
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub default_model: String,
    pub embedding_model: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for AIConfig {
    fn default() -> Self {
        Self {
            openai_api_key: std::env::var("OPENAI_API_KEY").ok(),
            anthropic_api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            default_model: "gpt-3.5-turbo".to_string(),
            embedding_model: "text-embedding-ada-002".to_string(),
            max_tokens: 1000,
            temperature: 0.7,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIResponse {
    pub content: String,
    pub model: String,
    pub usage: AIUsage,
    pub confidence: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub text: String,
    pub model: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub embedding: Vec<f32>,
    pub model: String,
    pub usage: AIUsage,
}

pub trait AIProvider: Send + Sync {
    async fn generate(&self, request: AIRequest) -> Result<AIResponse>;
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;
    fn name(&self) -> &'static str;
}

pub struct AIEngine {
    config: AIConfig,
    providers: HashMap<String, Box<dyn AIProvider>>,
}

impl AIEngine {
    pub fn new(config: AIConfig) -> Self {
        let mut engine = Self {
            config,
            providers: HashMap::new(),
        };
        
        engine.register_providers();
        engine
    }

    fn register_providers(&mut self) {
        // Register OpenAI provider if API key is available
        if self.config.openai_api_key.is_some() {
            let provider = openai::OpenAIProvider::new(self.config.clone());
            self.providers.insert("openai".to_string(), Box::new(provider));
        }

        // Register Anthropic provider if API key is available
        if self.config.anthropic_api_key.is_some() {
            let provider = anthropic::AnthropicProvider::new(self.config.clone());
            self.providers.insert("anthropic".to_string(), Box::new(provider));
        }

        // Always register mock provider for testing
        let mock_provider = MockAIProvider::new();
        self.providers.insert("mock".to_string(), Box::new(mock_provider));
    }

    pub async fn generate(&self, request: AIRequest) -> Result<AIResponse> {
        let provider_name = self.determine_provider(&request.model);
        
        if let Some(provider) = self.providers.get(&provider_name) {
            provider.generate(request).await
        } else {
            // Fallback to mock provider
            self.providers.get("mock").unwrap().generate(request).await
        }
    }

    pub async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let provider_name = "openai"; // OpenAI has the best embeddings currently
        
        if let Some(provider) = self.providers.get(provider_name) {
            provider.embed(request).await
        } else {
            // Fallback to mock provider
            self.providers.get("mock").unwrap().embed(request).await
        }
    }

    pub fn similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        embeddings::cosine_similarity(vec1, vec2)
    }

    fn determine_provider(&self, model: &Option<String>) -> String {
        match model {
            Some(model) if model.starts_with("gpt") => "openai".to_string(),
            Some(model) if model.starts_with("claude") => "anthropic".to_string(),
            Some(model) if model == "mock" => "mock".to_string(),
            _ => {
                // Use first available provider
                if self.providers.contains_key("openai") {
                    "openai".to_string()
                } else if self.providers.contains_key("anthropic") {
                    "anthropic".to_string()
                } else {
                    "mock".to_string()
                }
            }
        }
    }

    pub fn available_providers(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }
}

// Mock AI Provider for testing and development
pub struct MockAIProvider;

impl MockAIProvider {
    pub fn new() -> Self {
        Self
    }
}

impl AIProvider for MockAIProvider {
    async fn generate(&self, request: AIRequest) -> Result<AIResponse> {
        // Generate a realistic mock response
        let response_content = format!(
            "Mock AI Response to: '{}'\n\nThis is a simulated response from the SYNTH AI engine. \
            In a real implementation, this would be generated by an actual language model.",
            request.prompt
        );

        Ok(AIResponse {
            content: response_content,
            model: "synth-mock-v1".to_string(),
            usage: AIUsage {
                prompt_tokens: request.prompt.split_whitespace().count() as u32,
                completion_tokens: response_content.split_whitespace().count() as u32,
                total_tokens: (request.prompt.split_whitespace().count() + 
                              response_content.split_whitespace().count()) as u32,
            },
            confidence: Some(0.95),
        })
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        // Generate a deterministic mock embedding based on text hash
        let embedding = embeddings::mock_embedding(&request.text);

        Ok(EmbeddingResponse {
            embedding,
            model: "synth-mock-embedding-v1".to_string(),
            usage: AIUsage {
                prompt_tokens: request.text.split_whitespace().count() as u32,
                completion_tokens: 0,
                total_tokens: request.text.split_whitespace().count() as u32,
            },
        })
    }

    fn name(&self) -> &'static str {
        "mock"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_ai_generation() {
        let config = AIConfig::default();
        let engine = AIEngine::new(config);

        let request = AIRequest {
            prompt: "What is artificial intelligence?".to_string(),
            model: Some("mock".to_string()),
            temperature: Some(0.7),
            max_tokens: Some(100),
            parameters: HashMap::new(),
        };

        let response = engine.generate(request).await.unwrap();
        assert!(!response.content.is_empty());
        assert_eq!(response.model, "synth-mock-v1");
        assert!(response.confidence.unwrap() > 0.0);
    }

    #[tokio::test]
    async fn test_mock_embeddings() {
        let config = AIConfig::default();
        let engine = AIEngine::new(config);

        let request = EmbeddingRequest {
            text: "artificial intelligence".to_string(),
            model: Some("mock".to_string()),
        };

        let response = engine.embed(request).await.unwrap();
        assert_eq!(response.embedding.len(), 1536); // Standard embedding dimension
        assert_eq!(response.model, "synth-mock-embedding-v1");
    }

    #[test]
    fn test_similarity_calculation() {
        let config = AIConfig::default();
        let engine = AIEngine::new(config);

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![1.0, 0.0, 0.0];

        let similarity_different = engine.similarity(&vec1, &vec2);
        let similarity_same = engine.similarity(&vec1, &vec3);

        assert!(similarity_same > similarity_different);
        assert!((similarity_same - 1.0).abs() < 0.001); // Should be very close to 1.0
    }
}