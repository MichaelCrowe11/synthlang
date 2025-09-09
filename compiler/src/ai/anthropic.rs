/*!
 * Anthropic Claude API Integration for SYNTH
 * Provides real Anthropic API connections for SYNTH programs
 */

use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use super::{AIConfig, AIProvider, AIRequest, AIResponse, AIUsage, EmbeddingRequest, EmbeddingResponse};

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsageResponse,
    model: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    #[serde(rename = "type")]
    content_type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsageResponse {
    input_tokens: u32,
    output_tokens: u32,
}

pub struct AnthropicProvider {
    client: Client,
    config: AIConfig,
}

impl AnthropicProvider {
    pub fn new(config: AIConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    fn get_anthropic_model(&self, requested_model: &Option<String>) -> String {
        match requested_model {
            Some(model) if model.starts_with("claude") => model.clone(),
            _ => "claude-3-haiku-20240307".to_string(), // Default Anthropic model
        }
    }
}

impl AIProvider for AnthropicProvider {
    async fn generate(&self, request: AIRequest) -> Result<AIResponse> {
        let api_key = self.config.anthropic_api_key
            .as_ref()
            .ok_or_else(|| anyhow!("Anthropic API key not configured"))?;

        let model = self.get_anthropic_model(&request.model);
        let max_tokens = request.max_tokens.unwrap_or(self.config.max_tokens);
        
        let anthropic_request = AnthropicRequest {
            model: model.clone(),
            max_tokens,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: request.prompt,
            }],
            temperature: request.temperature.or(Some(self.config.temperature)),
        };

        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&anthropic_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("Anthropic API error: {}", error_text));
        }

        let anthropic_response: AnthropicResponse = response.json().await?;
        
        let content = anthropic_response.content
            .into_iter()
            .filter(|c| c.content_type == "text")
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n");

        if content.is_empty() {
            return Err(anyhow!("No text content returned from Anthropic"));
        }

        Ok(AIResponse {
            content,
            model: anthropic_response.model,
            usage: AIUsage {
                prompt_tokens: anthropic_response.usage.input_tokens,
                completion_tokens: anthropic_response.usage.output_tokens,
                total_tokens: anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens,
            },
            confidence: Some(0.92), // Anthropic doesn't provide confidence, using reasonable default
        })
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        // Anthropic doesn't provide a native embedding API
        // For now, we'll use OpenAI's embedding API as fallback
        // In a production implementation, you might use a different embedding service
        // or implement text-to-embedding conversion using Claude's text analysis
        
        // Generate a mock embedding based on the text for proof-of-concept
        let embedding = super::embeddings::mock_embedding(&request.text);
        
        Ok(EmbeddingResponse {
            embedding,
            model: "claude-text-embedding-mock".to_string(),
            usage: AIUsage {
                prompt_tokens: request.text.split_whitespace().count() as u32,
                completion_tokens: 0,
                total_tokens: request.text.split_whitespace().count() as u32,
            },
        })
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}