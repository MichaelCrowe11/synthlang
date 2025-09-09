/*! 
 * OpenAI API Integration for SYNTH
 * Provides real OpenAI API connections for SYNTH programs
 */

use std::collections::HashMap;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use super::{AIConfig, AIProvider, AIRequest, AIResponse, AIUsage, EmbeddingRequest, EmbeddingResponse};

#[derive(Debug, Serialize)]
struct OpenAICompletionRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAICompletionResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsageResponse,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsageResponse {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    usage: OpenAIUsageResponse,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

pub struct OpenAIProvider {
    client: Client,
    config: AIConfig,
}

impl OpenAIProvider {
    pub fn new(config: AIConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }
}

impl AIProvider for OpenAIProvider {
    async fn generate(&self, request: AIRequest) -> Result<AIResponse> {
        let api_key = self.config.openai_api_key
            .as_ref()
            .ok_or_else(|| anyhow!("OpenAI API key not configured"))?;

        let model = request.model.unwrap_or_else(|| self.config.default_model.clone());
        
        let openai_request = OpenAICompletionRequest {
            model: model.clone(),
            messages: vec![OpenAIMessage {
                role: "user".to_string(),
                content: request.prompt,
            }],
            temperature: request.temperature.or(Some(self.config.temperature)),
            max_tokens: request.max_tokens.or(Some(self.config.max_tokens)),
        };

        let response = self.client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("OpenAI API error: {}", error_text));
        }

        let openai_response: OpenAICompletionResponse = response.json().await?;
        
        let choice = openai_response.choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No choices returned from OpenAI"))?;

        Ok(AIResponse {
            content: choice.message.content,
            model: openai_response.model,
            usage: AIUsage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                completion_tokens: openai_response.usage.completion_tokens,
                total_tokens: openai_response.usage.total_tokens,
            },
            confidence: Some(0.9), // OpenAI doesn't provide confidence, using reasonable default
        })
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let api_key = self.config.openai_api_key
            .as_ref()
            .ok_or_else(|| anyhow!("OpenAI API key not configured"))?;

        let model = request.model.unwrap_or_else(|| self.config.embedding_model.clone());
        
        let openai_request = OpenAIEmbeddingRequest {
            model: model.clone(),
            input: request.text,
        };

        let response = self.client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&openai_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow!("OpenAI API error: {}", error_text));
        }

        let openai_response: OpenAIEmbeddingResponse = response.json().await?;
        
        let embedding_data = openai_response.data
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("No embedding data returned from OpenAI"))?;

        Ok(EmbeddingResponse {
            embedding: embedding_data.embedding,
            model: openai_response.model,
            usage: AIUsage {
                prompt_tokens: openai_response.usage.prompt_tokens,
                completion_tokens: openai_response.usage.completion_tokens,
                total_tokens: openai_response.usage.total_tokens,
            },
        })
    }

    fn name(&self) -> &'static str {
        "openai"
    }
}