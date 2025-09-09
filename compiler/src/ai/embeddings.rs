/*!
 * Embedding utilities for SYNTH AI operations
 * Provides vector operations and similarity calculations
 */

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() {
        return 0.0;
    }

    let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude1: f32 = vec1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let magnitude2: f32 = vec2.iter().map(|a| a * a).sum::<f32>().sqrt();

    if magnitude1 == 0.0 || magnitude2 == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude1 * magnitude2)
}

/// Calculate Euclidean distance between two vectors
pub fn euclidean_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() {
        return f32::INFINITY;
    }

    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Calculate Manhattan distance between two vectors
pub fn manhattan_distance(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() {
        return f32::INFINITY;
    }

    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum()
}

/// Generate a deterministic mock embedding based on text content
/// This is used for testing and when no real embedding API is available
pub fn mock_embedding(text: &str) -> Vec<f32> {
    const EMBEDDING_DIMENSIONS: usize = 1536; // Standard OpenAI embedding size
    
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let seed = hasher.finish();
    
    // Simple deterministic pseudo-random number generation
    let mut rng_state = seed;
    let mut embedding = Vec::with_capacity(EMBEDDING_DIMENSIONS);
    
    for i in 0..EMBEDDING_DIMENSIONS {
        // Linear congruential generator for reproducible "randomness"
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        
        // Convert to normalized float between -1 and 1
        let normalized = ((rng_state as i64) as f64) / (u64::MAX as f64);
        let value = (normalized * 2.0 - 1.0) as f32;
        
        // Add some text-based features for more realistic embeddings
        let char_influence = if i < text.len() {
            (text.chars().nth(i).unwrap() as u8 as f32) / 255.0 - 0.5
        } else {
            0.0
        };
        
        let word_influence = if i % 50 < text.split_whitespace().count() {
            let words: Vec<_> = text.split_whitespace().collect();
            let word_idx = i % words.len();
            (words[word_idx].len() as f32) / 100.0 - 0.5
        } else {
            0.0
        };
        
        embedding.push(value * 0.7 + char_influence * 0.2 + word_influence * 0.1);
    }
    
    // Normalize the vector to unit length (common for embeddings)
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for value in &mut embedding {
            *value /= magnitude;
        }
    }
    
    embedding
}

/// Find the most similar vectors in a collection
pub fn find_most_similar<T>(
    query_vector: &[f32],
    candidates: &[(T, Vec<f32>)],
    top_k: usize,
) -> Vec<(f32, &T)> 
where
    T: Clone,
{
    let mut similarities: Vec<(f32, &T)> = candidates
        .iter()
        .map(|(item, vector)| (cosine_similarity(query_vector, vector), item))
        .collect();
    
    similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(top_k);
    similarities
}

/// Embedding cache for performance optimization
pub struct EmbeddingCache {
    cache: std::collections::HashMap<String, Vec<f32>>,
    max_size: usize,
}

impl EmbeddingCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size,
        }
    }

    pub fn get(&self, text: &str) -> Option<&Vec<f32>> {
        self.cache.get(text)
    }

    pub fn insert(&mut self, text: String, embedding: Vec<f32>) {
        if self.cache.len() >= self.max_size {
            // Simple eviction: remove first entry (not LRU, but good enough for POC)
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(text, embedding);
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![1.0, 0.0, 0.0];

        let similarity_orthogonal = cosine_similarity(&vec1, &vec2);
        let similarity_identical = cosine_similarity(&vec1, &vec3);

        assert!(similarity_identical > similarity_orthogonal);
        assert!((similarity_identical - 1.0).abs() < 0.001);
        assert!(similarity_orthogonal.abs() < 0.001);
    }

    #[test]
    fn test_mock_embedding() {
        let embedding1 = mock_embedding("artificial intelligence");
        let embedding2 = mock_embedding("machine learning");
        let embedding3 = mock_embedding("artificial intelligence"); // Same text

        assert_eq!(embedding1.len(), 1536);
        assert_eq!(embedding2.len(), 1536);
        assert_eq!(embedding1, embedding3); // Deterministic

        let similarity = cosine_similarity(&embedding1, &embedding2);
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_embedding_cache() {
        let mut cache = EmbeddingCache::new(2);
        let embedding1 = vec![1.0, 0.0, 0.0];
        let embedding2 = vec![0.0, 1.0, 0.0];
        let embedding3 = vec![0.0, 0.0, 1.0];

        cache.insert("test1".to_string(), embedding1.clone());
        cache.insert("test2".to_string(), embedding2.clone());
        assert_eq!(cache.size(), 2);

        assert_eq!(cache.get("test1"), Some(&embedding1));

        // Should evict first entry when exceeding max size
        cache.insert("test3".to_string(), embedding3.clone());
        assert_eq!(cache.size(), 2);
    }
}