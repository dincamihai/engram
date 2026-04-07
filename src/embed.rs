//! Ollama embedding integration.

use serde::{Deserialize, Serialize};

pub struct Embedder {
    base_url: String,
    model: String,
    client: reqwest::blocking::Client,
    pub dimension: usize,
    max_chars: usize,
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f64>,
}

impl Embedder {
    pub fn new(base_url: &str, model: &str) -> Option<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok()?;

        let dimension = probe_dimension(&client, base_url, model)?;
        let max_chars = if dimension <= 384 { 1000 } else { 8000 };

        Some(Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client,
            dimension,
            max_chars,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let truncated = if text.len() > self.max_chars {
            let boundary = text
                .char_indices()
                .nth(self.max_chars)
                .map(|(i, _)| i)
                .unwrap_or(text.len());
            &text[..boundary]
        } else {
            text
        };

        let url = format!("{}/api/embeddings", self.base_url.trim_end_matches('/'));
        let resp = self
            .client
            .post(&url)
            .json(&EmbedRequest {
                model: &self.model,
                prompt: truncated,
            })
            .send()
            .map_err(|e| format!("embed request failed: {e}"))?;

        if !resp.status().is_success() {
            return Err(format!("Ollama returned status {}", resp.status()));
        }

        let data: EmbedResponse = resp
            .json()
            .map_err(|e| format!("parse embedding response: {e}"))?;

        Ok(data.embedding.into_iter().map(|v| v as f32).collect())
    }
}

fn probe_dimension(
    client: &reqwest::blocking::Client,
    base_url: &str,
    model: &str,
) -> Option<usize> {
    let url = format!("{}/api/embeddings", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .json(&EmbedRequest {
            model,
            prompt: "dimension probe",
        })
        .send()
        .ok()?;
    let data: EmbedResponse = resp.json().ok()?;
    Some(data.embedding.len())
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}
