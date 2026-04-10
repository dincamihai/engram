//! Local embedding via fastembed (ONNX inference, no external server needed).

use std::sync::Mutex;
use fastembed::{EmbeddingModel, TextEmbedding, InitOptions};

/// Default embedding dimension (BGE-small-en-v1.5).
pub const DIMENSION: usize = 384;

pub struct Embedder {
    model: Mutex<TextEmbedding>,
    pub dimension: usize,
    max_chars: usize,
}

impl Embedder {
    pub fn new() -> Option<Self> {
        let options = InitOptions::new(EmbeddingModel::BGESmallENV15)
            .with_show_download_progress(true);

        let model = TextEmbedding::try_new(options).ok()?;

        Some(Self {
            model: Mutex::new(model),
            dimension: DIMENSION,
            max_chars: 1000, // BGE-small has ~512 token context
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

        let mut model = self.model.lock().map_err(|e| format!("lock: {e}"))?;
        let results = model
            .embed(vec![truncated], None)
            .map_err(|e| format!("embed failed: {e}"))?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| "empty embedding result".into())
    }
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
    if denom == 0.0 { 0.0 } else { dot / denom }
}
