//! Aha-moment detector using archetype embeddings.
//!
//! Embeds a set of "insight archetypes" at startup, then compares incoming
//! messages against them. If any archetype matches above threshold,
//! it triggers spreading activation on the memory tree.

use crate::embed::{cosine_similarity, Embedder};

const ARCHETYPES: &[&str] = &[
    "I just realized this connects to something else",
    "This changes how I understand the problem",
    "Wait, this contradicts what I thought before",
    "This is a key decision that affects everything",
    "I learned something important and new",
    "This person has a skill or role I didn't know about",
    "We should remember this for later",
    "This pattern keeps coming up across different contexts",
];

pub struct AhaDetector {
    archetype_embeddings: Vec<Vec<f32>>,
    pub threshold: f32,
}

impl AhaDetector {
    /// Initialize by embedding all archetypes. Returns None if embedder fails.
    pub fn new(embedder: &Embedder, threshold: f32) -> Option<Self> {
        let mut archetype_embeddings = Vec::with_capacity(ARCHETYPES.len());
        for archetype in ARCHETYPES {
            match embedder.embed(archetype) {
                Ok(emb) => archetype_embeddings.push(emb),
                Err(_) => return None,
            }
        }
        Some(Self { archetype_embeddings, threshold })
    }

    /// Check if a message embedding triggers an aha moment.
    /// Returns the max archetype similarity if above threshold, None otherwise.
    pub fn check(&self, message_embedding: &[f32]) -> Option<f32> {
        let max_sim = self
            .archetype_embeddings
            .iter()
            .map(|arch| cosine_similarity(message_embedding, arch))
            .fold(f32::NEG_INFINITY, f32::max);

        eprintln!("[engram] aha check: max_sim={:.3} threshold={:.3}", max_sim, self.threshold);

        if max_sim >= self.threshold {
            Some(max_sim)
        } else {
            None
        }
    }
}
