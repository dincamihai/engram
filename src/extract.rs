//! Extract + classify pipeline using fastembed (extraction) + BERT NLI via ONNX (classification).
//! Zero additional dependencies — reuses ort, tokenizers, and hf-hub from fastembed.

use std::sync::Mutex;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

/// Binary relevance hypotheses — BERT decides "worth remembering?" not "what category?"
/// BIRCH handles categorization through embedding clustering.
pub const RELEVANCE_HYPOTHESES: &[&str] = &[
    "This contains useful technical information.",
    "This describes a person or team.",
    "This is about a customer or problem.",
    "This describes software or infrastructure.",
    "This is about a work process or decision.",
];

/// Relevance threshold — NLI entailment score above this means "worth storing".
pub const RELEVANCE_THRESHOLD: f32 = 0.1;

/// Split text into sentences (simple rule-based).
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') && current.trim().len() > 10 {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.len() > 10 {
        sentences.push(trimmed);
    }
    sentences
}

/// Extractive summarization: pick top N sentences closest to the centroid.
pub fn extractive_summary(sentences: &[String], embedder: &crate::embed::Embedder, top_n: usize) -> Vec<String> {
    if sentences.len() <= top_n {
        return sentences.to_vec();
    }

    // Embed all sentences
    let embeddings: Vec<Vec<f32>> = sentences.iter()
        .filter_map(|s| embedder.embed(s).ok())
        .collect();

    if embeddings.is_empty() || embeddings.len() != sentences.len() {
        return sentences[..top_n.min(sentences.len())].to_vec();
    }

    // Compute centroid
    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;
    let mut centroid = vec![0.0f32; dim];
    for emb in &embeddings {
        for (i, v) in emb.iter().enumerate() {
            centroid[i] += v / n;
        }
    }

    // Score each sentence by similarity to centroid
    let mut scored: Vec<(usize, f32)> = embeddings.iter()
        .enumerate()
        .map(|(i, emb)| (i, crate::embed::cosine_similarity(emb, &centroid)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return top N in original order
    let mut selected: Vec<usize> = scored[..top_n.min(scored.len())]
        .iter()
        .map(|(i, _)| *i)
        .collect();
    selected.sort();
    selected.iter().map(|&i| sentences[i].clone()).collect()
}

/// BERT NLI classifier using ONNX Runtime.
/// Downloads distilbert-base-uncased-mnli on first use.
pub struct NliClassifier {
    session: Mutex<Session>,
    tokenizer: Mutex<tokenizers::Tokenizer>,
}

impl NliClassifier {
    pub fn new() -> Result<Self, String> {
        // Download model from HuggingFace
        let api = hf_hub::api::sync::Api::new().map_err(|e| format!("hf api: {e}"))?;
        let repo = api.model("Xenova/distilbert-base-uncased-mnli".to_string());

        let model_path = repo.get("onnx/model.onnx")
            .map_err(|e| format!("download model: {e}"))?;
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| format!("download tokenizer: {e}"))?;

        let session = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort opt: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("ort load: {e}"))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("tokenizer: {e}"))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
        })
    }

    /// Classify text against a hypothesis using NLI.
    /// Returns (entailment, neutral, contradiction) scores.
    pub fn nli_scores(&self, premise: &str, hypothesis: &str) -> Result<[f32; 3], String> {
        let tokenizer = self.tokenizer.lock().map_err(|e| format!("lock: {e}"))?;

        // Tokenize premise + hypothesis pair
        let encoding = tokenizer.encode((premise, hypothesis), true)
            .map_err(|e| format!("tokenize: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let len = input_ids.len();

        let input_ids_array = ndarray::Array2::from_shape_vec((1, len), input_ids)
            .map_err(|e| format!("shape: {e}"))?;
        let attention_mask_array = ndarray::Array2::from_shape_vec((1, len), attention_mask)
            .map_err(|e| format!("shape: {e}"))?;

        let mut session = self.session.lock().map_err(|e| format!("lock: {e}"))?;
        let input_ids_val = Value::from_array(input_ids_array).map_err(|e| format!("value: {e}"))?;
        let attention_mask_val = Value::from_array(attention_mask_array).map_err(|e| format!("value: {e}"))?;
        let session_inputs = ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
        ];

        let outputs = session.run(session_inputs)
            .map_err(|e| format!("run: {e}"))?;

        // Extract logits [1, 3] → softmax → [entailment, neutral, contradiction]
        let (_shape, logits_data) = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| format!("extract: {e}"))?;
        let raw = [logits_data[0], logits_data[1], logits_data[2]];

        // Softmax — output order: [entailment, neutral, contradiction]
        let max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        Ok([exps[0] / sum, exps[1] / sum, exps[2] / sum])
    }

    /// Binary relevance check: is this text worth remembering?
    /// Tests against multiple hypotheses, returns max entailment score.
    pub fn is_relevant(&self, text: &str) -> Result<(f32, String), String> {
        let mut best_score = 0.0f32;
        let mut best_hyp = String::new();

        for &hypothesis in RELEVANCE_HYPOTHESES {
            let scores = self.nli_scores(text, hypothesis)?;
            let entailment = scores[0];
            if entailment > best_score {
                best_score = entailment;
                best_hyp = hypothesis.to_string();
            }
        }

        Ok((best_score, best_hyp))
    }
}

/// Full extraction pipeline: extract key sentences + classify + return storable facts.
pub struct Pipeline {
    pub embedder: crate::embed::Embedder,
    pub classifier: NliClassifier,
}

impl Pipeline {
    pub fn new() -> Result<Self, String> {
        let embedder = crate::embed::Embedder::new()
            .ok_or_else(|| "failed to init embedder".to_string())?;
        let classifier = NliClassifier::new()?;
        Ok(Self { embedder, classifier })
    }

    /// Process a text block: extract key sentences, check if relevant.
    /// Returns Some((summary, score)) if worth storing, None if trivial.
    /// BIRCH handles topic assignment — BERT only decides relevance.
    pub fn process(&self, text: &str) -> Result<Option<(String, f32)>, String> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Ok(None);
        }

        // Extract or use directly
        let summary = if sentences.len() <= 3 {
            sentences.join(" ")
        } else {
            extractive_summary(&sentences, &self.embedder, 2).join(" ")
        };

        // Binary relevance check
        let (score, _hyp) = self.classifier.is_relevant(&summary)?;
        if score >= RELEVANCE_THRESHOLD {
            Ok(Some((summary, score)))
        } else {
            Ok(None)
        }
    }
}
