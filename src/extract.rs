//! Extract + classify pipeline using fastembed (extraction) + BERT NLI via ONNX (classification).
//! Zero additional dependencies — reuses ort, tokenizers, and hf-hub from fastembed.

use std::sync::Mutex;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;

/// Binary relevance hypotheses — BERT decides "worth remembering?" not "what category?"
/// BIRCH handles categorization through embedding clustering.
pub const RELEVANCE_HYPOTHESES: &[&str] = &[
    "This is about a bug or failure.",
    "This describes a person or team.",
    "This is about a customer or problem.",
    "This describes a system or service.",
    "This is about a work process or decision.",
    "This describes a deployment or configuration.",
    "This is about a policy or security requirement.",
    "This mentions a number or quantity.",
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

/// Generative summarizer using FLAN-T5-small via ONNX Runtime.
/// Downloads encoder + decoder models from HuggingFace on first use (~240MB total).
pub struct Summarizer {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Mutex<tokenizers::Tokenizer>,
}

impl Summarizer {
    pub fn new() -> Result<Self, String> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| format!("hf api: {e}"))?;
        let repo = api.model("Xenova/flan-t5-small".to_string());

        let encoder_path = repo.get("onnx/encoder_model.onnx")
            .map_err(|e| format!("download encoder: {e}"))?;
        let decoder_path = repo.get("onnx/decoder_model_merged.onnx")
            .map_err(|e| format!("download decoder: {e}"))?;
        let tokenizer_path = repo.get("tokenizer.json")
            .map_err(|e| format!("download tokenizer: {e}"))?;

        let encoder = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort opt: {e}"))?
            .commit_from_file(encoder_path)
            .map_err(|e| format!("ort load encoder: {e}"))?;

        let decoder = Session::builder()
            .map_err(|e| format!("ort builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("ort opt: {e}"))?
            .commit_from_file(decoder_path)
            .map_err(|e| format!("ort load decoder: {e}"))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("tokenizer: {e}"))?;

        Ok(Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokenizer: Mutex::new(tokenizer),
        })
    }

    /// Generate a short summary of the given text.
    /// Uses greedy decoding (argmax at each step) for speed.
    pub fn summarize(&self, text: &str, max_tokens: usize) -> Result<String, String> {
        use ort::session::SessionInputValue;

        let prompt = format!("Summarize in one sentence: {}", text);

        let tokenizer = self.tokenizer.lock().map_err(|e| format!("lock: {e}"))?;
        let encoding = tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let src_len = input_ids.len();

        // --- Encoder: run once ---
        let mut encoder = self.encoder.lock().map_err(|e| format!("lock: {e}"))?;
        let enc_out = encoder.run(ort::inputs![
            "input_ids" => Value::from_array(
                ndarray::Array2::from_shape_vec((1, src_len), input_ids)
                    .map_err(|e| format!("shape: {e}"))?
            ).map_err(|e| format!("val: {e}"))?,
            "attention_mask" => Value::from_array(
                ndarray::Array2::from_shape_vec((1, src_len), attention_mask)
                    .map_err(|e| format!("shape: {e}"))?
            ).map_err(|e| format!("val: {e}"))?,
        ]).map_err(|e| format!("encoder run: {e}"))?;

        // Extract encoder hidden states as raw data (rebuild tensor each decoder step)
        let (enc_shape, enc_data) = enc_out[0].try_extract_tensor::<f32>()
            .map_err(|e| format!("extract enc: {e}"))?;
        let enc_shape_vec: Vec<usize> = enc_shape.iter().map(|&d| d as usize).collect();
        let enc_vec: Vec<f32> = enc_data.iter().copied().collect();
        drop(enc_out);
        drop(encoder);

        // --- Decoder: greedy decode loop ---
        let mut decoder = self.decoder.lock().map_err(|e| format!("lock: {e}"))?;

        // Discover KV cache input names and shapes from the model metadata
        let mut kv_names: Vec<String> = Vec::new();
        let mut kv_shapes: Vec<(usize, usize)> = Vec::new(); // (num_heads, head_dim) per tensor
        for input in decoder.inputs().iter() {
            let name = input.name().to_string();
            if name.starts_with("past_key_values") {
                if let ort::value::ValueType::Tensor { shape, .. } = input.dtype() {
                    // Shape is (batch, num_heads, seq_len, head_dim) — [1] and [3]
                    let dims: &[i64] = shape;
                    let heads = dims.get(1).map(|&d| d.max(1) as usize).unwrap_or(8);
                    let hdim = dims.get(3).map(|&d| d.max(1) as usize).unwrap_or(64);
                    kv_shapes.push((heads, hdim));
                } else {
                    kv_shapes.push((8, 64));
                }
                kv_names.push(name);
            }
        }
        let num_kv = kv_names.len();

        let has_use_cache = decoder.inputs().iter()
            .any(|i| i.name() == "use_cache_branch");

        // Build mapping: input kv name → output kv name
        // Input: "past_key_values.0.decoder.key" → Output: "present.0.decoder.key"
        let kv_output_names: Vec<String> = kv_names.iter()
            .map(|n| n.replace("past_key_values.", "present."))
            .collect();

        let eos_token_id = 1i64;
        let pad_token_id = 0i64;

        let mut generated_ids: Vec<i64> = Vec::new();
        let mut decoder_input_id = pad_token_id;
        let mut past_kvs: Vec<(Vec<usize>, Vec<f32>)> = Vec::new();

        for step in 0..max_tokens {
            let first_step = step == 0;

            let mut inputs: Vec<(std::borrow::Cow<str>, SessionInputValue)> = Vec::new();

            // input_ids
            inputs.push(("input_ids".into(), SessionInputValue::from(
                Value::from_array(
                    ndarray::Array2::from_shape_vec((1, 1), vec![decoder_input_id])
                        .map_err(|e| format!("shape: {e}"))?
                ).map_err(|e| format!("val: {e}"))?
            )));

            // encoder_attention_mask
            inputs.push(("encoder_attention_mask".into(), SessionInputValue::from(
                Value::from_array(
                    ndarray::Array2::from_shape_vec((1, src_len), vec![1i64; src_len])
                        .map_err(|e| format!("shape: {e}"))?
                ).map_err(|e| format!("val: {e}"))?
            )));

            // encoder_hidden_states
            inputs.push(("encoder_hidden_states".into(), SessionInputValue::from(
                Value::from_array(
                    ndarray::ArrayD::from_shape_vec(
                        ndarray::IxDyn(&enc_shape_vec), enc_vec.clone()
                    ).map_err(|e| format!("shape: {e}"))?
                ).map_err(|e| format!("val: {e}"))?
            )));

            // use_cache_branch (merged decoder model toggle)
            if has_use_cache {
                inputs.push(("use_cache_branch".into(), SessionInputValue::from(
                    Value::from_array(
                        ndarray::Array1::from_vec(vec![!first_step])
                    ).map_err(|e| format!("val: {e}"))?
                )));
            }

            // Past KV cache
            if first_step {
                for (i, name) in kv_names.iter().enumerate() {
                    let (heads, hdim) = kv_shapes[i];
                    let arr = ndarray::ArrayD::<f32>::from_shape_vec(
                        ndarray::IxDyn(&[1, heads, 0, hdim]), Vec::new()
                    ).map_err(|e| format!("shape: {e}"))?;
                    inputs.push((name.as_str().into(), SessionInputValue::from(
                        Value::from_array(arr).map_err(|e| format!("val: {e}"))?
                    )));
                }
            } else {
                for (i, name) in kv_names.iter().enumerate() {
                    let (shape, data) = &past_kvs[i];
                    let arr = ndarray::ArrayD::<f32>::from_shape_vec(
                        ndarray::IxDyn(shape), data.clone()
                    ).map_err(|e| format!("shape: {e}"))?;
                    inputs.push((name.as_str().into(), SessionInputValue::from(
                        Value::from_array(arr).map_err(|e| format!("val: {e}"))?
                    )));
                }
            }

            let outputs = decoder.run(inputs)
                .map_err(|e| format!("decoder step {step}: {e}"))?;

            // Logits: first output, shape (1, 1, vocab_size)
            let (_, logits) = outputs[0].try_extract_tensor::<f32>()
                .map_err(|e| format!("logits: {e}"))?;

            let next_token = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(eos_token_id);

            if next_token == eos_token_id { break; }

            // Break on any n-gram repetition (n=1..8)
            generated_ids.push(next_token);
            let mut repeated = false;
            for n in 1..=8 {
                let len = generated_ids.len();
                if len >= n * 2 {
                    if generated_ids[len-n..] == generated_ids[len-2*n..len-n] {
                        generated_ids.truncate(len - n);
                        repeated = true;
                        break;
                    }
                }
            }
            if repeated { break; }
            decoder_input_id = next_token;

            // Extract present KV values for next step, matched by name
            // Encoder KV (cross-attention) is computed once at step 0 and reused —
            // subsequent steps return dummy [0,...] shapes, so keep the step-0 values.
            let old_kvs = std::mem::take(&mut past_kvs);
            for (i, out_name) in kv_output_names.iter().enumerate() {
                let is_encoder_kv = out_name.contains(".encoder.");
                let (shape, data) = outputs[out_name.as_str()].try_extract_tensor::<f32>()
                    .map_err(|e| format!("kv {out_name}: {e}"))?;
                let shape_vec: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

                if is_encoder_kv && !first_step && shape_vec.first() == Some(&0) {
                    // Dummy output — reuse cached encoder KV from step 0
                    if i < old_kvs.len() {
                        past_kvs.push(old_kvs[i].clone());
                    } else {
                        past_kvs.push((shape_vec, data.iter().copied().collect()));
                    }
                } else {
                    past_kvs.push((shape_vec, data.iter().copied().collect()));
                }
            }
        }

        let generated_u32: Vec<u32> = generated_ids.iter().map(|&id| id as u32).collect();
        let result = tokenizer.decode(&generated_u32, true)
            .map_err(|e| format!("decode: {e}"))?;

        Ok(result)
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

        // Extract or use directly — only summarize truly long content
        let summary = if sentences.len() <= 5 {
            sentences.join(" ")
        } else {
            extractive_summary(&sentences, &self.embedder, 3).join(" ")
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
