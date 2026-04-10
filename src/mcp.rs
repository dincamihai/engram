//! MCP server — JSON-RPC over stdin/stdout. Two tools: store, search.

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::activation::AhaDetector;
use crate::birch::Tree;
use crate::embed::Embedder;

#[derive(Deserialize)]
struct Request {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Serialize)]
struct Response {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<serde_json::Value>,
}

impl Response {
    fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }
    fn error(id: Option<serde_json::Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(json!({"code": code, "message": message})),
        }
    }
}

pub fn run(tree: Tree, embedder: Embedder) -> Result<(), String> {
    // Initialize aha detector for automatic spreading activation
    let aha = AhaDetector::new(&embedder, 0.40);
    if aha.is_some() {
        eprintln!("[engram] aha detector ready ({} archetypes)", 8);
    }

    // Start queue watcher thread
    let queue_path = {
        let db_dir = std::path::Path::new(&tree.db_path())
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf();
        db_dir.join("queue.jsonl")
    };
    eprintln!("[engram] queue watcher: {}", queue_path.display());

    // Share tree and embedder with queue thread via Arc<Mutex<>>
    let tree = std::sync::Arc::new(std::sync::Mutex::new(tree));
    let embedder = std::sync::Arc::new(embedder);

    // Initialize NLI classifier in queue thread (lazy, only if queue has items)
    let tree_q = tree.clone();
    let embedder_q = embedder.clone();
    let queue_path_q = queue_path.clone();
    std::thread::spawn(move || {
        queue_watcher(&queue_path_q, &tree_q, &embedder_q);
    });

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line.map_err(|e| format!("stdin: {e}"))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let request: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let resp = Response::error(None, -32700, format!("Parse error: {e}"));
                writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).ok();
                stdout.flush().ok();
                continue;
            }
        };

        let tree_lock = tree.lock().map_err(|e| format!("lock: {e}"))?;
        let response = handle(&request, &tree_lock, &embedder, aha.as_ref());
        drop(tree_lock);
        if let Some(resp) = response {
            writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).ok();
            stdout.flush().ok();
        }
    }

    Ok(())
}

/// Queue watcher: polls queue.jsonl for new entries, processes with BERT NLI.
fn queue_watcher(
    queue_path: &std::path::Path,
    tree: &std::sync::Arc<std::sync::Mutex<Tree>>,
    embedder: &std::sync::Arc<Embedder>,
) {
    use std::io::BufRead;

    // Lazy-init classifier on first queue item
    let mut classifier: Option<crate::extract::NliClassifier> = None;

    loop {
        std::thread::sleep(std::time::Duration::from_secs(2));

        // Check if queue file exists and has content
        let content = match std::fs::read_to_string(queue_path) {
            Ok(c) if !c.trim().is_empty() => c,
            _ => continue,
        };

        // Clear queue immediately (atomic: truncate)
        std::fs::write(queue_path, "").ok();

        // Lazy-init classifier
        if classifier.is_none() {
            eprintln!("[engram] queue: loading NLI classifier...");
            match crate::extract::NliClassifier::new() {
                Ok(c) => {
                    eprintln!("[engram] queue: classifier ready");
                    classifier = Some(c);
                }
                Err(e) => {
                    eprintln!("[engram] queue: classifier failed: {e}");
                    continue;
                }
            }
        }
        let nli = classifier.as_ref().unwrap();

        // Process each line
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }

            // Parse queue entry: {"text": "...", "source": "..."}
            #[derive(serde::Deserialize)]
            struct QueueEntry {
                text: String,
                source: Option<String>,
            }

            let entry: QueueEntry = match serde_json::from_str(line) {
                Ok(e) => e,
                Err(_) => {
                    // Fallback: treat raw line as text
                    QueueEntry { text: line.to_string(), source: Some("queue".to_string()) }
                }
            };

            // Extract + classify
            let sentences = crate::extract::split_sentences(&entry.text);
            let summary = if sentences.len() <= 3 {
                sentences.join(" ")
            } else {
                crate::extract::extractive_summary(&sentences, &embedder, 2).join(" ")
            };

            if summary.is_empty() { continue; }

            let (score, _hyp) = match nli.is_relevant(&summary) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[engram] queue: classify error: {e}");
                    continue;
                }
            };

            if score < crate::extract::RELEVANCE_THRESHOLD {
                eprintln!("[engram] queue: skip ({score:.2}) {}", &summary[..summary.len().min(60)]);
                continue;
            }

            // Store
            match embedder.embed(&summary) {
                Ok(embedding) => {
                    let src = entry.source.as_deref().unwrap_or("auto-extract");
                    let tree_lock = tree.lock().unwrap();
                    match tree_lock.store(&summary, &embedding, Some(src)) {
                        Ok((_id, label)) => eprintln!("[engram] queue: stored ({score:.2}) → {label}"),
                        Err(e) => eprintln!("[engram] queue: store error: {e}"),
                    }
                }
                Err(e) => eprintln!("[engram] queue: embed error: {e}"),
            }
        }
    }
}

fn handle(req: &Request, tree: &Tree, embedder: &Embedder, aha: Option<&AhaDetector>) -> Option<Response> {
    match req.method.as_str() {
        "initialize" => Some(Response::success(
            req.id.clone(),
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "engram", "version": env!("CARGO_PKG_VERSION") }
            }),
        )),
        "notifications/initialized" => None,
        "tools/list" => Some(Response::success(req.id.clone(), json!({ "tools": tools() }))),
        "tools/call" => {
            let name = req.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = req.params.get("arguments").cloned().unwrap_or(json!({}));
            let result = dispatch(name, &args, tree, embedder, aha);
            Some(Response::success(
                req.id.clone(),
                json!({ "content": [{ "type": "text", "text": result }] }),
            ))
        }
        method => Some(Response::error(req.id.clone(), -32601, format!("Unknown method: {method}"))),
    }
}

fn dispatch(name: &str, args: &serde_json::Value, tree: &Tree, embedder: &Embedder, aha: Option<&AhaDetector>) -> String {
    let result = match name {
        "engram_store" => {
            let content = args.get("content").and_then(|v| v.as_str()).unwrap_or("");
            let source = args.get("source").and_then(|v| v.as_str());

            if content.is_empty() {
                json!({"error": "content is required"})
            } else {
                let dated = format!("{}: {}", chrono::Utc::now().format("%Y-%m-%d"), content);
                match embedder.embed(&dated) {
                    Ok(embedding) => {
                        // Check for aha moment against raw content (undated)
                        let aha_triggered = if let Some(detector) = aha {
                            let raw_emb = embedder.embed(content).ok();
                            let check_emb = raw_emb.as_deref().unwrap_or(&embedding);
                            if let Some(sim) = detector.check(check_emb) {
                                let activated = tree.replay(&embedding, 0.2, 5).unwrap_or(0);
                                Some((sim, activated))
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        match tree.store(&dated, &embedding, source) {
                            Ok((_id, _topic)) => {
                                // Silently trigger spreading activation on aha moments
                                if let Some((_sim, _activated)) = aha_triggered {
                                    // activation already happened above
                                }
                                json!({"stored": true})
                            }
                            Err(e) => json!({"error": e}),
                        }
                    }
                    Err(e) => json!({"error": format!("embedding failed: {e}")}),
                }
            }
        }

        "engram_search" => {
            let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

            if query.is_empty() {
                json!({"error": "query is required"})
            } else {
                match embedder.embed(query) {
                    Ok(embedding) => match tree.recall(&embedding, limit) {
                        Ok(recall) => json!({"recall": recall}),
                        Err(e) => json!({"error": e}),
                    },
                    Err(e) => json!({"error": format!("embedding failed: {e}")}),
                }
            }
        }

        _ => json!({"error": format!("unknown tool: {name}")}),
    };

    serde_json::to_string_pretty(&result).unwrap_or_else(|_| "{}".into())
}

fn tools() -> Vec<serde_json::Value> {
    vec![
        json!({
            "name": "engram_store",
            "description": "Store a memory. Content is embedded and self-organized into topics.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content to store"},
                    "source": {"type": "string", "description": "Optional source reference"}
                },
                "required": ["content"]
            }
        }),
        json!({
            "name": "engram_search",
            "description": "Search memories by semantic similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (natural language)"},
                    "limit": {"type": "integer", "description": "Max results (default 5)"}
                },
                "required": ["query"]
            }
        }),
    ]
}
