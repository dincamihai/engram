//! MCP server — JSON-RPC over stdin/stdout. Four tools: store, search, topics, forget.

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
    let aha = AhaDetector::new(&embedder, 0.55);
    if aha.is_some() {
        eprintln!("[engram] aha detector ready ({} archetypes)", 8);
    }

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

        let response = handle(&request, &tree, &embedder, aha.as_ref());
        if let Some(resp) = response {
            writeln!(stdout, "{}", serde_json::to_string(&resp).unwrap()).ok();
            stdout.flush().ok();
        }
    }

    Ok(())
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
                        // Check for aha moment → auto-replay
                        let aha_triggered = if let Some(detector) = aha {
                            if let Some(sim) = detector.check(&embedding) {
                                let activated = tree.replay(&embedding, 0.2, 5).unwrap_or(0);
                                Some((sim, activated))
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        match tree.store(&dated, &embedding, source) {
                            Ok((id, topic)) => {
                                let mut result = json!({"id": id, "topic": topic});
                                if let Some((sim, activated)) = aha_triggered {
                                    result["aha"] = json!({"similarity": format!("{:.3}", sim), "activated": activated});
                                }
                                result
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
                    Ok(embedding) => match tree.search(&embedding, limit) {
                        Ok(entries) => {
                            let results: Vec<serde_json::Value> = entries
                                .iter()
                                .map(|e| {
                                    json!({
                                        "id": e.id,
                                        "content": e.content,
                                        "source": e.source,
                                        "similarity": format!("{:.3}", e.similarity),
                                        "when": e.when,
                                    })
                                })
                                .collect();
                            json!({"query": query, "count": results.len(), "results": results})
                        }
                        Err(e) => json!({"error": e}),
                    },
                    Err(e) => json!({"error": format!("embedding failed: {e}")}),
                }
            }
        }

        "engram_topics" => match tree.topics() {
            Ok(topics) => {
                let total = tree.count().unwrap_or(0);
                json!({"total_entries": total, "topics": topics})
            }
            Err(e) => json!({"error": e}),
        },

        "engram_forget" => {
            let id = args.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
            if id == 0 {
                json!({"error": "id is required"})
            } else {
                match tree.forget(id) {
                    Ok(true) => json!({"forgotten": true}),
                    Ok(false) => json!({"forgotten": false, "reason": "not found"}),
                    Err(e) => json!({"error": e}),
                }
            }
        }

        "engram_replay" => {
            let context = args.get("context").and_then(|v| v.as_str()).unwrap_or("");
            let noise = args.get("noise").and_then(|v| v.as_f64()).unwrap_or(0.2) as f32;
            let count = args.get("count").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

            if context.is_empty() {
                json!({"error": "context is required"})
            } else {
                match embedder.embed(context) {
                    Ok(embedding) => match tree.replay(&embedding, noise, count) {
                        Ok(activated) => json!({"activated": activated, "noise": noise}),
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
        json!({
            "name": "engram_topics",
            "description": "List self-organized memory topics with counts.",
            "inputSchema": {"type": "object", "properties": {}}
        }),
        json!({
            "name": "engram_forget",
            "description": "Delete a memory by ID.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Entry ID to forget"}
                },
                "required": ["id"]
            }
        }),
        json!({
            "name": "engram_replay",
            "description": "Spreading activation: rehearse memories related to context. Keeps relevant memories alive organically.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "Current context or topic to activate around"},
                    "noise": {"type": "number", "description": "Probability of random activation (0-1, default 0.2)"},
                    "count": {"type": "integer", "description": "Number of memories to activate (default 5)"}
                },
                "required": ["context"]
            }
        }),
    ]
}
