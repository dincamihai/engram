use engram::birch;
use engram::embed;
use engram::mcp;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "engram")]
#[command(about = "Self-organizing AI agent memory")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Data directory (default: ~/.engram)
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run as MCP server (stdin/stdout JSON-RPC)
    Serve,
    /// Store a memory
    Store {
        content: String,
        #[arg(long)]
        source: Option<String>,
    },
    /// Search memories
    Search {
        query: String,
        #[arg(short, long, default_value = "5")]
        limit: usize,
    },
    /// List topics
    Topics,
    /// Show stats
    Stats,
    /// Forget (delete) memories by ID or source pattern
    Forget {
        /// Memory ID to delete
        id: Option<i64>,
        /// Delete all entries whose source matches this pattern (SQL LIKE, e.g. "%SKILL.md")
        #[arg(long)]
        source: Option<String>,
    },
    /// Rebalance clusters
    Rebalance,
    /// Rebuild tree from scratch (re-cluster all entries)
    Rebuild,
    /// Ingest files from a directory
    Ingest {
        dir: PathBuf,
        #[arg(long, default_value = "0")]
        limit: usize,
    },
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let data_dir = cli.data_dir.unwrap_or_else(|| {
        let home = dirs_next::home_dir().expect("cannot determine home directory");
        home.join(".engram")
    });
    std::fs::create_dir_all(&data_dir).expect("cannot create data directory");

    let db_path = data_dir.join("engram.db");
    let db_str = db_path.to_str().expect("invalid path");

    let ollama_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".into());
    let ollama_model =
        std::env::var("OLLAMA_EMBED_MODEL").unwrap_or_else(|_| "embeddinggemma".into());

    match cli.command {
        Commands::Serve => {
            let embedder = embed::Embedder::new(&ollama_url, &ollama_model)
                .expect("Ollama not available — is it running?");
            let tree = birch::Tree::open(db_str, embedder.dimension, birch::Config::default())
                .expect("cannot open tree");
            eprintln!("[engram] ready — model={ollama_model} dim={}", embedder.dimension);
            mcp::run(tree, embedder).expect("MCP server error");
        }

        Commands::Store { content, source } => {
            let embedder = embed::Embedder::new(&ollama_url, &ollama_model)
                .expect("Ollama not available");
            let tree = birch::Tree::open(db_str, embedder.dimension, birch::Config::default())
                .expect("cannot open tree");
            let embedding = embedder.embed(&content).expect("embedding failed");
            let (id, topic) = tree.store(&content, &embedding, source.as_deref()).expect("store failed");
            println!("stored #{id} in topic: {topic}");
        }

        Commands::Search { query, limit } => {
            let embedder = embed::Embedder::new(&ollama_url, &ollama_model)
                .expect("Ollama not available");
            let tree = birch::Tree::open(db_str, embedder.dimension, birch::Config::default())
                .expect("cannot open tree");
            let embedding = embedder.embed(&query).expect("embedding failed");
            let results = tree.search(&embedding, limit).expect("search failed");

            if results.is_empty() {
                println!("no results for: \"{query}\"");
            } else {
                for (i, e) in results.iter().enumerate() {
                    println!("[{}] (sim={:.3}) {}", i + 1, e.similarity, e.when);
                    if let Some(ref src) = e.source {
                        println!("    source: {src}");
                    }
                    for line in e.content.lines().take(3) {
                        println!("    {line}");
                    }
                    println!();
                }
            }
        }

        Commands::Topics => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            let topics = tree.topics().expect("topics failed");
            let total = tree.count().unwrap_or(0);
            println!("total entries: {total}\n");
            print_topics(&topics, 0);
        }

        Commands::Stats => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            let total = tree.count().unwrap_or(0);
            let topics = tree.topics().unwrap_or_default();
            let leaf_count = count_leaves(&topics);
            println!("entries:  {total}");
            println!("topics:   {leaf_count}");
        }

        Commands::Forget { id, source } => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            if let Some(pattern) = source {
                match tree.forget_by_source(&pattern) {
                    Ok(n) => println!("forgot {n} entries matching source \"{pattern}\""),
                    Err(e) => eprintln!("error: {e}"),
                }
            } else if let Some(id) = id {
                match tree.forget(id) {
                    Ok(true) => println!("forgot #{id}"),
                    Ok(false) => println!("#{id} not found"),
                    Err(e) => eprintln!("error: {e}"),
                }
            } else {
                eprintln!("provide an ID or --source pattern");
            }
        }

        Commands::Rebalance => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            match tree.rebalance() {
                Ok(msg) => println!("{msg}"),
                Err(e) => eprintln!("rebalance error: {e}"),
            }
        }

        Commands::Rebuild => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            match tree.rebuild() {
                Ok(msg) => println!("{msg}"),
                Err(e) => eprintln!("error: {e}"),
            }
        }

        Commands::Ingest { dir, limit } => {
            let embedder = embed::Embedder::new(&ollama_url, &ollama_model)
                .expect("Ollama not available");
            let tree = birch::Tree::open(db_str, embedder.dimension, birch::Config::default())
                .expect("cannot open tree");
            ingest(&dir, &tree, &embedder, limit);
        }
    }
}

fn ingest(dir: &std::path::Path, tree: &birch::Tree, embedder: &embed::Embedder, limit: usize) {
    let mut count = 0;
    let mut errors = 0;

    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "md" | "txt" | "rs" | "py" | "ts" | "js" | "toml" | "yaml" | "yml" | "json" | "sh") {
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => continue,
        };

        if content.trim().is_empty() || content.len() < 50 {
            continue;
        }

        // Skip binary-looking or non-text content
        if content.bytes().filter(|b| *b == 0).count() > 0 {
            continue;
        }

        // Get file modification date for temporal embedding
        let file_date = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .map(|t| {
                let dt: chrono::DateTime<chrono::Utc> = t.into();
                dt.format("%Y-%m-%d").to_string()
            })
            .unwrap_or_else(|| chrono::Utc::now().format("%Y-%m-%d").to_string());

        let chunks = semantic_chunk(&content, embedder);
        let source = path.to_string_lossy().to_string();

        for (chunk, _chunk_embedding) in &chunks {
            let dated = format!("{}: {}", file_date, chunk);
            let dated_embedding = match embedder.embed(&dated) {
                Ok(e) => e,
                Err(_) => continue,
            };
            match tree.store(&dated, &dated_embedding, Some(&source)) {
                Ok(_) => {
                    count += 1;
                    if count % 10 == 0 {
                        eprint!("\r  stored {count} entries...");
                    }
                }
                Err(e) => {
                    errors += 1;
                    eprintln!("\n  error storing {}: {e}", path.display());
                }
            }

            if limit > 0 && count >= limit {
                break;
            }
        }

        if limit > 0 && count >= limit {
            break;
        }
    }

    eprintln!();
    println!("ingested {count} entries ({errors} errors) from {}", dir.display());

    if count > 0 {
        match tree.rebalance() {
            Ok(msg) => println!("rebalance: {msg}"),
            Err(e) => eprintln!("rebalance error: {e}"),
        }
    }
}

/// Semantic chunking: split text into passages by detecting topic shifts via embedding similarity.
fn semantic_chunk(text: &str, embedder: &embed::Embedder) -> Vec<(String, Vec<f32>)> {
    const SIMILARITY_THRESHOLD: f32 = 0.75;
    const MIN_CHUNK_CHARS: usize = 30;

    // Split into paragraphs (double newline or single newline for short blocks)
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .flat_map(|block| {
            // If a block is very large, split on single newlines too
            if block.len() > 3000 {
                block.split('\n').collect::<Vec<_>>()
            } else {
                vec![block]
            }
        })
        .map(|p| p.trim())
        .filter(|p| {
            let alpha = p.chars().filter(|c| c.is_alphanumeric()).count();
            alpha >= MIN_CHUNK_CHARS
        })
        .collect();

    if paragraphs.is_empty() {
        return Vec::new();
    }

    // Embed each paragraph
    let mut para_embeddings: Vec<Option<Vec<f32>>> = Vec::with_capacity(paragraphs.len());
    for p in &paragraphs {
        para_embeddings.push(embedder.embed(p).ok());
    }

    // Group consecutive paragraphs into chunks, splitting at topic boundaries
    let mut chunks: Vec<(String, Vec<f32>)> = Vec::new();
    let mut current_paras: Vec<&str> = Vec::new();
    let mut current_embedding: Option<Vec<f32>> = None;

    for i in 0..paragraphs.len() {
        let emb = &para_embeddings[i];

        if current_paras.is_empty() {
            current_paras.push(paragraphs[i]);
            current_embedding = emb.clone();
            continue;
        }

        // Check if this paragraph belongs with the current chunk
        let should_split = match (&current_embedding, emb) {
            (Some(prev), Some(curr)) => {
                let sim = embed::cosine_similarity(prev, curr);
                sim < SIMILARITY_THRESHOLD
            }
            _ => false, // if embedding failed, keep grouping
        };

        // Also split if current chunk is getting too large
        let current_len: usize = current_paras.iter().map(|p| p.len()).sum();
        let too_large = current_len > 2000;

        if should_split || too_large {
            // Flush current chunk
            let text = current_paras.join("\n\n");
            if let Some(emb) = current_embedding.take() {
                chunks.push((text, emb));
            } else if let Ok(emb) = embedder.embed(&text) {
                chunks.push((text, emb));
            }
            current_paras.clear();
        }

        current_paras.push(paragraphs[i]);
        current_embedding = emb.clone();
    }

    // Flush remaining
    if !current_paras.is_empty() {
        let text = current_paras.join("\n\n");
        if let Some(emb) = current_embedding {
            chunks.push((text, emb));
        } else if let Ok(emb) = embedder.embed(&text) {
            chunks.push((text, emb));
        }
    }

    chunks
}

fn print_topics(topics: &[birch::Topic], indent: usize) {
    for t in topics {
        let pad = "  ".repeat(indent);
        println!("{pad}{} ({} entries)", t.label, t.count);
        if !t.children.is_empty() {
            print_topics(&t.children, indent + 1);
        }
    }
}

fn count_leaves(topics: &[birch::Topic]) -> usize {
    topics
        .iter()
        .map(|t| {
            if t.children.is_empty() {
                1
            } else {
                count_leaves(&t.children)
            }
        })
        .sum()
}
