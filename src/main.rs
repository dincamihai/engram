mod birch;
mod embed;
mod mcp;

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
    /// Rebalance clusters
    Rebalance,
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
                    println!("[{}] (sim={:.3}) {}", i + 1, e.similarity, e.created_at);
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

        Commands::Rebalance => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            match tree.rebalance() {
                Ok(msg) => println!("{msg}"),
                Err(e) => eprintln!("rebalance error: {e}"),
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

        let chunks = chunk_text(&content, 2000);
        let source = path.to_string_lossy().to_string();

        for chunk in &chunks {
            // Skip chunks with too little actual text
            let alpha_count = chunk.chars().filter(|c| c.is_alphanumeric()).count();
            if alpha_count < 30 {
                continue;
            }

            match embedder.embed(chunk) {
                Ok(embedding) => match tree.store(chunk, &embedding, Some(&source)) {
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
                },
                Err(e) => {
                    errors += 1;
                    eprintln!("\n  embed error {}: {e}", path.display());
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

fn chunk_text(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + max_chars).min(text.len());
        let break_at = text[start..end]
            .rfind('\n')
            .map(|i| start + i + 1)
            .unwrap_or(end);

        let chunk = text[start..break_at].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }
        start = break_at;
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
