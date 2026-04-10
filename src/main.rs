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
    /// X-ray: visualize tree structure with content
    Tree {
        /// Max characters per entry to show
        #[arg(long, default_value = "80")]
        width: usize,
        /// Watch mode: refresh every N seconds
        #[arg(short, long)]
        watch: Option<f64>,
    },
    /// Ingest files from a directory
    Ingest {
        dir: PathBuf,
        #[arg(long, default_value = "0")]
        limit: usize,
    },
    /// X-ray live: animated BIRCH tree visualization
    Viz,
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
            let recall = tree.recall(&embedding, limit).expect("recall failed");

            if recall.is_empty() {
                println!("no results for: \"{query}\"");
            } else {
                let term_width = terminal_width();
                let prefix_width = 4; // "├── " or "└── "
                let content_width = term_width.saturating_sub(prefix_width).max(40);

                let blocks: Vec<&str> = recall.split("\n\n").collect();
                for (i, block) in blocks.iter().enumerate() {
                    let mut lines = block.lines();
                    if let Some(header) = lines.next() {
                        let label = header.trim_end_matches(':');
                        println!("* ({label})");
                        let entries: Vec<&str> = lines.collect();
                        for (j, line) in entries.iter().enumerate() {
                            let is_last = j == entries.len() - 1;
                            let branch = if is_last { "└── " } else { "├── " };
                            let cont   = if is_last { "    " } else { "│   " };
                            let wrapped = wrap_text(line, content_width);
                            for (k, wline) in wrapped.iter().enumerate() {
                                if k == 0 {
                                    println!("{branch}{wline}");
                                } else {
                                    println!("{cont}{wline}");
                                }
                            }
                        }
                        if i < blocks.len() - 1 {
                            println!();
                        }
                    }
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
            let embedder = embed::Embedder::new(&ollama_url, &ollama_model)
                .expect("Ollama not available — is it running?");
            let tree = birch::Tree::open(db_str, embedder.dimension, birch::Config::default())
                .expect("cannot open tree");
            match tree.rebalance_with_embedder(&embedder) {
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

        Commands::Tree { width, watch } => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            if let Some(secs) = watch {
                let duration = std::time::Duration::from_secs_f64(secs);
                loop {
                    print!("\x1B[2J\x1B[H"); // clear screen, move cursor home
                    print_tree(&tree, width);
                    std::thread::sleep(duration);
                }
            } else {
                print_tree(&tree, width);
            }
        }

        Commands::Viz => {
            let tree = birch::Tree::open(db_str, 768, birch::Config::default())
                .expect("cannot open tree");
            if let Err(e) = engram::viz::run_viz(&tree) {
                eprintln!("viz error: {e}");
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
        match tree.rebalance_with_embedder(&embedder) {
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

fn print_tree(tree: &birch::Tree, content_limit: usize) {
    let topic_tree = match tree.topic_tree(content_limit) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            return;
        }
    };
    let total = tree.count().unwrap_or(0);
    println!("engram x-ray — {} entries\n", total);
    let rendered = render_topic(&topic_tree);
    println!("{rendered}");
}

fn render_topic(node: &birch::TopicTree) -> termtree::Tree<String> {
    let label = if node.label.is_empty() || node.label.starts_with("topic_") {
        format!("[{}]", node.id)
    } else {
        node.label.clone()
    };

    let mut children: Vec<termtree::Tree<String>> = Vec::new();

    // Add entries as leaves
    for entry in &node.entries {
        let access = if entry.access_count > 0 {
            format!(" (x{})", entry.access_count)
        } else {
            String::new()
        };
        let source_tag = entry.source
            .as_deref()
            .map(|s| format!(" [{}]", s.split('/').last().unwrap_or(s)))
            .unwrap_or_default();
        children.push(termtree::Tree::new(format!("#{}{}{} {}", entry.id, source_tag, access, entry.content)));
    }

    // Add child topics
    for child in &node.children {
        children.push(render_topic(child));
    }

    let count_label = if children.is_empty() && node.entries.is_empty() {
        label
    } else {
        format!("{} ({} entries)", label, node.count)
    };

    termtree::Tree::new(count_label).with_leaves(children)
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

fn terminal_width() -> usize {
    // Try ioctl to get actual terminal width
    #[cfg(unix)]
    {
        use std::mem::zeroed;
        unsafe {
            let mut ws: libc::winsize = zeroed();
            if libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                return ws.ws_col as usize;
            }
        }
    }
    // Fall back to COLUMNS env var, then 120
    std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120)
}

fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if text.len() <= width {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    let mut remaining = text;
    while remaining.len() > width {
        // Find last space before width
        let break_at = remaining[..width]
            .rfind(' ')
            .unwrap_or(width);
        lines.push(remaining[..break_at].to_string());
        remaining = remaining[break_at..].trim_start();
    }
    if !remaining.is_empty() {
        lines.push(remaining.to_string());
    }
    lines
}
