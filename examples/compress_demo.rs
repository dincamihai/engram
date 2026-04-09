use std::io::{self, Read};

fn main() {
    let text = if std::env::args().len() > 1 {
        std::env::args().skip(1).collect::<Vec<_>>().join(" ")
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).expect("read stdin");
        buf
    };

    if text.trim().is_empty() {
        eprintln!("usage: compress-demo <text>  or  echo text | compress-demo");
        std::process::exit(1);
    }

    let char_count = text.chars().count();
    println!("=== INPUT ({} chars) ===\n{}\n", char_count, text);

    // Detect content type
    let is_code = detect_code(&text);
    println!("=== CONTENT TYPE: {} ===\n", if is_code { "CODE" } else { "PROSE" });

    if is_code {
        // Structural compression for code
        let structural = compress_structural(&text);
        let structural_chars = structural.chars().count();
        println!("=== STRUCTURAL ({} / {} = {:.0}%) ===\n{}\n",
            structural_chars, char_count,
            (structural_chars as f64 / char_count as f64) * 100.0,
            structural
        );
    }

    // Level 1: Caveman — always applied
    let (caveman, lang) = compress_caveman(&text);
    let caveman_chars = caveman.chars().count();
    println!("=== CAVEMAN [{}] ({} / {} = {:.0}%) ===\n{}\n",
        lang, caveman_chars, char_count,
        (caveman_chars as f64 / char_count as f64) * 100.0,
        caveman
    );

    // Level 2: Keywords — YAKE extraction
    let keywords = extract_keywords_yake(&text);
    if !keywords.is_empty() {
        println!("=== KEYWORDS (YAKE) ===");
        for (i, k) in keywords.iter().take(15).enumerate() {
            println!("  {:2}. {}  (score: {:.4})", i + 1, k.raw, k.score);
        }
        println!();
    }

    // Summary: what would engram store?
    println!("=== ENGRAM STORAGE ===");
    if is_code {
        println!("  content_display: <structural compression>");
        println!("  content_raw:     <original, for dedup>");
    } else {
        println!("  content_display:  <caveman or YAKE top-N>");
        println!("  content_raw:      <original, for dedup>");
    }
}

/// Detect if text looks like code
fn detect_code(text: &str) -> bool {
    let indicators = [
        "fn ", "def ", "function ", "class ", "pub ", "async ",
        "impl ", "struct ", "enum ", "interface ", "import ",
        "->", "=>", "}:",
    ];
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() { return false; }

    let hits = indicators.iter().filter(|pat| text.contains(**pat)).count();
    let code_line_ratio = lines.iter()
        .filter(|l| {
            let t = l.trim();
            t.ends_with('{') || t.ends_with('}') || t.ends_with(';') || t.ends_with(':')
            || t.starts_with("def ") || t.starts_with("fn ") || t.starts_with("class ")
            || t.starts_with("pub ") || t.starts_with("impl ")
            || t.starts_with("import ") || t.starts_with("use ")
            || t.starts_with('#') || t.starts_with("//")
        })
        .count() as f64 / lines.len() as f64;

    hits >= 2 || code_line_ratio > 0.3
}

/// Structural compression for code: extract signatures and structure
fn compress_structural(code: &str) -> String {
    let mut results: Vec<String> = Vec::new();
    let mut in_block_comment = false;

    for line in code.lines() {
        let trimmed = line.trim();

        // Skip empty lines
        if trimmed.is_empty() { continue; }

        // Skip block comments
        if trimmed.starts_with("/*") { in_block_comment = true; }
        if in_block_comment {
            if trimmed.ends_with("*/") { in_block_comment = false; }
            continue;
        }
        if trimmed.starts_with("/*") && trimmed.ends_with("*/") { continue; }

        // Skip line comments
        if trimmed.starts_with('#') || trimmed.starts_with("//") || trimmed.starts_with("--") {
            continue;
        }

        // Keep structural lines: definitions, declarations, control flow
        let is_structural = is_structural_line(trimmed);
        if is_structural {
            results.push(trimmed.to_string());
        }
    }

    results.join("\n")
}

fn is_structural_line(line: &str) -> bool {
    // Definition patterns (start of line)
    let definition_starts = [
        "def ", "class ", "fn ", "pub fn ", "pub struct ", "pub enum ",
        "impl ", "interface ", "async fn ", "async def ",
        "use ", "import ", "from ", "require ",
        "var ", "let ", "const ", "type ", "trait ",
        "module ", "package ",
        "@",  // decorators
    ];

    // Control flow (start of line)
    let control_starts = [
        "if ", "else", "elif ", "for ", "while ", "match ", "switch ",
        "try ", "catch ", "except ", "finally ", "return ", "raise ",
    ];

    // Check if line starts with a definition or control keyword
    for pat in &definition_starts {
        if line.starts_with(pat) { return true; }
    }
    for pat in &control_starts {
        if line.starts_with(pat) { return true; }
    }

    // Lines ending with : (Python blocks, type annotations)
    if line.ends_with(':') && !line.starts_with('#') { return true; }

    // Lines containing => or -> (type signatures, match arms)
    if line.contains("=>") || line.contains("->") { return true; }

    // Lines with = that look like type/constant definitions (not assignments in expressions)
    if line.contains('=') && !line.contains("==") {
        // Only if it looks like a module-level definition
        let before_eq = line.split('=').next().unwrap_or("");
        if before_eq.trim().chars().all(|c| c.is_alphanumeric() || c == '_' || c == ' ' || c == ':') {
            // Likely a definition like "CONSTANT = value" or "x: int = 5"
            if before_eq.contains(':') || before_eq.chars().next().map_or(false, |c| c.is_uppercase()) {
                return true;
            }
        }
    }

    false
}

/// Caveman compression: remove stop-words, uppercase remaining
fn compress_caveman(text: &str) -> (String, String) {
    let lang = detect_language(text);
    let stops = stop_words::get(&lang);

    let stop_set: std::collections::HashSet<&str> = stops.iter().copied().collect();

    let result = text.split_whitespace()
        .filter(|w| {
            let lower = w.to_lowercase();
            let clean: String = lower.chars().filter(|c| c.is_alphabetic()).collect();
            !stop_set.contains(clean.as_str()) && !clean.is_empty()
        })
        .map(|w| w.to_uppercase())
        .collect::<Vec<_>>()
        .join(" ");

    (result, lang)
}

/// Simple language detection: count stop-word hits per language
fn detect_language(text: &str) -> String {
    let lower = text.to_lowercase();
    let words: std::collections::HashSet<String> = lower
        .split_whitespace()
        .map(|w| w.chars().filter(|c| c.is_alphabetic()).collect::<String>())
        .filter(|w: &String| !w.is_empty())
        .collect();

    let candidates = ["en", "ro", "fr", "de", "es", "it", "pt", "nl"];

    let mut best_lang = "en".to_string();
    let mut best_count = 0;

    for lang in candidates {
        let stops = stop_words::get(lang);
        let hits = words.iter().filter(|w| stops.contains(&w.as_str())).count();
        if hits > best_count {
            best_count = hits;
            best_lang = lang.to_string();
        }
    }

    best_lang
}

/// YAKE keyword extraction
fn extract_keywords_yake(text: &str) -> Vec<yake_rust::ResultItem> {
    let lang = detect_language(text);

    let stop_words = yake_rust::StopWords::predefined(&lang)
        .unwrap_or_else(|| yake_rust::StopWords::predefined("en").unwrap());

    let config = yake_rust::Config::default();
    yake_rust::get_n_best(15, text, &stop_words, &config)
}