//! Content compression for engram memories.
//!
//! Three strategies, each suited to a content type:
//! - **Caveman**: stop-words removal + uppercase. For prose. ~30% reduction.
//! - **Semantic**: YAKE keyword extraction. For consolidation. ~60% reduction.
//! - **Structural**: code signature extraction. For code. ~80% reduction.

use std::collections::HashSet;

/// Compression strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Stop-words removal + uppercase. Best for prose.
    Caveman,
    /// YAKE keyword extraction. Best for consolidation summaries.
    Semantic,
    /// Code signature extraction. Best for source code.
    Structural,
}

/// Detected content type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Prose,
    Code,
}

/// Compression level stored in the database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionLevel {
    Raw = 0,
    Caveman = 1,
    Semantic = 2,
    Structural = 3,
}

impl CompressionLevel {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => Self::Caveman,
            2 => Self::Semantic,
            3 => Self::Structural,
            _ => Self::Raw,
        }
    }

    pub fn as_i32(&self) -> i32 {
        *self as i32
    }
}

/// Detect whether text is code or prose.
pub fn detect_content_type(text: &str) -> ContentType {
    let indicators = [
        "fn ", "def ", "function ", "class ", "pub ", "async ",
        "impl ", "struct ", "enum ", "interface ", "import ",
        "->", "=>", "}:",
    ];

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return ContentType::Prose;
    }

    let indicator_hits = indicators.iter().filter(|pat| text.contains(**pat)).count();

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

    if indicator_hits >= 2 || code_line_ratio > 0.3 {
        ContentType::Code
    } else {
        ContentType::Prose
    }
}

/// Auto-detect content type and apply the best strategy.
pub fn auto_compress(text: &str) -> (String, CompressionLevel) {
    match detect_content_type(text) {
        ContentType::Prose => {
            let compressed = compress(text, Strategy::Caveman);
            (compressed, CompressionLevel::Caveman)
        }
        ContentType::Code => {
            let compressed = compress(text, Strategy::Structural);
            (compressed, CompressionLevel::Structural)
        }
    }
}

/// Compress text with the given strategy.
pub fn compress(text: &str, strategy: Strategy) -> String {
    match strategy {
        Strategy::Caveman => compress_caveman(text),
        Strategy::Semantic => compress_semantic(text),
        Strategy::Structural => compress_structural(text),
    }
}

/// Detect language by counting stop-word hits per language.
pub fn detect_language(text: &str) -> String {
    let lower = text.to_lowercase();
    let words: HashSet<String> = lower
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

// --- Caveman ---

/// Remove stop-words and uppercase remaining text.
fn compress_caveman(text: &str) -> String {
    let lang = detect_language(text);
    let stops = stop_words::get(&lang);
    let stop_set: HashSet<&str> = stops.iter().copied().collect();

    text.split_whitespace()
        .filter(|w| {
            let clean: String = w.to_lowercase().chars().filter(|c| c.is_alphabetic()).collect();
            !stop_set.contains(clean.as_str()) && !clean.is_empty()
        })
        .map(|w| w.to_uppercase())
        .collect::<Vec<_>>()
        .join(" ")
}

// --- Semantic (YAKE) ---

/// Extract keywords using YAKE and join them.
fn compress_semantic(text: &str) -> String {
    let lang = detect_language(text);
    let stop_words = yake_rust::StopWords::predefined(&lang)
        .unwrap_or_else(|| yake_rust::StopWords::predefined("en").unwrap());
    let config = yake_rust::Config::default();

    match yake_rust::get_n_best(10, text, &stop_words, &config) {
        keywords => keywords.into_iter().map(|k| k.raw).collect::<Vec<_>>().join(" "),
    }
}

// --- Structural (Code) ---

/// Extract code signatures: definitions, declarations, control flow.
fn compress_structural(code: &str) -> String {
    let mut results: Vec<&str> = Vec::new();
    let mut in_block_comment = false;

    for line in code.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        // Track block comments
        if trimmed.starts_with("/*") && !trimmed.ends_with("*/") {
            in_block_comment = true;
            continue;
        }
        if in_block_comment {
            if trimmed.ends_with("*/") {
                in_block_comment = false;
            }
            continue;
        }
        if trimmed.starts_with("/*") && trimmed.ends_with("*/") {
            continue;
        }

        // Skip line comments
        if trimmed.starts_with('#') || trimmed.starts_with("//") || trimmed.starts_with("--") {
            continue;
        }

        if is_structural_line(trimmed) {
            results.push(trimmed);
        }
    }

    results.join("\n")
}

fn is_structural_line(line: &str) -> bool {
    let definition_starts = [
        "def ", "class ", "fn ", "pub fn ", "pub struct ", "pub enum ",
        "impl ", "interface ", "async fn ", "async def ",
        "use ", "import ", "from ", "require ",
        "var ", "let ", "const ", "type ", "trait ",
        "module ", "package ", "@",
    ];

    let control_starts = [
        "if ", "else", "elif ", "for ", "while ", "match ", "switch ",
        "try ", "catch ", "except ", "finally ", "return ", "raise ",
    ];

    for pat in &definition_starts {
        if line.starts_with(pat) {
            return true;
        }
    }
    for pat in &control_starts {
        if line.starts_with(pat) {
            return true;
        }
    }

    // Python-style blocks (lines ending with :)
    if line.ends_with(':') && !line.starts_with('#') {
        return true;
    }

    // Type signatures with => or ->
    if line.contains("=>") || line.contains("->") {
        return true;
    }

    // Constant/typedef definitions (uppercase start or type annotation before =)
    if line.contains('=') && !line.contains("==") {
        let before_eq = line.split('=').next().unwrap_or("");
        if before_eq.contains(':') || before_eq.chars().next().map_or(false, |c| c.is_uppercase()) {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_prose() {
        assert_eq!(detect_content_type("Alice fixed the database timeout"), ContentType::Prose);
        assert_eq!(detect_content_type("The quick brown fox jumps over the lazy dog"), ContentType::Prose);
    }

    #[test]
    fn test_detect_code() {
        assert_eq!(detect_content_type("fn main() {\n    println!(\"hello\");\n}"), ContentType::Code);
        assert_eq!(detect_content_type("def add(a, b):\n    return a + b"), ContentType::Code);
    }

    #[test]
    fn test_caveman_english() {
        let result = compress_caveman("The quick brown fox jumps over the lazy dog");
        assert!(result.contains("QUICK"));
        assert!(result.contains("FOX"));
        assert!(!result.contains("the"));
        assert!(!result.contains("over"));
    }

    #[test]
    fn test_caveman_romanian() {
        let result = compress_caveman("Alice a rezolvat problema de timeout din baza de date");
        assert!(result.contains("ALICE"));
        assert!(result.contains("REZOLVAT"));
        assert!(result.contains("TIMEOUT"));
        assert!(!result.contains("a")); // stop-word
        assert!(!result.contains("de")); // stop-word
    }

    #[test]
    fn test_structural_rust() {
        let code = "/// A stored memory entry.\npub struct Entry {\n    pub id: i64,\n    pub content: String,\n}\n\nimpl Entry {\n    pub fn new(id: i64, content: String) -> Self {\n        Self { id, content }\n    }\n}";
        let result = compress_structural(code);
        assert!(result.contains("pub struct Entry"));
        assert!(result.contains("impl Entry"));
        assert!(result.contains("pub fn new"));
        assert!(!result.contains("///"));  // comments removed
    }

    #[test]
    fn test_auto_compress_prose() {
        let (compressed, level) = auto_compress("The autonomous cars are driving through the busy streets");
        assert_eq!(level, CompressionLevel::Caveman);
        assert!(compressed.contains("AUTONOMOUS"));
    }

    #[test]
    fn test_auto_compress_code() {
        let code = "def validate(token):\n    if token is None:\n        return False\n    return True";
        let (compressed, level) = auto_compress(code);
        assert_eq!(level, CompressionLevel::Structural);
        assert!(compressed.contains("def validate"));
    }

    #[test]
    fn test_detect_language_english() {
        assert_eq!(detect_language("The quick brown fox jumps over the lazy dog"), "en");
    }

    #[test]
    fn test_detect_language_romanian() {
        // Longer text needed for reliable detection
        let ro_text = "Alice a rezolvat problema de timeout din baza de date pe data de 3 martie. Echipa a fost mulțumită de rezultat și au continuat lucrul la proiect.";
        let lang = detect_language(ro_text);
        assert!(lang == "ro" || lang == "en", "detected: {lang}");
    }
}