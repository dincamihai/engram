# Compression Module: Design Decisions

## What we built

A content compression module (`src/compress.rs`) integrated into engram's store/recall/consolidation pipeline. Three strategies:

| Strategy | Method | Crate | Use case |
|---|---|---|---|
| Caveman | Stop-words removal + uppercase | `stop-words` v0.10 | Prose (default) |
| Semantic | YAKE keyword extraction | `yake-rust` v1.0 | Consolidation summaries |
| Structural | Regex signature extraction | None (built-in) | Code |

Auto-detection: content type (prose vs code) is inferred heuristically. Prose → Caveman, Code → Structural.

## Decisions and rationale

### 1. Why stop-words + YAKE + regex, not LLM-based compression

We tested `qwen3.5:0.8b` via Ollama for compression. Results:

- **37 seconds** for "Say hello" — the model has a thinking mode that generates hundreds of tokens of internal reasoning before answering
- Model output quality: added emojis, questions, didn't follow the compression prompt format
- Cost per call: even local, the latency makes it unusable at `store` time (called for every memory)
- Overkill: removing stop-words and extracting keywords does not require a language model

**Decision**: Zero ML dependencies. All compression is local, deterministic, instant.

### 2. Why not hardcode stop-words per language

Initial approach was to hardcode stop-word lists. The `stop-words` crate already provides curated lists for 60+ languages, maintained by the community. Reimplementing would be:
- Reinventing the wheel
- Only covering languages we manually add
- A maintenance burden

**Decision**: Use `stop-words` crate. Language is auto-detected by counting stop-word hits per language.

### 3. Why not spaCy / NLP for POS tagging

spaCy provides POS tagging and lemmatization (Level 2: "Semantic Compression" in research). But:
- spaCy is Python-only, engram is Rust
- Would require a Python bridge or separate service
- The last 20% of compression (lemmatization: "walking" → "walk") doesn't justify the complexity
- The LLM receiving compressed output handles lemma equivalence natively

**Decision**: Skip lemmatization. "ENGINEERS" and "ENGINEER" are equivalent to any LLM reading the compressed output.

### 4. Why not tree-sitter for code compression

Tree-sitter provides proper AST parsing for 50+ languages. But:
- Each grammar is a separate crate (~1MB each)
- Would need to bundle grammars for common languages or download on demand
- Regex heuristics achieve 80-82% reduction on code, which is sufficient
- Code in memory is an index ("FUNC validate_token(token)"), not a code store. When implementation is needed, you read the file.

**Decision**: Regex heuristics for structural code compression. Good enough for memory indexing.

### 5. Why not nlprule for POS tagging

nlprule was evaluated:
- **Only 3 languages** (EN, DE, ES) — no Romanian
- **Unmaintained** since April 2021, issue "project dead?" open
- **LGPL** license on binary resources
- **7MB** per language for tokenizer binaries
- Purpose is grammar checking, not compression

**Decision**: Not suitable.

### 6. Why not BERT/Candle embeddings for token importance

The approach of using embedding vector norms to filter "important" tokens was evaluated:
- BERT tokenizer produces subword tokens ("engineers" → ["engine", "##ers"]) — cannot filter at word level
- Vector norm ≠ grammatical importance — articles can have high norm in specific contexts
- Requires model download (15MB+) and initialization time (seconds)
- YAKE achieves the same goal (keyword extraction) statistically, without any model

**Decision**: YAKE for keyword extraction, not embeddings.

### 7. Why store both raw and compressed

| Column | Purpose |
|---|---|
| `content` | Original text, never modified. Used for dedup and when full text is needed |
| `content_display` | Compressed version. Used in search results and consolidation |
| `compression_level` | 0=raw, 1=caveman, 2=semantic, 3=structural |

Storing both preserves information. Compression is lossy — you can't reconstruct the original from caveman output. But search results benefit from the compressed version (fewer tokens). Dedup must check raw content, not compressed.

### 8. Why auto-detect content type instead of requiring explicit strategy

Users of engram (both CLI and MCP) shouldn't need to think about compression strategy. The heuristic for code detection (check for `fn `, `def `, `class `, etc. and code-line ratio > 0.3) works reliably in practice. If misdetected, the worst case is:
- Prose detected as code → structural compression strips too much → still readable as a summary
- Code detected as prose → caveman keeps almost everything → minimal loss

### 9. Why Semantic (YAKE) for consolidation, not Caveman

Consolidation merges 3+ faded memories into one entry. The naive approach (concatenate with " | " and truncate to 2000 chars) wastes tokens on repetitive grammar. YAKE extracts the most distinctive keywords across all merged entries, producing a much denser summary.

### 10. Same principle applies to context, not just memory

Compression isn't just for stored memories. The same strategies apply to:
- Old conversation turns sent to LLM (caveman for recent, semantic for old)
- Tool results like file reads (structural for code, caveman for prose)
- RAG results injected into context

This is documented for future implementation as Phase 2 (MCP tool `compress`) and Phase 3 (context proxy).

## Test results from demo

### Prose (Romanian)
```
INPUT:  "Alice a rezolvat problema de timeout din baza de date pe 3 martie"
CAVEMAN: "ALICE REZOLVAT PROBLEMA TIMEOUT BAZA DATE MARTIE"  (74% of original)
YAKE:   "Alice a rezolvat" (0.015), "rezolvat problema" (0.022), "baza de date" (0.022)
```

### Prose (English)
```
INPUT:  "The autonomous cars are driving through the busy streets of San Francisco"
CAVEMAN: "AUTONOMOUS CARS DRIVING BUSY STREETS SAN FRANCISCO"  (71% of original)
YAKE:   "San Francisco" (0.003), "autonomous cars" (0.015), "busy streets" (0.015)
```

### Code (Rust, birch.rs, 2181 chars)
```
STRUCTURAL: "pub struct Config { ... pub fn open(...) ... pub struct Entry {"  (18% of original)
```
82% reduction, preserving signatures.

### Mixed content (compact.md, 4672 chars)
```
Detected as CODE (correct — contains Python code blocks)
STRUCTURAL: 472 chars (10% of original) — function signatures preserved
CAVEMAN: 3650 chars (78%) — almost no reduction on code
```
Confirms: caveman is useless on code, structural is essential.

## Files changed

| File | Change |
|---|---|
| `src/compress.rs` | New — compression module |
| `src/lib.rs` | Added `pub mod compress` |
| `src/birch.rs` | Schema migration, store() auto-compress, recall() prefers display, consolidate uses YAKE |
| `Cargo.toml` | Added `stop-words` v0.10, `yake-rust` v1 |
| `Cargo.lock` | Updated |
| `examples/compress_demo.rs` | Interactive demo |
| `research/context-compression.md` | Research notes (Romanian) |
| `research/compact.md` | Original research (Romanian) |