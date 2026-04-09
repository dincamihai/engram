# Context Compression for LLM Calls

## Decizie finală: Stack-ul de implementare

**Stop-words + YAKE + Regex.** Zero dependențe grele, compilare statică, performanță predictibilă.

| Componentă | Crate | Ce face | Limbi |
|---|---|---|---|
| Caveman (stop-words) | `stop-words` v0.10 | Elimină noise gramatical, uppercase | 60+ |
| Semantic (keywords) | `yake-rust` v1.0 | Extrage keyword-uri după importanță | Multi |
| Structural (code) | Regex hardcoded | Extrage semnături din cod | Agnostic |

Lemmatizarea ("walking" → "walk") e scopul ultimilor 20% din compresie. Nu merită complexitatea. LLM-ul care primește output-ul comprimat va înțelege că "engineers" și "engineer" sunt același lucru.

---

## De ce context compression > memory compression

| Ce comprimi | Frecvență | Câștig per apel |
|---|---|---|
| Rezultate engram search | la fiecare sesiune | mic (10-20 tokeni) |
| Tool results (Read, Grep) | zeci de ori pe sesiune | mare (100-500 tokeni) |
| Conversation history (turnuri vechi) | permanent | foarte mare (1000+ tokeni) |
| RAG / documente injectate | la fiecare query relevant | enorm (1000-5000 tokeni) |

Pe memorie câștigi spațiu de stocare. Pe context câștigi **tokeni activi** — bani și capacitate pe fiecare apel.

---

## Strategie adaptivă: nu comprimi uniform

### Pe context (conversație cu LLM)

```
turn curent (instructions, query)    → RAW             (0% reducere)
turn anterior                        → caveman light   (~40% reducere)
turnuri medii (3-6 înapoi)           → semantic/POS    (~70% reducere)
turnuri vechi (7+ înapoi)            → triplet SPO    (~90% reducere)
tool results mari (file reads)        → structural     (~80% reducere)
rezultate search/RAG                  → triplet SPO   (~90% reducere)
```

### Pe memorie (engram storage)

Același principiu, bazat pe retention score (Weibull decay):

```
retention > 0.7  → RAW          memorie fresh, detaliată
retention > 0.4  → caveman      noise gramatical eliminat
retention > 0.15 → semantic     doar faptele esențiale
retention ≤ 0.15 → triplet      fapte pure → consolidation merge
```

### Pe cod (ingest/file storage)

```
detect code → structural (semnături, definiții, control flow)
detect prose → caveman + YAKE
```

Codul în memorie = index, nu depozit. "FUNC validate_token(token)" nu "47 de linii de implementare". Când ai nevoie de implementare, cauți fișierul.

---

## Rezultate din demo

### Proză română
```
INPUT:  "Alice a rezolvat problema de timeout din baza de date pe 3 martie"
CAVEMAN: "ALICE REZOLVAT PROBLEMA TIMEOUT BAZA DATE MARTIE"  (74%)
YAKE:   "Alice a rezolvat" (0.015), "rezolvat problema" (0.022), "baza de date" (0.022)
```

### Proză engleză
```
INPUT:  "The autonomous cars are driving through the busy streets of San Francisco"
CAVEMAN: "AUTONOMOUS CARS DRIVING BUSY STREETS SAN FRANCISCO"  (71%)
YAKE:   "San Francisco" (0.003), "autonomous cars" (0.015), "busy streets" (0.015)
```

### Cod Rust (birch.rs, 2181 chars)
```
STRUCTURAL: "pub struct Config { ... pub fn open(...) ... pub struct Entry {"  (18%)
```
82% reducere, păstrând semnăturile.

### Cod Python (compact.md, 4672 chars) — detectat ca CODE
```
STRUCTURAL: "def caveman_basic(text, lang_stops): ... def compress_semantic(text): ..."  (10%)
```
90% reducere.

### Observații
- Caveman pe cod e inutil — codul n-are stop-words
- YAKE pe cod produce noise ("pub", "String", "usize") — nu e pentru cod
- Structural e excelent pe cod — 82-90% reducere cu informație utilă păstrată
- Detecția automată CODE vs PROSE funcționează corect

---

## Ce NU folosim și de ce

| Abordare | De ce nu |
|---|---|
| LLM local (qwen3.5:0.8b) | 37 secunde pentru "hello", thinking mode, overkill pentru stop-words |
| nlprule | 3 limbi, nementinut din 2021, LGPL |
| Tree-sitter | Corect pentru cod, dar adaugă 1MB+ per limbaj. Regex face 80% din job la zero cost |
| BERT/Candle embeddings | Norma vectorului ≠ importanță gramaticală. Subword tokens sparg cuvintele |
| spaCy | Python, nu Rust. Ar fi bridge overkill |
| Lemmatizare | Ultimele 20%. LLM-ul face asta implicit când primește output-ul |

---

## Arhitectura engram cu compression

### Modul `compress.rs`

```rust
pub enum Strategy {
    Caveman,       // stop-words removal + uppercase (prose)
    Semantic,      // YAKE keyword extraction (prose, consolidation)
    Structural,    // regex signatures (code)
}

pub fn compress(text: &str, strategy: Strategy) -> String { ... }
pub fn detect_content_type(text: &str) -> ContentType { ... }  // CODE vs PROSE
```

### Flow

```
store("Alice fixed the database timeout")
    │
    ├── detect: PROSE
    ├── compress with Caveman → "ALICE FIXED DATABASE TIMEOUT"
    ├── embed with embeddinggemma → vector
    │
    └── store raw + compressed in SQLite

store("def validate_token(token):\n    ...47 lines...")
    │
    ├── detect: CODE
    ├── compress with Structural → "FUNC validate_token(token)"
    ├── embed with embeddinggemma → vector
    │
    └── store raw + compressed in SQLite

rebalance:
    │
    ├── retention < 0.4 → re-compress with Semantic (YAKE keywords)
    ├── retention < 0.15 → consolidate cluster → merged entry
    │
    └── content_display updated, content_raw intact

search:
    │
    └── return content_display (compressed) → denser results in fewer tokens
```

### Schema SQLite

```
entries table:
  content_raw:        TEXT   — original, never modified
  content_display:    TEXT   — compressed version
  compression_level: INTEGER — 0=raw, 1=caveman, 2=semantic, 3=structural
  embedding:         BLOB   — from content_raw always
```

### Config

```rust
pub ollama_base_url: String,       // default: localhost:11434
pub ollama_embed_model: String,    // default: embeddinggemma
// No compress_model needed — all local, no LLM
```

---

## Arhitectura: trei faze de adoptare

### Phase 1: Compression în engram (store + rebalance + search)

- La `store`: detectează tipul, comprimă, stochează raw + display
- La `rebalance`: re-comprima intrări cu retention scăzut
- La `search`: returnează content_display
- La `consolidation`: merge pe versiunile comprimate

### Phase 2: MCP tool `compress` + hooks

- Agentul apelează explicit `engram_compress(text, strategy)` pe orice text
- Hook PreCompact: reminder să comprime înainte de compaction
- Hook PostToolUse: comprimă automat tool results mari

### Phase 3: Context proxy (complet automat)

```
Claude Code → localhost:8080 (compressor proxy) → api.anthropic.com
```

---

## Reguli de aur

1. **Raw la store, compressed la display** — originalul intact, display-ul dens
2. **Stop-words + YAKE + Regex** — zero dependențe grele, compilare statică, 60+ limbi
3. **Lemmă nu facem** — LLM-ul face asta implicit. "engineers" și "engineer" sunt același lucru pentru el
4. **compression_level derivat din retention** — e consecința naturală a decay-ului
5. **Detect code vs prose** — structural pe cod, caveman pe proză
6. **Nu comprima instrucțiunile curente** — doar history și tool results
7. **Fallback la raw** — dacă compresia eșuează, nu crash