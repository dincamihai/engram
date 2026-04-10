# engram

Self-organizing memory for AI agents. Two tools. Everything else emerges.

## Interface

```
store(content)   → embed → cluster → done
search(query)    → embed → nearest → results
```

That's it.

## Principles

**No manual organization.** There are no folders, tags, categories, or labels to manage. Content goes in; the BIRCH clustering tree decides where it belongs. Topics emerge from the data, not from configuration.

**Organic decay.** Memories fade over time following a Weibull curve — fast initial drop, then survivors stabilize. Memories that get searched are rehearsed and resist decay. Unused memories fade quietly.

**Temporal bit-shift.** Each memory stores when it happened as epoch seconds. Over time, the precision degrades via bit-shifting — you remember "yesterday at 3pm" but only "sometime in 2024." Accessing a memory keeps its content alive but doesn't protect temporal precision. You remember your old password, not when you set it.

**Memory consolidation.** When a cluster of memories all fade below a retention threshold, they merge into a single condensed entry. Individual episodes compress into general knowledge. The consolidated memory gets a fresh start.

**Spreading activation.** When content is stored that resembles an "aha moment" (detected via archetype embeddings), nearby memories are automatically rehearsed — keeping related knowledge alive. Occasionally a random distant memory is activated too, enabling serendipitous connections.

**Merge on insert.** When new content makes two clusters converge (their centroids become similar enough), they merge automatically. The tree breathes — splitting when diverse, merging when content converges.

**Dedup through clustering.** Identical content routed to the same cluster is detected and skipped. No extra indexes or hashing — the tree structure itself prevents duplicates.

## Architecture

| File | Purpose |
|---|---|
| `birch.rs` | BIRCH CF-tree + SQLite persistence. Clustering, decay, consolidation, activation. |
| `embed.rs` | fastembed (BGE-small-en-v1.5, 384-dim ONNX) + cosine similarity. No external server. |
| `extract.rs` | BERT NLI classification (DistilBERT-MNLI) + extractive summarization. Decides what's worth remembering. |
| `activation.rs` | Aha-moment detector using archetype embeddings. |
| `viz.rs` | Defrag-style terminal visualization — one row per cluster, color = freshness, flicker = new. |
| `compress.rs` | YAKE keyword extraction for memory compression. |
| `mcp.rs` | MCP server — JSON-RPC over stdin/stdout. |

One database: `~/.engram/engram.db` (SQLite).

## How it works

```
           store("Alice fixed the database timeout")
                              │
                         embed content
                              │
                     aha detector checks ──→ spreading activation
                              │                    (if triggered)
                     BIRCH finds nearest leaf
                              │
                    ┌─── similar enough? ───┐
                    │                       │
                yes: join cluster     no: new cluster
                    │                       │
              update centroid          split if full
                    │
              dedup check ──→ skip if identical
                    │
                  store in SQLite with epoch
                    │
              merge siblings if converged
```

```
           search("who works on infrastructure")
                              │
                         embed query
                              │
               tree-guided beam search:
               compare query vs node centroids,
               descend into promising branches,
               gap detection prunes at largest
               similarity drop-off
                              │
               score entries in candidate leaves:
               cosine similarity + Weibull decay + access boost
                              │
               read fuzzy date from bit-shifted epoch
                              │
                      return top-k results
                              │
               bump access_count on returned entries
```

**Neighborhood recall.** The MCP `search` tool returns cluster neighborhoods instead of flat ranked lists. It finds the best-matching entries, walks up to their parent node, and returns sibling clusters — giving associative context beyond the direct matches. The response is a single text blob with cluster labels, not a structured array of individual entries.

```
                        rebalance
                              │
       ┌──────────┬───────────┼───────────┬──────────┐
       │          │           │           │          │
   prune ghost  degrade    consolidate  merge     re-label
   empty nodes  temporal   faded        small     all
                epochs     memories     clusters  leaves
```

## Install

No external dependencies — embeddings and classification run in-process via ONNX Runtime. Models are downloaded from HuggingFace on first use (~33MB for embeddings, ~250MB for BERT classifier).

```bash
curl -fsSL https://raw.githubusercontent.com/dincamihai/engram/main/install.sh | bash
```

## Running

### As MCP server (for Claude Code)

```bash
engram serve
```

### CLI

```bash
engram store "Alice leads the database migration"
engram search "who works on infrastructure"
engram forget 42                        # delete entry by ID
engram forget --source "%SKILL.md"      # bulk delete by source pattern
engram rebuild                          # re-embed all entries (for model migration)
engram rebalance                        # consolidate, merge, prune, re-label
engram ingest ~/path/to/files
engram extract "some text"              # classify relevance with BERT NLI
engram extract --store "some text"      # classify + auto-store if relevant
engram topics
engram stats
engram tree                             # text tree view of all entries
engram viz                              # defrag-style terminal visualization
```

## Design decisions

**Why BIRCH over BERTopic/HDBSCAN?** BIRCH handles both batch and incremental inserts with the same algorithm. No separate batch-vs-online modes. One algorithm, consistent behavior.

**Why Weibull decay over exponential?** Weibull with k=0.5 matches human forgetting curves — steep initial drop, then stabilization. Exponential decay is too uniform.

**Why bit-shift for temporal decay?** It's the simplest possible precision degradation. Right-shift an integer and information is irreversibly lost — just like memory. Same-period events become identical after enough shifting, enabling natural temporal merging.

**Why no manual organization tools?** The agent's job is to remember and recall, not to curate. Curation is a human abstraction. The embedding space and clustering tree handle organization better than any tag system could.

**Why gap detection over fixed beam width?** A fixed beam explores the same number of branches regardless of query specificity. Gap detection finds the natural elbow in the similarity curve — specific queries narrow to 1-2 clusters, broad queries explore many. No parameter to tune; the data decides.

**Why neighborhood recall over flat ranked results?** Flat results are a database query. Neighborhood recall returns the cluster context — sibling memories the tree grouped together, even if they weren't in the top-K by similarity. This gives associative context, like how remembering one thing naturally brings related things to mind.

**Why fastembed over Ollama?** In-process ONNX inference is ~10x faster than HTTP roundtrips to Ollama and removes the external server dependency. BGE-small-en-v1.5 (384-dim) is smaller than embeddinggemma (768-dim) but quality is comparable for short text. `rebuild` re-embeds all entries for model migration.

**Why binary relevance over multi-class?** BERT NLI decides one thing: "is this worth remembering?" BIRCH handles categorization through embedding similarity — it already does this better than any label system. Splitting the responsibilities keeps both simple. Short hypotheses work best with DistilBERT ("This describes a person or team." scores 0.96, while "This text describes relationships between people, teams, or components." scores 0.006).

## Extract Pipeline

The `extract` command uses a two-stage pipeline for automatic memory creation:

```
text input
    │
    ▼
extractive summarization (fastembed)
    │ split into sentences
    │ embed each with BGE-small
    │ pick top 2 by centroid proximity
    ▼
binary relevance (BERT NLI)
    │ DistilBERT-MNLI via ONNX Runtime
    │ test against 5 hypotheses
    │ max entailment score > threshold?
    ▼
    yes → engram store (BIRCH clusters it)
    no  → discard
```

The 5 relevance hypotheses:
- "This contains useful technical information."
- "This describes a person or team."
- "This is about a customer or problem."
- "This describes software or infrastructure."
- "This is about a work process or decision."

## Visualization

`engram viz` shows a defrag-style terminal view — one row per cluster:

```
   saas-role.. ████████████████████████
  production.. ██████████████████████████████████████
  team-colim.. ████████████
     memory-.. ██████████
```

Visual encoding:
- **Color** = freshness (warm gold → cool steel blue)
- **Brightness** = proven (accessed) vs dim (unproven)
- **Flicker** = brand new entry (< 12 hours, never accessed) — speed decays with age
- **Blue-tinted** = consolidated (merged from multiple entries)

## Emergent behaviors

These behaviors aren't coded as features — they emerge from the interaction between the agent, the embedding space, and the BIRCH tree.

**Synthesis through bridging.** When the agent notices a connection between two distant topics and stores an insight ("Alice is the go-to for incident response"), that memory's embedding sits between the "people" cluster and the "procedures" cluster. The nearest cluster's centroid shifts toward the other. Over time, related clusters drift closer in vector space and may merge. The bridge is the memory itself — no explicit linking mechanism needed.

**Self-correcting associations.** If a bridging memory was wrong or irrelevant, nobody searches for it. It fades via Weibull decay. The clusters drift back apart. Bad associations dissolve on their own.

**Constructive interference.** When a cluster accumulates many entries about the same theme, the centroid becomes a purer representation of the shared concept than any individual entry. Abstract queries match the centroid's neighborhood more strongly, naturally surfacing the right cluster. The synthesis is implicit in the geometry.

**Schema formation through consolidation.** When individual episode memories fade ("Alice fixed timeout on March 3", "Alice debugged API connection issue"), they consolidate into a compressed summary. The summary is the schema — general knowledge extracted from specific episodes. The cluster centroid preserves the abstract meaning even as details are lost.

**Temporal merging.** As temporal precision degrades via bit-shifting, memories from nearby time periods become identical in their temporal dimension. "March 2025" and "April 2025" both become "2025" — enabling the system to treat same-era memories as contemporaneous, just as human memory does.

**The agent is the intelligence.** The engine provides two tools: `store` and `search`. Everything else — synthesis, linking, contradiction detection, importance judgments — is the agent's responsibility. The engine handles physics (clustering, decay, activation). The agent handles meaning. This separation keeps the engine simple and the agent powerful.

## Claude Code Hooks

Engram integrates with Claude Code's hook system for automatic memory injection and storage — the agent doesn't need to explicitly call engram tools, memories just appear and get saved.

### SessionStart — Memory Injection

Injects relevant memories at the start of every session using the current date and working directory as query bias. Results are already compressed (caveman for prose, structural for code) for maximum information density.

```bash
#!/bin/bash
WORKDIR=$(basename "$PWD")
TODAY=$(date +%Y-%m-%d)
MONTH=$(date +"%B %Y")
QUERY="$MONTH recent work context $TODAY $WORKDIR"
RESULT=$(engram search "$QUERY" --limit 10 2>/dev/null | head -120)
if [ -n "$RESULT" ]; then
  ESCAPED=$(echo "$RESULT" | python3 -c "import sys,json; s=json.dumps(sys.stdin.read()); print(s[1:-1])")
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"== Engram Memory (auto-loaded) ==\n${ESCAPED}\n== End Engram Memory ==\"}}"
fi
```

### PostToolUse — Auto-extract with BERT

**Bash hook** (async) — on git commit/push, Jira commands:
- Extracts commit message or command context
- Runs `engram extract --store` — BERT classifies relevance
- Stores automatically if relevant

**Edit hook** (async) — on vault note edits:
- Extracts the new content
- Runs `engram extract --store` — BERT classifies
- Stores if relevant

**Read hook** (sync) — on vault/repo file reads:
- Injects an LLM suggestion to store if useful
- Does NOT run extract pipeline (too slow per file read)

### PreCompact — Save Before Compaction

Stores the conversation summary before compaction and approves the compaction. No information is lost — the summary is saved to engram.

```bash
#!/bin/bash
SUMMARY=$(cat)
if [ -n "$SUMMARY" ]; then
    engram store "$SUMMARY" --source "claude-code-compact" 2>/dev/null &
fi
echo '{"decision":"approve"}'
```

### Stop — Session End

Stores the final session context on conversation stop.

### Hook Configuration

```json
{
  "hooks": {
    "SessionStart": [{ "hooks": [{ "type": "command", "command": "engram-session-start.sh", "timeout": 15 }] }],
    "PreCompact": [{ "hooks": [{ "type": "command", "command": "engram-pre-compact.sh" }] }],
    "PostToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "..." }] },
      { "matcher": "Edit", "hooks": [{ "type": "command", "command": "..." }] }
    ]
  }
}
```

## Credits

Built with [fastembed](https://github.com/Anush008/fastembed-rs) for embeddings, [ONNX Runtime](https://onnxruntime.ai/) for BERT inference, [ratatui](https://ratatui.rs/) for terminal UI, and [rusqlite](https://github.com/rusqlite/rusqlite) for persistence.
