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

Four source files:

| File | Purpose |
|---|---|
| `birch.rs` | BIRCH CF-tree + SQLite persistence. Clustering, decay, consolidation, activation. |
| `embed.rs` | Ollama embeddings + cosine similarity. |
| `activation.rs` | Aha-moment detector using archetype embeddings. |
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

Requires [Ollama](https://ollama.ai) with an embedding model:

```bash
ollama pull embeddinggemma
```

One-command install for Claude Code (downloads the binary, registers the MCP server, configures hooks):

```bash
curl -fsSL https://raw.githubusercontent.com/dincamihai/engram/main/install.sh | bash
```

Or a specific version:

```bash
curl -fsSL https://raw.githubusercontent.com/dincamihai/engram/main/install.sh | bash -s v0.1.0
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
engram rebuild                          # re-cluster all entries from scratch
engram ingest ~/path/to/files
engram topics
engram stats
engram rebalance
```

### Environment

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_EMBED_MODEL` | `embeddinggemma` | Embedding model name |

## Design decisions

**Why BIRCH over BERTopic/HDBSCAN?** BIRCH handles both batch and incremental inserts with the same algorithm. No separate batch-vs-online modes. One algorithm, consistent behavior.

**Why Weibull decay over exponential?** Weibull with k=0.5 matches human forgetting curves — steep initial drop, then stabilization. Exponential decay is too uniform.

**Why bit-shift for temporal decay?** It's the simplest possible precision degradation. Right-shift an integer and information is irreversibly lost — just like memory. Same-period events become identical after enough shifting, enabling natural temporal merging.

**Why no manual organization tools?** The agent's job is to remember and recall, not to curate. Curation is a human abstraction. The embedding space and clustering tree handle organization better than any tag system could.

**Why gap detection over fixed beam width?** A fixed beam explores the same number of branches regardless of query specificity. Gap detection finds the natural elbow in the similarity curve — specific queries narrow to 1-2 clusters, broad queries explore many. No parameter to tune; the data decides.

**Why neighborhood recall over flat ranked results?** Flat results are a database query. Neighborhood recall returns the cluster context — sibling memories the tree grouped together, even if they weren't in the top-K by similarity. This gives associative context, like how remembering one thing naturally brings related things to mind.

## Emergent behaviors

These behaviors aren't coded as features — they emerge from the interaction between the agent, the embedding space, and the BIRCH tree.

**Synthesis through bridging.** When the agent notices a connection between two distant topics and stores an insight ("Alice is the go-to for incident response"), that memory's embedding sits between the "people" cluster and the "procedures" cluster. The nearest cluster's centroid shifts toward the other. Over time, related clusters drift closer in vector space and may merge. The bridge is the memory itself — no explicit linking mechanism needed.

**Self-correcting associations.** If a bridging memory was wrong or irrelevant, nobody searches for it. It fades via Weibull decay. The clusters drift back apart. Bad associations dissolve on their own.

**Constructive interference.** When a cluster accumulates many entries about the same theme, the centroid becomes a purer representation of the shared concept than any individual entry. Abstract queries match the centroid's neighborhood more strongly, naturally surfacing the right cluster. The synthesis is implicit in the geometry.

**Schema formation through consolidation.** When individual episode memories fade ("Alice fixed timeout on March 3", "Alice debugged API connection issue"), they consolidate into a compressed summary. The summary is the schema — general knowledge extracted from specific episodes. The cluster centroid preserves the abstract meaning even as details are lost.

**Temporal merging.** As temporal precision degrades via bit-shifting, memories from nearby time periods become identical in their temporal dimension. "March 2025" and "April 2025" both become "2025" — enabling the system to treat same-era memories as contemporaneous, just as human memory does.

**The agent is the intelligence.** The engine provides two tools: `store` and `search`. Everything else — synthesis, linking, contradiction detection, importance judgments — is the agent's responsibility. The engine handles physics (clustering, decay, activation). The agent handles meaning. This separation keeps the engine simple and the agent powerful.

## Claude Code Hooks

Engram integrates with Claude Code's hook system to provide autoMemory-like behavior — automatic recall at session start and memory nudges during work.

### SessionStart — Memory Injection

A `SessionStart` hook searches engram using the current month and date as a query bias, returning the most relevant recent memories and injecting them as context at the start of every session.

**Trick:** Entries are prefixed with timestamps (e.g., `2026-04-08:`). Passing a date string as part of the search query biases semantic similarity toward memories from that time period. Use recent dates for current context, older dates for distant recall.

```bash
#!/bin/bash
TODAY=$(date +%Y-%m-%d)
MONTH=$(date +"%B %Y")
QUERY="$MONTH recent work context $TODAY"
RESULT=$(engram search "$QUERY" --limit 10 2>/dev/null | head -100)
if [ -n "$RESULT" ]; then
  ESCAPED=$(echo "$RESULT" | python3 -c "import sys,json; s=json.dumps(sys.stdin.read()); print(s[1:-1])")
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"== Engram Memory (auto-loaded) ==\n${ESCAPED}\n== End Engram Memory ==\"}}"
fi
```

### PreCompact — Save Before Compaction

A `PreCompact` hook fires before Claude Code compacts the conversation context. It injects a reminder to review the conversation and save important information (decisions, outcomes, patterns) to engram before context is lost.

```bash
#!/bin/bash
echo '{"hookSpecificOutput":{"hookEventName":"PreCompact","additionalContext":"Context is about to be compacted. Save any important information to engram memory before proceeding."}}'
```

### PostToolUse — Work Completion Nudges

PostToolUse hooks nudge Claude to consider saving memories after significant work. For example, matching Bash commands containing `git push`, `git commit`, or CI/CD tool invocations:

```json
{
  "matcher": "Bash",
  "hooks": [{
    "type": "command",
    "command": "jq -r '.tool_input.command' | grep -qE 'git (push|commit)' && echo '{\"hookSpecificOutput\":{\"hookEventName\":\"PostToolUse\",\"additionalContext\":\"Significant work detected. Consider saving memories via engram_store.\"}}' || true"
  }]
}
```

Similarly, an Edit matcher can fire after editing files in specific directories (e.g., a notes vault).

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

Built with [Ollama](https://ollama.ai) for embeddings and [rusqlite](https://github.com/rusqlite/rusqlite) for persistence.
