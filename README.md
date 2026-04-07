# engram

Self-organizing memory for AI agents. Two tools. Everything else is physics.

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
                  cosine similarity vs all entries
                              │
                  apply Weibull decay + access boost
                              │
                  read fuzzy date from bit-shifted epoch
                              │
                      return top-k results
                              │
                  bump access_count on returned entries
```

```
                        rebalance
                              │
              ┌───────────────┼───────────────┐
              │               │               │
      degrade temporal   consolidate      merge small
      epochs (bit-shift) faded memories   clusters
              │               │               │
              └───────────────┼───────────────┘
                              │
                      re-label all leaves
```

## Running

Requires [Ollama](https://ollama.ai) with an embedding model:

```bash
ollama pull embeddinggemma
```

### As MCP server (for Claude Code)

```bash
engram serve
```

### CLI

```bash
engram store "Alice leads the database migration"
engram search "who works on infrastructure"
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

## Emergent behaviors

These behaviors aren't coded as features — they emerge from the interaction between the agent, the embedding space, and the BIRCH tree.

**Synthesis through bridging.** When the agent notices a connection between two distant topics and stores an insight ("Alice is the go-to for incident response"), that memory's embedding sits between the "people" cluster and the "procedures" cluster. The nearest cluster's centroid shifts toward the other. Over time, related clusters drift closer in vector space and may merge. The bridge is the memory itself — no explicit linking mechanism needed.

**Self-correcting associations.** If a bridging memory was wrong or irrelevant, nobody searches for it. It fades via Weibull decay. The clusters drift back apart. Bad associations dissolve on their own.

**Constructive interference.** When a cluster accumulates many entries about the same theme, the centroid becomes a purer representation of the shared concept than any individual entry. Abstract queries match the centroid's neighborhood more strongly, naturally surfacing the right cluster. The synthesis is implicit in the geometry.

**Schema formation through consolidation.** When individual episode memories fade ("Alice fixed timeout on March 3", "Alice debugged API connection issue"), they consolidate into a compressed summary. The summary is the schema — general knowledge extracted from specific episodes. The cluster centroid preserves the abstract meaning even as details are lost.

**Temporal merging.** As temporal precision degrades via bit-shifting, memories from nearby time periods become identical in their temporal dimension. "March 2025" and "April 2025" both become "2025" — enabling the system to treat same-era memories as contemporaneous, just as human memory does.

**The agent is the intelligence.** The engine provides two tools: `store` and `search`. Everything else — synthesis, linking, contradiction detection, importance judgments — is the agent's responsibility. The engine handles physics (clustering, decay, activation). The agent handles meaning. This separation keeps the engine simple and the agent powerful.

## Credits

Built with [Ollama](https://ollama.ai) for embeddings and [rusqlite](https://github.com/rusqlite/rusqlite) for persistence.
