//! BIRCH CF-tree with SQLite persistence.
//!
//! A self-organizing hierarchical clustering tree for AI agent memory.
//! Content is stored with embeddings; the tree structure emerges from the data.

use rusqlite::{params, Connection};

use crate::embed::cosine_similarity;

/// Configuration for the BIRCH tree.
pub struct Config {
    /// Cosine similarity threshold: above this → join existing cluster.
    pub threshold: f32,
    /// Max entries per leaf before splitting.
    pub leaf_capacity: usize,
    /// Max children per internal node before splitting.
    pub branch_factor: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            threshold: 0.65,
            leaf_capacity: 50,
            branch_factor: 10,
        }
    }
}

/// A stored memory entry.
#[derive(Debug, Clone)]
pub struct Entry {
    pub id: i64,
    pub content: String,
    pub content_display: Option<String>,
    pub compression_level: i32,
    pub source: Option<String>,
    pub created_at: String,
    pub when: String,
    pub similarity: f32,
}

/// A cluster node in the tree.
#[derive(Debug, Clone)]
pub struct Node {
    pub parent_id: Option<i64>,
    pub centroid: Vec<f32>,
    pub count: i64,
    pub label: String,
}

/// Summary of a topic (cluster) for browsing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Topic {
    pub id: i64,
    pub label: String,
    pub count: i64,
    pub depth: i32,
    pub children: Vec<Topic>,
}

/// The BIRCH tree.
pub struct Tree {
    conn: Connection,
    config: Config,
    dimension: usize,
}

impl Tree {
    /// Open or create a BIRCH tree backed by SQLite.
    pub fn open(path: &str, dimension: usize, config: Config) -> Result<Self, String> {
        let conn = Connection::open(path).map_err(|e| format!("open db: {e}"))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .map_err(|e| format!("pragma: {e}"))?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id INTEGER REFERENCES nodes(id),
                centroid BLOB NOT NULL,
                count INTEGER DEFAULT 0,
                radius REAL DEFAULT 0.0,
                label TEXT NOT NULL DEFAULT '',
                depth INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL REFERENCES nodes(id),
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                epoch INTEGER,
                temporal_epoch INTEGER,
                temporal_shift INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_entries_node ON entries(node_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);",
        )
        .map_err(|e| format!("create tables: {e}"))?;

        // Migrate: add columns if they don't exist (for existing DBs)
        for col in &[
            "ALTER TABLE entries ADD COLUMN temporal_epoch INTEGER",
            "ALTER TABLE entries ADD COLUMN temporal_shift INTEGER DEFAULT 0",
            "ALTER TABLE entries ADD COLUMN content_display TEXT",
            "ALTER TABLE entries ADD COLUMN compression_level INTEGER DEFAULT 0",
        ] {
            conn.execute_batch(col).ok(); // ignore "duplicate column" errors
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_entries_temporal ON entries(temporal_epoch);",
        ).ok();

        let tree = Self { conn, config, dimension };
        tree.ensure_root()?;
        Ok(tree)
    }

    fn ensure_root(&self) -> Result<(), String> {
        let exists: bool = self
            .conn
            .query_row("SELECT COUNT(*) > 0 FROM nodes WHERE parent_id IS NULL", [], |r| r.get(0))
            .map_err(|e| format!("check root: {e}"))?;

        if !exists {
            let zero_centroid = vec![0.0f32; self.dimension];
            self.conn
                .execute(
                    "INSERT INTO nodes (parent_id, centroid, count, label, depth) VALUES (NULL, ?1, 0, 'root', 0)",
                    params![centroid_to_blob(&zero_centroid)],
                )
                .map_err(|e| format!("create root: {e}"))?;
        }
        Ok(())
    }

    /// Store content with its embedding. Returns (entry_id, node_label).
    pub fn store(&self, content: &str, embedding: &[f32], source: Option<&str>) -> Result<(i64, String), String> {
        // Find the best leaf node for this embedding
        let (node_id, label) = self.find_or_create_leaf(embedding)?;

        // Dedup: skip if identical content already exists in this leaf
        let exists: bool = self
            .conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM entries WHERE node_id = ?1 AND content = ?2",
                params![node_id, content],
                |r| r.get(0),
            )
            .map_err(|e| format!("dedup check: {e}"))?;

        if exists {
            return Ok((-1, label));
        }

        // Compress content for display
        let (content_display, compression_level) = crate::compress::auto_compress(content);
        let compression_level_i32 = compression_level.as_i32();

        // Insert the entry
        let epoch = chrono::Utc::now().timestamp();
        self.conn
            .execute(
                "INSERT INTO entries (node_id, content, content_display, compression_level, embedding, source, epoch, temporal_epoch) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
                params![node_id, content, content_display, compression_level_i32, centroid_to_blob(embedding), source, epoch],
            )
            .map_err(|e| format!("insert entry: {e}"))?;

        let entry_id = self.conn.last_insert_rowid();

        // Update node centroid and count
        self.update_node_stats(node_id, embedding)?;

        // Check if leaf needs splitting
        let count = self.node_entry_count(node_id)?;
        let was_split = if count > self.config.leaf_capacity as i64 {
            self.split_leaf(node_id)?;
            true
        } else {
            false
        };

        // Check if any siblings should merge (skip if node was just split/deleted)
        if !was_split {
            self.try_merge_siblings(node_id)?;
        }

        Ok((entry_id, label))
    }

    /// Search for the top-k most similar entries using tree-guided beam search
    /// with gap-detection pruning. Descends the BIRCH tree comparing query
    /// against node centroids, cutting at the largest similarity drop-off.
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<Entry>, String> {
        let root_id = self.root_id()?;

        // Initial beam: root's children scored by centroid similarity
        let mut beam: Vec<(f32, i64)> = self
            .children_of(root_id)?
            .into_iter()
            .map(|(id, centroid)| (cosine_similarity(query_embedding, &centroid), id))
            .collect();

        // Empty tree or flat (entries directly on root)
        if beam.is_empty() {
            return self.score_entries_in_nodes(query_embedding, &[root_id], limit);
        }

        // Descend: expand non-leaf nodes, prune by gap detection
        loop {
            let mut next_beam: Vec<(f32, i64)> = Vec::new();
            let mut all_leaves = true;

            for (score, node_id) in &beam {
                let children = self.children_of(*node_id)?;
                if children.is_empty() {
                    next_beam.push((*score, *node_id));
                } else {
                    all_leaves = false;
                    for (child_id, centroid) in children {
                        let sim = cosine_similarity(query_embedding, &centroid);
                        next_beam.push((sim, child_id));
                    }
                }
            }

            if all_leaves {
                break;
            }

            next_beam.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let cut = gap_cut(&next_beam, limit);
            next_beam.truncate(cut);
            beam = next_beam;
        }

        let leaf_ids: Vec<i64> = beam.iter().map(|(_, id)| *id).collect();
        self.score_entries_in_nodes(query_embedding, &leaf_ids, limit)
    }

    /// Fetch child node IDs and centroids for a given parent.
    fn children_of(&self, parent_id: i64) -> Result<Vec<(i64, Vec<f32>)>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, centroid FROM nodes WHERE parent_id = ?1")
            .map_err(|e| format!("children_of: {e}"))?;
        let rows: Vec<(i64, Vec<f32>)> = stmt
            .query_map(params![parent_id], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob_to_centroid(&blob)))
            })
            .map_err(|e| format!("query children: {e}"))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Score entries in the given nodes with decay, access boost, and fuzzy timestamps.
    fn score_entries_in_nodes(
        &self,
        query_embedding: &[f32],
        node_ids: &[i64],
        limit: usize,
    ) -> Result<Vec<Entry>, String> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }

        let now = chrono::Utc::now();
        let placeholders: String = node_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, node_id, content, content_display, compression_level, embedding, source, created_at, access_count, last_accessed, epoch, temporal_epoch, temporal_shift FROM entries WHERE node_id IN ({})",
            placeholders
        );

        let mut stmt = self.conn.prepare(&sql).map_err(|e| format!("prepare score: {e}"))?;
        let mut scored: Vec<(f32, Entry)> = stmt
            .query_map(rusqlite::params_from_iter(node_ids), |row| {
                let id: i64 = row.get(0)?;
                let _node_id: i64 = row.get(1)?;
                let content: String = row.get(2)?;
                let content_display: Option<String> = row.get(3)?;
                let compression_level: i32 = row.get(4)?;
                let blob: Vec<u8> = row.get(5)?;
                let source: Option<String> = row.get(6)?;
                let created_at: String = row.get(7)?;
                let access_count: i64 = row.get(8)?;
                let last_accessed: Option<String> = row.get(9)?;
                let _epoch: Option<i64> = row.get(10)?;
                let temporal_epoch: Option<i64> = row.get(11)?;
                let temporal_shift: u32 = row.get::<_, u32>(12)?;

                let embedding = blob_to_centroid(&blob);
                let raw_sim = cosine_similarity(query_embedding, &embedding);

                let ref_date = last_accessed.as_deref().unwrap_or(&created_at);
                let days_ago = chrono::DateTime::parse_from_rfc3339(ref_date)
                    .map(|dt| (now - dt.to_utc()).num_days().max(0) as f32)
                    .unwrap_or(0.0);

                let k: f32 = 0.5;
                let lambda: f32 = 30.0;
                let decay = (-(days_ago / lambda).powf(k)).exp();

                let access_boost = (access_count as f32).ln_1p() * 0.03;
                let score = raw_sim * (0.7 + 0.3 * decay) + access_boost;

                let when = temporal_epoch
                    .map(|te| fuzzy_date_from_shift(te, temporal_shift))
                    .unwrap_or_else(|| created_at.clone());

                Ok((score, Entry { id, content, content_display, compression_level, source, created_at, when, similarity: score }))
            })
            .map_err(|e| format!("score query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let ids: Vec<i64> = scored.iter().map(|(_, e)| e.id).collect();
        let now_str = now.to_rfc3339();
        for id in &ids {
            self.conn
                .execute(
                    "UPDATE entries SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
                    params![now_str, id],
                )
                .ok();
        }

        Ok(scored.into_iter().map(|(_, e)| e).collect())
    }

    /// Recall: search and return results as a neighborhood subtree.
    /// Finds the best matching entries, then returns their clusters with nearby sibling clusters,
    /// capped at `limit` total entries.
    pub fn recall(&self, query_embedding: &[f32], limit: usize) -> Result<String, String> {
        let entries = self.search(query_embedding, limit)?;
        if entries.is_empty() {
            return Ok(String::new());
        }

        // Collect the node_ids for matched entries
        let matched_node_ids: Vec<i64> = {
            let ids: Vec<i64> = entries.iter().map(|e| e.id).collect();
            let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!("SELECT DISTINCT node_id FROM entries WHERE id IN ({})", placeholders);
            let mut stmt = self.conn.prepare(&sql).map_err(|e| format!("prepare nodes: {e}"))?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(&ids), |row| row.get::<_, i64>(0))
                .map_err(|e| format!("query nodes: {e}"))?;
            rows.filter_map(|r| r.ok()).collect()
        };

        // Go up one level: collect siblings, ranked by centroid similarity to query
        let mut neighborhood: Vec<(f32, i64)> = Vec::new(); // (similarity, node_id)
        let mut seen = std::collections::HashSet::new();

        for node_id in &matched_node_ids {
            let parent_id: Option<i64> = self
                .conn
                .query_row("SELECT parent_id FROM nodes WHERE id = ?1", params![node_id], |row| row.get(0))
                .ok()
                .flatten();

            let siblings = if let Some(pid) = parent_id {
                let mut stmt = self.conn.prepare("SELECT id, centroid FROM nodes WHERE parent_id = ?1")
                    .map_err(|e| format!("siblings: {e}"))?;
                stmt.query_map(params![pid], |row| {
                    let id: i64 = row.get(0)?;
                    let blob: Vec<u8> = row.get(1)?;
                    Ok((id, blob))
                })
                .map_err(|e| format!("query siblings: {e}"))?
                .filter_map(|r| r.ok())
                .collect::<Vec<_>>()
            } else {
                vec![(*node_id, Vec::new())]
            };

            for (sid, centroid_blob) in siblings {
                if !seen.insert(sid) {
                    continue;
                }
                let sim = if centroid_blob.is_empty() {
                    0.0
                } else {
                    let centroid = blob_to_centroid(&centroid_blob);
                    cosine_similarity(query_embedding, &centroid)
                };
                // Matched nodes always come first
                let boost = if matched_node_ids.contains(&sid) { 1.0 } else { 0.0 };
                neighborhood.push((sim + boost, sid));
            }
        }

        neighborhood.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Build output, capping total entries at limit
        let mut output = String::new();
        let mut total_entries = 0;

        for (_sim, node_id) in &neighborhood {
            if total_entries >= limit {
                break;
            }

            let label: String = self
                .conn
                .query_row("SELECT label FROM nodes WHERE id = ?1", params![node_id], |row| row.get(0))
                .unwrap_or_else(|_| "~".into());

            let mut stmt = self
                .conn
                .prepare("SELECT content, content_display FROM entries WHERE node_id = ?1")
                .map_err(|e| format!("entries: {e}"))?;
            let cluster_entries: Vec<String> = stmt
                .query_map(params![node_id], |row| {
                    let content: String = row.get(0)?;
                    let display: Option<String> = row.get(1)?;
                    Ok(display.unwrap_or(content))
                })
                .map_err(|e| format!("query entries: {e}"))?
                .filter_map(|r| r.ok())
                .collect();

            if cluster_entries.is_empty() {
                continue;
            }

            if !output.is_empty() {
                output.push_str("\n\n");
            }
            output.push_str(&label);
            output.push(':');
            for content in &cluster_entries {
                if total_entries >= limit {
                    break;
                }
                output.push('\n');
                output.push_str(content);
                total_entries += 1;
            }
        }

        Ok(output)
    }

    /// List topics (cluster nodes) as a tree.
    pub fn topics(&self) -> Result<Vec<Topic>, String> {
        let root_id = self.root_id()?;
        self.topics_under(root_id)
    }

    /// Forget (delete) an entry by ID.
    pub fn forget(&self, entry_id: i64) -> Result<bool, String> {
        let affected = self
            .conn
            .execute("DELETE FROM entries WHERE id = ?1", params![entry_id])
            .map_err(|e| format!("delete entry: {e}"))?;
        Ok(affected > 0)
    }

    /// Forget (delete) all entries whose source matches a SQL LIKE pattern.
    /// Returns the number of entries deleted.
    pub fn forget_by_source(&self, pattern: &str) -> Result<usize, String> {
        let affected = self
            .conn
            .execute(
                "DELETE FROM entries WHERE source LIKE ?1",
                params![pattern],
            )
            .map_err(|e| format!("delete by source: {e}"))?;
        Ok(affected)
    }

    /// Delete leaf nodes that have no entries, then recurse up to prune empty parents.
    fn prune_empty_leaves(&self) -> Result<usize, String> {
        let mut pruned = 0;
        loop {
            // Find nodes with no entries and no children (true leaves), excluding root
            let empty: Vec<i64> = self
                .conn
                .prepare(
                    "SELECT n.id FROM nodes n
                     WHERE n.parent_id IS NOT NULL
                       AND NOT EXISTS (SELECT 1 FROM entries e WHERE e.node_id = n.id)
                       AND NOT EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n.id)",
                )
                .map_err(|e| format!("find empty: {e}"))?
                .query_map([], |row| row.get(0))
                .map_err(|e| format!("query empty: {e}"))?
                .filter_map(|r| r.ok())
                .collect();

            if empty.is_empty() {
                break;
            }
            for id in &empty {
                self.conn
                    .execute("DELETE FROM nodes WHERE id = ?1", params![id])
                    .map_err(|e| format!("prune node: {e}"))?;
                pruned += 1;
            }
        }
        Ok(pruned)
    }

    /// Rebuild the tree from scratch: extract all entries, drop nodes, re-insert.
    pub fn rebuild(&self) -> Result<String, String> {
        // 1. Extract all entries
        let mut stmt = self
            .conn
            .prepare("SELECT content, embedding, source, created_at FROM entries ORDER BY id")
            .map_err(|e| format!("prepare extract: {e}"))?;
        let entries: Vec<(String, Vec<u8>, Option<String>, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(|e| format!("query entries: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        let count = entries.len();
        if count == 0 {
            return Ok("nothing to rebuild".into());
        }

        // 2. Nuke everything
        self.conn
            .execute_batch("DELETE FROM entries; DELETE FROM nodes;")
            .map_err(|e| format!("clear tables: {e}"))?;

        // 3. Re-create root
        self.ensure_root()?;

        // 4. Re-insert each entry through the normal store path
        let dim = self.dimension;
        for (content, emb_bytes, source, _created_at) in &entries {
            let embedding: Vec<f32> = emb_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            if embedding.len() != dim {
                continue;
            }
            self.store(content, &embedding, source.as_deref())?;
        }

        Ok(format!("rebuilt tree with {count} entries"))
    }

    /// Spreading activation: rehearse memories near the seed, with stochastic noise.
    /// Returns the number of entries activated.
    pub fn replay(&self, seed_embedding: &[f32], noise: f32, count: usize) -> Result<usize, String> {
        let total = self.count()?;
        if total == 0 {
            return Ok(0);
        }

        let now_str = chrono::Utc::now().to_rfc3339();
        let mut activated = 0;

        // Collect all entries with embeddings
        let mut stmt = self
            .conn
            .prepare("SELECT id, embedding FROM entries")
            .map_err(|e| format!("prepare replay: {e}"))?;

        let entries: Vec<(i64, Vec<f32>)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob_to_centroid(&blob)))
            })
            .map_err(|e| format!("query replay: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        if entries.is_empty() {
            return Ok(0);
        }

        // Score all entries by similarity to seed
        let mut scored: Vec<(f32, i64)> = entries
            .iter()
            .map(|(id, emb)| (cosine_similarity(seed_embedding, emb), *id))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Simple deterministic RNG seeded from the embedding
        let mut rng_state: u64 = seed_embedding
            .iter()
            .fold(0u64, |acc, &v| acc.wrapping_add((v * 1e6) as u64));

        for _ in 0..count.min(entries.len()) {
            // Cheap xorshift RNG
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let rand_val = (rng_state % 1000) as f32 / 1000.0;

            let id = if rand_val < noise {
                // Random entry (exploration)
                let idx = (rng_state as usize) % entries.len();
                entries[idx].0
            } else {
                // Nearest entry (exploitation)
                let idx = activated.min(scored.len() - 1);
                scored[idx].1
            };

            self.conn
                .execute(
                    "UPDATE entries SET access_count = access_count + 1, last_accessed = ?1 WHERE id = ?2",
                    params![now_str, id],
                )
                .ok();
            activated += 1;
        }

        Ok(activated)
    }

    /// Total number of stored entries.
    pub fn count(&self) -> Result<i64, String> {
        self.conn
            .query_row("SELECT COUNT(*) FROM entries", [], |r| r.get(0))
            .map_err(|e| format!("count: {e}"))
    }

    /// Execute raw SQL (for testing — manipulate timestamps, access counts, etc.)
    pub fn conn_exec(&self, sql: &str) -> Result<(), String> {
        self.conn.execute_batch(sql).map_err(|e| format!("exec: {e}"))
    }

    /// Rebalance the tree: prune ghost nodes, consolidate faded memories, merge small clusters, re-label.
    pub fn rebalance(&self) -> Result<String, String> {
        let mut merged = 0;
        let mut relabeled = 0;
        let temporal_degraded = self.degrade_temporal_epochs()?;
        let consolidated = self.consolidate_faded()?;

        // Prune empty leaf nodes left behind by forget/decay
        let pruned = self.prune_empty_leaves()?;

        // Merge leaves with fewer than 3 entries into their nearest sibling
        let small_leaves = self.find_small_leaves(3)?;
        for leaf_id in small_leaves {
            if self.merge_into_nearest_sibling(leaf_id)? {
                merged += 1;
            }
        }

        // Re-label all leaves
        let leaves = self.all_leaves()?;
        for leaf_id in leaves {
            if self.auto_label(leaf_id)? {
                relabeled += 1;
            }
        }

        let mut msg = format!("merged {merged} small clusters, relabeled {relabeled} nodes");
        if pruned > 0 {
            msg = format!("pruned {pruned} empty nodes, {msg}");
        }
        if temporal_degraded > 0 {
            msg = format!("degraded {temporal_degraded} timestamps, {msg}");
        }
        if consolidated > 0 {
            msg = format!("consolidated {consolidated} faded memories, {msg}");
        }
        Ok(msg)
    }

    /// Degrade temporal_epoch precision based on age (not access count).
    fn degrade_temporal_epochs(&self) -> Result<usize, String> {
        let now = chrono::Utc::now();
        let mut count = 0;

        let mut stmt = self
            .conn
            .prepare("SELECT id, epoch, temporal_shift, created_at FROM entries WHERE epoch IS NOT NULL")
            .map_err(|e| format!("prepare temporal: {e}"))?;

        let entries: Vec<(i64, i64, u32, String)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get::<_, u32>(2)?, row.get(3)?))
            })
            .map_err(|e| format!("query temporal: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        for (id, epoch, current_shift, created_at) in &entries {
            let days_ago = chrono::DateTime::parse_from_rfc3339(created_at)
                .map(|dt| (now - dt.to_utc()).num_days().max(0) as f32)
                .unwrap_or(0.0);

            let new_shift = temporal_shift_for_age(days_ago);

            if new_shift > *current_shift {
                let new_temporal = epoch >> new_shift;
                self.conn
                    .execute(
                        "UPDATE entries SET temporal_epoch = ?1, temporal_shift = ?2 WHERE id = ?3",
                        params![new_temporal, new_shift, id],
                    )
                    .map_err(|e| format!("update temporal: {e}"))?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Consolidate faded memories: group old unaccessed entries by cluster,
    /// merge each group into a single condensed entry.
    fn consolidate_faded(&self) -> Result<usize, String> {
        let now = chrono::Utc::now();
        let k: f32 = 0.5;
        let lambda: f32 = 30.0;
        let retention_threshold: f32 = 0.15; // consolidate below this retention
        let min_group_size = 3; // need at least 3 faded entries to consolidate

        // Find all entries with low retention and no access
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, node_id, content, embedding, created_at FROM entries
                 WHERE access_count = 0",
            )
            .map_err(|e| format!("prepare consolidate: {e}"))?;

        let candidates: Vec<(i64, i64, String, Vec<f32>, String)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let node_id: i64 = row.get(1)?;
                let content: String = row.get(2)?;
                let blob: Vec<u8> = row.get(3)?;
                let created_at: String = row.get(4)?;
                Ok((id, node_id, content, blob_to_centroid(&blob), created_at))
            })
            .map_err(|e| format!("query consolidate: {e}"))?
            .filter_map(|r| r.ok())
            .filter(|(_, _, _, _, created_at)| {
                let days_ago = chrono::DateTime::parse_from_rfc3339(created_at)
                    .map(|dt| (now - dt.to_utc()).num_days().max(0) as f32)
                    .unwrap_or(0.0);
                let retention = (-(days_ago / lambda).powf(k)).exp();
                retention < retention_threshold
            })
            .collect();

        // Group by node_id
        let mut groups: std::collections::HashMap<i64, Vec<(i64, String, Vec<f32>, String)>> =
            std::collections::HashMap::new();
        for (id, node_id, content, embedding, created_at) in candidates {
            groups.entry(node_id).or_default().push((id, content, embedding, created_at));
        }

        let mut total_consolidated = 0;

        for (node_id, entries) in &groups {
            if entries.len() < min_group_size {
                continue;
            }

            // Find the date range of the faded memories
            let mut earliest: Option<chrono::DateTime<chrono::Utc>> = None;
            let mut latest: Option<chrono::DateTime<chrono::Utc>> = None;
            for (_, _, _, created_at) in entries {
                if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(created_at) {
                    let dt = dt.to_utc();
                    earliest = Some(earliest.map_or(dt, |e: chrono::DateTime<chrono::Utc>| e.min(dt)));
                    latest = Some(latest.map_or(dt, |l: chrono::DateTime<chrono::Utc>| l.max(dt)));
                }
            }

            // Fuzzy date range using bit-shift appropriate for faded memories
            let date_prefix = match (earliest, latest) {
                (Some(e), Some(l)) => {
                    let shift = temporal_shift_for_age(
                        (chrono::Utc::now() - e).num_days().max(0) as f32,
                    );
                    let e_fuzzy = fuzzy_date_from_shift(e.timestamp() >> shift, shift);
                    let l_fuzzy = fuzzy_date_from_shift(l.timestamp() >> shift, shift);
                    if e_fuzzy == l_fuzzy {
                        e_fuzzy
                    } else {
                        format!("{}..{}", e_fuzzy, l_fuzzy)
                    }
                }
                _ => "long ago".to_string(),
            };

            // Compress contents with semantic extraction, average embeddings
            let raw_content: String = entries
                .iter()
                .map(|(_, c, _, _)| c.as_str())
                .collect::<Vec<_>>()
                .join(" | ");
            let combined_content = format!("{}: {}", date_prefix, raw_content);

            // Use semantic (YAKE) compression for consolidated memories
            let compressed = crate::compress::compress(&combined_content, crate::compress::Strategy::Semantic);
            let condensed = if compressed.len() > 2000 {
                &compressed[..compressed.char_indices().nth(2000).map(|(i, _)| i).unwrap_or(compressed.len())]
            } else {
                &compressed
            };

            // Average embedding
            let dim = entries[0].2.len();
            let mut avg = vec![0.0f32; dim];
            for (_, _, emb, _) in entries {
                for (i, v) in emb.iter().enumerate() {
                    avg[i] += v;
                }
            }
            let n = entries.len() as f32;
            for v in &mut avg {
                *v /= n;
            }

            // Insert consolidated entry with display version
            let (consolidated_display, consolidated_level) = crate::compress::auto_compress(condensed);
            let consolidated_level_i32 = consolidated_level.as_i32();
            let epoch = earliest.map(|e| e.timestamp()).unwrap_or_else(|| now.timestamp());
            self.conn
                .execute(
                    "INSERT INTO entries (node_id, content, content_display, compression_level, embedding, source, epoch, temporal_epoch) VALUES (?1, ?2, ?3, ?4, ?5, 'consolidated', ?6, ?6)",
                    params![node_id, condensed, consolidated_display, consolidated_level_i32, centroid_to_blob(&avg), epoch],
                )
                .map_err(|e| format!("insert consolidated: {e}"))?;

            // Delete originals
            for (id, _, _, _) in entries {
                self.conn
                    .execute("DELETE FROM entries WHERE id = ?1", params![id])
                    .ok();
            }

            total_consolidated += entries.len();
        }

        Ok(total_consolidated)
    }

    // --- Internal methods ---

    fn root_id(&self) -> Result<i64, String> {
        self.conn
            .query_row("SELECT id FROM nodes WHERE parent_id IS NULL LIMIT 1", [], |r| r.get(0))
            .map_err(|e| format!("find root: {e}"))
    }

    fn find_or_create_leaf(&self, embedding: &[f32]) -> Result<(i64, String), String> {
        let leaves = self.all_leaves()?;

        if leaves.is_empty() {
            // First entry — create first leaf under root
            return self.create_leaf(self.root_id()?, embedding);
        }

        // Find nearest leaf by centroid similarity
        let mut best_id = leaves[0];
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_label = String::new();

        for leaf_id in &leaves {
            let node = self.get_node(*leaf_id)?;
            let sim = cosine_similarity(embedding, &node.centroid);
            if sim > best_sim {
                best_sim = sim;
                best_id = *leaf_id;
                best_label = node.label;
            }
        }

        if best_sim >= self.config.threshold {
            Ok((best_id, best_label))
        } else {
            // Too far from any existing cluster — create new leaf
            // Find the best internal node to attach to
            let parent_id = self.find_best_parent(embedding)?;
            self.create_leaf(parent_id, embedding)
        }
    }

    fn find_best_parent(&self, embedding: &[f32]) -> Result<i64, String> {
        // For now, attach to root. With deeper trees, would traverse internals.
        let root_id = self.root_id()?;

        // Check if root has too many children
        let child_count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(*) FROM nodes WHERE parent_id = ?1",
                params![root_id],
                |r| r.get(0),
            )
            .map_err(|e| format!("count children: {e}"))?;

        if child_count >= self.config.branch_factor as i64 {
            // Find the internal node whose centroid is nearest
            let mut stmt = self
                .conn
                .prepare("SELECT id, centroid FROM nodes WHERE parent_id = ?1")
                .map_err(|e| format!("prepare internals: {e}"))?;

            let mut best_id = root_id;
            let mut best_sim = f32::NEG_INFINITY;

            let rows: Vec<(i64, Vec<u8>)> = stmt
                .query_map(params![root_id], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(|e| format!("query internals: {e}"))?
                .filter_map(|r| r.ok())
                .collect();

            for (id, blob) in &rows {
                let centroid = blob_to_centroid(blob);
                let sim = cosine_similarity(embedding, &centroid);
                if sim > best_sim {
                    best_sim = sim;
                    best_id = *id;
                }
            }

            Ok(best_id)
        } else {
            Ok(root_id)
        }
    }

    fn create_leaf(&self, parent_id: i64, embedding: &[f32]) -> Result<(i64, String), String> {
        let parent_depth: i32 = self
            .conn
            .query_row("SELECT depth FROM nodes WHERE id = ?1", params![parent_id], |r| r.get(0))
            .map_err(|e| format!("get parent depth: {e}"))?;

        let label = format!("topic_{}", self.next_topic_number()?);

        self.conn
            .execute(
                "INSERT INTO nodes (parent_id, centroid, count, label, depth) VALUES (?1, ?2, 0, ?3, ?4)",
                params![parent_id, centroid_to_blob(embedding), label, parent_depth + 1],
            )
            .map_err(|e| format!("create leaf: {e}"))?;

        let id = self.conn.last_insert_rowid();
        Ok((id, label))
    }

    fn next_topic_number(&self) -> Result<i64, String> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes WHERE parent_id IS NOT NULL", [], |r| r.get(0))
            .map_err(|e| format!("count nodes: {e}"))?;
        Ok(count + 1)
    }

    fn update_node_stats(&self, node_id: i64, new_embedding: &[f32]) -> Result<(), String> {
        let node = self.get_node(node_id)?;
        let new_count = node.count + 1;

        // Incremental centroid update: new_centroid = old_centroid + (embedding - old_centroid) / new_count
        let mut new_centroid = vec![0.0f32; self.dimension];
        for i in 0..self.dimension {
            new_centroid[i] = node.centroid[i] + (new_embedding[i] - node.centroid[i]) / new_count as f32;
        }

        self.conn
            .execute(
                "UPDATE nodes SET centroid = ?1, count = ?2 WHERE id = ?3",
                params![centroid_to_blob(&new_centroid), new_count, node_id],
            )
            .map_err(|e| format!("update stats: {e}"))?;

        Ok(())
    }

    fn node_entry_count(&self, node_id: i64) -> Result<i64, String> {
        self.conn
            .query_row(
                "SELECT COUNT(*) FROM entries WHERE node_id = ?1",
                params![node_id],
                |r| r.get(0),
            )
            .map_err(|e| format!("count entries: {e}"))
    }

    fn split_leaf(&self, node_id: i64) -> Result<(), String> {
        // Get all entries in this leaf
        let mut stmt = self
            .conn
            .prepare("SELECT id, embedding FROM entries WHERE node_id = ?1")
            .map_err(|e| format!("prepare split: {e}"))?;

        let entries: Vec<(i64, Vec<f32>)> = stmt
            .query_map(params![node_id], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob_to_centroid(&blob)))
            })
            .map_err(|e| format!("query split: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        if entries.len() < 2 {
            return Ok(());
        }

        // Simple 2-means split: pick two most distant entries as seeds
        let (seed_a, seed_b) = find_most_distant(&entries);

        let node = self.get_node(node_id)?;
        let parent_id = node.parent_id.unwrap_or(node_id);

        // Create two new leaves
        let (leaf_a, _) = self.create_leaf(parent_id, &entries[seed_a].1)?;
        let (leaf_b, _) = self.create_leaf(parent_id, &entries[seed_b].1)?;

        // Assign entries to nearest seed
        for (entry_id, embedding) in &entries {
            let sim_a = cosine_similarity(embedding, &entries[seed_a].1);
            let sim_b = cosine_similarity(embedding, &entries[seed_b].1);
            let target = if sim_a >= sim_b { leaf_a } else { leaf_b };

            self.conn
                .execute(
                    "UPDATE entries SET node_id = ?1 WHERE id = ?2",
                    params![target, entry_id],
                )
                .map_err(|e| format!("reassign entry: {e}"))?;
        }

        // Recompute centroids for new leaves
        self.recompute_centroid(leaf_a)?;
        self.recompute_centroid(leaf_b)?;

        // Auto-label new leaves
        let _ = self.auto_label(leaf_a);
        let _ = self.auto_label(leaf_b);

        // Delete the old leaf (now empty)
        self.conn
            .execute("DELETE FROM nodes WHERE id = ?1", params![node_id])
            .map_err(|e| format!("delete old leaf: {e}"))?;

        Ok(())
    }

    fn recompute_centroid(&self, node_id: i64) -> Result<(), String> {
        let mut stmt = self
            .conn
            .prepare("SELECT embedding FROM entries WHERE node_id = ?1")
            .map_err(|e| format!("prepare recompute: {e}"))?;

        let embeddings: Vec<Vec<f32>> = stmt
            .query_map(params![node_id], |row| {
                let blob: Vec<u8> = row.get(0)?;
                Ok(blob_to_centroid(&blob))
            })
            .map_err(|e| format!("query recompute: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        if embeddings.is_empty() {
            return Ok(());
        }

        let n = embeddings.len() as f32;
        let mut centroid = vec![0.0f32; self.dimension];
        for emb in &embeddings {
            for (i, v) in emb.iter().enumerate() {
                centroid[i] += v;
            }
        }
        for v in &mut centroid {
            *v /= n;
        }

        self.conn
            .execute(
                "UPDATE nodes SET centroid = ?1, count = ?2 WHERE id = ?3",
                params![centroid_to_blob(&centroid), embeddings.len() as i64, node_id],
            )
            .map_err(|e| format!("update centroid: {e}"))?;

        Ok(())
    }

    fn get_node(&self, node_id: i64) -> Result<Node, String> {
        self.conn
            .query_row(
                "SELECT id, parent_id, centroid, count, radius, label, depth FROM nodes WHERE id = ?1",
                params![node_id],
                |row| {
                    Ok(Node {
                        parent_id: row.get(1)?,
                        centroid: blob_to_centroid(&row.get::<_, Vec<u8>>(2)?),
                        count: row.get(3)?,
                        label: row.get(5)?,
                    })
                },
            )
            .map_err(|e| format!("get node {node_id}: {e}"))
    }

    fn all_leaves(&self) -> Result<Vec<i64>, String> {
        // Leaves = nodes with no children (except root if it has no children AND no entries)
        let mut stmt = self
            .conn
            .prepare(
                "SELECT n.id FROM nodes n
                 WHERE n.parent_id IS NOT NULL
                 AND NOT EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n.id)",
            )
            .map_err(|e| format!("prepare leaves: {e}"))?;

        let ids: Vec<i64> = stmt
            .query_map([], |row| row.get(0))
            .map_err(|e| format!("query leaves: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(ids)
    }

    fn find_small_leaves(&self, min_entries: i64) -> Result<Vec<i64>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT n.id FROM nodes n
                 WHERE n.parent_id IS NOT NULL
                 AND NOT EXISTS (SELECT 1 FROM nodes c WHERE c.parent_id = n.id)
                 AND (SELECT COUNT(*) FROM entries e WHERE e.node_id = n.id) < ?1
                 AND (SELECT COUNT(*) FROM entries e WHERE e.node_id = n.id) > 0",
            )
            .map_err(|e| format!("prepare small: {e}"))?;

        let ids: Vec<i64> = stmt
            .query_map(params![min_entries], |row| row.get(0))
            .map_err(|e| format!("query small: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        Ok(ids)
    }

    /// After an insert, check if this node's parent has two siblings close enough to merge.
    fn try_merge_siblings(&self, node_id: i64) -> Result<(), String> {
        let node = self.get_node(node_id)?;
        let parent_id = match node.parent_id {
            Some(p) => p,
            None => return Ok(()),
        };

        // Get all siblings under the same parent
        let mut stmt = self
            .conn
            .prepare("SELECT id, centroid FROM nodes WHERE parent_id = ?1")
            .map_err(|e| format!("prepare siblings: {e}"))?;

        let siblings: Vec<(i64, Vec<f32>)> = stmt
            .query_map(params![parent_id], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob_to_centroid(&blob)))
            })
            .map_err(|e| format!("query siblings: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        if siblings.len() < 2 {
            return Ok(());
        }

        // Find the most similar pair
        let merge_threshold = self.config.threshold + 0.15; // merge when clearly same topic
        let mut best_sim = f32::NEG_INFINITY;
        let mut merge_a = 0i64;
        let mut merge_b = 0i64;

        for i in 0..siblings.len() {
            for j in (i + 1)..siblings.len() {
                let sim = cosine_similarity(&siblings[i].1, &siblings[j].1);
                if sim > best_sim {
                    best_sim = sim;
                    merge_a = siblings[i].0;
                    merge_b = siblings[j].0;
                }
            }
        }

        if best_sim >= merge_threshold {
            // Merge b into a
            self.conn
                .execute(
                    "UPDATE entries SET node_id = ?1 WHERE node_id = ?2",
                    params![merge_a, merge_b],
                )
                .map_err(|e| format!("merge entries: {e}"))?;

            self.recompute_centroid(merge_a)?;

            self.conn
                .execute("DELETE FROM nodes WHERE id = ?1", params![merge_b])
                .map_err(|e| format!("delete merged: {e}"))?;

            let _ = self.auto_label(merge_a);
        }

        Ok(())
    }

    fn merge_into_nearest_sibling(&self, leaf_id: i64) -> Result<bool, String> {
        let node = self.get_node(leaf_id)?;
        let parent_id = match node.parent_id {
            Some(p) => p,
            None => return Ok(false),
        };

        // Find nearest sibling
        let mut stmt = self
            .conn
            .prepare("SELECT id, centroid FROM nodes WHERE parent_id = ?1 AND id != ?2")
            .map_err(|e| format!("prepare siblings: {e}"))?;

        let siblings: Vec<(i64, Vec<f32>)> = stmt
            .query_map(params![parent_id, leaf_id], |row| {
                let id: i64 = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                Ok((id, blob_to_centroid(&blob)))
            })
            .map_err(|e| format!("query siblings: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        if siblings.is_empty() {
            return Ok(false);
        }

        let mut best_id = siblings[0].0;
        let mut best_sim = f32::NEG_INFINITY;
        for (id, centroid) in &siblings {
            let sim = cosine_similarity(&node.centroid, centroid);
            if sim > best_sim {
                best_sim = sim;
                best_id = *id;
            }
        }

        // Move all entries to the nearest sibling
        self.conn
            .execute(
                "UPDATE entries SET node_id = ?1 WHERE node_id = ?2",
                params![best_id, leaf_id],
            )
            .map_err(|e| format!("merge entries: {e}"))?;

        // Recompute sibling centroid
        self.recompute_centroid(best_id)?;

        // Delete the empty leaf
        self.conn
            .execute("DELETE FROM nodes WHERE id = ?1", params![leaf_id])
            .map_err(|e| format!("delete merged: {e}"))?;

        Ok(true)
    }

    /// Generate a label for a leaf from its entry contents.
    fn auto_label(&self, node_id: i64) -> Result<bool, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT content FROM entries WHERE node_id = ?1 LIMIT 20")
            .map_err(|e| format!("prepare label: {e}"))?;

        let texts: Vec<String> = stmt
            .query_map(params![node_id], |row| row.get(0))
            .map_err(|e| format!("query label: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        if texts.is_empty() {
            return Ok(false);
        }

        let label = extract_top_terms(&texts, 3);

        self.conn
            .execute(
                "UPDATE nodes SET label = ?1 WHERE id = ?2",
                params![label, node_id],
            )
            .map_err(|e| format!("update label: {e}"))?;

        Ok(true)
    }

    fn topics_under(&self, node_id: i64) -> Result<Vec<Topic>, String> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, label, count, depth FROM nodes WHERE parent_id = ?1")
            .map_err(|e| format!("prepare topics: {e}"))?;

        let nodes: Vec<(i64, String, i64, i32)> = stmt
            .query_map(params![node_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .map_err(|e| format!("query topics: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        let mut topics = Vec::new();
        for (id, label, count, depth) in nodes {
            let children = self.topics_under(id)?;
            // Use entry count for leaves
            let entry_count = if children.is_empty() {
                self.node_entry_count(id).unwrap_or(count)
            } else {
                count
            };
            topics.push(Topic { id, label, count: entry_count, depth, children });
        }

        Ok(topics)
    }
}

// --- Helpers ---

/// Compute the bit-shift amount for temporal decay based purely on age.
/// Access count does NOT protect temporal precision — only content retention.
pub fn temporal_shift_for_age(days_since_created: f32) -> u32 {
    // Gradual precision loss based on age alone
    // Each shift doubles the time window that looks "the same"
    if days_since_created < 1.0 { 0 }       // today: full second precision
    else if days_since_created < 7.0 { 4 }   // this week: ~16 second blocks
    else if days_since_created < 30.0 { 8 }   // this month: ~4 minute blocks
    else if days_since_created < 90.0 { 12 }  // this quarter: ~1 hour blocks
    else if days_since_created < 365.0 { 16 } // this year: ~18 hour blocks
    else if days_since_created < 1095.0 { 20 } // ~3 years: ~12 day blocks
    else if days_since_created < 3650.0 { 24 } // ~10 years: ~6 month blocks
    else { 28 }                                // ancient: ~8 year blocks
}

/// Reconstruct a fuzzy date string from a shifted epoch and its shift amount.
pub fn fuzzy_date_from_shift(temporal_epoch: i64, shift: u32) -> String {
    if temporal_epoch == 0 {
        return "long ago".to_string();
    }

    // Shift back left to get approximate epoch
    let approx_epoch = temporal_epoch << shift;

    use chrono::{DateTime, Utc};
    let dt = match DateTime::<Utc>::from_timestamp(approx_epoch, 0) {
        Some(dt) => dt,
        None => return "long ago".to_string(),
    };

    match shift {
        0..=4 => dt.format("%Y-%m-%d %H:%M").to_string(),
        5..=8 => dt.format("%Y-%m-%d %H:xx").to_string(),
        9..=12 => dt.format("%Y-%m-%d").to_string(),
        13..=16 => dt.format("%Y-%m").to_string(),
        17..=20 => dt.format("%Y").to_string(),
        _ => "long ago".to_string(),
    }
}

/// Find the natural cut point in a sorted (descending) scored list by largest gap.
/// Returns the number of items to keep. Always keeps at least `min_keep`.
fn gap_cut(sorted: &[(f32, i64)], min_keep: usize) -> usize {
    let n = sorted.len();
    let min_keep = min_keep.max(1).min(n);

    if n <= min_keep {
        return n;
    }

    // Find the largest gap between consecutive scores
    let mut max_gap = 0.0f32;
    let mut cut_at = n; // default: keep all

    for i in min_keep..n {
        let gap = sorted[i - 1].0 - sorted[i].0;
        if gap > max_gap {
            max_gap = gap;
            cut_at = i;
        }
    }

    // Only cut if the gap is meaningful (> 10% of the score range)
    let range = sorted[0].0 - sorted[n - 1].0;
    if range > 0.0 && max_gap / range > 0.1 {
        cut_at
    } else {
        n // no clear gap — keep all
    }
}

fn centroid_to_blob(centroid: &[f32]) -> Vec<u8> {
    centroid.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_centroid(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn find_most_distant(entries: &[(i64, Vec<f32>)]) -> (usize, usize) {
    let mut worst_sim = f32::INFINITY;
    let mut a = 0;
    let mut b = 1;

    // Sample for efficiency if many entries
    let step = if entries.len() > 100 { entries.len() / 20 } else { 1 };

    for i in (0..entries.len()).step_by(step) {
        for j in (i + 1..entries.len()).step_by(step) {
            let sim = cosine_similarity(&entries[i].1, &entries[j].1);
            if sim < worst_sim {
                worst_sim = sim;
                a = i;
                b = j;
            }
        }
    }

    (a, b)
}

/// Extract top-N distinctive terms from a set of texts.
fn extract_top_terms(texts: &[String], n: usize) -> String {
    use std::collections::HashMap;

    let stop_words = [
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "and", "but", "or", "not", "so", "yet", "this", "that", "these", "those", "it", "its",
    ];

    let mut freq: HashMap<String, usize> = HashMap::new();
    let mut doc_freq: HashMap<String, usize> = HashMap::new();

    for text in texts {
        let mut seen = std::collections::HashSet::new();
        for word in text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
            let w = word.to_lowercase();
            if stop_words.contains(&w.as_str()) {
                continue;
            }
            *freq.entry(w.clone()).or_default() += 1;
            if seen.insert(w.clone()) {
                *doc_freq.entry(w).or_default() += 1;
            }
        }
    }

    // Score: term frequency * inverse document frequency
    let num_docs = texts.len() as f64;
    let mut scored: Vec<(String, f64)> = freq
        .iter()
        .map(|(term, &tf)| {
            let df = *doc_freq.get(term).unwrap_or(&1) as f64;
            let idf = (num_docs / df).ln() + 1.0;
            (term.clone(), tf as f64 * idf)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(n)
        .map(|(term, _)| term)
        .collect::<Vec<_>>()
        .join("-")
}
