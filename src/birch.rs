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
    pub source: Option<String>,
    pub created_at: String,
    pub node_id: i64,
    pub similarity: f32,
}

/// A cluster node in the tree.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: i64,
    pub parent_id: Option<i64>,
    pub centroid: Vec<f32>,
    pub count: i64,
    pub radius: f32,
    pub label: String,
    pub depth: i32,
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_entries_node ON entries(node_id);",
        )
        .map_err(|e| format!("create tables: {e}"))?;

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

        // Insert the entry
        self.conn
            .execute(
                "INSERT INTO entries (node_id, content, embedding, source) VALUES (?1, ?2, ?3, ?4)",
                params![node_id, content, centroid_to_blob(embedding), source],
            )
            .map_err(|e| format!("insert entry: {e}"))?;

        let entry_id = self.conn.last_insert_rowid();

        // Update node centroid and count
        self.update_node_stats(node_id, embedding)?;

        // Check if leaf needs splitting
        let count = self.node_entry_count(node_id)?;
        if count > self.config.leaf_capacity as i64 {
            self.split_leaf(node_id)?;
        }

        Ok((entry_id, label))
    }

    /// Search for the top-k most similar entries to a query embedding.
    pub fn search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<Entry>, String> {
        // Brute-force cosine similarity over all entries.
        // Fine for <10k entries. For larger scale, traverse tree to prune.
        let mut stmt = self
            .conn
            .prepare("SELECT id, node_id, content, embedding, source, created_at FROM entries")
            .map_err(|e| format!("prepare search: {e}"))?;

        let mut scored: Vec<(f32, Entry)> = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let node_id: i64 = row.get(1)?;
                let content: String = row.get(2)?;
                let blob: Vec<u8> = row.get(3)?;
                let source: Option<String> = row.get(4)?;
                let created_at: String = row.get(5)?;
                let embedding = blob_to_centroid(&blob);
                let sim = cosine_similarity(query_embedding, &embedding);
                Ok((sim, Entry { id, content, source, created_at, node_id, similarity: sim }))
            })
            .map_err(|e| format!("search query: {e}"))?
            .filter_map(|r| r.ok())
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored.into_iter().map(|(_, e)| e).collect())
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

    /// Total number of stored entries.
    pub fn count(&self) -> Result<i64, String> {
        self.conn
            .query_row("SELECT COUNT(*) FROM entries", [], |r| r.get(0))
            .map_err(|e| format!("count: {e}"))
    }

    /// Rebalance the tree: merge small clusters, re-label.
    pub fn rebalance(&self) -> Result<String, String> {
        let mut merged = 0;
        let mut relabeled = 0;

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

        Ok(format!("merged {merged} small clusters, relabeled {relabeled} nodes"))
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
                        id: row.get(0)?,
                        parent_id: row.get(1)?,
                        centroid: blob_to_centroid(&row.get::<_, Vec<u8>>(2)?),
                        count: row.get(3)?,
                        radius: row.get(4)?,
                        label: row.get(5)?,
                        depth: row.get(6)?,
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
