use engram::birch::{Config, Tree};
use engram::embed::cosine_similarity;

/// Create a test tree with an in-memory SQLite database.
fn test_tree(dim: usize) -> Tree {
    Tree::open(":memory:", dim, Config::default()).expect("open tree")
}

/// Generate a synthetic embedding: a unit vector with a spike at the given index.
fn make_embedding(dim: usize, spike: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[spike] = 1.0;
    v
}

/// Generate an embedding with multiple spikes — simulates a "topic blend".
fn make_blended(dim: usize, spikes: &[usize], weight: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for &s in spikes {
        v[s] = weight;
    }
    v
}

// --- Store and retrieve ---

#[test]
fn store_and_search() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);
    let (id, _topic) = tree.store("hello world", &emb, None).unwrap();
    assert!(id > 0);

    let results = tree.search(&emb, 5).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "hello world");
    assert!(results[0].similarity > 0.5);
}

#[test]
fn search_returns_most_similar_first() {
    let tree = test_tree(8);
    let emb_a = make_embedding(8, 0);
    let emb_b = make_embedding(8, 1);
    let emb_c = make_embedding(8, 2);

    tree.store("topic A", &emb_a, None).unwrap();
    tree.store("topic B", &emb_b, None).unwrap();
    tree.store("topic C", &emb_c, None).unwrap();

    // Search for something close to A
    let results = tree.search(&emb_a, 3).unwrap();
    assert_eq!(results[0].content, "topic A");
}

#[test]
fn search_with_limit() {
    let tree = test_tree(8);
    for i in 0..8 {
        let emb = make_embedding(8, i);
        tree.store(&format!("entry {i}"), &emb, None).unwrap();
    }

    let results = tree.search(&make_embedding(8, 0), 3).unwrap();
    assert_eq!(results.len(), 3);
}

// --- Dedup ---

#[test]
fn duplicate_content_is_skipped() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);

    let (id1, _) = tree.store("same content", &emb, None).unwrap();
    let (id2, _) = tree.store("same content", &emb, None).unwrap();

    assert!(id1 > 0);
    assert_eq!(id2, -1); // duplicate

    assert_eq!(tree.count().unwrap(), 1);
}

#[test]
fn similar_but_different_content_is_not_deduped() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);

    tree.store("content version 1", &emb, None).unwrap();
    tree.store("content version 2", &emb, None).unwrap();

    assert_eq!(tree.count().unwrap(), 2);
}

// --- Clustering ---

#[test]
fn similar_entries_cluster_together() {
    let tree = test_tree(8);

    // Two similar embeddings (both spike at index 0, slight variation)
    let mut emb1 = make_embedding(8, 0);
    emb1[1] = 0.1;
    let mut emb2 = make_embedding(8, 0);
    emb2[1] = 0.2;

    let (_, topic1) = tree.store("related A", &emb1, None).unwrap();
    let (_, topic2) = tree.store("related B", &emb2, None).unwrap();

    // Should land in the same topic (high cosine similarity)
    assert_eq!(topic1, topic2);
}

#[test]
fn dissimilar_entries_get_different_clusters() {
    let tree = test_tree(8);
    let emb_a = make_embedding(8, 0);
    let emb_b = make_embedding(8, 4); // orthogonal

    let (_, topic_a) = tree.store("topic A content", &emb_a, None).unwrap();
    let (_, topic_b) = tree.store("topic B content", &emb_b, None).unwrap();

    assert_ne!(topic_a, topic_b);
}

// --- Split ---

#[test]
fn leaf_splits_when_full() {
    let config = Config {
        threshold: 0.3, // low threshold so everything clusters together
        leaf_capacity: 5,
        branch_factor: 10,
    };
    let tree = Tree::open(":memory:", 8, config).expect("open tree");

    // Insert 6 entries with slightly varying embeddings — all cluster together
    for i in 0..6 {
        let mut emb = vec![1.0f32; 8];
        emb[i % 8] += 0.1 * i as f32;
        tree.store(&format!("entry {i}"), &emb, None).unwrap();
    }

    // After split, should have more than 1 topic
    let topics = tree.topics().unwrap();
    assert!(topics.len() >= 2, "expected split, got {} topics", topics.len());
}

// --- Merge on insert ---

#[test]
fn siblings_merge_when_converging() {
    let config = Config {
        threshold: 0.5,
        leaf_capacity: 50,
        branch_factor: 10,
    };
    let tree = Tree::open(":memory:", 8, config).expect("open tree");

    // Create two separate clusters
    let emb_a = make_embedding(8, 0);
    let emb_b = make_embedding(8, 4);
    tree.store("cluster A", &emb_a, None).unwrap();
    tree.store("cluster B", &emb_b, None).unwrap();

    let topics_before = tree.topics().unwrap();

    // Now insert entries that bridge A and B — their centroids converge
    for _ in 0..10 {
        let bridge = make_blended(8, &[0, 4], 0.7);
        tree.store("bridging content", &bridge, Some("bridge")).unwrap();
    }

    // After many bridging inserts, clusters may have merged
    let topics_after = tree.topics().unwrap();
    assert!(
        topics_after.len() <= topics_before.len(),
        "expected merge: before={}, after={}",
        topics_before.len(),
        topics_after.len()
    );
}

// --- Topics ---

#[test]
fn topics_returns_tree_structure() {
    let tree = test_tree(8);
    let emb_a = make_embedding(8, 0);
    let emb_b = make_embedding(8, 4);

    tree.store("topic A entry", &emb_a, None).unwrap();
    tree.store("topic B entry", &emb_b, None).unwrap();

    let topics = tree.topics().unwrap();
    assert!(!topics.is_empty());

    let total_entries: i64 = topics.iter().map(|t| t.count).sum();
    assert_eq!(total_entries, 2);
}

// --- Forget ---

#[test]
fn forget_removes_entry() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);
    let (id, _) = tree.store("to be forgotten", &emb, None).unwrap();

    assert_eq!(tree.count().unwrap(), 1);
    assert!(tree.forget(id).unwrap());
    assert_eq!(tree.count().unwrap(), 0);
}

#[test]
fn forget_nonexistent_returns_false() {
    let tree = test_tree(8);
    assert!(!tree.forget(9999).unwrap());
}

// --- Decay ---

#[test]
fn recently_stored_ranks_higher_than_old() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);

    // Store an "old" entry by manipulating created_at
    tree.store("old memory", &emb, None).unwrap();
    // Backdate it
    let old_date = (chrono::Utc::now() - chrono::Duration::days(120)).to_rfc3339();
    tree.conn_exec(
        &format!("UPDATE entries SET created_at = '{old_date}' WHERE content = 'old memory'"),
    ).unwrap();

    // Store a fresh entry with same embedding
    let mut emb2 = make_embedding(8, 0);
    emb2[1] = 0.01; // tiny variation so it's not deduped
    tree.store("fresh memory", &emb2, None).unwrap();

    let results = tree.search(&emb, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].content, "fresh memory");
}

// --- Access boost ---

#[test]
fn accessed_memory_resists_decay() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);

    tree.store("accessed memory", &emb, None).unwrap();
    tree.store("ignored memory", &make_blended(8, &[0, 1], 0.9), None).unwrap();

    // Backdate both
    let old_date = (chrono::Utc::now() - chrono::Duration::days(90)).to_rfc3339();
    tree.conn_exec(
        &format!("UPDATE entries SET created_at = '{old_date}'"),
    ).unwrap();

    // Search multiple times to boost "accessed memory"
    for _ in 0..5 {
        tree.search(&emb, 1).unwrap();
    }

    // Reset the other one's access count
    tree.conn_exec(
        "UPDATE entries SET access_count = 0, last_accessed = NULL WHERE content = 'ignored memory'",
    ).unwrap();

    let results = tree.search(&emb, 2).unwrap();
    assert_eq!(results[0].content, "accessed memory");
}

// --- Consolidation ---

#[test]
fn faded_memories_consolidate() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);

    // Store 5 entries and backdate them to trigger consolidation
    for i in 0..5 {
        let mut e = emb.clone();
        e[1] = 0.01 * i as f32; // slight variation to avoid dedup
        tree.store(&format!("faded memory {i}"), &e, None).unwrap();
    }
    assert_eq!(tree.count().unwrap(), 5);

    // Backdate all entries to 120 days ago (Weibull retention < 0.15)
    let old_date = (chrono::Utc::now() - chrono::Duration::days(120)).to_rfc3339();
    tree.conn_exec(
        &format!("UPDATE entries SET created_at = '{old_date}', access_count = 0, last_accessed = NULL"),
    ).unwrap();

    let result = tree.rebalance().unwrap();
    assert!(result.contains("consolidated"), "expected consolidation: {result}");

    // Should have fewer entries now (5 → 1 consolidated)
    let count = tree.count().unwrap();
    assert!(count < 5, "expected consolidation to reduce entries, got {count}");
}

#[test]
fn accessed_memories_are_not_consolidated() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);

    for i in 0..5 {
        let mut e = emb.clone();
        e[1] = 0.01 * i as f32;
        tree.store(&format!("active memory {i}"), &e, None).unwrap();
    }

    // Backdate but set access_count > 0
    let old_date = (chrono::Utc::now() - chrono::Duration::days(120)).to_rfc3339();
    tree.conn_exec(
        &format!("UPDATE entries SET created_at = '{old_date}', access_count = 3"),
    ).unwrap();

    tree.rebalance().unwrap();

    // All 5 should survive — they were accessed
    assert_eq!(tree.count().unwrap(), 5);
}

// --- Rebalance ---

#[test]
fn rebalance_merges_small_clusters() {
    let tree = test_tree(8);

    // Create two singleton clusters
    tree.store("lonely A", &make_embedding(8, 0), None).unwrap();
    tree.store("lonely B", &make_embedding(8, 4), None).unwrap();

    let topics_before = tree.topics().unwrap();
    tree.rebalance().unwrap();
    let topics_after = tree.topics().unwrap();

    // Small clusters (< 3 entries) should merge
    assert!(
        topics_after.len() <= topics_before.len(),
        "expected merge: before={}, after={}",
        topics_before.len(),
        topics_after.len()
    );
}

// --- Cosine similarity ---

#[test]
fn cosine_similarity_identical() {
    let a = vec![1.0, 0.0, 0.0];
    assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert!(cosine_similarity(&a, &b).abs() < 1e-6);
}

#[test]
fn cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
}

// --- Source tracking ---

#[test]
fn source_is_preserved() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);
    tree.store("with source", &emb, Some("test.md")).unwrap();

    let results = tree.search(&emb, 1).unwrap();
    assert_eq!(results[0].source.as_deref(), Some("test.md"));
}

#[test]
fn source_is_optional() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);
    tree.store("no source", &emb, None).unwrap();

    let results = tree.search(&emb, 1).unwrap();
    assert!(results[0].source.is_none());
}

// --- Temporal bit-shift decay ---

#[test]
fn fresh_memory_has_full_temporal_precision() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);
    tree.store("just now", &emb, None).unwrap();

    let results = tree.search(&emb, 1).unwrap();
    // Fresh memory: when should contain full date + time
    assert!(results[0].when.contains(':'), "expected time precision, got: {}", results[0].when);
}

#[test]
fn temporal_degrades_with_age() {
    use engram::birch::{temporal_shift_for_age, fuzzy_date_from_shift};

    let epoch: i64 = 1780000000; // some fixed epoch

    // Fresh: no shift
    let shift = temporal_shift_for_age(0.5);
    assert_eq!(shift, 0);
    let date = fuzzy_date_from_shift(epoch >> shift, shift);
    assert!(date.contains(':'), "fresh should have time: {date}");

    // 1 week old: shift 4
    let shift = temporal_shift_for_age(5.0);
    assert_eq!(shift, 4);

    // 2 months old: shift 12
    let shift = temporal_shift_for_age(60.0);
    assert_eq!(shift, 12);
    let date = fuzzy_date_from_shift(epoch >> shift, shift);
    assert!(!date.contains(':'), "2 months should not have time: {date}");

    // 2 years old: shift 20
    let shift = temporal_shift_for_age(730.0);
    assert_eq!(shift, 20);
    let date = fuzzy_date_from_shift(epoch >> shift, shift);
    // Should be just a year
    assert!(date.len() == 4, "2 years old should be just year: {date}");
}

#[test]
fn rebalance_degrades_old_timestamps() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);
    tree.store("old memory", &emb, None).unwrap();

    // Backdate to 60 days ago
    let old_date = (chrono::Utc::now() - chrono::Duration::days(60)).to_rfc3339();
    tree.conn_exec(
        &format!("UPDATE entries SET created_at = '{old_date}'"),
    ).unwrap();

    let result = tree.rebalance().unwrap();
    assert!(result.contains("degraded"), "expected temporal degradation: {result}");

    // Check that temporal_shift increased
    let results = tree.search(&emb, 1).unwrap();
    // 60 days old → shift 12 → day precision, no time
    assert!(!results[0].when.contains(':'), "should lose time precision: {}", results[0].when);
}

#[test]
fn frequently_accessed_old_memory_still_loses_temporal_precision() {
    let tree = test_tree(8);
    let emb = make_embedding(8, 0);
    tree.store("daily password", &emb, None).unwrap();

    // Backdate to 2 years ago but set high access count
    let old_date = (chrono::Utc::now() - chrono::Duration::days(730)).to_rfc3339();
    tree.conn_exec(
        &format!("UPDATE entries SET created_at = '{old_date}', access_count = 1000, last_accessed = '{}'",
            chrono::Utc::now().to_rfc3339()),
    ).unwrap();

    tree.rebalance().unwrap();

    let results = tree.search(&emb, 1).unwrap();

    // Content should still rank high (access boost)
    assert!(results[0].similarity > 0.5, "accessed memory should rank high");

    // But temporal precision should be degraded (age-based, not access-based)
    assert!(
        results[0].when.len() <= 4,
        "2-year old memory should have year-only precision even if accessed daily: {}",
        results[0].when
    );
}

#[test]
fn same_period_entries_get_identical_temporal_epoch_after_shift() {
    use engram::birch::temporal_shift_for_age;

    // Two entries 30 seconds apart
    let epoch_a: i64 = 1780000000;
    let epoch_b: i64 = 1780000030;

    // At shift 8 (~4 min blocks), they should be identical
    let shift = temporal_shift_for_age(15.0); // 15 days → shift 8
    assert_eq!(epoch_a >> shift, epoch_b >> shift, "same 4-min block should match after shift");

    // Two entries 2 hours apart
    let epoch_c: i64 = 1780007200;

    // At shift 12 (~1 hour blocks), they should be different
    let shift12 = temporal_shift_for_age(60.0); // 60 days → shift 12
    assert_ne!(epoch_a >> shift12, epoch_c >> shift12, "2 hours apart should differ at hour blocks");

    // At shift 16 (~18 hour blocks), they should be identical
    let shift16 = temporal_shift_for_age(200.0); // 200 days → shift 16
    assert_eq!(epoch_a >> shift16, epoch_c >> shift16, "2 hours apart should merge at day blocks");
}

// --- Tree-guided beam search ---

#[test]
fn gap_detection_finds_correct_cluster() {
    // Many orthogonal clusters — gap detection should narrow to the right one
    let config = Config {
        threshold: 0.8,
        leaf_capacity: 50,
        branch_factor: 10,
    };
    let tree = Tree::open(":memory:", 8, config).expect("open tree");

    // Create distinct clusters across orthogonal dimensions
    for i in 0..8 {
        let emb = make_embedding(8, i);
        tree.store(&format!("cluster {i} entry"), &emb, None).unwrap();
    }

    // Search for cluster 3 — gap detection should find the clear winner
    let query = make_embedding(8, 3);
    let results = tree.search(&query, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].content, "cluster 3 entry");
}

#[test]
fn beam_search_degrades_to_full_scan_for_small_trees() {
    // With beam_width=10 and only 3 clusters, all are explored
    let tree = test_tree(8);
    tree.store("alpha", &make_embedding(8, 0), None).unwrap();
    tree.store("beta", &make_embedding(8, 3), None).unwrap();
    tree.store("gamma", &make_embedding(8, 6), None).unwrap();

    // Should find all 3 regardless of query
    let results = tree.search(&make_embedding(8, 0), 3).unwrap();
    assert_eq!(results.len(), 3);
}
