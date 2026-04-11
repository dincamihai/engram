//! Animated BIRCH tree visualization — Conway-style x-ray of memory.
//!
//! Uses dotmax for braille canvas rendering, ascii-petgraph for force-directed
//! layout, and ratatui for terminal management.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::{Duration, Instant};

use ascii_petgraph::physics::{PhysicsConfig, PhysicsEngine, Vec2};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use dotmax::grid::Color as DotColor;
// draw_line_colored no longer needed — edges drawn per-pixel for pulse effect
use dotmax::BrailleGrid;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::Terminal;

use crate::birch;

/// Node data in the visualization graph.
#[derive(Debug, Clone)]
struct VizNode {
    id: i64,
    label: String,
    count: i64,
    depth: i32,
    #[allow(dead_code)]
    access: i64,
    /// Freshness: 1.0 = just touched, 0.0 = very old. Based on most recent entry age.
    freshness: f32,
    /// Number of consolidated (merged) entries in this node.
    consolidated: i64,
    /// Number of never-accessed entries (unproven knowledge).
    unproven: i64,
    /// 2D position from embedding projection (semantic map).
    semantic_x: f64,
    semantic_y: f64,
}

/// Animation state for a node.
#[derive(Debug, Clone)]
enum NodeAnim {
    /// Node was added — green color fading to normal.
    Growing { progress: f32 },
    /// Node lost entries — red color fading out.
    Shrinking { progress: f32 },
    /// Branch vibration — node shakes slightly (reserved for future use).
    #[allow(dead_code)]
    Vibrating { progress: f32 },
    /// Normal, stable.
    Stable,
}

const ANIM_SPEED: f32 = 0.005; // progress per frame (~6.7s at 30fps for full cycle) — slower for visibility
const VIBRATE_AMPLITUDE: f64 = 5.0; // pixels of shake — increased for prominence

/// State for the visualization.
struct VizState {
    graph: DiGraph<VizNode, ()>,
    physics: PhysicsEngine,
    grid: BrailleGrid,
    zoom: f64,
    pan_x: f64,
    pan_y: f64,
    show_labels: bool,
    show_edges: bool,
    paused: bool,
    grid_w: usize,
    grid_h: usize,
    /// Per-node animation state (keyed by petgraph NodeIndex).
    animations: HashMap<NodeIndex, NodeAnim>,
    /// Last known set of BIRCH node IDs (for polling diff).
    known_ids: std::collections::HashSet<i64>,
    /// When we last polled the DB for changes.
    last_poll: Instant,
    /// Global frame counter for continuous animations (orbits, breathing).
    frame: u64,
    /// Particle trails: recent pixel positions per node for comet effect.
    trails: HashMap<NodeIndex, VecDeque<(usize, usize)>>,
    /// Cursor position: index into the cluster list (for label highlight).
    cursor: usize,
    /// Number of rows in the current multi-column layout (for cursor navigation).
    layout_rows: usize,
    /// Number of columns in the current multi-column layout.
    layout_cols: usize,
    /// Cached preview for the selected cluster: (cursor_idx, label, summary line, entry previews).
    preview_cache: Option<(usize, String, String, Vec<String>)>,
    /// When the cursor last moved (for delayed summarization).
    cursor_moved_at: Instant,
    /// The cursor index when it last moved.
    cursor_at_move: usize,
    /// Background summarizer: sends (cursor_idx, summary) when ready.
    summary_rx: Option<std::sync::mpsc::Receiver<(usize, String)>>,
    /// Sender for background summarizer thread.
    summary_tx: Option<std::sync::mpsc::Sender<(usize, i64, String, String)>>,
    /// Whether the summarizer background thread is running.
    summarizer_running: bool,
    /// Whether the summarizer model has finished loading.
    summarizer_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub fn run_viz(db_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Open tree and build initial state
    let tree = birch::Tree::open(db_path, crate::embed::DIMENSION, birch::Config::default())
        .map_err(|e| format!("open tree: {e}"))?;
    let size = terminal.size()?;
    let state = build_state(&tree, size.width as usize, size.height as usize)?;

    // Run the event loop with polling
    let result = run_loop(&mut terminal, state, db_path);

    // Restore terminal
    terminal::disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;

    result
}

fn build_state(tree: &birch::Tree, width: usize, height: usize) -> Result<VizState, Box<dyn std::error::Error>> {
    let nodes = tree.all_nodes().map_err(|e| format!("all_nodes: {e}"))?;

    if nodes.is_empty() {
        return Err("No nodes in tree".into());
    }

    // Build petgraph
    let mut graph: DiGraph<VizNode, ()> = DiGraph::new();
    let mut node_map: Vec<(i64, petgraph::graph::NodeIndex)> = Vec::new();

    // PCA-like 2D projection of centroids:
    // 1. Compute mean centroid (only nodes with correct dimension)
    let dim = crate::embed::DIMENSION;
    let valid_nodes: Vec<&_> = nodes.iter().filter(|n| n.centroid.len() == dim).collect();
    let n = valid_nodes.len().max(1) as f64;
    let mut mean = vec![0.0f64; dim];
    for node in &valid_nodes {
        for (i, v) in node.centroid.iter().enumerate() {
            mean[i] += *v as f64 / n;
        }
    }

    // 2. Find the two dimensions with highest variance
    let mut variance = vec![0.0f64; dim];
    for node in &valid_nodes {
        for (i, v) in node.centroid.iter().enumerate() {
            let d = *v as f64 - mean[i];
            variance[i] += d * d / n;
        }
    }
    let mut dim_indices: Vec<usize> = (0..dim).collect();
    dim_indices.sort_by(|&a, &b| variance[b].partial_cmp(&variance[a]).unwrap_or(std::cmp::Ordering::Equal));
    let dim_x = dim_indices[0];
    let dim_y = if dim_indices.len() > 1 { dim_indices[1] } else { 0 };

    // 3. Project to polar: angle from PCA (semantic direction), radius from freshness
    //    Center = fresh/active, edge = aging/fading
    let projections: Vec<(f64, f64)> = nodes.iter()
        .map(|node| {
            let px = if dim_x < node.centroid.len() { node.centroid[dim_x] as f64 - mean[dim_x] } else { 0.0 };
            let py = if dim_y < node.centroid.len() { node.centroid[dim_y] as f64 - mean[dim_y] } else { 0.0 };
            // Angle = semantic direction (preserved from PCA)
            let angle = py.atan2(px);
            // Freshness for radius: compute here (same formula as VizNode)
            let age_hours = tree.node_freshness_hours(node.id)
                .unwrap_or(None)
                .unwrap_or(720.0);
            let freshness = 1.0 / (1.0 + age_hours / 72.0);
            // Radius: fresh = outer edge (new arrival), old+proven = center (core knowledge)
            // Fresh memories start at the periphery and migrate inward as they get accessed
            let radius = 0.2 + 0.8 * freshness; // fresh(1.0) = outer(1.0), old(0.0) = core(0.2)
            (angle.cos() * radius, angle.sin() * radius)
        })
        .collect();

    // Add nodes with freshness + semantic position
    for (i, node) in nodes.iter().enumerate() {
        let access = tree.node_total_access(node.id).unwrap_or(0);
        let age_hours = tree.node_freshness_hours(node.id)
            .unwrap_or(None)
            .unwrap_or(720.0);
        let freshness = (1.0 / (1.0 + age_hours / 72.0)) as f32;

        let consolidated = tree.node_consolidated_count(node.id).unwrap_or(0);
        let unproven = tree.node_unproven_count(node.id).unwrap_or(0);
        let (sx, sy) = projections[i];
        let viz_node = VizNode {
            id: node.id,
            label: node.label.clone(),
            count: node.count,
            depth: node.depth,
            access,
            freshness,
            consolidated,
            unproven,
            semantic_x: sx,
            semantic_y: sy,
        };
        let idx = graph.add_node(viz_node);
        node_map.push((node.id, idx));
    }

    // Add edges (parent -> child)
    for node in &nodes {
        if let Some(parent_id) = node.parent_id {
            let parent_idx = node_map.iter().find(|(id, _)| *id == parent_id).map(|(_, idx)| *idx);
            let child_idx = node_map.iter().find(|(id, _)| *id == node.id).map(|(_, idx)| *idx);
            if let (Some(p), Some(c)) = (parent_idx, child_idx) {
                graph.add_edge(p, c, ());
            }
        }
    }

    // Setup physics engine (still needed for the data structure, but positions are semantic)
    let physics_config = PhysicsConfig {
        spring_constant: 0.0,
        spring_length: 0.0,
        repulsion_constant: 0.0,
        gravity: 0.0,
        damping: 0.99,
        dt: 1.0,
        velocity_threshold: 0.01,
        max_iterations: 0,
    };
    let mut physics = PhysicsEngine::new(&graph, physics_config);

    // Seed positions from semantic projection — deterministic, no simulation needed
    for node_idx in graph.node_indices() {
        let node = &graph[node_idx];
        let i = node_idx.index();
        if i < physics.nodes.len() {
            physics.nodes[i].position = ascii_petgraph::physics::Vec2::new(
                node.semantic_x * 100.0, // scale up for screen space
                node.semantic_y * 100.0,
            );
            physics.nodes[i].velocity = ascii_petgraph::physics::Vec2::new(0.0, 0.0);
        }
    }
    physics.normalize_positions();

    let mut grid = BrailleGrid::new(width, height)?;
    grid.enable_color_support();

    // All initial nodes are stable
    let animations: HashMap<NodeIndex, NodeAnim> = graph.node_indices()
        .map(|idx| (idx, NodeAnim::Stable))
        .collect();
    let known_ids: std::collections::HashSet<i64> = nodes.iter().map(|n| n.id).collect();

    Ok(VizState {
        graph,
        physics,
        grid,
        zoom: 0.7,
        pan_x: 0.0,
        pan_y: 0.0,
        show_labels: false,
        show_edges: false,
        paused: false,
        grid_w: width,
        grid_h: height,
        animations,
        known_ids,
        last_poll: Instant::now(),
        frame: 0,
        trails: HashMap::new(),
        cursor: 0,
        layout_rows: 1,
        layout_cols: 1,
        preview_cache: None,
        summary_rx: None,
        summary_tx: None,
        summarizer_running: false,
        summarizer_ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        cursor_moved_at: Instant::now(),
        cursor_at_move: 0,
    })
}

/// Poll the DB for changes and update the graph.
/// Detects: new nodes (coagulate), removed nodes (disperse), changed nodes (pulse).
fn poll_changes(state: &mut VizState, db_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tree = birch::Tree::open(db_path, crate::embed::DIMENSION, birch::Config::default())
        .map_err(|e| format!("reopen tree: {e}"))?;
    let nodes = tree.all_nodes().map_err(|e| format!("all_nodes: {e}"))?;
    let current_ids: std::collections::HashSet<i64> = nodes.iter().map(|n| n.id).collect();

    // Detect structural changes (splits, merges, new/removed nodes)
    let has_new = current_ids.iter().any(|id| !state.known_ids.contains(id));
    let has_removed = state.known_ids.iter().any(|id| !current_ids.contains(id));

    if has_new || has_removed {
        // Structural change — full rebuild to get correct PCA, freshness, counts
        let new_state = build_state(&tree, state.grid_w, state.grid_h)?;

        // Detect which nodes grew/shrank for animations
        let old_counts: std::collections::HashMap<i64, i64> = state.graph.node_indices()
            .map(|idx| (state.graph[idx].id, state.graph[idx].count))
            .collect();

        state.graph = new_state.graph;
        state.known_ids = new_state.known_ids;
        state.physics = new_state.physics;

        // Trigger animations for changed/new nodes
        for idx in state.graph.node_indices() {
            let node = &state.graph[idx];
            match old_counts.get(&node.id) {
                Some(&old_count) if old_count < node.count => {
                    state.animations.insert(idx, NodeAnim::Growing { progress: 0.0 });
                }
                Some(&old_count) if old_count > node.count => {
                    state.animations.insert(idx, NodeAnim::Shrinking { progress: 0.0 });
                }
                None => {
                    // Brand new node
                    state.animations.insert(idx, NodeAnim::Growing { progress: 0.0 });
                }
                _ => {}
            }
        }

        return Ok(());
    }

    // No structural change — lightweight update of counts, labels, freshness
    for node_idx in state.graph.node_indices() {
        let viz_id = state.graph[node_idx].id;
        if let Some(db_node) = nodes.iter().find(|n| n.id == viz_id) {
            let old_count = state.graph[node_idx].count;
            let new_count = db_node.count;
            if old_count != new_count {
                state.graph[node_idx].count = new_count;
                if new_count > old_count {
                    state.animations.insert(node_idx, NodeAnim::Growing { progress: 0.0 });
                } else {
                    state.animations.insert(node_idx, NodeAnim::Shrinking { progress: 0.0 });
                }
            }
            if state.graph[node_idx].label != db_node.label {
                state.graph[node_idx].label = db_node.label.clone();
            }
            // Update freshness, consolidated, unproven
            let age_hours = tree.node_freshness_hours(db_node.id)
                .unwrap_or(None)
                .unwrap_or(720.0);
            state.graph[node_idx].freshness = (1.0 / (1.0 + age_hours / 72.0)) as f32;
            state.graph[node_idx].consolidated = tree.node_consolidated_count(db_node.id).unwrap_or(0);
            state.graph[node_idx].unproven = tree.node_unproven_count(db_node.id).unwrap_or(0);
        }
    }
    state.known_ids = current_ids;

    Ok(())
}

/// Remove fully shrunk nodes from the graph.
fn remove_dispersed(state: &mut VizState) {
    let to_remove: Vec<NodeIndex> = state
        .animations
        .iter()
        .filter_map(|(&idx, anim)| {
            if let NodeAnim::Shrinking { progress } = anim {
                if *progress >= 1.0 { Some(idx) } else { None }
            } else {
                None
            }
        })
        .collect();

    if to_remove.is_empty() {
        return;
    }

    for idx in &to_remove {
        state.graph.remove_node(*idx);
        state.animations.remove(idx);
    }

    // Rebuild physics
    let config = state.physics.config.clone();
    state.physics = PhysicsEngine::new(&state.graph, config);
    for _ in 0..5 {
        state.physics.tick(&state.graph);
    }
    state.physics.normalize_positions();
}

/// Vibrate the branch (parent and siblings) of a changed node.
/// Only applies Vibrating to nodes that are currently Stable (doesn't override Growing/Shrinking).

/// Advance animation progress by one frame.
fn tick_animations(state: &mut VizState) {
    for anim in state.animations.values_mut() {
        match anim {
            NodeAnim::Growing { progress } => {
                *progress += ANIM_SPEED;
                if *progress >= 1.0 {
                    *anim = NodeAnim::Stable;
                }
            }
            NodeAnim::Shrinking { progress } => {
                *progress += ANIM_SPEED;
                if *progress >= 1.0 {
                    *anim = NodeAnim::Stable;
                }
            }
            NodeAnim::Vibrating { progress } => {
                *progress += ANIM_SPEED * 1.5;
                if *progress >= 1.0 {
                    *anim = NodeAnim::Stable;
                }
            }
            NodeAnim::Stable => {}
        }
    }
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut state: VizState,
    db_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Start background summarizer thread immediately (downloads model while user browses)
    {
        let (request_tx, request_rx) = std::sync::mpsc::channel::<(usize, i64, String, String)>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<(usize, String)>();
        state.summary_tx = Some(request_tx);
        state.summary_rx = Some(result_rx);
        state.summarizer_running = true;

        let ready_flag = state.summarizer_ready.clone();
        std::thread::spawn(move || {
            // hf-hub may print download progress — redirect only stderr in this thread
            let saved_stderr = {
                use std::os::unix::io::AsRawFd;
                std::fs::File::open("/dev/null").ok().map(|f| {
                    let saved = unsafe { libc::dup(2) };
                    unsafe { libc::dup2(f.as_raw_fd(), 2); }
                    saved
                })
            };
            let summarizer = crate::extract::Summarizer::new().ok();
            if let Some(saved) = saved_stderr {
                unsafe { libc::dup2(saved, 2); libc::close(saved); }
            }
            ready_flag.store(summarizer.is_some(), std::sync::atomic::Ordering::Relaxed);
            while let Ok((idx, nid, db, lbl)) = request_rx.recv() {
                if let Some(ref s) = summarizer {
                    let text = match birch::Tree::open(&db, crate::embed::DIMENSION, birch::Config::default()) {
                        Ok(tree) => match tree.node_entries(nid, 300) {
                            Ok(entries) => {
                                let combined: String = entries.iter()
                                    .map(|e| e.content.split_whitespace().collect::<Vec<_>>().join(" "))
                                    .collect::<Vec<_>>().join(". ");
                                let input: String = combined.chars().take(400).collect();
                                match s.summarize(&input, 40) {
                                    Ok(s) if !s.trim().is_empty() => s,
                                    Ok(_) => format!("[empty summary] {}", lbl),
                                    Err(e) => format!("[summarize err: {}]", e),
                                }
                            }
                            Err(e) => format!("[entries err: {}]", e),
                        },
                        Err(e) => format!("[tree err: {}]", e),
                    };
                    let _ = result_tx.send((idx, text));
                } else {
                    let _ = result_tx.send((idx, lbl.replace('-', " ")));
                }
            }
        });
    }

    loop {
        // Advance animations
        tick_animations(&mut state);

        // Remove fully dispersed nodes
        remove_dispersed(&mut state);

        // No physics simulation — positions are semantic (deterministic from embeddings)

        // Poll DB for changes every 2 seconds
        if state.last_poll.elapsed() >= Duration::from_secs(1) {
            if let Err(e) = poll_changes(&mut state, db_path) {
                // Don't crash on poll errors, just log
                eprintln!("[engram viz] poll error: {e}");
            }
            state.last_poll = Instant::now();
        }

        // Advance frame counter
        state.frame = state.frame.wrapping_add(1);

        // Render
        terminal.draw(|frame| {
            render_frame(frame, &mut state, db_path);
        })?;

        // Handle events
        if event::poll(Duration::from_millis(33))? {
            match event::read()? {
                Event::Resize(_, _) => {
                    // Let ratatui recalculate its area; grid resize handled below
                    let _ = terminal.size();
                }
                Event::Key(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                        KeyCode::Char('r') => {
                            let config = state.physics.config.clone();
                            state.physics = PhysicsEngine::new(&state.graph, config);
                            state.physics.normalize_positions();
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            state.zoom *= 1.2;
                        }
                        KeyCode::Char('-') => {
                            state.zoom /= 1.2;
                        }
                        KeyCode::Left => {
                            let old = state.cursor;
                            if state.cursor > 0 { state.cursor -= 1; }
                            if state.cursor != old { state.cursor_moved_at = Instant::now(); state.cursor_at_move = state.cursor; }
                        }
                        KeyCode::Right => {
                            let old = state.cursor;
                            state.cursor += 1; // clamped in render
                            if state.cursor != old { state.cursor_moved_at = Instant::now(); state.cursor_at_move = state.cursor; }
                        }
                        KeyCode::Up => {
                            let old = state.cursor;
                            if state.cursor >= state.layout_cols {
                                state.cursor -= state.layout_cols;
                            }
                            if state.cursor != old { state.cursor_moved_at = Instant::now(); state.cursor_at_move = state.cursor; }
                        }
                        KeyCode::Down => {
                            let old = state.cursor;
                            state.cursor += state.layout_cols; // clamped in render
                            if state.cursor != old { state.cursor_moved_at = Instant::now(); state.cursor_at_move = state.cursor; }
                        }
                        KeyCode::Char('l') => state.show_labels = !state.show_labels,
                        KeyCode::Char('e') => state.show_edges = !state.show_edges,
                        KeyCode::Char(' ') => state.paused = !state.paused,
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Resize grid if terminal size changed
        let size = terminal.size().unwrap_or(ratatui::layout::Size::new(state.grid_w as u16, state.grid_h as u16));
        let new_w = size.width as usize;
        let new_h = size.height as usize;
        if new_w != state.grid_w || new_h != state.grid_h {
            if new_w > 4 && new_h > 4 {
                if let Ok(mut new_grid) = BrailleGrid::new(new_w, new_h) {
                    new_grid.enable_color_support();
                    state.grid = new_grid;
                    state.grid_w = new_w;
                    state.grid_h = new_h;
                }
            }
        }
    }
}

fn render_frame(frame: &mut ratatui::Frame, state: &mut VizState, db_path: &str) {
    let area = frame.area();
    let w = area.width as usize;
    let preview_lines = 2_usize; // header + summary line
    let h = area.height.saturating_sub(preview_lines as u16) as usize;

    // Collect all entries as blocks from leaf nodes, grouped by cluster
    struct Block { freshness: f32, is_unproven: bool, is_consolidated: bool, anim_color: Option<(u8, u8, u8)>, seed: f64 }
    let mut blocks: Vec<Block> = Vec::new();
    let mut cluster_boundaries: Vec<usize> = Vec::new(); // index where each cluster starts
    let mut total_entries: usize = 0;

    let leaf_nodes: std::collections::HashSet<NodeIndex> = state.graph.node_indices()
        .filter(|&idx| state.graph.neighbors_directed(idx, petgraph::Direction::Outgoing).next().is_none())
        .collect();

    let mut leaf_list: Vec<NodeIndex> = leaf_nodes.iter().copied().collect();
    leaf_list.sort_by_key(|&idx| (state.graph[idx].depth, state.graph[idx].id));

    for &node_idx in &leaf_list {
        let node = &state.graph[node_idx];
        if node.count <= 0 { continue; }
        let proven = (node.count - node.consolidated - node.unproven).max(0) as usize;
        let unproven = node.unproven as usize;
        let consol = node.consolidated as usize;

        let anim_color = match state.animations.get(&node_idx) {
            Some(NodeAnim::Growing { progress }) => {
                let p = 0.5 + 0.5 * ((progress * std::f32::consts::PI * 3.0).sin());
                Some(((80.0 * p) as u8, (255.0 * p.max(0.5)) as u8, (60.0 * p) as u8))
            }
            Some(NodeAnim::Shrinking { progress }) => {
                let p = 0.5 + 0.5 * ((progress * std::f32::consts::PI * 3.0).sin());
                Some(((255.0 * p) as u8, (40.0 * (1.0 - progress)) as u8, (30.0 * (1.0 - progress)) as u8))
            }
            _ => None,
        };

        cluster_boundaries.push(blocks.len());
        for i in 0..proven {
            blocks.push(Block { freshness: node.freshness, is_unproven: false, is_consolidated: false, anim_color, seed: i as f64 * 7.3 + node.id as f64 * 3.1 });
        }
        for i in 0..unproven {
            blocks.push(Block { freshness: node.freshness, is_unproven: true, is_consolidated: false, anim_color, seed: (proven + i) as f64 * 7.3 + node.id as f64 * 3.1 });
        }
        for i in 0..consol {
            blocks.push(Block { freshness: node.freshness, is_unproven: false, is_consolidated: true, anim_color, seed: (proven + unproven + i) as f64 * 7.3 + node.id as f64 * 3.1 });
        }
        total_entries += proven + unproven + consol;
    }
    // Add a gap block between clusters
    let cluster_starts: std::collections::HashSet<usize> = cluster_boundaries.into_iter().collect();

    // Freshness to RGB
    let freshness_rgb = |f: f32| -> (u8, u8, u8) {
        let f = f.clamp(0.0, 1.0);
        (
            (100.0 + 155.0 * f) as u8,
            (160.0 + 70.0 * f * f) as u8,
            (220.0 - 80.0 * f) as u8,
        )
    };

    // Render: compact multi-column layout — no labels, cursor selects

    // Group blocks by cluster — (label, birch_node_id, blocks)
    let mut cluster_blocks: Vec<(String, i64, Vec<&Block>)> = Vec::new();
    let mut bi = 0;
    for &node_idx in &leaf_list {
        let node = &state.graph[node_idx];
        if node.count <= 0 { continue; }
        let count = (node.count) as usize;
        let label = node.label.clone();
        let cluster: Vec<&Block> = blocks[bi..bi + count.min(blocks.len() - bi)].iter().collect();
        bi += cluster.len();
        cluster_blocks.push((label, node.id, cluster));
    }

    let num_clusters = cluster_blocks.len();

    // Clamp cursor
    if num_clusters > 0 {
        state.cursor = state.cursor.min(num_clusters - 1);
    }
    let cursor_idx = state.cursor;

    // Brain scan: one solid █ block, each cell colored by which cluster owns it.
    // Clusters are laid out contiguously — each gets N cells proportional to entry count.
    // Baseline is dim, activity (anim) lights up regions, cursor highlights a region.

    // Total cells in the solid block
    let total_cells: usize = cluster_blocks.iter().map(|(_, _, c)| c.len()).sum();

    // Target a roughly 2:1 width:height rectangle (chars are ~2x tall)
    // Aspect ratio ~3:1 width:height so it looks square (chars are ~2x tall)
    let block_w = ((total_cells as f64 * 3.0).sqrt() as usize)
        .clamp(10, w.saturating_sub(4));
    let block_h = ((total_cells + block_w - 1) / block_w).max(1);

    // Center on screen
    let pad_x = w.saturating_sub(block_w) / 2;
    let pad_y = h.saturating_sub(block_h) / 2;

    // Build a flat cell map: for each cell, which cluster index owns it
    let mut cell_cluster: Vec<usize> = Vec::with_capacity(total_cells);
    let mut cell_entry: Vec<usize> = Vec::with_capacity(total_cells); // entry index within cluster
    for (idx, (_, _, cluster)) in cluster_blocks.iter().enumerate() {
        for ei in 0..cluster.len() {
            cell_cluster.push(idx);
            cell_entry.push(ei);
        }
    }

    // For cursor nav: estimate grid columns
    state.layout_cols = block_w;
    state.layout_rows = (num_clusters + block_w - 1) / block_w;

    // Precompute animation epicenters: center (row, col) and color/intensity for each active cluster
    struct Ripple { center_row: f64, center_col: f64, r: u8, g: u8, b: u8, progress: f32 }
    let mut ripples: Vec<Ripple> = Vec::new();
    {
        let mut cell_offset = 0usize;
        for (idx, (_, _, cluster)) in cluster_blocks.iter().enumerate() {
            let count = cluster.len();
            if count == 0 { cell_offset += count; continue; }
            // Check if this cluster has an active animation
            if let Some((ar, ag, ab)) = cluster.first().and_then(|b| b.anim_color) {
                // Find center cell of this cluster's region
                let mid = cell_offset + count / 2;
                let center_row = (mid / block_w) as f64;
                let center_col = (mid % block_w) as f64;
                // Get animation progress (0.0 = just started, 1.0 = done)
                let progress = match cluster.first().and_then(|_| {
                    // Find the node animation for this cluster
                    let node_idx = leaf_list.iter()
                        .filter(|&&ni| state.graph[ni].count > 0)
                        .nth(idx);
                    node_idx.and_then(|&ni| state.animations.get(&ni))
                }) {
                    Some(NodeAnim::Growing { progress }) => *progress,
                    Some(NodeAnim::Shrinking { progress }) => *progress,
                    _ => 1.0,
                };
                ripples.push(Ripple { center_row, center_col, r: ar, g: ag, b: ab, progress });
            }
            cell_offset += count;
        }
    }

    // Baseline dim color
    let base_r = 30u8;
    let base_g = 35u8;
    let base_bl = 50u8;

    let mut lines: Vec<ratatui::text::Line> = Vec::with_capacity(h);

    // Top padding
    for _ in 0..pad_y.min(h) {
        lines.push(ratatui::text::Line::from(" ".repeat(w)));
    }

    for row in 0..block_h {
        if lines.len() >= h { break; }
        let mut spans: Vec<Span> = Vec::with_capacity(w);

        // Left padding
        if pad_x > 0 {
            spans.push(Span::raw(" ".repeat(pad_x)));
        }

        for col in 0..block_w {
            let cell_idx = row * block_w + col;
            if cell_idx < total_cells {
                let cidx = cell_cluster[cell_idx];
                let b = &cluster_blocks[cidx].2[cell_entry[cell_idx]];
                let is_cursor = cidx == cursor_idx;

                // Start with baseline + subtle freshness tint
                let (fr, fg, fbl) = freshness_rgb(b.freshness);
                let mix = 0.15;
                let mut r = (base_r as f64 * (1.0 - mix) + fr as f64 * mix) as f64;
                let mut g = (base_g as f64 * (1.0 - mix) + fg as f64 * mix) as f64;
                let mut bl = (base_bl as f64 * (1.0 - mix) + fbl as f64 * mix) as f64;

                // Apply ripples from all active animations
                for ripple in &ripples {
                    let dr = row as f64 - ripple.center_row;
                    let dc = col as f64 - ripple.center_col;
                    let dist = (dr * dr + dc * dc).sqrt();

                    // Ripple ring: expanding wavefront based on progress
                    // Wavefront radius grows with progress, ring width is ~3 cells
                    let max_radius = 20.0;
                    let wavefront = ripple.progress as f64 * max_radius;
                    let ring_dist = (dist - wavefront).abs();
                    let ring_width = 3.0;

                    if ring_dist < ring_width {
                        // Intensity: strongest at wavefront, fades with distance from ring
                        let ring_intensity = 1.0 - ring_dist / ring_width;
                        // Also fade out as animation progresses
                        let fade = 1.0 - ripple.progress as f64;
                        let intensity = ring_intensity * fade;

                        // Blend ripple color
                        r = r + (ripple.r as f64 - r) * intensity;
                        g = g + (ripple.g as f64 - g) * intensity;
                        bl = bl + (ripple.b as f64 - bl) * intensity;
                    }
                }

                // Cursor highlight
                if is_cursor {
                    let (cr, cg, cbl) = freshness_rgb(b.freshness);
                    r = r + (cr as f64 - r) * 0.6;
                    g = g + (cg as f64 - g) * 0.6;
                    bl = bl + (cbl as f64 - bl) * 0.6;
                }

                // Subtle breathing
                let breath = 1.0 + 0.03 * (state.frame as f64 * 0.02 + cell_idx as f64 * 0.01).sin();
                r = (r * breath).min(255.0);
                g = (g * breath).min(255.0);
                bl = (bl * breath).min(255.0);

                spans.push(Span::styled("█", Style::default().fg(Color::Rgb(r as u8, g as u8, bl as u8))));
            } else {
                spans.push(Span::raw(" "));
            }
        }

        // Right padding
        let right_pad = w.saturating_sub(pad_x + block_w);
        if right_pad > 0 {
            spans.push(Span::raw(" ".repeat(right_pad)));
        }

        lines.push(ratatui::text::Line::from(spans));
    }

    // Bottom padding
    while lines.len() < h {
        lines.push(ratatui::text::Line::from(" ".repeat(w)));
    }

    // Render bars area
    let bars_area = Rect::new(0, 0, area.width, h as u16);
    frame.render_widget(ratatui::text::Text::from(lines), bars_area);

    // Preview panel — summarize selected cluster with NLI pipeline
    if cursor_idx < cluster_blocks.len() {
        let (ref label, node_id, ref cluster) = cluster_blocks[cursor_idx];
        let entry_count = cluster.len();

        // Update preview cache if cursor changed or summary not yet generated
        let has_cache = state.preview_cache.as_ref().map(|(idx, _, _, _)| *idx == cursor_idx).unwrap_or(false);
        let dwell_time = state.cursor_moved_at.elapsed();
        let dwell_threshold = Duration::from_millis(800);
        let has_summary = state.preview_cache.as_ref()
            .map(|(idx, _, summary, _)| *idx == cursor_idx && !summary.is_empty() && summary != "<requested>")
            .unwrap_or(false);

        // Phase 1: immediately load entry previews on cursor change
        if !has_cache {
            let mut entry_previews = Vec::new();
            if let Ok(tree) = birch::Tree::open(db_path, crate::embed::DIMENSION, birch::Config::default()) {
                if let Ok(entries) = tree.node_entries(node_id, 300) {
                    for e in entries.iter().take(3) {
                        let oneline: String = e.content.split_whitespace().collect::<Vec<_>>().join(" ");
                        entry_previews.push(oneline);
                    }
                }
            }
            // Empty summary — will be filled when dwell threshold is met
            state.preview_cache = Some((cursor_idx, label.clone(), String::new(), entry_previews));
        }

        // Phase 2: send summary request to background thread after dwell (only once per cursor pos)
        let model_ready = state.summarizer_ready.load(std::sync::atomic::Ordering::Relaxed);
        let already_requested = state.preview_cache.as_ref()
            .map(|(idx, _, s, _)| *idx == cursor_idx && s == "<requested>")
            .unwrap_or(false);
        if !has_summary && !already_requested && dwell_time >= dwell_threshold && model_ready {
            if let Some(ref tx) = state.summary_tx {
                let _ = tx.send((cursor_idx, node_id, db_path.to_string(), label.clone()));
                // Mark as requested
                if let Some(ref mut cache) = state.preview_cache {
                    if cache.0 == cursor_idx {
                        cache.2 = "<requested>".to_string();
                    }
                }
            }
        }

        // Check for completed summaries from background thread
        // Only apply if the result matches the CURRENT cursor position
        if let Some(ref rx) = state.summary_rx {
            while let Ok((idx, summary)) = rx.try_recv() {
                if idx == cursor_idx {
                    if let Some(ref mut cache) = state.preview_cache {
                        if cache.0 == idx {
                            cache.2 = summary;
                        }
                    }
                }
                // else: stale result for a different cursor position — discard
            }
        }

        let cached = state.preview_cache.as_ref().unwrap();
        let cached_label = &cached.1;
        let summary = &cached.2;
        let previews = &cached.3;

        // Render preview lines
        let preview_y = area.height.saturating_sub(preview_lines as u16);

        // Line 1: header
        let header = format!(
            " ▸ {} ({} entries)  ─  entries:{} clusters:{} [q]uit [↑↓←→]nav",
            cached_label, entry_count, total_entries, num_clusters,
        );
        let header_widget = ratatui::widgets::Paragraph::new(
            Span::styled(header, Style::default().fg(Color::White)),
        );
        frame.render_widget(header_widget, Rect::new(0, preview_y, area.width, 1));

        // Line 2: summary (or loading status)
        let model_ready = state.summarizer_ready.load(std::sync::atomic::Ordering::Relaxed);
        let summary_text = if !model_ready {
            "<loading summarizer>".to_string()
        } else if summary.is_empty() || summary == "<requested>" {
            String::new()
        } else {
            summary.clone()
        };
        if !summary_text.is_empty() {
            let truncated: String = summary_text.chars().take(w.saturating_sub(4)).collect();
            let color = if model_ready { Color::Rgb(180, 200, 220) } else { Color::DarkGray };
            let summary_widget = ratatui::widgets::Paragraph::new(
                Span::styled(format!("   {}", truncated), Style::default().fg(color)),
            );
            frame.render_widget(summary_widget, Rect::new(0, preview_y + 1, area.width, 1));
        }
    } else {
        // No clusters — just show basic HUD
        let status = format!(" entries:{} clusters:{} [q]uit", total_entries, num_clusters);
        let bottom = Rect::new(0, area.height.saturating_sub(1), area.width, 1);
        frame.render_widget(
            ratatui::widgets::Paragraph::new(Span::styled(status, Style::default().fg(Color::DarkGray))),
            bottom,
        );
    }

    // Skip old braille rendering entirely
    return;

    // Braille grid pixel dimensions (2 wide x 4 tall per character)
    let grid_pixel_w = (state.grid_w * 2) as f64;
    let grid_pixel_h = (state.grid_h * 4) as f64;

    // Find bounds of node positions
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for node_idx in state.graph.node_indices() {
        let pos = state.physics.position(node_idx);
        min_x = min_x.min(pos.x);
        max_x = max_x.max(pos.x);
        min_y = min_y.min(pos.y);
        max_y = max_y.max(pos.y);
    }

    let range_x = (max_x - min_x).max(1.0);
    let range_y = (max_y - min_y).max(1.0);

    // Scale to fit with padding
    let padding = 10.0;
    let scale_x = (grid_pixel_w - 2.0 * padding) / range_x;
    let scale_y = (grid_pixel_h - 2.0 * padding) / range_y;
    let scale = scale_x.min(scale_y) * state.zoom;

    // Center offset
    let center_x = grid_pixel_w / 2.0 + state.pan_x;
    let center_y = grid_pixel_h / 2.0 + state.pan_y;
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Convert physics position to braille pixel coordinates
    let to_pixel = |pos: ascii_petgraph::physics::Vec2| -> (i32, i32) {
        let px = center_x + (pos.x - mid_x) * scale;
        let py = center_y + (pos.y - mid_y) * scale;
        (px.round() as i32, py.round() as i32)
    };

    // Freshness-based color for leaf nodes (living memories):
    //   freshness 1.0 (just touched):  bright warm white-gold
    //   freshness 0.5 (few days old):  teal/cyan
    //   freshness 0.0 (very old):      dim steel blue
    // Internal/structural nodes: neutral gray
    let freshness_color = |f: f32, is_leaf: bool| -> DotColor {
        if !is_leaf {
            return DotColor::rgb(90, 90, 100); // structural gray
        }
        let f = f.clamp(0.0, 1.0);
        DotColor::rgb(
            (100.0 + 155.0 * f) as u8,           // R: 100 (old) → 255 (fresh)
            (160.0 + 70.0 * f * f) as u8,         // G: 160 (old) → 230 (fresh)
            (220.0 - 80.0 * f) as u8,             // B: 220 (old/cool) → 140 (fresh/warm)
        )
    };

    // Identify leaf nodes (no outgoing edges = no children) — needed for color + rendering
    let leaf_nodes: std::collections::HashSet<NodeIndex> = state.graph.node_indices()
        .filter(|&idx| {
            state.graph.neighbors_directed(idx, petgraph::Direction::Outgoing).next().is_none()
        })
        .collect();

    // Determine node colors based on animation state + freshness
    let node_colors: HashMap<NodeIndex, Option<DotColor>> = state.graph.node_indices()
        .map(|idx| {
            let node = &state.graph[idx];
            let anim = state.animations.get(&idx).unwrap_or(&NodeAnim::Stable);
            let is_leaf = leaf_nodes.contains(&idx);
            let base = freshness_color(node.freshness, is_leaf);
            let color = match anim {
                NodeAnim::Growing { progress } => {
                    // Bright green flash → freshness color
                    let t = *progress;
                    let pulse = 0.5 + 0.5 * ((t * std::f32::consts::PI * 3.0).sin());
                    Some(DotColor::rgb(
                        ((base.r as f32 * t + 80.0 * (1.0 - t)) * pulse) as u8,
                        ((base.g as f32 * t + 255.0 * (1.0 - t)) * pulse.max(0.5)) as u8,
                        ((base.b as f32 * t + 60.0 * (1.0 - t)) * pulse) as u8,
                    ))
                }
                NodeAnim::Shrinking { progress } => {
                    // Red flash → dark
                    let t = *progress;
                    let pulse = 0.5 + 0.5 * ((t * std::f32::consts::PI * 3.0).sin());
                    Some(DotColor::rgb(
                        (255.0 * (1.0 - t * 0.5) * pulse) as u8,
                        (base.g as f32 * (1.0 - t) * 0.3) as u8,
                        (base.b as f32 * (1.0 - t) * 0.2) as u8,
                    ))
                }
                NodeAnim::Vibrating { progress } => {
                    let t = *progress;
                    Some(DotColor::rgb(
                        (base.r as f32 * (0.7 + 0.3 * t)) as u8,
                        (base.g as f32 * (0.7 + 0.3 * t)) as u8,
                        ((base.b as f32).min(220.0) + 35.0 * t) as u8,
                    ))
                }
                NodeAnim::Stable => Some(base),
            };
            (idx, color)
        })
        .collect();

    // Draw edges (hidden by default, toggle with 'e')
    if state.show_edges {
    for edge in state.graph.edge_references() {
        let source_pos = state.physics.position(edge.source());
        let target_pos = state.physics.position(edge.target());
        let (x1, y1) = to_pixel(source_pos);
        let (x2, y2) = to_pixel(target_pos);

        if x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0 { continue; }
        if (x1 as usize) >= state.grid_w * 2 || (y1 as usize) >= state.grid_h * 4 { continue; }
        if (x2 as usize) >= state.grid_w * 2 || (y2 as usize) >= state.grid_h * 4 { continue; }

        let src_leaf = leaf_nodes.contains(&edge.source());
        let tgt_leaf = leaf_nodes.contains(&edge.target());
        let src_color = freshness_color(state.graph[edge.source()].freshness, src_leaf);
        let tgt_color = freshness_color(state.graph[edge.target()].freshness, tgt_leaf);

        // Single-pixel edge with color gradient
        let steps = ((x2 - x1).abs().max((y2 - y1).abs())) as usize;
        let steps = steps.max(1);
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let ex = (x1 as f64 + (x2 - x1) as f64 * t).round() as usize;
            let ey = (y1 as f64 + (y2 - y1) as f64 * t).round() as usize;
            if ex >= state.grid_w * 2 || ey >= state.grid_h * 4 { continue; }
            let edge_color = DotColor::rgb(
                ((src_color.r as f64 * (1.0 - t) + tgt_color.r as f64 * t) * 0.5) as u8,
                ((src_color.g as f64 * (1.0 - t) + tgt_color.g as f64 * t) * 0.5) as u8,
                ((src_color.b as f64 * (1.0 - t) + tgt_color.b as f64 * t) * 0.5) as u8,
            );
            state.grid.set_dot(ex, ey).ok();
            state.grid.set_cell_color(ex / 2, ey / 4, edge_color).ok();
        }
    }
    } // end show_edges

    // Draw particle trails — fading dots behind moving nodes
    let max_trail = 12usize;
    for node_idx in state.graph.node_indices() {
        let pos = state.physics.position(node_idx);
        let (px, py) = to_pixel(pos);
        if px < 0 || py < 0 { continue; }
        let ux = px as usize;
        let uy = py as usize;
        if ux >= state.grid_w * 2 || uy >= state.grid_h * 4 { continue; }

        // Record current position (every 3rd frame to space out trail dots)
        if state.frame % 3 == 0 {
            let trail = state.trails.entry(node_idx).or_insert_with(VecDeque::new);
            // Only record if position actually changed
            let should_record = trail.back().map_or(true, |&(lx, ly)| lx != ux || ly != uy);
            if should_record {
                trail.push_back((ux, uy));
                if trail.len() > max_trail {
                    trail.pop_front();
                }
            }
        }

        // Draw the trail
        let node = &state.graph[node_idx];
        let base_c = freshness_color(node.freshness, leaf_nodes.contains(&node_idx));
        if let Some(trail) = state.trails.get(&node_idx) {
            let trail_len = trail.len();
            for (i, &(tx, ty)) in trail.iter().enumerate() {
                if tx >= state.grid_w * 2 || ty >= state.grid_h * 4 { continue; }
                // Fade: oldest = dimmest, newest = brightest
                let fade = (i as f64 + 1.0) / (trail_len as f64 + 1.0);
                let fade = fade * fade; // quadratic falloff — sharper tail
                let trail_color = DotColor::rgb(
                    (base_c.r as f64 * fade * 0.7) as u8,
                    (base_c.g as f64 * fade * 0.7) as u8,
                    (base_c.b as f64 * fade * 0.7) as u8,
                );
                state.grid.set_dot(tx, ty).ok();
                state.grid.set_cell_color(tx / 2, ty / 4, trail_color).ok();
            }
        }
    }



    // Draw nodes as braille dots — sized by entry count
    for node_idx in state.graph.node_indices() {
        let pos = state.physics.position(node_idx);
        let (mut px, mut py) = to_pixel(pos);
        let node = &state.graph[node_idx];

        // Apply vibration offset if vibrating
        let anim = state.animations.get(&node_idx).unwrap_or(&NodeAnim::Stable);
        if let NodeAnim::Vibrating { progress } = anim {
            let phase = *progress as f64 * 20.0;
            px += (phase.sin() * VIBRATE_AMPLITUDE) as i32;
            py += (phase.cos() * VIBRATE_AMPLITUDE) as i32;
        }

        if px < 0 || py < 0 {
            continue;
        }
        let ux = px as usize;
        let uy = py as usize;
        if ux >= state.grid_w * 2 || uy >= state.grid_h * 4 {
            continue;
        }

        let node_color = node_colors.get(&node_idx).and_then(|c| *c);
        // Animated nodes get a size pulse for extra visibility
        let anim_boost = match anim {
            NodeAnim::Growing { progress } => ((1.0 - *progress) * 4.0) as usize,
            NodeAnim::Shrinking { progress } => ((1.0 - *progress) * 3.0) as usize,
            _ => 0,
        };

        // Scale node size by entry count (log scale): 1 entry = 1px, 10 = 2px, 50 = 3px, etc.
        let count_size = if node.count <= 0 {
            1
        } else {
            1 + ((node.count as f64).ln().max(0.0) / 1.5) as usize
        };
        let base_size = if node.depth == 0 { count_size + 2 } else { count_size };
        let dot_size = base_size + anim_boost;

        // Draw the node core — filled square proportional to count
        for dx in 0..dot_size {
            for dy in 0..dot_size {
                let nx = ux + dx;
                let ny = uy + dy;
                if nx < state.grid_w * 2 && ny < state.grid_h * 4 {
                    state.grid.set_dot(nx, ny).ok();
                    if let Some(color) = node_color {
                        state.grid.set_cell_color(nx / 2, ny / 4, color).ok();
                    }
                }
            }
        }

        // Draw glow halo around nodes — dimmer ring of dots
        let is_leaf = leaf_nodes.contains(&node_idx);
        let base_depth_color = freshness_color(node.freshness, is_leaf);
        let node_breath = 1.0_f64;
        let halo_radius = dot_size + 1;
        let halo_color = DotColor::rgb(
            (base_depth_color.r as f64 * node_breath * 0.5) as u8,
            (base_depth_color.g as f64 * node_breath * 0.5) as u8,
            (base_depth_color.b as f64 * node_breath * 0.5) as u8,
        );
        for a in 0..12 {
            let angle = (a as f64 / 12.0) * std::f64::consts::TAU;
            let hx = (ux as f64 + (dot_size as f64 / 2.0) + halo_radius as f64 * angle.cos()).round() as usize;
            let hy = (uy as f64 + (dot_size as f64 / 2.0) + halo_radius as f64 * angle.sin()).round() as usize;
            if hx < state.grid_w * 2 && hy < state.grid_h * 4 {
                state.grid.set_dot(hx, hy).ok();
                state.grid.set_cell_color(hx / 2, hy / 4, halo_color).ok();
            }
        }

        // Draw entries as a defrag-style rectangular grid of blocks
        // Packed tight, one block per entry. Color = state.
        if is_leaf && node.count > 0 {
            let proven_count = (node.count - node.consolidated - node.unproven).max(0) as usize;
            let unproven_count = node.unproven.min(40) as usize;
            let consol_count = node.consolidated.min(40) as usize;
            let total = (proven_count + unproven_count + consol_count).min(40);

            // Temperature for jitter
            let temperature = node.freshness as f64;
            let jitter_amp = 0.2 + temperature * 1.0;

            // Grid dimensions: roughly square, 2px per block with 1px gap
            let cell = 3; // cell size in braille pixels (2px block + 1px gap)
            let cols = (total as f64).sqrt().ceil() as usize;
            let cols = cols.max(1);

            // Center the grid on the node
            let grid_w = cols * cell;
            let grid_h = ((total + cols - 1) / cols) * cell;
            let start_x = ux.saturating_sub(grid_w / 2);
            let start_y = uy.saturating_sub(grid_h / 2);

            for i in 0..total {
                let is_unproven = i >= proven_count && i < proven_count + unproven_count;
                let is_consolidated = i >= proven_count + unproven_count;

                let col = i % cols;
                let row = i / cols;
                let bx = start_x + col * cell;
                let by = start_y + row * cell;

                // Jitter
                let seed = i as f64 * 7.3 + node.id as f64 * 3.1;
                let jx = (jitter_amp * (state.frame as f64 * 0.06 + seed).sin()
                       * (state.frame as f64 * 0.09 + seed * 1.7).cos()) as i32;
                let jy = (jitter_amp * (state.frame as f64 * 0.07 + seed * 2.3).cos()
                       * (state.frame as f64 * 0.05 + seed * 0.9).sin()) as i32;

                // Color by state
                let block_color = if is_consolidated {
                    // Consolidated = distinct cooler tint
                    DotColor::rgb(
                        (base_depth_color.r as f64 * 0.5) as u8,
                        (base_depth_color.g as f64 * 0.5) as u8,
                        (base_depth_color.b as f64 * 0.7) as u8,
                    )
                } else if is_unproven {
                    // Unproven = dimmer
                    DotColor::rgb(
                        (base_depth_color.r as f64 * 0.45) as u8,
                        (base_depth_color.g as f64 * 0.45) as u8,
                        (base_depth_color.b as f64 * 0.45) as u8,
                    )
                } else {
                    // Proven = full brightness
                    DotColor::rgb(
                        (base_depth_color.r as f64 * 0.9) as u8,
                        (base_depth_color.g as f64 * 0.9) as u8,
                        (base_depth_color.b as f64 * 0.9) as u8,
                    )
                };

                // Draw 2x2 block with jitter
                for dx in 0..2_usize {
                    for dy in 0..2_usize {
                        let px = (bx + dx) as i32 + jx;
                        let py = (by + dy) as i32 + jy;
                        if px >= 0 && py >= 0 {
                            let px = px as usize;
                            let py = py as usize;
                            if px < state.grid_w * 2 && py < state.grid_h * 4 {
                                state.grid.set_dot(px, py).ok();
                                state.grid.set_cell_color(px / 2, py / 4, block_color).ok();
                            }
                        }
                    }
                }
            }
        }
    }

    // Build styled text from grid (chars + colors)
    let unicode_grid = state.grid.to_unicode_grid();
    let mut lines: Vec<ratatui::text::Line> = Vec::with_capacity(state.grid_h);
    for y in 0..state.grid_h {
        let mut spans: Vec<ratatui::text::Span> = Vec::with_capacity(state.grid_w);
        for x in 0..state.grid_w {
            let ch = unicode_grid[y][x];
            let dot_color = state.grid.get_color(x, y);
            let span = if let Some(c) = dot_color {
                Span::styled(
                    ch.to_string(),
                    Style::default().fg(Color::Rgb(c.r, c.g, c.b)),
                )
            } else {
                Span::raw(ch.to_string())
            };
            spans.push(span);
        }
        lines.push(ratatui::text::Line::from(spans));
    }
    let text = ratatui::text::Text::from(lines);
    frame.render_widget(text, area);

    // Overlay labels if enabled
    if state.show_labels {
        for node_idx in state.graph.node_indices() {
            let pos = state.physics.position(node_idx);
            let (px, py) = to_pixel(pos);
            let node = &state.graph[node_idx];
            if !node.label.is_empty() && node.label != "root" && node.label != "topic_0" {
                // Convert braille pixel to character position
                let col = (px as u16) / 2;
                let row = (py as u16) / 4;
                if col < area.width.saturating_sub(2) && row < area.height.saturating_sub(1) {
                    let short_label = if node.label.chars().count() > 15 {
                        let truncated: String = node.label.chars().take(13).collect();
                        format!("{}..", truncated)
                    } else {
                        node.label.clone()
                    };
                    let label_len = short_label.len() as u16 + 2;
                    // Color label to match node animation
                    let label_fg = node_colors.get(&node_idx)
                        .and_then(|c| *c)
                        .map(|c| Color::Rgb(c.r, c.g, c.b))
                        .unwrap_or(Color::Yellow);
                    let span = Span::styled(
                        format!(" {short_label} "),
                        Style::default().fg(label_fg),
                    );
                    frame.render_widget(
                        ratatui::widgets::Paragraph::new(span),
                        Rect::new(col, row, label_len, 1),
                    );
                }
            }
        }
    }

    // HUD: status line at the bottom
    let anim_count = state.animations.values().filter(|a| !matches!(a, NodeAnim::Stable)).count();
    let total_entries: i64 = state.graph.node_indices()
        .filter(|&idx| leaf_nodes.contains(&idx))
        .map(|idx| state.graph[idx].count)
        .sum();
    let status = format!(
        " nodes:{} entries:{} zoom:{:.1}x anims:{} [q]uit [+/-]zoom [arrows]pan [l]abels [e]dges ",
        state.graph.node_count(),
        total_entries,
        state.zoom,
        anim_count,
    );
    let status_widget = ratatui::widgets::Paragraph::new(
        Span::styled(status, Style::default().fg(Color::DarkGray)),
    );
    let bottom = Rect::new(0, area.height.saturating_sub(1), area.width, 1);
    frame.render_widget(status_widget, bottom);
}