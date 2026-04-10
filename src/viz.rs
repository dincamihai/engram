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
    let tree = birch::Tree::open(db_path, 768, birch::Config::default())
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
    // 1. Compute mean centroid
    let dim = nodes[0].centroid.len();
    let n = nodes.len() as f64;
    let mut mean = vec![0.0f64; dim];
    for node in &nodes {
        for (i, v) in node.centroid.iter().enumerate() {
            mean[i] += *v as f64 / n;
        }
    }

    // 2. Find the two dimensions with highest variance
    let mut variance = vec![0.0f64; dim];
    for node in &nodes {
        for (i, v) in node.centroid.iter().enumerate() {
            let d = *v as f64 - mean[i];
            variance[i] += d * d / n;
        }
    }
    let mut dim_indices: Vec<usize> = (0..dim).collect();
    dim_indices.sort_by(|&a, &b| variance[b].partial_cmp(&variance[a]).unwrap_or(std::cmp::Ordering::Equal));
    let dim_x = dim_indices[0];
    let dim_y = if dim_indices.len() > 1 { dim_indices[1] } else { 0 };

    // 3. Project each node to 2D using top-2 variance dimensions
    let projections: Vec<(f64, f64)> = nodes.iter()
        .map(|node| {
            let x = node.centroid[dim_x] as f64 - mean[dim_x];
            let y = node.centroid[dim_y] as f64 - mean[dim_y];
            (x, y)
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
    })
}

/// Poll the DB for changes and update the graph.
/// Detects: new nodes (coagulate), removed nodes (disperse), changed nodes (pulse).
fn poll_changes(state: &mut VizState, db_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let tree = birch::Tree::open(db_path, 768, birch::Config::default())
        .map_err(|e| format!("reopen tree: {e}"))?;
    let nodes = tree.all_nodes().map_err(|e| format!("all_nodes: {e}"))?;
    let current_ids: std::collections::HashSet<i64> = nodes.iter().map(|n| n.id).collect();

    // Find new nodes (in DB but not in graph)
    let new_ids: Vec<i64> = current_ids.difference(&state.known_ids).copied().collect();
    // Find removed nodes (in graph but not in DB)
    let removed_ids: Vec<i64> = state.known_ids.difference(&current_ids).copied().collect();
    // Find changed nodes (count changed)
    let mut _changed = false;

    if new_ids.is_empty() && removed_ids.is_empty() {
        // Check for count changes on existing nodes — triggers a pulse animation
        for node_idx in state.graph.node_indices() {
            let viz_id = state.graph[node_idx].id;
            if let Some(db_node) = nodes.iter().find(|n| n.id == viz_id) {
                let old_count = state.graph[node_idx].count;
                let new_count = db_node.count;
                if old_count != new_count {
                    state.graph[node_idx].count = new_count;
                    let idx_usize = node_idx.index();
                    if idx_usize < state.animations.len() {
                        // Growing = blink+pulse, shrinking = blink+shrink
                        if new_count > old_count {
                            state.animations.insert(node_idx, NodeAnim::Growing { progress: 0.0 });
                        } else {
                            state.animations.insert(node_idx, NodeAnim::Shrinking { progress: 0.0 });
                        }
                    }
                    _changed = true;
                }
                // Also update label if it changed (e.g. after rebalance)
                if state.graph[node_idx].label != db_node.label {
                    state.graph[node_idx].label = db_node.label.clone();
                    _changed = true;
                }
            }
        }
        state.known_ids = current_ids;
        // No physics ticks — positions are semantic
        return Ok(());
    }

    // Add new nodes with Coagulating animation
    for &id in &new_ids {
        if let Some(node) = nodes.iter().find(|n| n.id == id) {
            let access = tree.node_total_access(node.id).unwrap_or(0);
            let age_hours = tree.node_freshness_hours(node.id)
                .unwrap_or(None)
                .unwrap_or(0.0);
            let freshness = (1.0 / (1.0 + age_hours / 72.0)) as f32;
            let consolidated = tree.node_consolidated_count(node.id).unwrap_or(0);
            let unproven = tree.node_unproven_count(node.id).unwrap_or(0);

            // Compute semantic position for new node using same projection
            // Use centroid dimensions 0 and 1 as simple fallback for new nodes
            let sx = if node.centroid.len() > 0 { node.centroid[0] as f64 } else { 0.0 };
            let sy = if node.centroid.len() > 1 { node.centroid[1] as f64 } else { 0.0 };

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
            let idx = state.graph.add_node(viz_node);
            // Add edge to parent if parent exists in graph
            if let Some(parent_id) = node.parent_id {
                if let Some(parent_idx) = state.graph.node_indices().find(|&ni| state.graph[ni].id == parent_id) {
                    state.graph.add_edge(parent_idx, idx, ());
                }
            }
            state.animations.insert(idx, NodeAnim::Growing { progress: 0.0 });
        }
    }

    // Mark removed nodes as Shrinking
    for &id in &removed_ids {
        if let Some(idx) = state.graph.node_indices().find(|&ni| state.graph[ni].id == id) {
            state.animations.insert(idx, NodeAnim::Shrinking { progress: 0.0 });
        }
    }

    state.known_ids = current_ids;

    // Place new nodes at their semantic position
    for &id in &new_ids {
        if let Some(idx) = state.graph.node_indices().find(|&ni| state.graph[ni].id == id) {
            let node = &state.graph[idx];
            let new_pos = Vec2::new(node.semantic_x * 100.0, node.semantic_y * 100.0);

            // Insert position into physics engine
            // We need to extend the nodes vec - pad with positions matching graph order
            let idx_i = idx.index();
            while state.physics.nodes.len() <= idx_i {
                state.physics.nodes.push(ascii_petgraph::physics::NodePhysics::new(0.0, 0.0));
            }
            state.physics.nodes[idx_i].position = new_pos;
            state.physics.nodes[idx_i].velocity = Vec2::new(0.0, 0.0);
        }
    }

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
            render_frame(frame, &mut state);
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
                        KeyCode::Up => state.pan_y -= 10.0,
                        KeyCode::Down => state.pan_y += 10.0,
                        KeyCode::Left => state.pan_x -= 10.0,
                        KeyCode::Right => state.pan_x += 10.0,
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

fn render_frame(frame: &mut ratatui::Frame, state: &mut VizState) {
    let area = frame.area();
    state.grid.clear();
    state.grid.clear_colors();

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

    // Draw atmosphere around all nodes — color encodes cluster health
    // Leaf nodes: health from own freshness + proven ratio
    // Structural nodes: average health of their children
    for node_idx in state.graph.node_indices() {
        let node = &state.graph[node_idx];
        let is_leaf = leaf_nodes.contains(&node_idx);

        let pos = state.physics.position(node_idx);
        let (px, py) = to_pixel(pos);
        if px < 0 || py < 0 { continue; }
        let ux = px as usize;
        let uy = py as usize;
        if ux >= state.grid_w * 2 || uy >= state.grid_h * 4 { continue; }

        // Health score: 0.0 = danger, 1.0 = healthy
        let health = if is_leaf {
            if node.count <= 0 { continue; }
            let proven_ratio = 1.0 - (node.unproven as f64 / node.count as f64);
            (node.freshness as f64 * 0.5 + proven_ratio * 0.5).clamp(0.0, 1.0)
        } else {
            // Structural: average health of child nodes
            let children: Vec<NodeIndex> = state.graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                .collect();
            if children.is_empty() { continue; }
            let sum: f64 = children.iter().map(|&c| {
                let cn = &state.graph[c];
                let pr = if cn.count > 0 { 1.0 - (cn.unproven as f64 / cn.count as f64) } else { 1.0 };
                (cn.freshness as f64 * 0.5 + pr * 0.5).clamp(0.0, 1.0)
            }).sum();
            sum / children.len() as f64
        };

        // Color: teal (healthy) → amber (warning) → magenta (danger)
        // Chosen for contrast on dark terminals
        let atmo_color = if health > 0.6 {
            // Teal/cyan — healthy, calm
            DotColor::rgb(40, 200, 180)
        } else if health > 0.3 {
            // Amber/orange — warning
            DotColor::rgb(240, 180, 50)
        } else {
            // Magenta/pink — danger, visible on dark bg
            DotColor::rgb(240, 60, 140)
        };

        // Atmosphere is steady — only brightens when the node is animating (add/delete)
        let anim = state.animations.get(&node_idx).unwrap_or(&NodeAnim::Stable);
        let breath = match anim {
            NodeAnim::Growing { progress } => 0.8 + 0.4 * ((progress * std::f32::consts::PI * 4.0).sin() as f64).abs(),
            NodeAnim::Shrinking { progress } => 0.8 + 0.4 * ((progress * std::f32::consts::PI * 4.0).sin() as f64).abs(),
            _ => 0.7, // steady baseline
        };

        // Atmosphere radius based on node size
        let count_size = 1 + ((node.count as f64).ln().max(0.0) / 1.5) as usize;
        let atmo_radius = (count_size as f64 + 5.0 + (node.count as f64).sqrt() * 2.0) as usize;

        // Draw dense cloud of dots filling the atmosphere area
        let num_atmo_dots = 40 + node.count.min(30) as usize * 3;
        for i in 0..num_atmo_dots {
            // Distribute dots across the full disk, denser toward the edge
            let angle = (i as f64 / num_atmo_dots as f64) * std::f64::consts::TAU
                + (node.id as f64 * 0.5) + (i as f64 * 1.618); // golden ratio scatter
            let r_variation = 0.3 + 0.7 * ((i as f64 * 2.7 + state.frame as f64 * 0.005).sin() * 0.5 + 0.5);
            let r = atmo_radius as f64 * r_variation;

            let ax = (ux as f64 + r * angle.cos()).round() as usize;
            let ay = (uy as f64 + r * angle.sin()).round() as usize;
            if ax >= state.grid_w * 2 || ay >= state.grid_h * 4 { continue; }

            // Apply breathing to atmosphere brightness
            let dot_color = DotColor::rgb(
                (atmo_color.r as f64 * breath * 0.6) as u8,
                (atmo_color.g as f64 * breath * 0.6) as u8,
                (atmo_color.b as f64 * breath * 0.6) as u8,
            );
            state.grid.set_dot(ax, ay).ok();
            state.grid.set_cell_color(ax / 2, ay / 4, dot_color).ok();
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

        // Draw orbiting data point satellites around leaf nodes
        // Fresh memories orbit actively; old memories slow down and become static
        // Consolidated memories appear as clumps (2-3 dots stuck together)
        // Unproven (never-accessed) memories orbit further out — loosely held
        if is_leaf && node.count > 0 {
            let proven_count = (node.count - node.consolidated - node.unproven).max(0) as usize;
            let unproven_count = node.unproven.min(24) as usize;
            let consol_count = node.consolidated.min(24) as usize;
            let total_points = (proven_count + unproven_count + consol_count).min(24);
            let inner_radius = (dot_size as f64) + 3.0 + (node.count as f64).sqrt() * 1.2;
            let outer_radius = inner_radius * 1.8; // unproven orbit 80% further out

            // Entropy/temperature: fresh = hot vibrating particles, old = cooled, still
            let temperature = node.freshness as f64; // 0.0 (frozen) → 1.0 (boiling)
            let jitter_amp = 0.5 + temperature * 4.0; // 0.5px (cold) → 4.5px (hot)

            for i in 0..total_points {
                // Classify: proven first, then unproven, then consolidated
                let is_unproven = i >= proven_count && i < proven_count + unproven_count;
                let is_consolidated = i >= proven_count + unproven_count;

                // Fixed base position on a ring (deterministic, no orbit)
                let base_angle = (i as f64 / total_points as f64) * std::f64::consts::TAU
                    + (node.id as f64 * 0.7); // per-node offset

                // Proven = inner, unproven = outer, consolidated = inner
                let base_r = if is_unproven { outer_radius } else { inner_radius };

                // Brownian jitter: random-looking vibration from fast sin/cos with unique seeds
                let seed = i as f64 * 7.3 + node.id as f64 * 3.1;
                let jx = jitter_amp * (state.frame as f64 * 0.15 + seed).sin()
                       * (state.frame as f64 * 0.23 + seed * 1.7).cos();
                let jy = jitter_amp * (state.frame as f64 * 0.19 + seed * 2.3).cos()
                       * (state.frame as f64 * 0.13 + seed * 0.9).sin();

                let cx = ux as f64 + (dot_size as f64 / 2.0) + base_r * base_angle.cos() + jx;
                let cy = uy as f64 + (dot_size as f64 / 2.0) + base_r * base_angle.sin() + jy;

                if is_consolidated {
                    // Draw clump: 3 dots stuck together in a tight triangle
                    let clump_offsets: [(f64, f64); 3] = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
                    for (dx, dy) in &clump_offsets {
                        let sx = (cx + dx).round() as usize;
                        let sy = (cy + dy).round() as usize;
                        if sx < state.grid_w * 2 && sy < state.grid_h * 4 {
                            let sat_color = DotColor::rgb(
                                (base_depth_color.r as f64 * 0.6) as u8,
                                (base_depth_color.g as f64 * 0.6) as u8,
                                (base_depth_color.b as f64 * 0.7) as u8,
                            );
                            state.grid.set_dot(sx, sy).ok();
                            state.grid.set_cell_color(sx / 2, sy / 4, sat_color).ok();
                        }
                    }
                } else {
                    // Single dot satellite
                    let sx = cx.round() as usize;
                    let sy = cy.round() as usize;
                    if sx < state.grid_w * 2 && sy < state.grid_h * 4 {
                        let sat_bright = 0.5 + 0.5 * ((base_angle * 2.0 + state.frame as f64 * 0.1).sin()).abs();
                        let sat_color = DotColor::rgb(
                            (base_depth_color.r as f64 * (0.5 + 0.5 * sat_bright)) as u8,
                            (base_depth_color.g as f64 * (0.5 + 0.5 * sat_bright)) as u8,
                            (base_depth_color.b as f64 * (0.5 + 0.5 * sat_bright)) as u8,
                        );
                        state.grid.set_dot(sx, sy).ok();
                        state.grid.set_cell_color(sx / 2, sy / 4, sat_color).ok();
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
                    let short_label = if node.label.len() > 15 {
                        format!("{}..", &node.label[..13])
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