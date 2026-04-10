//! Animated BIRCH tree visualization — Conway-style x-ray of memory.
//!
//! Uses dotmax for braille canvas rendering, ascii-petgraph for force-directed
//! layout, and ratatui for terminal management.

use std::collections::HashMap;
use std::io;
use std::time::{Duration, Instant};

use ascii_petgraph::physics::{PhysicsConfig, PhysicsEngine, Vec2};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use dotmax::grid::Color as DotColor;
use dotmax::primitives::{draw_line, draw_line_colored};
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

const ANIM_SPEED: f32 = 0.008; // progress per frame (~4s at 30fps for full cycle)
const VIBRATE_AMPLITUDE: f64 = 3.0; // pixels of shake

/// State for the visualization.
struct VizState {
    graph: DiGraph<VizNode, ()>,
    physics: PhysicsEngine,
    grid: BrailleGrid,
    zoom: f64,
    pan_x: f64,
    pan_y: f64,
    show_labels: bool,
    paused: bool,
    grid_w: usize,
    grid_h: usize,
    /// Per-node animation state (keyed by petgraph NodeIndex).
    animations: HashMap<NodeIndex, NodeAnim>,
    /// Last known set of BIRCH node IDs (for polling diff).
    known_ids: std::collections::HashSet<i64>,
    /// When we last polled the DB for changes.
    last_poll: Instant,
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

    // Add nodes
    for node in &nodes {
        let access = tree.node_total_access(node.id).unwrap_or(0);
        let viz_node = VizNode {
            id: node.id,
            label: node.label.clone(),
            count: node.count,
            depth: node.depth,
            access,
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

    // Setup physics engine — tuned for tree layout
    let physics_config = PhysicsConfig {
        spring_constant: 0.08,
        spring_length: 80.0,
        repulsion_constant: 5000.0,
        gravity: 0.0, // no downward gravity — radial tree
        damping: 0.85,
        dt: 1.0,
        velocity_threshold: 0.1,
        max_iterations: 5000,
    };
    let mut physics = PhysicsEngine::new(&graph, physics_config);
    physics.normalize_positions();

    // Run initial ticks so nodes spread out
    for _ in 0..20 {
        physics.tick(&graph);
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
        zoom: 1.0,
        pan_x: 0.0,
        pan_y: 0.0,
        show_labels: false,
        paused: false,
        grid_w: width,
        grid_h: height,
        animations,
        known_ids,
        last_poll: Instant::now(),
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
    let mut changed = false;

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
                    changed = true;
                }
                // Also update label if it changed (e.g. after rebalance)
                if state.graph[node_idx].label != db_node.label {
                    state.graph[node_idx].label = db_node.label.clone();
                    changed = true;
                }
            }
        }
        state.known_ids = current_ids;
        if changed {
            // Small nudge: give a few physics ticks to absorb the change
            for _ in 0..3 {
                state.physics.tick(&state.graph);
            }
        }
        return Ok(());
    }

    // Add new nodes with Coagulating animation
    for &id in &new_ids {
        if let Some(node) = nodes.iter().find(|n| n.id == id) {
            let access = tree.node_total_access(node.id).unwrap_or(0);
            let viz_node = VizNode {
                id: node.id,
                label: node.label.clone(),
                count: node.count,
                depth: node.depth,
                access,
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

    // Instead of rebuilding physics, add new node positions near their parents
    // and let the existing simulation absorb them
    for &id in &new_ids {
        if let Some(idx) = state.graph.node_indices().find(|&ni| state.graph[ni].id == id) {
            // Find parent position and place new node nearby
            let parent_pos = state.graph.neighbors_directed(idx, petgraph::Direction::Incoming)
                .next()
                .and_then(|pidx| {
                    let p_i = pidx.index();
                    if p_i < state.physics.nodes.len() {
                        Some(state.physics.nodes[p_i].position)
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| {
                    // No parent or parent not found: random position near center
                    let angle = (id as f64).to_radians();
                    Vec2::new(angle.cos() * 50.0, angle.sin() * 50.0)
                });

            // Place near parent with small random offset
            let offset = Vec2::new(
                ((id * 17) % 20) as f64 - 10.0,
                ((id * 31) % 20) as f64 - 10.0,
            );
            let new_pos = parent_pos + offset;

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

        // Physics step
        if !state.paused && !state.physics.is_stable() {
            state.physics.tick(&state.graph);
            state.physics.normalize_positions();
        }

        // Poll DB for changes every 2 seconds
        if state.last_poll.elapsed() >= Duration::from_secs(1) {
            if let Err(e) = poll_changes(&mut state, db_path) {
                // Don't crash on poll errors, just log
                eprintln!("[engram viz] poll error: {e}");
            }
            state.last_poll = Instant::now();
        }

        // Render
        terminal.draw(|frame| {
            render_frame(frame, &mut state);
        })?;

        // Handle events
        if event::poll(Duration::from_millis(33))? {
            if let Event::Key(key) = event::read()? {
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
                    KeyCode::Char(' ') => state.paused = !state.paused,
                    _ => {}
                }
            }
        }

        // Resize grid if terminal size changed
        let size = terminal.size()?;
        let new_w = size.width as usize;
        let new_h = size.height as usize;
        if new_w != state.grid_w || new_h != state.grid_h {
            if new_w > 2 && new_h > 2 {
                let mut new_grid = BrailleGrid::new(new_w, new_h)?;
                new_grid.enable_color_support();
                state.grid = new_grid;
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

    // Determine node colors based on animation state
    let node_colors: HashMap<NodeIndex, Option<DotColor>> = state.graph.node_indices()
        .map(|idx| {
            let anim = state.animations.get(&idx).unwrap_or(&NodeAnim::Stable);
            let color = match anim {
                NodeAnim::Growing { progress } => {
                    // Bright green fading to default as progress → 1
                    let t = *progress;
                    Some(DotColor::rgb(
                        (30.0 + 170.0 * t) as u8,   // R: 30 → 200
                        (255.0 - 55.0 * t) as u8,    // G: 255 → 200
                        (50.0 + 150.0 * t) as u8,    // B: 50 → 200
                    ))
                }
                NodeAnim::Shrinking { progress } => {
                    // Bright red fading to dim as progress → 1
                    let t = *progress;
                    Some(DotColor::rgb(
                        (255.0 * (1.0 - t * 0.7)) as u8, // R: 255 → 76
                        (40.0 * (1.0 - t)) as u8,         // G: 40 → 0
                        (20.0 * (1.0 - t)) as u8,          // B: 20 → 0
                    ))
                }
                NodeAnim::Vibrating { progress } => {
                    // Subtle blue tint
                    let t = *progress;
                    Some(DotColor::rgb(
                        (100.0 + 80.0 * t) as u8,
                        (100.0 + 80.0 * t) as u8,
                        (200.0 - 20.0 * t) as u8,
                    ))
                }
                NodeAnim::Stable => None,
            };
            (idx, color)
        })
        .collect();

    // Draw edges — colored if either endpoint is animated
    for edge in state.graph.edge_references() {
        let source_pos = state.physics.position(edge.source());
        let target_pos = state.physics.position(edge.target());
        let (x1, y1) = to_pixel(source_pos);
        let (x2, y2) = to_pixel(target_pos);

        // Only draw if both endpoints are in bounds
        if x1 >= 0 && y1 >= 0 && x2 >= 0 && y2 >= 0
            && (x1 as usize) < state.grid_w * 2
            && (y1 as usize) < state.grid_h * 4
            && (x2 as usize) < state.grid_w * 2
            && (y2 as usize) < state.grid_h * 4
        {
            // Color edge if either endpoint is animated
            let edge_color = node_colors.get(&edge.source())
                .or_else(|| node_colors.get(&edge.target()))
                .and_then(|c| *c);

            if let Some(color) = edge_color {
                draw_line_colored(&mut state.grid, x1, y1, x2, y2, color, None).ok();
            } else {
                draw_line(&mut state.grid, x1, y1, x2, y2).ok();
            }
        }
    }

    // Draw nodes as braille dots with color
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
            NodeAnim::Growing { progress } => ((1.0 - *progress) * 2.0) as usize,
            NodeAnim::Shrinking { progress } => ((1.0 - *progress) * 1.5) as usize,
            _ => 0,
        };
        let dot_size = if node.depth == 0 { 3 + anim_boost } else { 1 + anim_boost };

        for dx in 0..dot_size {
            for dy in 0..dot_size.min(2) {
                let nx = ux + dx;
                let ny = uy + dy;
                if nx < state.grid_w * 2 && ny < state.grid_h * 4 {
                    state.grid.set_dot(nx, ny).ok();
                    if let Some(color) = node_color {
                        // Braille pixel (nx, ny) → cell (nx/2, ny/4)
                        let cell_x = nx / 2;
                        let cell_y = ny / 4;
                        state.grid.set_cell_color(cell_x, cell_y, color).ok();
                    }
                }
            }
        }

        // Extra dots for big clusters
        if node.count > 5 && dot_size <= 1 {
            if ux + 1 < state.grid_w * 2 {
                state.grid.set_dot(ux + 1, uy).ok();
                if let Some(color) = node_color {
                    state.grid.set_cell_color((ux + 1) / 2, uy / 4, color).ok();
                }
            }
            if uy + 1 < state.grid_h * 4 {
                state.grid.set_dot(ux, uy + 1).ok();
                if let Some(color) = node_color {
                    state.grid.set_cell_color(ux / 2, (uy + 1) / 4, color).ok();
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
    let status = format!(
        " nodes:{} edges:{} zoom:{:.1}x {} anims:{} [q]uit [r]eset [+/-]zoom [arrows]pan [l]abels [space]pause ",
        state.graph.node_count(),
        state.graph.edge_count(),
        state.zoom,
        if state.paused { "PAUSED" } else if state.physics.is_stable() { "STABLE" } else { "SETTLING" },
        anim_count,
    );
    let status_widget = ratatui::widgets::Paragraph::new(
        Span::styled(status, Style::default().fg(Color::DarkGray)),
    );
    let bottom = Rect::new(0, area.height.saturating_sub(1), area.width, 1);
    frame.render_widget(status_widget, bottom);
}