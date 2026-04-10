//! Animated BIRCH tree visualization — Conway-style x-ray of memory.
//!
//! Uses dotmax for braille canvas rendering, ascii-petgraph for force-directed
//! layout, and ratatui for terminal management.

use std::io;
use std::time::Duration;

use ascii_petgraph::physics::{PhysicsConfig, PhysicsEngine};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use dotmax::primitives::draw_line;
use dotmax::BrailleGrid;
use petgraph::graph::DiGraph;
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
    access: i64,
}

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
}

pub fn run_viz(tree: &birch::Tree) -> Result<(), Box<dyn std::error::Error>> {
    // Setup terminal
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Build graph from BIRCH tree
    let size = terminal.size()?;
    let state = build_state(tree, size.width as usize, size.height as usize)?;

    // Run the event loop
    let result = run_loop(&mut terminal, state);

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

    let grid = BrailleGrid::new(width, height)?;

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
    })
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    mut state: VizState,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        // Physics step
        if !state.paused && !state.physics.is_stable() {
            state.physics.tick(&state.graph);
            state.physics.normalize_positions();
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
                state.grid = BrailleGrid::new(new_w, new_h)?;
                state.grid_w = new_w;
                state.grid_h = new_h;
            }
        }
    }
}

fn render_frame(frame: &mut ratatui::Frame, state: &mut VizState) {
    let area = frame.area();
    state.grid.clear();

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

    // Draw edges
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
            draw_line(&mut state.grid, x1, y1, x2, y2).ok();
        }
    }

    // Draw nodes as braille dots
    for node_idx in state.graph.node_indices() {
        let pos = state.physics.position(node_idx);
        let (px, py) = to_pixel(pos);
        let node = &state.graph[node_idx];

        // Bounds check (braille pixels)
        if px < 0 || py < 0 {
            continue;
        }
        let ux = px as usize;
        let uy = py as usize;
        if ux >= state.grid_w * 2 || uy >= state.grid_h * 4 {
            continue;
        }

        if node.depth == 0 {
            // Root node: 3x2 cluster
            for dx in 0..3 {
                for dy in 0..2 {
                    let nx = ux + dx;
                    let ny = uy + dy;
                    if nx < state.grid_w * 2 && ny < state.grid_h * 4 {
                        state.grid.set_dot(nx, ny).ok();
                    }
                }
            }
        } else {
            // Single dot
            state.grid.set_dot(ux, uy).ok();

            // Bigger cluster for nodes with many entries
            if node.count > 5 {
                if ux + 1 < state.grid_w * 2 {
                    state.grid.set_dot(ux + 1, uy).ok();
                }
                if uy + 1 < state.grid_h * 4 {
                    state.grid.set_dot(ux, uy + 1).ok();
                }
                if ux + 1 < state.grid_w * 2 && uy + 1 < state.grid_h * 4 {
                    state.grid.set_dot(ux + 1, uy + 1).ok();
                }
            }
        }
    }

    // Render braille grid to string
    let unicode_grid = state.grid.to_unicode_grid();
    let text: String = unicode_grid
        .iter()
        .map(|row| row.iter().collect::<String>())
        .collect::<Vec<_>>()
        .join("\n");

    let paragraph = ratatui::text::Text::from(text);
    frame.render_widget(paragraph, area);

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
                    let span = Span::styled(
                        format!(" {short_label} "),
                        Style::default().fg(Color::Yellow),
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
    let status = format!(
        " nodes:{} edges:{} zoom:{:.1}x {} [q]uit [r]eset [+/-]zoom [arrows]pan [l]abels [space]pause ",
        state.graph.node_count(),
        state.graph.edge_count(),
        state.zoom,
        if state.paused { "PAUSED" } else if state.physics.is_stable() { "STABLE" } else { "SETTLING" },
    );
    let status_widget = ratatui::widgets::Paragraph::new(
        Span::styled(status, Style::default().fg(Color::DarkGray)),
    );
    let bottom = Rect::new(0, area.height.saturating_sub(1), area.width, 1);
    frame.render_widget(status_widget, bottom);
}