# Terminal Animation Research for Engram Viz

## Stack Chosen

| Crate | Role | Why |
|---|---|---|
| **dotmax** v0.1.7 | Braille canvas rendering, AnimationLoop, drawing primitives | High-res braille dots (2x4 per cell), `draw_line()`, `set_dot()`, `draw_circle()`, DifferentialRenderer |
| **ascii-petgraph** v0.2.0 | Force-directed layout (PhysicsEngine) | Incremental `tick()`, `position() -> Vec2`, `PhysicsConfig` tunable, built on petgraph |
| **ratatui** v0.29 | Terminal management, event loop, text overlay | Standard TUI framework, crossterm backend |
| **petgraph** v0.8 | Graph data structure | `DiGraph` for BIRCH tree nodes/edges, `NodeIndex` for stable references |

## Rejected Alternatives

| Crate | Why rejected |
|---|---|
| **img2art** | Python CLI tool, not a Rust library |
| **rustyAscii** | Binary-only player, no library API |
| **drawille** v0.3 | Minimal braille canvas, no animation loop |
| **fdg-sim** | Force-directed layout only (no rendering), 3D positions (Vec3), heavier |
| **tachyonfx** | Ratatui effects (fade, dissolve), could complement but not needed yet |
| **ratatui widgets** | Blocky ASCII/Unicode rendering, not smooth braille |

## Animation Approach

Current implementation uses:
- **Braille dots** (`set_dot()`) for nodes — minimal, fast
- **Braille lines** (`draw_line()`) for edges
- **Force-directed physics** (ascii-petgraph `PhysicsEngine`) with incremental `tick()` at ~30fps
- **ratatui** for terminal setup/teardown, event handling, text overlay

### Node Animations

| Event | Visual Effect |
|---|---|
| **New node** (Growing) | Size pulse (starts big → settles) + blink (alternating visibility) |
| **Removed entries** (Shrinking) | Blink (slower) until animation completes |
| **Branch change** (Vibrating) | Small position jitter (3px sinusoidal offset) on node and siblings |
| **Stable** | Normal rendering, no animation |

### Physics Tuning

```rust
PhysicsConfig {
    spring_constant: 0.08,    // edges pull nodes together
    spring_length: 80.0,      // ideal edge length
    repulsion_constant: 5000.0, // nodes push apart
    gravity: 0.0,             // no downward pull (radial tree)
    damping: 0.85,            // velocity decay
}
```

### Polling

- Every 2 seconds, reopen SQLite and diff nodes
- New nodes: add to graph, start Growing animation, place near parent
- Removed nodes: mark Shrinking, remove after animation completes
- Changed count: Growing (increase) or Shrinking (decrease) + vibrate branch

## Future Ideas

1. **Color schemes** — dotmax supports `draw_line_colored()` and `draw_circle_colored()` with `Color` struct. Could use terminal truecolor for depth-based coloring.
2. **DifferentialRenderer** — dotmax has this for efficient partial redraws. Could skip `grid.clear()` and only redraw changed cells.
3. **Web UI** — serve a local HTML page with canvas/JS for smooth 2D animation with real physics. Would enable mouse interaction.
4. **Sound** — subtle terminal bell on new node creation.
5. **Mouse interaction** — crossterm supports mouse events. Could click a node to see label/content.