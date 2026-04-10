# Engram Viz Test Script

## Setup

Open `engram viz` in terminal 1, then run store/delete commands in terminal 2.

## Test 1: Add memories (watch coagulation)

```bash
engram store "Test viz: Alice works on the database cluster" --source "viz-test"
engram store "Test viz: Bob manages the API gateway" --source "viz-test"
engram store "Test viz: Charlie deployed the monitoring stack" --source "viz-test"
engram store "Test viz: Diana wrote the authentication module" --source "viz-test"
```

Expected: New dots appear with coagulation animation (scattered pixels converging to center).
Polling interval is 2 seconds.

## Test 2: Wait and observe

Wait ~5 seconds. The graph should settle as the force-directed layout adjusts.

## Test 3: Delete memories (watch dispersion)

```bash
engram forget --source "%viz-test%"
```

Expected: Dots disperse (pixels scatter outward) then vanish.

## Test 4: Add more, then delete individually

```bash
engram store "Test viz: Eve handles incident response" --source "viz-test"
engram store "Test viz: Frank runs the CI pipeline" --source "viz-test"
# Note the entry IDs from the output, then:
engram forget <ID1>
engram forget <ID2>
```

Expected: Each deletion triggers dispersion animation for that node.

## Cleanup

```bash
engram forget --source "%viz-test%"
```

## Viz Controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `r` | Reset layout |
| `+` / `-` | Zoom |
| Arrows | Pan |
| `l` | Toggle labels |
| `Space` | Pause/resume physics |
