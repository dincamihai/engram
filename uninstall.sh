#!/usr/bin/env bash
set -euo pipefail

SETTINGS="$HOME/.claude/settings.local.json"

echo "[engram] Unregistering MCP server..."
claude mcp remove memory --scope user 2>/dev/null || true

echo "[engram] Removing hooks from settings..."
python3 - "$SETTINGS" <<'PYEOF'
import json, sys, os

settings_path = sys.argv[1]

if not os.path.exists(settings_path):
    print("[engram] No settings file found, nothing to clean up.")
    sys.exit(0)

with open(settings_path) as f:
    settings = json.load(f)

# Remove engram permissions
perms = settings.get("permissions", {})
allow = perms.get("allow", [])
allow = [p for p in allow if not p.startswith("mcp__memory__")]
if allow:
    perms["allow"] = allow
else:
    perms.pop("allow", None)
if not perms:
    settings.pop("permissions", None)

# Remove engram hooks from each event
hooks = settings.get("hooks", {})
for event in ["Stop", "PreCompact", "SessionStart", "PostToolUse"]:
    if event not in hooks:
        continue
    entries = hooks[event]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_hooks = entry.get("hooks", [])
        entry["hooks"] = [h for h in entry_hooks if "engram" not in h.get("command", "")]
    # Remove empty matcher entries
    hooks[event] = [e for e in entries if isinstance(e, dict) and e.get("hooks")]
    if not hooks[event]:
        del hooks[event]

if not hooks:
    settings.pop("hooks", None)

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")

print("[engram] Hooks and permissions removed.")
PYEOF

echo "[engram] Done! MCP server and hooks removed."
echo "[engram] Data directory preserved at ~/.engram/ (remove manually if desired)."