#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/target/release/engram"
HOOKS_DIR="$SCRIPT_DIR/hooks"
SETTINGS="$HOME/.claude/settings.local.json"

echo "[engram] Building release binary..."
cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml"

echo "[engram] Registering MCP server with Claude Code..."
claude mcp remove memory --scope user 2>/dev/null || true
claude mcp add --scope user memory \
  -e OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}" \
  -e OLLAMA_EMBED_MODEL="${OLLAMA_EMBED_MODEL:-embeddinggemma}" \
  -- "$BINARY" serve

echo "[engram] Configuring hooks and permissions..."
mkdir -p "$HOME/.claude"

python3 - "$SETTINGS" "$HOOKS_DIR" <<'PYEOF'
import json, sys, os

settings_path = sys.argv[1]
hooks_dir = sys.argv[2]

if os.path.exists(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)
else:
    settings = {}

# Permissions
perms = settings.setdefault("permissions", {})
allow = perms.setdefault("allow", [])
for tool in ["mcp__memory__engram_store", "mcp__memory__engram_search"]:
    if tool not in allow:
        allow.append(tool)

# Hooks
hooks = settings.setdefault("hooks", {})

# Stop — save session context
stop_hooks = hooks.setdefault("Stop", [{"hooks": []}])
if not any("engram_save_hook" in h.get("command", "") for hh in stop_hooks for h in hh.get("hooks", []) if isinstance(hh, dict)):
    pass
stop_list = stop_hooks[0] if stop_hooks and isinstance(stop_hooks[0], dict) else {"hooks": []}
if not isinstance(stop_list, dict):
    stop_list = {"hooks": []}
stop_hooks_list = stop_list.setdefault("hooks", [])
if not any("engram_save_hook" in h.get("command", "") for h in stop_hooks_list):
    stop_hooks_list.append({"type": "command", "command": f"{hooks_dir}/engram_save_hook.sh"})

# PreCompact — store summary before compaction
precompact_hooks = hooks.setdefault("PreCompact", [{"hooks": []}])
pc_list = precompact_hooks[0] if precompact_hooks and isinstance(precompact_hooks[0], dict) else {"hooks": []}
if not isinstance(pc_list, dict):
    pc_list = {"hooks": []}
pc_hooks_list = pc_list.setdefault("hooks", [])
if not any("engram_precompact_hook" in h.get("command", "") for h in pc_hooks_list):
    pc_hooks_list.append({"type": "command", "command": f"{hooks_dir}/engram_precompact_hook.sh"})

# SessionStart — inject memories
session_hooks = hooks.setdefault("SessionStart", [{"hooks": []}])
ss_list = session_hooks[0] if session_hooks and isinstance(session_hooks[0], dict) else {"hooks": []}
if not isinstance(ss_list, dict):
    ss_list = {"hooks": []}
ss_hooks_list = ss_list.setdefault("hooks", [])
if not any("engram_session_start_hook" in h.get("command", "") for h in ss_hooks_list):
    ss_hooks_list.append({"type": "command", "command": f"{hooks_dir}/engram_session_start_hook.sh", "timeout": 15})

# PostToolUse — store + activate (Bash and Agent)
post_list = hooks.setdefault("PostToolUse", [])
for tool_name in ["Bash", "Agent"]:
    matcher = {"toolName": tool_name}
    existing = next((e for e in post_list if isinstance(e, dict) and e.get("matcher", {}).get("toolName") == tool_name), None)
    if existing:
        entry_hooks = existing.setdefault("hooks", [])
        if not any("engram_post_tool_use_hook" in h.get("command", "") for h in entry_hooks):
            entry_hooks.append({"type": "command", "command": f"{hooks_dir}/engram_post_tool_use_hook.sh"})
    else:
        post_list.append({
            "matcher": {"toolName": tool_name},
            "hooks": [{"type": "command", "command": f"{hooks_dir}/engram_post_tool_use_hook.sh"}]
        })

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")

print("[engram] Settings updated.")
PYEOF

echo ""
echo "[engram] Done! MCP server registered and hooks configured."
echo "[engram] Make sure Ollama is running with: ollama pull embeddinggemma"