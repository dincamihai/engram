#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/target/release/engram"
CLAUDE_HOOKS="$HOME/.claude/hooks"

echo "[engram] Building release binary..."
cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml"

echo "[engram] Registering MCP server with Claude Code..."
claude mcp remove memory --scope user 2>/dev/null || true
claude mcp add --scope user memory -- "$BINARY" serve

echo "[engram] Installing hooks..."
mkdir -p "$CLAUDE_HOOKS"

# SessionStart — load memories
cat > "$CLAUDE_HOOKS/engram-session-start.sh" <<'HOOK'
#!/bin/bash
ENGRAM=~/Repos/engram/target/release/engram
WORKDIR=$(basename "$PWD")
TODAY=$(date +%Y-%m-%d)
MONTH=$(date +"%B %Y")
QUERY="$MONTH recent work context $TODAY $WORKDIR"
RESULT=$($ENGRAM search "$QUERY" --limit 10 2>/dev/null | head -120)
if [ -n "$RESULT" ]; then
  ESCAPED=$(echo "$RESULT" | python3 -c "import sys,json; s=json.dumps(sys.stdin.read()); print(s[1:-1])")
  echo "{\"hookSpecificOutput\":{\"hookEventName\":\"SessionStart\",\"additionalContext\":\"== Engram Memory (auto-loaded) ==\n${ESCAPED}\n== End Engram Memory ==\"}}"
fi
exit 0
HOOK

# PreCompact — save summary before compaction
cat > "$CLAUDE_HOOKS/engram-pre-compact.sh" <<'HOOK'
#!/bin/bash
ENGRAM=~/Repos/engram/target/release/engram
SUMMARY=$(cat | head -c 2000)
if [ -n "$SUMMARY" ]; then
    $ENGRAM queue "$SUMMARY" --source pre-compact 2>/dev/null
fi
echo '{"decision":"approve"}'
exit 0
HOOK

# PostToolUse/Bash — queue git push + jira
cat > "$CLAUDE_HOOKS/engram-post-bash.sh" <<'HOOK'
#!/bin/bash
INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
CWD=$(echo "$INPUT" | jq -r '.cwd // "."' 2>/dev/null)
ENGRAM=~/Repos/engram/target/release/engram
if echo "$CMD" | grep -qE 'git push'; then
    BRANCH=$(cd "$CWD" && git branch --show-current 2>/dev/null)
    MSG=$(cd "$CWD" && git log -1 --format='%s' 2>/dev/null)
    [ -n "$MSG" ] && $ENGRAM queue "Pushed $BRANCH: $MSG" --source git-hook 2>/dev/null
elif echo "$CMD" | grep -qE 'acli jira'; then
    TICKET=$(echo "$CMD" | grep -oE 'L3-[0-9]+|SPOT-[0-9]+')
    [ -n "$TICKET" ] && $ENGRAM queue "Investigated Jira ticket $TICKET" --source jira-hook 2>/dev/null
fi
exit 0
HOOK

# PostToolUse/Edit — queue vault edits
cat > "$CLAUDE_HOOKS/engram-post-edit.sh" <<'HOOK'
#!/bin/bash
INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
ENGRAM=~/Repos/engram/target/release/engram
if echo "$FILE" | grep -q '/exasol/'; then
    CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // empty' 2>/dev/null | head -c 500)
    [ -n "$CONTENT" ] && $ENGRAM queue "$CONTENT" --source vault-hook 2>/dev/null
fi
exit 0
HOOK

# UserPromptSubmit — queue substantial prompts
cat > "$CLAUDE_HOOKS/engram-user-prompt.sh" <<'HOOK'
#!/bin/bash
INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty' 2>/dev/null | head -c 300)
if [ ${#PROMPT} -gt 30 ]; then
    ~/Repos/engram/target/release/engram queue "$PROMPT" --source user-prompt 2>/dev/null
fi
exit 0
HOOK

# Stop — queue session summary
cat > "$CLAUDE_HOOKS/engram-post-stop.sh" <<'HOOK'
#!/bin/bash
INPUT=$(cat)
SUMMARY=$(echo "$INPUT" | jq -r '.transcript_summary // .summary // empty' 2>/dev/null | head -c 500)
if [ -n "$SUMMARY" ]; then
    ~/Repos/engram/target/release/engram queue "$SUMMARY" --source session-end 2>/dev/null
fi
exit 0
HOOK

# SubagentStop — queue agent results
cat > "$CLAUDE_HOOKS/engram-subagent-stop.sh" <<'HOOK'
#!/bin/bash
INPUT=$(cat)
RESULT=$(echo "$INPUT" | jq -r '.result // empty' 2>/dev/null | head -c 500)
if [ -n "$RESULT" ]; then
    ~/Repos/engram/target/release/engram queue "$RESULT" --source subagent 2>/dev/null
fi
exit 0
HOOK

chmod +x "$CLAUDE_HOOKS"/engram-*.sh

echo "[engram] Configuring settings..."
python3 - <<'PYEOF'
import json, os

settings_path = os.path.expanduser("~/.claude/settings.json")
if os.path.exists(settings_path):
    with open(settings_path) as f:
        settings = json.load(f)
else:
    settings = {}

hooks_dir = os.path.expanduser("~/.claude/hooks")
hooks = settings.setdefault("hooks", {})

hooks["SessionStart"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram-session-start.sh", "timeout": 15, "statusMessage": "Loading memories..."}]}]
hooks["PreCompact"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram-pre-compact.sh"}]}]
hooks["Stop"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram-post-stop.sh", "timeout": 5}]}]
hooks["UserPromptSubmit"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram-user-prompt.sh", "timeout": 5}]}]
hooks["SubagentStop"] = [{"hooks": [{"type": "command", "command": f"{hooks_dir}/engram-subagent-stop.sh", "timeout": 5}]}]

post = hooks.setdefault("PostToolUse", [])
# Remove old entries, add new
post[:] = [e for e in post if e.get("matcher") not in ("Bash", "Edit")]
post.append({"matcher": "Bash", "hooks": [{"type": "command", "command": f"{hooks_dir}/engram-post-bash.sh", "timeout": 5}]})
post.append({"matcher": "Edit", "hooks": [{"type": "command", "command": f"{hooks_dir}/engram-post-edit.sh", "timeout": 5}]})

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)

print("[engram] Settings updated.")
PYEOF

echo ""
echo "[engram] Done! No external dependencies needed."
echo "[engram] Models download automatically on first use (~33MB embeddings, ~250MB classifier)."
echo "[engram] Run 'engram rebuild' if migrating from an older version."
